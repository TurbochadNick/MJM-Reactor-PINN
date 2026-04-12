#!/usr/bin/env python3
"""Benchmark-aware upgrade of the legacy MJM 2-group solver.

This module preserves the legacy solver as a reference and implements a more
transparent v2 path with:

- corrected power normalization using group-wise fission sources
- explicit extrapolated-boundary leakage treatment
- benchmark-calibrated delayed-neutron reduction for circulating fuel
- field exporters for phi1(r,z), phi2(r,z), total flux, and power density
- temperature sweeps and machine-readable dataset generation

The cross-section model remains a homogenized 2-group approximation. Where the
repo lacks transport-derived constants, this module applies benchmark-guided
corrections and records that provenance in exported metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve

from monster_v2_package.leu_transport_xs import (
    DEFAULT_LEU_XS_LIBRARY_PATH,
    LEU_TRANSPORT_PROXY_LIBRARY_ID,
    REFERENCE_LATTICE_FUEL_VOLUME_FRACTION,
    REFERENCE_LATTICE_GRAPHITE_VOLUME_FRACTION,
    detect_transport_tooling,
    resolve_transport_proxy_xs,
)
from legacy.v1.mjm_solver_v1 import (
    BARN,
    E_FISSION,
    FLiBeProperties,
    IsotopeXS,
    MEV_TO_J,
    N_A,
    get_xs_database,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "monster_v2_package" / "benchmarks" / "msre_fuel_c_reference.json"
BENCHMARK_DATA = json.loads(BENCHMARK_PATH.read_text())
FUEL_C = BENCHMARK_DATA["fuel_c"]

THERMAL_FLUX_TARGET = FUEL_C["thermal_fluxes_at_10MW"]["maximum_n_cm2_s"]
ALPHA_TARGET_PCM_PER_K = FUEL_C["reactivity_coefficients"]["total_temperature_pcm_per_K"]
DELAYED_RATIO_TARGET = FUEL_C["delayed_neutrons"]["beta_ratio_circulating_to_static"]
BENCHMARK_STATIC_BETA = FUEL_C["delayed_neutrons"]["beta_eff_static"]
BENCHMARK_CIRCULATING_BETA = FUEL_C["delayed_neutrons"]["beta_eff_circulating"]
BENCHMARK_PROMPT_LIFETIME = FUEL_C["kinetics"]["prompt_neutron_lifetime_s"]
LEU_MIN_ENRICHMENT = 0.03
LEU_MAX_ENRICHMENT = 0.05
LEU_DESIGN_MODE = "leu_design"
FUEL_C_BENCHMARK_MODE = "fuel_c_benchmark"

# Explicit 2-group interpretation used for reporting, exported metadata, and
# presentation support. These bounds make the fast/thermal split explicit in the
# codebase, but the current homogenized XS set is still benchmark-guided rather
# than transport-regenerated for these exact cutoffs.
ENERGY_GROUPS = {
    "group_1_fast": {
        "symbol": "phi1",
        "name": "Fast neutron group",
        "energy_range_eV": {
            "lower": 1.0e3,
            "upper": 1.0e7,
        },
        "energy_range_text": "approximately 1 keV to 10 MeV",
        "purpose": "Represents the higher-energy neutron population in the 2-group diffusion model.",
    },
    "group_2_thermal": {
        "symbol": "phi2",
        "name": "Thermal neutron group",
        "energy_range_eV": {
            "lower": 2.5e-2,
            "upper": 1.0e3,
        },
        "energy_range_text": "approximately 0.025 eV to 1 keV",
        "purpose": "Represents the lower-energy neutron population in the 2-group diffusion model.",
    },
    "note": (
        "These are explicit assumed 2-group reporting bounds for this project. "
        "The solver cross sections are not transport-regenerated from a library "
        "tied to these exact cutoffs."
    ),
}


@dataclass(frozen=True)
class DelayedNeutronGroupV2:
    group: int
    half_life_s: float
    beta_i: float
    mean_energy_MeV: float
    age_cm2: float
    beta_i_star_static: float
    beta_i_star_circulating: float


DELAYED_GROUPS_V2 = [
    DelayedNeutronGroupV2(
        group=item["group"],
        half_life_s=item["half_life_s"],
        beta_i=item["beta_i"],
        mean_energy_MeV=item["mean_energy_MeV"],
        age_cm2=item["age_cm2"],
        beta_i_star_static=item["beta_i_star_static"],
        beta_i_star_circulating=item["beta_i_star_circulating"],
    )
    for item in FUEL_C["delayed_neutrons"]["groups"]
]


def _fit_delayed_penalty_constant() -> float:
    """Fit the benchmark penalty constant once from Fuel C reference data."""

    beta_static = np.array([g.beta_i_star_static for g in DELAYED_GROUPS_V2], dtype=float)
    age = np.array([g.age_cm2 for g in DELAYED_GROUPS_V2], dtype=float)
    half_lives = np.array([g.half_life_s for g in DELAYED_GROUPS_V2], dtype=float)
    ref_radius_cm = FUEL_C["geometry"]["delayed_neutron_model_radius_in"] * 2.54
    ref_height_cm = FUEL_C["geometry"]["delayed_neutron_model_height_in"] * 2.54
    bg2_ref = (2.405 / ref_radius_cm) ** 2 + (math.pi / ref_height_cm) ** 2
    t_core = FUEL_C["delayed_neutrons"]["core_residence_s"]
    t_loop = FUEL_C["delayed_neutrons"]["external_loop_residence_s"]
    target = DELAYED_RATIO_TARGET

    def predicted_ratio(penalty: float) -> float:
        ratios = []
        for half_life, age_i in zip(half_lives, age, strict=True):
            lam = math.log(2.0) / half_life
            in_core = (1.0 - math.exp(-lam * t_core)) / (1.0 - math.exp(-lam * (t_core + t_loop)))
            leakage_penalty = math.exp(-penalty * age_i * bg2_ref * (1.0 - in_core))
            ratios.append(in_core * leakage_penalty)
        ratios_arr = np.array(ratios, dtype=float)
        return float(np.sum(beta_static * ratios_arr) / np.sum(beta_static))

    lo = 0.0
    hi = 25.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if predicted_ratio(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


DELAYED_LEAKAGE_PENALTY = _fit_delayed_penalty_constant()


@dataclass
class XSCalibration:
    """Repository-local benchmark-guided corrections for the homogenized model."""

    benchmark_fuel_volume_fraction: float = REFERENCE_LATTICE_FUEL_VOLUME_FRACTION
    benchmark_graphite_volume_fraction: float = REFERENCE_LATTICE_GRAPHITE_VOLUME_FRACTION
    thermal_fission_factor: float = 0.82
    thermal_absorption_factor: float = 1.18
    fast_absorption_factor: float = 1.10
    downscatter_base: float = 0.28
    downscatter_graphite_gain: float = 0.08
    downscatter_water_gain: float = 0.18
    d2_ref_cm: float = 0.42
    d2_temp_coeff: float = 0.06
    s12_temp_coeff: float = 0.32
    fast_doppler_coeff: float = 0.24
    thermal_fission_temp_coeff: float = 0.08
    extrapolation_factor: float = 2.131


CALIBRATION = XSCalibration()


def get_xs_database_v2() -> Dict[str, IsotopeXS]:
    """Extend the legacy library with graphite as the default moderator."""

    xs_db = get_xs_database()
    xs_db["C12"] = IsotopeXS(
        "C-12",
        12,
        sigma_f1=0.0,
        sigma_c1=0.003 * BARN,
        sigma_s1=4.80 * BARN,
        sigma_f2=0.0,
        sigma_c2=0.0035 * BARN,
        sigma_s2=4.70 * BARN,
        nu1=0.0,
        nu2=0.0,
    )
    return xs_db


@dataclass
class HomogenizedMSRMaterial:
    """Salt + moderator homogenization with benchmark-guided corrections.

    This keeps the legacy salt chemistry as the base input model while adding
    the graphite-dominated volume fractions called out in ORNL-TM-730.
    """

    enrichment: float = LEU_MAX_ENRICHMENT
    uf4_mol_frac: float = 0.04
    temperature: float = 900.0
    water_vol_frac: float = 0.0
    fuel_volume_fraction: float = CALIBRATION.benchmark_fuel_volume_fraction
    graphite_density_gcc: float = 1.86
    graphite_temperature: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    M_Li7: float = 7.016
    M_Be9: float = 9.012
    M_F19: float = 18.998
    M_U235: float = 235.044
    M_U238: float = 238.051
    M_O16: float = 15.999
    M_H1: float = 1.008
    M_C12: float = 12.011

    def __post_init__(self) -> None:
        if self.graphite_temperature is None:
            self.graphite_temperature = float(self.temperature)

    @property
    def graphite_volume_fraction(self) -> float:
        return max(0.0, 1.0 - self.fuel_volume_fraction - self.water_vol_frac)

    def salt_density_gcc(self) -> float:
        rho_base = FLiBeProperties.density_gcc(self.temperature)
        return rho_base * (1.0 + 1.5 * self.uf4_mol_frac)

    def volume_fractions(self) -> Dict[str, float]:
        return {
            "fuel": self.fuel_volume_fraction,
            "graphite": self.graphite_volume_fraction,
            "water": self.water_vol_frac,
        }

    def compute_number_densities(self) -> Dict[str, float]:
        fuel_frac = self.fuel_volume_fraction
        graphite_frac = self.graphite_volume_fraction
        water_frac = self.water_vol_frac

        x = self.uf4_mol_frac
        enr = self.enrichment
        molar_mass_flibe = 2.0 * self.M_Li7 + self.M_Be9 + 4.0 * self.M_F19
        molar_mass_u_avg = enr * self.M_U235 + (1.0 - enr) * self.M_U238
        molar_mass_uf4 = molar_mass_u_avg + 4.0 * self.M_F19
        molar_mass_mix = (1.0 - x) * molar_mass_flibe + x * molar_mass_uf4

        salt_density = self.salt_density_gcc()
        n_mix = salt_density * N_A / molar_mass_mix

        rho_water = max(0.7, 1.0 - 4.0e-4 * (self.temperature - 293.15))
        n_h2o = rho_water * N_A / (2.0 * self.M_H1 + self.M_O16)

        return {
            "U235": fuel_frac * enr * x * n_mix,
            "U238": fuel_frac * (1.0 - enr) * x * n_mix,
            "Li7": fuel_frac * 2.0 * (1.0 - x) * n_mix,
            "Be9": fuel_frac * (1.0 - x) * n_mix,
            "F19": fuel_frac * 4.0 * n_mix,
            "C12": graphite_frac * self.graphite_density_gcc * N_A / self.M_C12,
            "H1": water_frac * 2.0 * n_h2o,
            "O16": water_frac * n_h2o,
        }

    def _compute_benchmark_guided_xs(self) -> Dict[str, Any]:
        xs_db = get_xs_database_v2()
        number_densities = self.compute_number_densities()
        temp_ratio = max(self.temperature, 1.0) / 900.0
        graphite_temp_ratio = max(self.graphite_temperature or self.temperature, 1.0) / 900.0
        graphite_frac = self.graphite_volume_fraction
        water_frac = self.water_vol_frac
        fuel_frac = self.fuel_volume_fraction
        benchmark_fuel_frac = CALIBRATION.benchmark_fuel_volume_fraction
        benchmark_graphite_frac = CALIBRATION.benchmark_graphite_volume_fraction

        sigma_f1 = sigma_a1 = sigma_s1 = nu_sigma_f1 = sigma_tr1 = 0.0
        sigma_f2 = sigma_a2 = sigma_s2 = nu_sigma_f2 = sigma_tr2 = 0.0

        resonance_factor = 1.0 + CALIBRATION.fast_doppler_coeff * math.log(temp_ratio)
        thermal_fission_factor = (
            CALIBRATION.thermal_fission_factor
            * temp_ratio ** (-CALIBRATION.thermal_fission_temp_coeff)
            * (fuel_frac / benchmark_fuel_frac) ** 0.08
        )
        thermal_absorption_factor = (
            CALIBRATION.thermal_absorption_factor
            * (1.0 + 0.10 * (graphite_frac - benchmark_graphite_frac))
            * (1.0 + 0.05 * water_frac)
        )
        fast_absorption_factor = CALIBRATION.fast_absorption_factor * temp_ratio ** 0.06
        downscatter_efficiency = (
            CALIBRATION.downscatter_base
            + CALIBRATION.downscatter_graphite_gain * (graphite_frac / max(benchmark_graphite_frac, 1.0e-8))
            + CALIBRATION.downscatter_water_gain * water_frac
        )
        downscatter_efficiency *= graphite_temp_ratio ** (-CALIBRATION.s12_temp_coeff)
        downscatter_efficiency = float(np.clip(downscatter_efficiency, 0.20, 0.45))

        for iso, density in number_densities.items():
            if density <= 0.0:
                continue
            xs = xs_db[iso]
            mu = 2.0 / (3.0 * xs.A)

            sigma_f1_iso = density * xs.sigma_f1
            sigma_a1_iso = density * xs.sigma_a1
            sigma_s1_iso = density * xs.sigma_s1
            nu_sigma_f1_iso = density * xs.nu1 * xs.sigma_f1

            sigma_f2_iso = density * xs.sigma_f2
            sigma_a2_iso = density * xs.sigma_a2
            sigma_s2_iso = density * xs.sigma_s2
            nu_sigma_f2_iso = density * xs.nu2 * xs.sigma_f2

            if iso == "U238":
                sigma_a1_iso *= resonance_factor * fast_absorption_factor
            if iso in {"U235", "U238"}:
                sigma_f2_iso *= thermal_fission_factor
                nu_sigma_f2_iso *= thermal_fission_factor
                sigma_a2_iso *= thermal_absorption_factor
            elif iso in {"Li7", "Be9", "F19", "C12", "H1", "O16"}:
                sigma_a2_iso *= 1.0 + 0.06 * water_frac

            sigma_f1 += sigma_f1_iso
            sigma_a1 += sigma_a1_iso
            sigma_s1 += sigma_s1_iso
            nu_sigma_f1 += nu_sigma_f1_iso
            sigma_tr1 += sigma_s1_iso * (1.0 - mu)

            sigma_f2 += sigma_f2_iso
            sigma_a2 += sigma_a2_iso
            sigma_s2 += sigma_s2_iso
            nu_sigma_f2 += nu_sigma_f2_iso
            sigma_tr2 += sigma_s2_iso * (1.0 - mu)

        sigma_s12 = sigma_s1 * downscatter_efficiency
        sigma_r1 = sigma_a1 + sigma_s12
        d1 = 1.0 / max(3.0 * sigma_tr1, 1.0e-12)
        d2_raw = 1.0 / max(3.0 * sigma_tr2, 1.0e-12)
        d2 = CALIBRATION.d2_ref_cm * graphite_temp_ratio ** CALIBRATION.d2_temp_coeff
        d2 *= (benchmark_fuel_frac / max(fuel_frac, 1.0e-6)) ** 0.03
        d2 *= (1.0 - 0.05 * water_frac)
        d2 = float(np.clip(d2, 0.32, 0.50))
        k_inf = (nu_sigma_f1 + nu_sigma_f2 * sigma_s12 / max(sigma_a2, 1.0e-12)) / max(sigma_r1, 1.0e-12)

        return {
            "D1": d1,
            "D2": d2,
            "D2_raw": d2_raw,
            "Sigma_a1": sigma_a1,
            "Sigma_a2": sigma_a2,
            "Sigma_f1": sigma_f1,
            "Sigma_f2": sigma_f2,
            "nu_Sigma_f1": nu_sigma_f1,
            "nu_Sigma_f2": nu_sigma_f2,
            "Sigma_s12": sigma_s12,
            "Sigma_r1": sigma_r1,
            "chi1": 1.0,
            "chi2": 0.0,
            "k_inf": k_inf,
            "N": number_densities,
            "volume_fractions": self.volume_fractions(),
            "xs_model": "benchmark_guided",
            "reactor_mode": FUEL_C_BENCHMARK_MODE,
            "xs_source": "benchmark-guided homogenized 2-group corrections anchored to ORNL Fuel C",
            "corrections": {
                "resonance_factor": resonance_factor,
                "thermal_fission_factor": thermal_fission_factor,
                "thermal_absorption_factor": thermal_absorption_factor,
                "fast_absorption_factor": fast_absorption_factor,
                "downscatter_efficiency": downscatter_efficiency,
            },
        }

    def compute_macroscopic_xs(
        self,
        *,
        reactor_mode: str = LEU_DESIGN_MODE,
        xs_model: str = "auto",
        xs_library_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        benchmark_guided_xs = self._compute_benchmark_guided_xs()
        benchmark_guided_xs["reactor_mode"] = reactor_mode

        resolved_xs_model = xs_model
        if xs_model == "auto":
            resolved_xs_model = "benchmark_guided" if reactor_mode == FUEL_C_BENCHMARK_MODE else "leu_transport_proxy"

        if resolved_xs_model == "benchmark_guided":
            benchmark_guided_xs["xs_model"] = "benchmark_guided"
            benchmark_guided_xs["reactor_mode"] = reactor_mode
            benchmark_guided_xs["transport_tooling"] = detect_transport_tooling()
            return benchmark_guided_xs

        if resolved_xs_model != "leu_transport_proxy":
            raise ValueError(f"Unsupported xs_model '{resolved_xs_model}'")

        state = {
            "enrichment": self.enrichment,
            "uf4_mol_frac": self.uf4_mol_frac,
            "temperature_K": self.temperature,
            "fuel_volume_fraction": self.fuel_volume_fraction,
            "water_vol_frac": self.water_vol_frac,
        }
        leu_xs = resolve_transport_proxy_xs(
            state=state,
            benchmark_guided_xs=benchmark_guided_xs,
            library_path=xs_library_path or DEFAULT_LEU_XS_LIBRARY_PATH,
        )
        leu_xs.update(
            {
                "N": benchmark_guided_xs["N"],
                "volume_fractions": benchmark_guided_xs["volume_fractions"],
                "xs_model": "leu_transport_proxy",
                "reactor_mode": reactor_mode,
                "xs_source": (
                    "transport-informed LEU 2-group proxy interpolated from explicit project library "
                    f"'{LEU_TRANSPORT_PROXY_LIBRARY_ID}'"
                ),
                "transport_tooling": detect_transport_tooling(),
            }
        )
        return leu_xs


@dataclass
class CylinderGeometryV2:
    radius_cm: float
    height_cm: float
    nr: int = 40
    nz: int = 60

    @property
    def dr(self) -> float:
        return self.radius_cm / self.nr

    @property
    def dz(self) -> float:
        return self.height_cm / self.nz

    @property
    def r_centers(self) -> np.ndarray:
        return np.linspace(self.dr / 2.0, self.radius_cm - self.dr / 2.0, self.nr)

    @property
    def z_centers(self) -> np.ndarray:
        return np.linspace(self.dz / 2.0, self.height_cm - self.dz / 2.0, self.nz)

    @property
    def volume_cm3(self) -> float:
        return math.pi * self.radius_cm ** 2 * self.height_cm

    @property
    def surface_area_cm2(self) -> float:
        return 2.0 * math.pi * self.radius_cm * self.height_cm + 2.0 * math.pi * self.radius_cm ** 2

    def volume_elements(self) -> np.ndarray:
        r = self.r_centers
        vol = np.zeros((self.nr, self.nz), dtype=float)
        for i, radius in enumerate(r):
            vol[i, :] = 2.0 * math.pi * radius * self.dr * self.dz
        return vol


def _build_diffusion_operator(
    geom: CylinderGeometryV2,
    diffusion_coeff: float,
    removal_xs: float,
    extrapolation_length_cm: float,
) -> sparse.csr_matrix:
    nr, nz = geom.nr, geom.nz
    dr, dz = geom.dr, geom.dz
    r = geom.r_centers
    n = nr * nz
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    def idx(i: int, j: int) -> int:
        return i * nz + j

    delta_r = max(extrapolation_length_cm, 0.0)
    delta_z = max(extrapolation_length_cm, 0.0)

    for i in range(nr):
        for j in range(nz):
            cell = idx(i, j)
            ri = r[i]
            diag = removal_xs

            if i > 0:
                rf = 0.5 * (r[i] + r[i - 1])
                coeff = diffusion_coeff * rf / (ri * dr ** 2)
                rows.append(cell)
                cols.append(idx(i - 1, j))
                vals.append(-coeff)
                diag += coeff
            else:
                # Axis symmetry: dphi/dr = 0
                diag += diffusion_coeff / dr ** 2

            if i < nr - 1:
                rf = 0.5 * (r[i] + r[i + 1])
                coeff = diffusion_coeff * rf / (ri * dr ** 2)
                rows.append(cell)
                cols.append(idx(i + 1, j))
                vals.append(-coeff)
                diag += coeff
            else:
                rf = ri + dr / 2.0
                leak_coeff = diffusion_coeff * rf / (ri * dr * (dr / 2.0 + delta_r))
                diag += leak_coeff

            axial_coeff = diffusion_coeff / dz ** 2
            if j > 0:
                rows.append(cell)
                cols.append(idx(i, j - 1))
                vals.append(-axial_coeff)
                diag += axial_coeff
            else:
                leak_coeff = diffusion_coeff / (dz * (dz / 2.0 + delta_z))
                diag += leak_coeff

            if j < nz - 1:
                rows.append(cell)
                cols.append(idx(i, j + 1))
                vals.append(-axial_coeff)
                diag += axial_coeff
            else:
                leak_coeff = diffusion_coeff / (dz * (dz / 2.0 + delta_z))
                diag += leak_coeff

            rows.append(cell)
            cols.append(cell)
            vals.append(diag)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def solve_two_group_diffusion(
    geom: CylinderGeometryV2,
    xs: Dict[str, Any],
    tol: float = 1.0e-7,
    max_iter: int = 500,
) -> Dict[str, Any]:
    """Solve the 2-group eigenproblem with extrapolated-vacuum leakage."""

    n = geom.nr * geom.nz
    delta1 = CALIBRATION.extrapolation_factor * xs["D1"]
    delta2 = CALIBRATION.extrapolation_factor * xs["D2"]
    a1 = _build_diffusion_operator(geom, xs["D1"], xs["Sigma_r1"], delta1)
    a2 = _build_diffusion_operator(geom, xs["D2"], xs["Sigma_a2"], delta2)

    f1 = np.full(n, xs["nu_Sigma_f1"], dtype=float)
    f2 = np.full(n, xs["nu_Sigma_f2"], dtype=float)
    s12 = xs["Sigma_s12"]

    r = geom.r_centers
    z = geom.z_centers
    re = geom.radius_cm + delta1
    he = geom.height_cm + 2.0 * delta1
    rr, zz = np.meshgrid(r, z, indexing="ij")
    phi_guess = np.maximum(
        np.cos(0.5 * math.pi * rr / max(re, 1.0e-8))
        * np.cos(math.pi * (zz - geom.height_cm / 2.0) / max(he, 1.0e-8)),
        1.0e-12,
    )
    phi1 = phi_guess.reshape(-1).copy()
    phi2 = phi_guess.reshape(-1).copy()

    cell_volumes = geom.volume_elements().reshape(-1)
    k_eff = float(np.clip(xs["k_inf"], 0.5, 2.0))
    history = {"k": [], "res": []}
    source = f1 * phi1 + f2 * phi2
    source_norm = float(np.sum(source * cell_volumes))

    for _ in range(max_iter):
        rhs1 = (xs["chi1"] / max(k_eff, 1.0e-12)) * source
        phi1_new = np.maximum(spsolve(a1, rhs1), 0.0)
        rhs2 = (xs["chi2"] / max(k_eff, 1.0e-12)) * source + s12 * phi1_new
        phi2_new = np.maximum(spsolve(a2, rhs2), 0.0)

        new_source = f1 * phi1_new + f2 * phi2_new
        new_norm = float(np.sum(new_source * cell_volumes))
        k_new = k_eff * new_norm / max(source_norm, 1.0e-18)
        residual = abs(k_new - k_eff) / max(abs(k_new), 1.0e-12)
        history["k"].append(k_new)
        history["res"].append(residual)

        norm = math.sqrt(float(np.sum((phi1_new ** 2 + phi2_new ** 2) * cell_volumes)))
        if norm > 0.0:
            phi1 = phi1_new / norm
            phi2 = phi2_new / norm
        else:
            phi1 = phi1_new
            phi2 = phi2_new

        source = f1 * phi1 + f2 * phi2
        source_norm = float(np.sum(source * cell_volumes))
        k_eff = k_new
        if residual < tol and len(history["k"]) > 6:
            break

    return {
        "k_eff": k_eff,
        "phi1_shape": phi1.reshape(geom.nr, geom.nz),
        "phi2_shape": phi2.reshape(geom.nr, geom.nz),
        "history": history,
        "iterations": len(history["k"]),
        "r_cm": r,
        "z_cm": z,
        "delta_r1_cm": delta1,
        "delta_r2_cm": delta2,
        "delta_z1_cm": delta1,
        "delta_z2_cm": delta2,
    }


def _buckling(radius_cm: float, height_cm: float, delta_r_cm: float, delta_z_cm: float) -> Dict[str, float]:
    radius_extrap_cm = radius_cm + delta_r_cm
    height_extrap_cm = height_cm + 2.0 * delta_z_cm
    br2 = (2.405 / max(radius_extrap_cm, 1.0e-8)) ** 2
    bz2 = (math.pi / max(height_extrap_cm, 1.0e-8)) ** 2
    return {
        "radius_extrap_cm": radius_extrap_cm,
        "height_extrap_cm": height_extrap_cm,
        "Br2_cm2_inv": br2,
        "Bz2_cm2_inv": bz2,
        "Bg2_cm2_inv": br2 + bz2,
    }


def _normalize_to_power(
    geom: CylinderGeometryV2,
    xs: Dict[str, Any],
    phi1_shape: np.ndarray,
    phi2_shape: np.ndarray,
    power_watts: float,
) -> Dict[str, Any]:
    cell_volumes = geom.volume_elements()
    fission_rate_shape = xs["Sigma_f1"] * phi1_shape + xs["Sigma_f2"] * phi2_shape
    unnormalized_fission_rate = float(np.sum(fission_rate_shape * cell_volumes))
    norm_factor = power_watts / max(E_FISSION * unnormalized_fission_rate, 1.0e-18)

    phi1_phys = phi1_shape * norm_factor
    phi2_phys = phi2_shape * norm_factor
    phi_total_phys = phi1_phys + phi2_phys
    fission_rate_phys = (xs["Sigma_f1"] * phi1_phys + xs["Sigma_f2"] * phi2_phys)
    power_density = E_FISSION * fission_rate_phys

    total_volume = float(np.sum(cell_volumes))
    avg_flux = float(np.sum(phi_total_phys * cell_volumes) / total_volume)
    avg_thermal_flux = float(np.sum(phi2_phys * cell_volumes) / total_volume)
    peak_flux = float(np.max(phi_total_phys))
    peak_thermal_flux = float(np.max(phi2_phys))
    avg_power_density = float(np.sum(power_density * cell_volumes) / total_volume)
    peak_power_density = float(np.max(power_density))

    return {
        "normalization_factor": norm_factor,
        "fission_rate_cm3_s": fission_rate_phys,
        "phi1_phys_n_cm2_s": phi1_phys,
        "phi2_phys_n_cm2_s": phi2_phys,
        "phi_total_phys_n_cm2_s": phi_total_phys,
        "power_density_W_cm3": power_density,
        "peak_flux_n_cm2_s": peak_flux,
        "peak_thermal_flux_n_cm2_s": peak_thermal_flux,
        "avg_flux_n_cm2_s": avg_flux,
        "avg_thermal_flux_n_cm2_s": avg_thermal_flux,
        "peak_power_density_W_cm3": peak_power_density,
        "avg_power_density_W_cm3": avg_power_density,
        "peaking_factor": peak_flux / max(avg_flux, 1.0e-18),
        "power_density_peaking_factor": peak_power_density / max(avg_power_density, 1.0e-18),
        "fission_rate_integral_s": float(np.sum(fission_rate_phys * cell_volumes)),
        "power_integral_W": float(np.sum(power_density * cell_volumes)),
    }


def _compute_group_currents(
    geom: CylinderGeometryV2,
    xs: Dict[str, Any],
    phi1_shape: np.ndarray,
    phi2_shape: np.ndarray,
) -> Dict[str, np.ndarray]:
    def gradients(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d_dr = np.zeros_like(field)
        d_dz = np.zeros_like(field)

        d_dr[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * geom.dr)
        d_dr[0, :] = 0.0
        d_dr[-1, :] = (field[-1, :] - field[-2, :]) / geom.dr

        d_dz[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * geom.dz)
        d_dz[:, 0] = (field[:, 1] - field[:, 0]) / geom.dz
        d_dz[:, -1] = (field[:, -1] - field[:, -2]) / geom.dz
        return d_dr, d_dz

    grad_r1, grad_z1 = gradients(phi1_shape)
    grad_r2, grad_z2 = gradients(phi2_shape)
    return {
        "Jr1_shape": -xs["D1"] * grad_r1,
        "Jz1_shape": -xs["D1"] * grad_z1,
        "Jr2_shape": -xs["D2"] * grad_r2,
        "Jz2_shape": -xs["D2"] * grad_z2,
    }


def _estimate_leakage_fractions(
    geom: CylinderGeometryV2,
    xs: Dict[str, Any],
    phi_total: np.ndarray,
) -> Dict[str, float]:
    dr = geom.dr
    dz = geom.dz
    r = geom.r_centers
    delta = CALIBRATION.extrapolation_factor * 0.5 * (xs["D1"] + xs["D2"])
    side_current = 0.0
    top_current = 0.0
    bottom_current = 0.0

    for i, radius in enumerate(r):
        area_ring = 2.0 * math.pi * radius * dr
        top_current += xs["D2"] * phi_total[i, -1] / max(dz / 2.0 + delta, 1.0e-8) * area_ring
        bottom_current += xs["D2"] * phi_total[i, 0] / max(dz / 2.0 + delta, 1.0e-8) * area_ring

    for j in range(geom.nz):
        area_side = 2.0 * math.pi * geom.radius_cm * dz
        side_current += xs["D2"] * phi_total[-1, j] / max(dr / 2.0 + delta, 1.0e-8) * area_side

    total_leakage = side_current + top_current + bottom_current
    if total_leakage <= 0.0:
        return {
            "leakage_fraction": 0.0,
            "leakage_top_fraction": 0.0,
            "leakage_side_fraction": 0.0,
            "leakage_bottom_fraction": 0.0,
        }

    return {
        "leakage_fraction": min(0.95, total_leakage / max(total_leakage + 1.0, 1.0)),
        "leakage_top_fraction": top_current / total_leakage,
        "leakage_side_fraction": side_current / total_leakage,
        "leakage_bottom_fraction": bottom_current / total_leakage,
    }


def effective_delayed_fraction_v2(
    core_residence_time_s: float,
    loop_residence_time_s: float,
    bg2_cm2_inv: float,
) -> Dict[str, Any]:
    """Benchmark-aware effective delayed fraction for circulating fuel.

    The repo does not contain a full precursor-transport solver, so this uses a
    calibrated in-core decay fraction combined with a leakage-importance penalty
    matched to the ORNL Fuel C benchmark. The benchmark match is explicit in the
    returned metadata.
    """

    beta_static_groups = np.array([g.beta_i_star_static for g in DELAYED_GROUPS_V2], dtype=float)
    beta_static = float(np.sum(beta_static_groups))
    beta_precursor_total = float(sum(g.beta_i for g in DELAYED_GROUPS_V2))
    beta_eff_groups: List[float] = []
    group_details: List[Dict[str, float]] = []

    for group in DELAYED_GROUPS_V2:
        lam = math.log(2.0) / group.half_life_s
        in_core = (1.0 - math.exp(-lam * core_residence_time_s)) / (
            1.0 - math.exp(-lam * (core_residence_time_s + loop_residence_time_s))
        )
        leakage_penalty = math.exp(
            -DELAYED_LEAKAGE_PENALTY * group.age_cm2 * bg2_cm2_inv * (1.0 - in_core)
        )
        ratio = float(np.clip(in_core * leakage_penalty, 0.0, 1.0))
        beta_eff_i = group.beta_i_star_static * ratio
        beta_eff_groups.append(beta_eff_i)
        group_details.append(
            {
                "group": group.group,
                "half_life_s": group.half_life_s,
                "beta_i": group.beta_i,
                "beta_i_star_static": group.beta_i_star_static,
                "beta_i_star_circulating_reference": group.beta_i_star_circulating,
                "ratio": ratio,
                "beta_i_star_effective": beta_eff_i,
            }
        )

    beta_eff = float(np.sum(beta_eff_groups))
    return {
        "beta_precursor_total": beta_precursor_total,
        "beta_static": beta_static,
        "beta_eff": beta_eff,
        "reduction_factor": beta_eff / max(beta_static, 1.0e-18),
        "core_time_s": core_residence_time_s,
        "loop_time_s": loop_residence_time_s,
        "benchmark_ratio_target": DELAYED_RATIO_TARGET,
        "penalty_constant": DELAYED_LEAKAGE_PENALTY,
        "groups": group_details,
    }


def compute_thermal_hydraulics_v2(
    result: Dict[str, Any],
    power_mw: float = 10.0,
    inlet_temperature_k: float = 873.0,
) -> Dict[str, float]:
    power_watts = power_mw * 1.0e6
    cp = FLiBeProperties.heat_capacity() * 1000.0
    delta_t = 50.0
    outlet_temperature_k = inlet_temperature_k + delta_t
    bulk_temperature = 0.5 * (inlet_temperature_k + outlet_temperature_k)
    density = FLiBeProperties.density(bulk_temperature)
    mass_flow = power_watts / max(cp * delta_t, 1.0e-8)
    volumetric_flow = mass_flow / max(density, 1.0e-8)

    fuel_fraction = result["xs"]["volume_fractions"]["fuel"]
    flow_area = math.pi * (result["params"]["radius_cm"] / 100.0) ** 2 * fuel_fraction
    velocity = volumetric_flow / max(flow_area, 1.0e-8)

    core_height_m = result["params"]["height_cm"] / 100.0
    core_residence = core_height_m / max(velocity, 1.0e-8)
    loop_residence = core_residence * 1.76

    viscosity = FLiBeProperties.viscosity(bulk_temperature) * 1.0e-3
    conductivity = FLiBeProperties.thermal_conductivity()
    hydraulic_diameter = 0.018
    reynolds = density * velocity * hydraulic_diameter / max(viscosity, 1.0e-12)
    prandtl = viscosity * cp / max(conductivity, 1.0e-12)

    return {
        "P_thermal_MW": power_mw,
        "T_inlet_K": inlet_temperature_k,
        "T_outlet_K": outlet_temperature_k,
        "T_avg_K": bulk_temperature,
        "deltaT_K": delta_t,
        "mass_flow_kg_s": mass_flow,
        "vol_flow_m3_s": volumetric_flow,
        "vol_flow_gpm": volumetric_flow * 15850.3,
        "flow_velocity_m_s": velocity,
        "core_residence_s": core_residence,
        "loop_transit_s": loop_residence,
        "density_kg_m3": density,
        "viscosity_mPa_s": FLiBeProperties.viscosity(bulk_temperature),
        "Cp_J_kgK": cp,
        "k_W_mK": conductivity,
        "Re": reynolds,
        "Pr": prandtl,
        "prompt_neutron_lifetime_s": BENCHMARK_PROMPT_LIFETIME,
    }


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_design_v2(
    *,
    enrichment: float = LEU_MAX_ENRICHMENT,
    uf4_mol_frac: float = 0.04,
    radius_cm: float = 50.0,
    height_cm: float = 100.0,
    temperature_k: float = 900.0,
    water_vol_frac: float = 0.0,
    fuel_volume_fraction: float = CALIBRATION.benchmark_fuel_volume_fraction,
    graphite_temperature_k: Optional[float] = None,
    nr: int = 40,
    nz: int = 60,
    power_watts: float = 10.0e6,
    reactor_mode: str = LEU_DESIGN_MODE,
    xs_model: str = "auto",
    xs_library_path: Optional[Path] = None,
    sample_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    material = HomogenizedMSRMaterial(
        enrichment=enrichment,
        uf4_mol_frac=uf4_mol_frac,
        temperature=temperature_k,
        water_vol_frac=water_vol_frac,
        fuel_volume_fraction=fuel_volume_fraction,
        graphite_temperature=graphite_temperature_k,
    )
    xs = material.compute_macroscopic_xs(
        reactor_mode=reactor_mode,
        xs_model=xs_model,
        xs_library_path=xs_library_path,
    )
    geom = CylinderGeometryV2(radius_cm=radius_cm, height_cm=height_cm, nr=nr, nz=nz)
    solution = solve_two_group_diffusion(geom, xs)
    buckling = _buckling(
        radius_cm,
        height_cm,
        delta_r_cm=solution["delta_r2_cm"],
        delta_z_cm=solution["delta_z2_cm"],
    )
    normalized = _normalize_to_power(
        geom,
        xs,
        solution["phi1_shape"],
        solution["phi2_shape"],
        power_watts,
    )
    currents = _compute_group_currents(geom, xs, solution["phi1_shape"], solution["phi2_shape"])
    leakage = _estimate_leakage_fractions(geom, xs, normalized["phi_total_phys_n_cm2_s"])

    result: Dict[str, Any] = {
        "solver_version": "mjm_v2",
        "energy_groups": ENERGY_GROUPS,
        "params": {
            "enrichment": enrichment,
            "uf4_mol_frac": uf4_mol_frac,
            "radius_cm": radius_cm,
            "height_cm": height_cm,
            "temperature_K": temperature_k,
            "graphite_temperature_K": graphite_temperature_k if graphite_temperature_k is not None else temperature_k,
            "water_vol_frac": water_vol_frac,
            "fuel_volume_fraction": fuel_volume_fraction,
            "power_W": power_watts,
            "reactor_mode": reactor_mode,
            "xs_model": xs["xs_model"],
        },
        "k_eff": solution["k_eff"],
        "k_inf": xs["k_inf"],
        "iterations": solution["iterations"],
        "history": solution["history"],
        "nr": nr,
        "nz": nz,
        "r_centers_cm": solution["r_cm"],
        "z_centers_cm": solution["z_cm"],
        "xs": xs,
        "buckling": buckling,
        "volume_fractions": xs["volume_fractions"],
        "core_volume_cm3": geom.volume_cm3,
        "core_volume_m3": geom.volume_cm3 / 1.0e6,
        "surface_area_cm2": geom.surface_area_cm2,
        "phi1_shape": solution["phi1_shape"],
        "phi2_shape": solution["phi2_shape"],
        "phi_total_shape": solution["phi1_shape"] + solution["phi2_shape"],
        **currents,
        **normalized,
        **leakage,
        "normalization_rule": "power-normalized with Sigma_f1*phi1 + Sigma_f2*phi2",
        "bc_type": "extrapolated vacuum on physical boundary",
        "xs_source": xs["xs_source"],
        "reactor_mode": reactor_mode,
        "xs_model": xs["xs_model"],
        "sample_metadata": sample_metadata or {},
    }
    th = compute_thermal_hydraulics_v2(result)
    dn = effective_delayed_fraction_v2(
        core_residence_time_s=th["core_residence_s"],
        loop_residence_time_s=th["loop_transit_s"],
        bg2_cm2_inv=buckling["Bg2_cm2_inv"],
    )
    result["thermal_hydraulics"] = th
    result["delayed_neutrons"] = dn
    return result


def find_critical_radius_v2(
    *,
    k_target: float = 1.03,
    aspect_ratio: float = 2.0,
    enrichment: float = LEU_MAX_ENRICHMENT,
    uf4_mol_frac: float = 0.04,
    temperature_k: float = 900.0,
    water_vol_frac: float = 0.0,
    fuel_volume_fraction: float = CALIBRATION.benchmark_fuel_volume_fraction,
    r_bounds_cm: Tuple[float, float] = (20.0, 140.0),
    reactor_mode: str = LEU_DESIGN_MODE,
    xs_model: str = "auto",
    xs_library_path: Optional[Path] = None,
) -> Dict[str, Any]:
    def residual(radius_cm: float) -> float:
        result = evaluate_design_v2(
            enrichment=enrichment,
            uf4_mol_frac=uf4_mol_frac,
            radius_cm=radius_cm,
            height_cm=aspect_ratio * radius_cm,
            temperature_k=temperature_k,
            water_vol_frac=water_vol_frac,
            fuel_volume_fraction=fuel_volume_fraction,
            nr=26,
            nz=40,
            reactor_mode=reactor_mode,
            xs_model=xs_model,
            xs_library_path=xs_library_path,
        )
        return result["k_eff"] - k_target

    lo, hi = r_bounds_cm
    f_lo = residual(lo)
    f_hi = residual(hi)
    if f_lo * f_hi > 0.0:
        for hi_candidate in (160.0, 180.0, 220.0):
            f_candidate = residual(hi_candidate)
            if f_lo * f_candidate <= 0.0:
                hi = hi_candidate
                f_hi = f_candidate
                break
        else:
            raise ValueError(
                f"Unable to bracket critical radius for enrichment={enrichment:.4f}, "
                f"aspect_ratio={aspect_ratio:.3f}, k_target={k_target:.4f}; residuals "
                f"were {f_lo:+.5f} at R={lo:.2f} cm and {f_hi:+.5f} at R={hi:.2f} cm."
            )

    radius_cm = brentq(residual, lo, hi, xtol=0.25)
    result = evaluate_design_v2(
        enrichment=enrichment,
        uf4_mol_frac=uf4_mol_frac,
        radius_cm=radius_cm,
        height_cm=aspect_ratio * radius_cm,
        temperature_k=temperature_k,
        water_vol_frac=water_vol_frac,
        fuel_volume_fraction=fuel_volume_fraction,
        nr=40,
        nz=60,
        reactor_mode=reactor_mode,
        xs_model=xs_model,
        xs_library_path=xs_library_path,
    )
    result["critical_radius_cm"] = radius_cm
    result["critical_height_cm"] = aspect_ratio * radius_cm
    result["k_target"] = k_target
    return result


def compute_temperature_sweep_v2(
    base_params: Dict[str, Any],
    temperatures_k: Iterable[float],
) -> Dict[str, Any]:
    sweep = []
    temps = [float(item) for item in temperatures_k]
    for temp in temps:
        params = dict(base_params)
        params["temperature_k"] = temp
        params.setdefault("nr", 28)
        params.setdefault("nz", 42)
        params.setdefault("reactor_mode", LEU_DESIGN_MODE)
        params.setdefault("xs_model", "auto")
        result = evaluate_design_v2(**params)
        sweep.append(
            {
                "temperature_K": temp,
                "k_eff": result["k_eff"],
                "k_inf": result["k_inf"],
                "peak_flux_n_cm2_s": result["peak_flux_n_cm2_s"],
            }
        )

    k_values = np.array([row["k_eff"] for row in sweep], dtype=float)
    temp_values = np.array(temps, dtype=float)
    dkdT = np.gradient(k_values, temp_values)
    alpha_pcm = dkdT / np.maximum(k_values, 1.0e-12) * 1.0e5
    for row, alpha, deriv in zip(sweep, alpha_pcm, dkdT, strict=True):
        row["dk_dT"] = float(deriv)
        row["alpha_pcm_per_K"] = float(alpha)

    return {
        "rows": sweep,
        "mean_alpha_pcm_per_K": float(np.mean(alpha_pcm)),
        "alpha_target_pcm_per_K": ALPHA_TARGET_PCM_PER_K,
    }


def compute_temperature_coefficient_v2(
    base_params: Dict[str, Any],
    dT: float = 20.0,
) -> Dict[str, float]:
    center_temperature = float(base_params["temperature_k"])
    lower_params = dict(base_params)
    upper_params = dict(base_params)
    lower_params["temperature_k"] = center_temperature - dT
    upper_params["temperature_k"] = center_temperature + dT
    lower_params.setdefault("nr", 22)
    lower_params.setdefault("nz", 32)
    upper_params.setdefault("nr", 22)
    upper_params.setdefault("nz", 32)
    lower_params.setdefault("reactor_mode", LEU_DESIGN_MODE)
    lower_params.setdefault("xs_model", "auto")
    upper_params.setdefault("reactor_mode", LEU_DESIGN_MODE)
    upper_params.setdefault("xs_model", "auto")

    lower = evaluate_design_v2(**lower_params)
    upper = evaluate_design_v2(**upper_params)
    k_lo = lower["k_eff"]
    k_hi = upper["k_eff"]
    dkdT = (k_hi - k_lo) / (2.0 * dT)
    k_avg = 0.5 * (k_lo + k_hi)
    alpha_pcm = dkdT / max(k_avg, 1.0e-12) * 1.0e5
    return {
        "T_center": center_temperature,
        "dT": dT,
        "k_lo": k_lo,
        "k_hi": k_hi,
        "dk_dT": dkdT,
        "alpha_pcm_per_K": alpha_pcm,
        "lower_xs_model": lower["xs_model"],
        "upper_xs_model": upper["xs_model"],
        "lower_proxy_method": lower["xs"].get("transport_proxy", {}).get("proxy_method"),
        "upper_proxy_method": upper["xs"].get("transport_proxy", {}).get("proxy_method"),
        "used_direct_transport_proxy": any(
            (item or "").startswith("direct_transport_proxy_formula")
            for item in (
                lower["xs"].get("transport_proxy", {}).get("proxy_method"),
                upper["xs"].get("transport_proxy", {}).get("proxy_method"),
            )
        ),
    }


def export_sample_record(
    result: Dict[str, Any],
    sample_id: str,
) -> Dict[str, Any]:
    xs = result["xs"]
    th = result["thermal_hydraulics"]
    dn = result["delayed_neutrons"]
    transport_proxy = xs.get("transport_proxy", {})
    notes = [
        "legacy v1 salt chemistry retained as a base input model",
        "project uses explicit 2-group reporting bounds and finite-cylinder buckling-aware leakage",
        ENERGY_GROUPS["note"],
    ]
    if result["reactor_mode"] == LEU_DESIGN_MODE:
        notes.append(
            "LEU design mode uses the transport-informed proxy XS path when full transport tooling is unavailable."
        )
    else:
        notes.append("Fuel C benchmark mode retains the benchmark-guided XS anchor for honest validation.")

    record = {
        "sample_id": sample_id,
        "solver_version": result["solver_version"],
        "energy_groups": result["energy_groups"],
        "source_document": BENCHMARK_DATA["source"]["document"],
        "notes": notes,
        "inputs": result["params"],
        "geometry": {
            "aspect_ratio": result["params"]["height_cm"] / max(result["params"]["radius_cm"], 1.0e-8),
            "core_volume_cm3": result["core_volume_cm3"],
            "surface_area_cm2": result["surface_area_cm2"],
            "delta_r_cm": result["buckling"]["radius_extrap_cm"] - result["params"]["radius_cm"],
            "delta_z_cm": 0.5 * (result["buckling"]["height_extrap_cm"] - result["params"]["height_cm"]),
            **result["buckling"],
        },
        "material_properties": {
            "rho_salt_g_cm3": FLiBeProperties.density_gcc(result["params"]["temperature_K"]),
            "rho_mix_g_cm3": FLiBeProperties.density_gcc(result["params"]["temperature_K"]),
            "cp_J_kgK": th["Cp_J_kgK"],
            "mu_Pa_s": th["viscosity_mPa_s"] * 1.0e-3,
            "k_W_mK": th["k_W_mK"],
            "graphite_volume_fraction": xs["volume_fractions"]["graphite"],
            "moderator_to_fuel_ratio": xs["volume_fractions"]["graphite"]
            / max(xs["volume_fractions"]["fuel"], 1.0e-8),
        },
        "number_densities": xs["N"],
        "xs": {
            "D1_cm": xs["D1"],
            "D2_cm": xs["D2"],
            "D2_raw_cm": xs["D2_raw"],
            "Sigma_a1_cm1": xs["Sigma_a1"],
            "Sigma_a2_cm1": xs["Sigma_a2"],
            "Sigma_s12_cm1": xs["Sigma_s12"],
            "Sigma_f1_cm1": xs["Sigma_f1"],
            "Sigma_f2_cm1": xs["Sigma_f2"],
            "nuSigma_f1_cm1": xs["nu_Sigma_f1"],
            "nuSigma_f2_cm1": xs["nu_Sigma_f2"],
            "Sigma_r1_cm1": xs["Sigma_r1"],
            "chi1": xs["chi1"],
            "chi2": xs["chi2"],
            "k_inf": xs["k_inf"],
            "corrections": xs["corrections"],
            "xs_model": xs["xs_model"],
            "reactor_mode": xs["reactor_mode"],
            "transport_proxy": transport_proxy,
        },
        "global_labels": {
            "k_eff": result["k_eff"],
            "k_inf": result["k_inf"],
            "reactivity_pcm": (result["k_eff"] - 1.0) / max(result["k_eff"], 1.0e-12) * 1.0e5,
            "alpha_T_pcm_per_K": result.get("alpha_T_pcm_per_K"),
            "peak_flux_n_cm2_s": result["peak_flux_n_cm2_s"],
            "peak_thermal_flux_n_cm2_s": result["peak_thermal_flux_n_cm2_s"],
            "avg_flux_n_cm2_s": result["avg_flux_n_cm2_s"],
            "avg_thermal_flux_n_cm2_s": result["avg_thermal_flux_n_cm2_s"],
            "peaking_factor": result["peaking_factor"],
            "peak_power_density_W_cm3": result["peak_power_density_W_cm3"],
            "avg_power_density_W_cm3": result["avg_power_density_W_cm3"],
            "power_density_peaking_factor": result["power_density_peaking_factor"],
            "beta_static": dn["beta_static"],
            "beta_eff": dn["beta_eff"],
            "beta_eff_ratio": dn["reduction_factor"],
            "leakage_fraction": result["leakage_fraction"],
            "leakage_top_fraction": result["leakage_top_fraction"],
            "leakage_side_fraction": result["leakage_side_fraction"],
            "leakage_bottom_fraction": result["leakage_bottom_fraction"],
            "prompt_neutron_lifetime_s": th["prompt_neutron_lifetime_s"],
        },
        "thermal_hydraulics": th,
        "mesh": {
            "nr": result["nr"],
            "nz": result["nz"],
            "r_centers_cm": result["r_centers_cm"],
            "z_centers_cm": result["z_centers_cm"],
        },
        "fields": {
            "phi1_shape": result["phi1_shape"],
            "phi2_shape": result["phi2_shape"],
            "phi_total_shape": result["phi_total_shape"],
            "phi1_phys_n_cm2_s": result["phi1_phys_n_cm2_s"],
            "phi2_phys_n_cm2_s": result["phi2_phys_n_cm2_s"],
            "phi_total_phys_n_cm2_s": result["phi_total_phys_n_cm2_s"],
            "power_density_W_cm3": result["power_density_W_cm3"],
            "power_density_W_cm3_shape": result["power_density_W_cm3"],
            "fission_rate_cm3_s": result["fission_rate_cm3_s"],
            "fission_rate_cm3_s_shape": result["fission_rate_cm3_s"],
            "Jr1_shape": result["Jr1_shape"],
            "Jz1_shape": result["Jz1_shape"],
            "Jr2_shape": result["Jr2_shape"],
            "Jz2_shape": result["Jz2_shape"],
        },
        "provenance": {
            "solver_version": result["solver_version"],
            "bc_type": result["bc_type"],
            "normalization_rule": result["normalization_rule"],
            "xs_source": result["xs_source"],
            "benchmark_path": str(BENCHMARK_PATH.relative_to(REPO_ROOT)),
            "reactor_mode": result["reactor_mode"],
            "xs_model": result["xs_model"],
            "transport_tooling": xs.get("transport_tooling"),
            "xs_library_id": transport_proxy.get("library_id"),
            "xs_library_path": transport_proxy.get("library_path", str(DEFAULT_LEU_XS_LIBRARY_PATH.relative_to(REPO_ROOT))),
            "enrichment_regime": "LEU" if result["params"]["enrichment"] <= LEU_MAX_ENRICHMENT + 1.0e-12 else "benchmark",
            "benchmark_anchor": result["reactor_mode"] == FUEL_C_BENCHMARK_MODE,
            "sampling_strategy": result.get("sample_metadata", {}).get("sampling_strategy"),
            "sample_family": result.get("sample_metadata", {}).get("sample_family"),
            "sample_metadata": result.get("sample_metadata", {}),
            "alpha_T_details": result.get("alpha_T_details"),
        },
    }
    return _to_builtin(record)


def _latin_hypercube_samples(
    bounds: Mapping[str, Tuple[float, float]],
    n_samples: int,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    keys = list(bounds)
    dim = len(keys)
    intervals = np.linspace(0.0, 1.0, n_samples + 1)
    unit = np.zeros((n_samples, dim), dtype=float)
    for axis in range(dim):
        unit[:, axis] = intervals[:-1] + rng.random(n_samples) * (1.0 / n_samples)
        rng.shuffle(unit[:, axis])

    samples: List[Dict[str, float]] = []
    for row in unit:
        sample = {}
        for axis, key in enumerate(keys):
            lo, hi = bounds[key]
            sample[key] = float(lo + row[axis] * (hi - lo))
        samples.append(sample)
    return samples


def _dataset_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def _write_dataset_manifest(output_path: Path, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    k_eff = np.array([row["global_labels"]["k_eff"] for row in records], dtype=float)
    alpha = np.array([row["global_labels"]["alpha_T_pcm_per_K"] for row in records], dtype=float)
    enrichment = np.array([row["inputs"]["enrichment"] for row in records], dtype=float)
    near_critical = int(np.sum(np.abs(k_eff - 1.0) <= 0.05))
    alpha_zero_count = int(np.sum(np.abs(alpha) <= 1.0e-12))
    alpha_direct_proxy_count = sum(
        1
        for row in records
        if row.get("provenance", {}).get("alpha_T_details", {}).get("used_direct_transport_proxy")
    )

    strategy_counts: Dict[str, int] = {}
    for row in records:
        strategy = row["provenance"].get("sampling_strategy") or "unspecified"
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    manifest = {
        "dataset_id": "leu_targeted_dataset_v3",
        "record_count": len(records),
        "strategy_counts": strategy_counts,
        "near_critical_count": near_critical,
        "alpha_zero_count": alpha_zero_count,
        "alpha_direct_proxy_count": alpha_direct_proxy_count,
        "k_eff": {
            "min": float(np.min(k_eff)),
            "max": float(np.max(k_eff)),
            "mean": float(np.mean(k_eff)),
        },
        "alpha_T_pcm_per_K": {
            "min": float(np.min(alpha)),
            "max": float(np.max(alpha)),
            "mean": float(np.mean(alpha)),
        },
        "enrichment": {
            "min": float(np.min(enrichment)),
            "max": float(np.max(enrichment)),
        },
        "xs_model": "leu_transport_proxy",
        "reactor_mode": LEU_DESIGN_MODE,
        "xs_library_path": str(DEFAULT_LEU_XS_LIBRARY_PATH.relative_to(REPO_ROOT)),
        "transport_tooling": detect_transport_tooling(),
        "alpha_T_note": (
            "alpha_T is computed by central-difference temperature perturbation; exact zeros are treated as suspicious and should be investigated."
        ),
        "alpha_direct_proxy_note": (
            "Counts samples whose alpha_T perturbation used the direct transport-proxy fallback outside the tabulated LEU XS grid; "
            "these rows are retained as explicit hard cases rather than silently clamped."
        ),
    }
    manifest_path = _dataset_manifest_path(output_path)
    manifest_path.write_text(json.dumps(_to_builtin(manifest), indent=2))
    return manifest


def generate_dataset_v2(
    output_path: Path,
    n_samples: int = 96,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    broad_count = max(8, int(round(0.28 * n_samples)))
    near_critical_count = max(12, int(round(0.34 * n_samples)))
    temperature_count = max(8, int(round(0.20 * n_samples)))
    leakage_count = max(6, int(round(0.10 * n_samples)))
    edge_count = max(4, n_samples - broad_count - near_critical_count - temperature_count - leakage_count)

    broad_bounds = {
        "enrichment": (LEU_MIN_ENRICHMENT, LEU_MAX_ENRICHMENT),
        "uf4_mol_frac": (0.025, 0.05),
        "radius_cm": (45.0, 115.0),
        "height_cm": (70.0, 220.0),
        "temperature_k": (780.0, 1050.0),
        "water_vol_frac": (0.0, 0.03),
        "fuel_volume_fraction": (0.18, 0.28),
    }

    planned_params: List[Dict[str, Any]] = []
    for sample in _latin_hypercube_samples(broad_bounds, broad_count, rng):
        sample.update(
            {
                "nr": 20,
                "nz": 30,
                "reactor_mode": LEU_DESIGN_MODE,
                "xs_model": "auto",
                "sample_metadata": {
                    "sampling_strategy": "broad_latin_hypercube",
                    "sample_family": "broad_leu",
                },
            }
        )
        planned_params.append(sample)

    def add_targeted_case(sample: Dict[str, Any]) -> None:
        sample.setdefault("nr", 20)
        sample.setdefault("nz", 30)
        sample.setdefault("reactor_mode", LEU_DESIGN_MODE)
        sample.setdefault("xs_model", "auto")
        planned_params.append(sample)

    for _ in range(near_critical_count):
        for attempt in range(10):
            enrichment = float(rng.uniform(0.035, LEU_MAX_ENRICHMENT))
            uf4_mol_frac = float(rng.uniform(0.03, 0.05))
            temperature_k = float(rng.uniform(820.0, 980.0))
            fuel_volume_fraction = float(rng.uniform(0.20, 0.26))
            water_vol_frac = float(rng.uniform(0.0, 0.01))
            aspect_ratio = float(rng.uniform(1.5, 2.7))
            try:
                critical = find_critical_radius_v2(
                    k_target=1.0,
                    aspect_ratio=aspect_ratio,
                    enrichment=enrichment,
                    uf4_mol_frac=uf4_mol_frac,
                    temperature_k=temperature_k,
                    water_vol_frac=water_vol_frac,
                    fuel_volume_fraction=fuel_volume_fraction,
                )
                delta = float(rng.uniform(-0.07, 0.06))
                radius_cm = critical["critical_radius_cm"] * (1.0 + delta)
                add_targeted_case(
                    {
                        "enrichment": enrichment,
                        "uf4_mol_frac": uf4_mol_frac,
                        "radius_cm": radius_cm,
                        "height_cm": aspect_ratio * radius_cm,
                        "temperature_k": temperature_k,
                        "water_vol_frac": water_vol_frac,
                        "fuel_volume_fraction": fuel_volume_fraction,
                        "sample_metadata": {
                            "sampling_strategy": "near_critical_rooted",
                            "sample_family": "near_critical",
                            "critical_radius_cm": critical["critical_radius_cm"],
                            "critical_delta_fraction": delta,
                            "critical_target": 1.0,
                        },
                    }
                )
                break
            except ValueError:
                if attempt == 9:
                    pass

    temperature_offsets = [-90.0, -55.0, -25.0, 0.0, 25.0, 55.0, 90.0]
    for _ in range(temperature_count):
        enrichment = float(rng.uniform(0.038, LEU_MAX_ENRICHMENT))
        uf4_mol_frac = float(rng.uniform(0.03, 0.045))
        reference_temperature = float(rng.uniform(860.0, 940.0))
        fuel_volume_fraction = float(rng.uniform(0.20, 0.25))
        aspect_ratio = float(rng.uniform(1.7, 2.4))
        critical = find_critical_radius_v2(
            k_target=1.0,
            aspect_ratio=aspect_ratio,
            enrichment=enrichment,
            uf4_mol_frac=uf4_mol_frac,
            temperature_k=reference_temperature,
            water_vol_frac=0.0,
            fuel_volume_fraction=fuel_volume_fraction,
        )
        offset = float(rng.choice(temperature_offsets))
        temperature_k = float(np.clip(reference_temperature + offset, 780.0, 1075.0))
        radius_cm = critical["critical_radius_cm"] * float(rng.uniform(0.97, 1.03))
        add_targeted_case(
            {
                "enrichment": enrichment,
                "uf4_mol_frac": uf4_mol_frac,
                "radius_cm": radius_cm,
                "height_cm": aspect_ratio * radius_cm,
                "temperature_k": temperature_k,
                "water_vol_frac": 0.0,
                "fuel_volume_fraction": fuel_volume_fraction,
                "sample_metadata": {
                    "sampling_strategy": "temperature_reactivity_band",
                    "sample_family": "temperature_sensitive",
                    "reference_temperature_K": reference_temperature,
                    "temperature_offset_K": offset,
                    "critical_radius_cm": critical["critical_radius_cm"],
                },
            }
        )

    for _ in range(leakage_count):
        enrichment = float(rng.uniform(0.04, LEU_MAX_ENRICHMENT))
        uf4_mol_frac = float(rng.uniform(0.03, 0.05))
        temperature_k = float(rng.uniform(840.0, 980.0))
        fuel_volume_fraction = float(rng.uniform(0.18, 0.24))
        aspect_ratio = float(rng.uniform(1.1, 3.2))
        critical = find_critical_radius_v2(
            k_target=1.0,
            aspect_ratio=aspect_ratio,
            enrichment=enrichment,
            uf4_mol_frac=uf4_mol_frac,
            temperature_k=temperature_k,
            water_vol_frac=0.0,
            fuel_volume_fraction=fuel_volume_fraction,
        )
        radius_cm = critical["critical_radius_cm"] * float(rng.uniform(0.90, 1.02))
        add_targeted_case(
            {
                "enrichment": enrichment,
                "uf4_mol_frac": uf4_mol_frac,
                "radius_cm": radius_cm,
                "height_cm": aspect_ratio * radius_cm,
                "temperature_k": temperature_k,
                "water_vol_frac": 0.0,
                "fuel_volume_fraction": fuel_volume_fraction,
                "sample_metadata": {
                    "sampling_strategy": "leakage_sensitive_geometry",
                    "sample_family": "boundary_sensitive",
                    "aspect_ratio": aspect_ratio,
                    "critical_radius_cm": critical["critical_radius_cm"],
                },
            }
        )

    for _ in range(edge_count):
        hot_edge = bool(rng.integers(0, 2))
        temperature_k = float(rng.uniform(1020.0, 1080.0) if hot_edge else rng.uniform(780.0, 830.0))
        radius_cm = float(rng.uniform(42.0, 120.0))
        aspect_ratio = float(rng.uniform(1.2, 3.0))
        add_targeted_case(
            {
                "enrichment": float(rng.uniform(LEU_MIN_ENRICHMENT, LEU_MAX_ENRICHMENT)),
                "uf4_mol_frac": float(rng.uniform(0.025, 0.05)),
                "radius_cm": radius_cm,
                "height_cm": aspect_ratio * radius_cm,
                "temperature_k": temperature_k,
                "water_vol_frac": float(rng.uniform(0.02, 0.05)),
                "fuel_volume_fraction": float(rng.uniform(0.18, 0.28)),
                "sample_metadata": {
                    "sampling_strategy": "edge_case_extremes",
                    "sample_family": "hard_case",
                    "hot_edge": hot_edge,
                },
            }
        )

    while len(planned_params) < n_samples:
        fallback = _latin_hypercube_samples(broad_bounds, 1, rng)[0]
        fallback.update(
            {
                "nr": 20,
                "nz": 30,
                "reactor_mode": LEU_DESIGN_MODE,
                "xs_model": "auto",
                "sample_metadata": {
                    "sampling_strategy": "broad_latin_hypercube",
                    "sample_family": "broad_leu",
                    "fallback_fill": True,
                },
            }
        )
        planned_params.append(fallback)

    planned_params = planned_params[:n_samples]
    for idx, params in enumerate(planned_params):
        result = evaluate_design_v2(**params)
        alpha = compute_temperature_coefficient_v2(
            {
                "enrichment": params["enrichment"],
                "uf4_mol_frac": params["uf4_mol_frac"],
                "radius_cm": params["radius_cm"],
                "height_cm": params["height_cm"],
                "temperature_k": params["temperature_k"],
                "water_vol_frac": params["water_vol_frac"],
                "fuel_volume_fraction": params["fuel_volume_fraction"],
                "reactor_mode": params["reactor_mode"],
                "xs_model": params["xs_model"],
            },
            dT=12.5,
        )
        result["alpha_T_pcm_per_K"] = alpha["alpha_pcm_per_K"]
        result["alpha_T_details"] = alpha
        record = export_sample_record(result, sample_id=f"v2-{idx:04d}")
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_builtin(records), indent=2))
    _write_dataset_manifest(output_path, records)
    return records
