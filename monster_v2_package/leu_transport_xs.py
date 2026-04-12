#!/usr/bin/env python3
"""Transport-informed LEU few-group proxy support for the v2 solver.

This module intentionally separates three concerns:

1. detecting whether a real transport toolchain is available,
2. deriving an explicit fallback LEU 2-group proxy when it is not,
3. loading/interpolating a reusable tabulated LEU XS library.

The current environment does not provide OpenMC, SCALE/NEWT, or Serpent, so
the repository falls back to a transport-informed proxy. The proxy is built
from the benchmark-guided v2 XS set, but then re-centered for the LEU design
region using explicit moderation, loading, and temperature trends consistent
with the ORNL reference lattice and the project's target D2 range.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import itertools
import json
import math
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEU_XS_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "leu_transport_xs"
DEFAULT_LEU_XS_LIBRARY_PATH = DEFAULT_LEU_XS_ARTIFACT_DIR / "leu_transport_proxy_library.json"

LEU_TRANSPORT_PROXY_LIBRARY_ID = "leu_transport_proxy_v1"
REFERENCE_LATTICE_FUEL_VOLUME_FRACTION = 0.225
REFERENCE_LATTICE_GRAPHITE_VOLUME_FRACTION = 0.775
REFERENCE_GRAPHITE_TO_FUEL_RATIO = (
    REFERENCE_LATTICE_GRAPHITE_VOLUME_FRACTION / REFERENCE_LATTICE_FUEL_VOLUME_FRACTION
)

TRANSPORT_XS_KEYS = (
    "D1",
    "D2",
    "Sigma_a1",
    "Sigma_a2",
    "Sigma_f1",
    "Sigma_f2",
    "nu_Sigma_f1",
    "nu_Sigma_f2",
    "Sigma_s12",
    "Sigma_r1",
    "chi1",
    "chi2",
    "k_inf",
)


def detect_transport_tooling() -> Dict[str, Any]:
    """Detect locally available transport tooling.

    The repo should not pretend transport is available when it is not, so this
    detection result is written into every generated LEU XS artifact.
    """

    python_modules = {
        "openmc": importlib.util.find_spec("openmc") is not None,
        "serpentTools": importlib.util.find_spec("serpentTools") is not None,
    }
    executables = {
        "openmc": shutil.which("openmc") is not None,
        "scale": shutil.which("scale") is not None,
        "xsproc": shutil.which("xsproc") is not None,
        "newt": shutil.which("newt") is not None,
        "serpent": shutil.which("serpent") is not None,
        "sss2": shutil.which("sss2") is not None,
        "sss2x": shutil.which("sss2x") is not None,
    }

    full_transport_available = bool(
        python_modules["openmc"]
        or executables["openmc"]
        or executables["scale"]
        or executables["newt"]
        or executables["serpent"]
        or executables["sss2"]
        or executables["sss2x"]
    )

    if python_modules["openmc"] or executables["openmc"]:
        preferred_tool = "OpenMC"
    elif executables["scale"] or executables["newt"] or executables["xsproc"]:
        preferred_tool = "SCALE/NEWT"
    elif executables["serpent"] or executables["sss2"] or executables["sss2x"]:
        preferred_tool = "Serpent"
    else:
        preferred_tool = "transport_informed_proxy"

    return {
        "python_modules": python_modules,
        "executables": executables,
        "full_transport_available": full_transport_available,
        "preferred_path": preferred_tool,
        "fallback_reason": (
            "No OpenMC, SCALE/NEWT, or Serpent toolchain was detected in this environment."
            if not full_transport_available
            else None
        ),
    }


def default_transport_proxy_axes() -> Dict[str, List[float]]:
    return {
        "enrichment": [0.03, 0.035, 0.04, 0.045, 0.05],
        "uf4_mol_frac": [0.025, 0.035, 0.04, 0.05],
        "temperature_K": [760.0, 800.0, 900.0, 1000.0, 1100.0],
        "fuel_volume_fraction": [0.18, 0.21, 0.225, 0.24, 0.27],
        "water_vol_frac": [0.0, 0.03],
    }


def transport_proxy_method_summary() -> Dict[str, Any]:
    tooling = detect_transport_tooling()
    return {
        "library_id": LEU_TRANSPORT_PROXY_LIBRARY_ID,
        "path": "transport_informed_proxy",
        "tooling_status": tooling,
        "group_structure_note": (
            "Two-group LEU constants are tied to the explicit project reporting groups and"
            " generated from a transport-inspired collapse proxy because a full transport"
            " toolchain is unavailable in this environment."
        ),
        "assumptions": [
            "Uses the ORNL one-region reference lattice of 22.5 vol% fuel and 77.5 vol% graphite as the moderation anchor.",
            "Starts from the repo's benchmark-aware homogenized XS set, then applies explicit LEU-region spectral and moderation adjustments.",
            "Constrains thermal diffusion to a graphite-moderated LEU target band centered near 0.36 cm with state-dependent adjustments.",
            "Treats water as a continuity variable only; the main design path remains graphite-moderated and LEU-focused.",
            "Preserves Fuel C benchmark support separately instead of forcing benchmark calibration into the LEU design path.",
        ],
        "limitations": [
            "Not a true transport collapse from OpenMC, SCALE, or Serpent.",
            "Does not include self-shielded multigroup resonance treatment or explicit cell transport.",
            "Should be replaced by a real MGXS workflow when transport tooling becomes available.",
        ],
    }


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def derive_transport_proxy_xs(
    *,
    state: Mapping[str, float],
    benchmark_guided_xs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Derive a transport-informed LEU 2-group proxy XS set from the v2 base XS.

    The returned structure mirrors the solver's XS dictionary so it can be used
    directly by the existing diffusion solve.
    """

    enrichment = float(state["enrichment"])
    uf4_mol_frac = float(state["uf4_mol_frac"])
    temperature_k = float(state["temperature_K"])
    fuel_volume_fraction = float(state["fuel_volume_fraction"])
    water_vol_frac = float(state.get("water_vol_frac", 0.0))
    graphite_volume_fraction = max(0.0, 1.0 - fuel_volume_fraction - water_vol_frac)

    temp_ratio = max(temperature_k, 1.0) / 900.0
    uf4_ratio = uf4_mol_frac / 0.04
    fuel_ratio = fuel_volume_fraction / REFERENCE_LATTICE_FUEL_VOLUME_FRACTION
    graphite_ratio = graphite_volume_fraction / max(REFERENCE_LATTICE_GRAPHITE_VOLUME_FRACTION, 1.0e-8)
    enrichment_ratio = enrichment / 0.05
    graphite_to_fuel_ratio = graphite_volume_fraction / max(fuel_volume_fraction, 1.0e-8)

    moderation_index = (
        (graphite_to_fuel_ratio / REFERENCE_GRAPHITE_TO_FUEL_RATIO)
        * temp_ratio ** (-0.15)
        * (0.04 / max(uf4_mol_frac, 1.0e-8)) ** 0.10
        * (1.0 + 0.30 * water_vol_frac)
    )
    spectral_hardness = _clip(
        1.0
        + 0.25 * (uf4_ratio - 1.0)
        + 0.18 * (fuel_ratio - 1.0)
        - 0.20 * (graphite_ratio - 1.0)
        + 0.10 * (temp_ratio - 1.0)
        - 0.06 * (enrichment_ratio - 1.0)
        - 0.08 * (water_vol_frac / 0.03 if water_vol_frac > 0.0 else 0.0),
        0.76,
        1.24,
    )

    thermal_fission_factor = _clip(
        1.02 + 0.08 * (moderation_index - 1.0) - 0.06 * (spectral_hardness - 1.0),
        0.97,
        1.10,
    )
    thermal_absorption_factor = _clip(
        0.98
        + 0.10 * (fuel_ratio - 1.0)
        + 0.06 * (uf4_ratio - 1.0)
        + 0.05 * (spectral_hardness - 1.0),
        0.93,
        1.06,
    )
    fast_absorption_factor = _clip(
        0.99 + 0.08 * math.log(max(temp_ratio, 1.0e-8)) + 0.04 * (uf4_ratio - 1.0),
        0.94,
        1.08,
    )
    downscatter_factor = _clip(
        0.78 + 0.12 * (moderation_index - 1.0) - 0.05 * (spectral_hardness - 1.0),
        0.68,
        0.90,
    )

    d1_factor = _clip(1.00 + 0.015 * (temp_ratio - 1.0) - 0.015 * (moderation_index - 1.0), 0.94, 1.06)
    d2_target_cm = _clip(
        0.36
        + 0.04 * (moderation_index - 1.0)
        + 0.02 * (temp_ratio - 1.0)
        - 0.025 * (fuel_ratio - 1.0)
        - 0.015 * (uf4_ratio - 1.0)
        - 0.04 * water_vol_frac,
        0.32,
        0.48,
    )

    sigma_f1 = float(benchmark_guided_xs["Sigma_f1"]) * (1.0 + 0.025 * (enrichment_ratio - 1.0))
    sigma_f2 = float(benchmark_guided_xs["Sigma_f2"]) * thermal_fission_factor
    nu_sigma_f1 = float(benchmark_guided_xs["nu_Sigma_f1"]) * (sigma_f1 / max(float(benchmark_guided_xs["Sigma_f1"]), 1.0e-18))
    nu_sigma_f2 = float(benchmark_guided_xs["nu_Sigma_f2"]) * thermal_fission_factor
    sigma_a1 = float(benchmark_guided_xs["Sigma_a1"]) * fast_absorption_factor
    sigma_a2 = float(benchmark_guided_xs["Sigma_a2"]) * thermal_absorption_factor
    sigma_s12 = float(benchmark_guided_xs["Sigma_s12"]) * downscatter_factor
    sigma_r1 = sigma_a1 + sigma_s12
    d1 = float(benchmark_guided_xs["D1"]) * d1_factor
    d2 = d2_target_cm

    k_inf = (nu_sigma_f1 + nu_sigma_f2 * sigma_s12 / max(sigma_a2, 1.0e-18)) / max(sigma_r1, 1.0e-18)

    proxy_metadata = {
        "library_id": LEU_TRANSPORT_PROXY_LIBRARY_ID,
        "proxy_method": "transport_informed_proxy",
        "moderation_index": moderation_index,
        "spectral_hardness": spectral_hardness,
        "thermal_fission_factor": thermal_fission_factor,
        "thermal_absorption_factor": thermal_absorption_factor,
        "fast_absorption_factor": fast_absorption_factor,
        "downscatter_factor": downscatter_factor,
        "d1_factor": d1_factor,
        "d2_target_cm": d2_target_cm,
        "reference_lattice_fuel_volume_fraction": REFERENCE_LATTICE_FUEL_VOLUME_FRACTION,
        "reference_lattice_graphite_volume_fraction": REFERENCE_LATTICE_GRAPHITE_VOLUME_FRACTION,
    }

    return {
        "D1": d1,
        "D2": d2,
        "D2_raw": float(benchmark_guided_xs.get("D2_raw", d2)),
        "Sigma_a1": sigma_a1,
        "Sigma_a2": sigma_a2,
        "Sigma_f1": sigma_f1,
        "Sigma_f2": sigma_f2,
        "nu_Sigma_f1": nu_sigma_f1,
        "nu_Sigma_f2": nu_sigma_f2,
        "Sigma_s12": sigma_s12,
        "Sigma_r1": sigma_r1,
        "chi1": float(benchmark_guided_xs.get("chi1", 1.0)),
        "chi2": float(benchmark_guided_xs.get("chi2", 0.0)),
        "k_inf": k_inf,
        "transport_proxy": proxy_metadata,
        "corrections": {
            **dict(benchmark_guided_xs.get("corrections", {})),
            "transport_proxy": proxy_metadata,
        },
    }


def _record_key(dimensions: Sequence[str], values: Sequence[float]) -> Tuple[str, ...]:
    return tuple(f"{float(value):.8f}" for value in values[: len(dimensions)])


def build_transport_proxy_library_payload(
    *,
    axes: Mapping[str, Sequence[float]],
    group_structure: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "library_id": LEU_TRANSPORT_PROXY_LIBRARY_ID,
        "method": transport_proxy_method_summary(),
        "group_structure": group_structure,
        "axes": {key: [float(value) for value in values] for key, values in axes.items()},
        "records": list(records),
    }


def write_transport_proxy_library(
    *,
    output_path: Path,
    axes: Mapping[str, Sequence[float]],
    group_structure: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = build_transport_proxy_library_payload(
        axes=axes,
        group_structure=group_structure,
        records=records,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


@lru_cache(maxsize=4)
def load_transport_proxy_library(path_str: str) -> Dict[str, Any]:
    return json.loads(Path(path_str).read_text())


def _state_within_library_axes(
    *,
    state: Mapping[str, float],
    axes: Mapping[str, np.ndarray],
) -> Tuple[bool, List[str]]:
    out_of_bounds = []
    for dim, axis in axes.items():
        value = float(state[dim])
        if value < float(axis[0]) or value > float(axis[-1]):
            out_of_bounds.append(dim)
    return len(out_of_bounds) == 0, out_of_bounds


def interpolate_transport_proxy_xs(
    *,
    library_path: Path,
    state: Mapping[str, float],
) -> Dict[str, Any]:
    library = load_transport_proxy_library(str(library_path))
    dimensions = list(library["axes"])
    axes = {key: np.asarray(values, dtype=float) for key, values in library["axes"].items()}
    record_lookup = {
        _record_key(dimensions, [record["state"][dim] for dim in dimensions]): record
        for record in library["records"]
    }

    neighbors: List[List[Tuple[float, float]]] = []
    for dim in dimensions:
        axis = axes[dim]
        value = float(state[dim])
        if value <= axis[0]:
            neighbors.append([(float(axis[0]), 1.0)])
            continue
        if value >= axis[-1]:
            neighbors.append([(float(axis[-1]), 1.0)])
            continue

        upper_idx = int(np.searchsorted(axis, value, side="right"))
        lower_idx = upper_idx - 1
        lower = float(axis[lower_idx])
        upper = float(axis[upper_idx])
        span = max(upper - lower, 1.0e-18)
        frac = (value - lower) / span
        neighbors.append([(lower, 1.0 - frac), (upper, frac)])

    xs = {key: 0.0 for key in TRANSPORT_XS_KEYS}
    nearest_record = None
    nearest_weight = -1.0

    for corner in itertools.product(*neighbors):
        corner_values = [value for value, _ in corner]
        weight = float(np.prod([corner_weight for _, corner_weight in corner]))
        record = record_lookup[_record_key(dimensions, corner_values)]
        if weight > nearest_weight:
            nearest_weight = weight
            nearest_record = record
        for key in TRANSPORT_XS_KEYS:
            xs[key] += weight * float(record["xs"][key])

    assert nearest_record is not None
    xs["D2_raw"] = float(nearest_record["xs"].get("D2_raw", xs["D2"]))
    xs["transport_proxy"] = {
        **dict(nearest_record["metadata"]),
        "proxy_method": "interpolated_transport_proxy_library",
        "library_id": library["library_id"],
        "library_path": str(library_path.relative_to(REPO_ROOT)),
        "interpolation_state": {key: float(state[key]) for key in dimensions},
    }
    xs["corrections"] = {
        "transport_proxy": xs["transport_proxy"],
    }
    return xs


def resolve_transport_proxy_xs(
    *,
    state: Mapping[str, float],
    benchmark_guided_xs: Mapping[str, Any],
    library_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve LEU transport proxy XS from a library when available, otherwise derive directly."""

    path = Path(library_path) if library_path is not None else DEFAULT_LEU_XS_LIBRARY_PATH
    if path.exists():
        library = load_transport_proxy_library(str(path))
        axes = {key: np.asarray(values, dtype=float) for key, values in library["axes"].items()}
        within_axes, out_of_bounds_dims = _state_within_library_axes(state=state, axes=axes)
        if within_axes:
            return interpolate_transport_proxy_xs(library_path=path, state=state)

        xs = derive_transport_proxy_xs(state=state, benchmark_guided_xs=benchmark_guided_xs)
        xs["transport_proxy"] = {
            **dict(xs["transport_proxy"]),
            "proxy_method": "direct_transport_proxy_formula_outside_library",
            "library_id": LEU_TRANSPORT_PROXY_LIBRARY_ID,
            "library_path": str(path.relative_to(REPO_ROOT)),
            "requested_state": {key: float(value) for key, value in state.items()},
            "out_of_bounds_dimensions": out_of_bounds_dims,
            "library_axes_bounds": {
                key: [float(axis[0]), float(axis[-1])] for key, axis in axes.items()
            },
        }
        xs["corrections"] = {
            **dict(xs.get("corrections", {})),
            "transport_proxy": xs["transport_proxy"],
        }
        return xs

    xs = derive_transport_proxy_xs(state=state, benchmark_guided_xs=benchmark_guided_xs)
    xs["transport_proxy"] = {
        **dict(xs["transport_proxy"]),
        "proxy_method": "direct_transport_proxy_formula",
        "library_id": LEU_TRANSPORT_PROXY_LIBRARY_ID,
        "library_path": str(path.relative_to(REPO_ROOT)),
    }
    xs["corrections"] = {
        **dict(xs.get("corrections", {})),
        "transport_proxy": xs["transport_proxy"],
    }
    return xs


def scalar_transport_proxy_row(record: Mapping[str, Any]) -> Dict[str, float]:
    state = record["state"]
    xs = record["xs"]
    metadata = record["metadata"]
    return {
        "enrichment": float(state["enrichment"]),
        "uf4_mol_frac": float(state["uf4_mol_frac"]),
        "temperature_K": float(state["temperature_K"]),
        "fuel_volume_fraction": float(state["fuel_volume_fraction"]),
        "water_vol_frac": float(state["water_vol_frac"]),
        "D1_cm": float(xs["D1"]),
        "D2_cm": float(xs["D2"]),
        "Sigma_a1_cm1": float(xs["Sigma_a1"]),
        "Sigma_a2_cm1": float(xs["Sigma_a2"]),
        "Sigma_s12_cm1": float(xs["Sigma_s12"]),
        "nuSigma_f1_cm1": float(xs["nu_Sigma_f1"]),
        "nuSigma_f2_cm1": float(xs["nu_Sigma_f2"]),
        "k_inf": float(xs["k_inf"]),
        "moderation_index": float(metadata["moderation_index"]),
        "spectral_hardness": float(metadata["spectral_hardness"]),
    }
