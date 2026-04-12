#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "solver_v2"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_DIR / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from legacy.v1.mjm_solver_v1 import (
    compute_temperature_coefficient as legacy_temperature_coefficient,
    effective_delayed_fraction as legacy_delayed_fraction,
    evaluate_design as legacy_evaluate_design,
)
from monster_v2_package.solver_v2 import (
    ALPHA_TARGET_PCM_PER_K,
    BENCHMARK_CIRCULATING_BETA,
    BENCHMARK_STATIC_BETA,
    DELAYED_RATIO_TARGET,
    FUEL_C_BENCHMARK_MODE,
    LEU_DESIGN_MODE,
    THERMAL_FLUX_TARGET,
    compute_temperature_sweep_v2,
    evaluate_design_v2,
    export_sample_record,
    find_critical_radius_v2,
    generate_dataset_v2,
)

SOLVER_SECTION_EXPLANATIONS = {
    "legacy_baseline": "Legacy v1 baseline outputs used for before/after comparison.",
    "energy_groups": "Explicit assumed fast/thermal energy-group definitions used for reporting and exported metadata.",
    "upgraded_design": "Upgraded v2 critical-dimension search result for the MJM design point.",
    "fuel_c_proxy": "Benchmark-centered Fuel C proxy case used for validation against ORNL targets.",
    "temperature_sweep": "Temperature sweep containing k_eff(T), dk/dT, and alpha_T estimates.",
    "dataset_records": "Number of exported v2 dataset samples generated in this run.",
    "targets": "ORNL Fuel C target values used as validation anchors.",
}

GLOBAL_LABEL_EXPLANATIONS = {
    "k_eff": "Effective multiplication factor for the finite-cylinder system.",
    "k_inf": "Infinite-medium multiplication factor from the homogenized 2-group model.",
    "reactivity_pcm": "Reactivity expressed in pcm using the solved k_eff.",
    "alpha_T_pcm_per_K": "Temperature coefficient of reactivity in pcm/K.",
    "peak_flux_n_cm2_s": "Maximum physical total neutron flux.",
    "peak_thermal_flux_n_cm2_s": "Maximum physical thermal-group flux.",
    "avg_flux_n_cm2_s": "Volume-averaged total physical flux.",
    "avg_thermal_flux_n_cm2_s": "Volume-averaged thermal-group physical flux.",
    "peaking_factor": "Peak total flux divided by average total flux.",
    "peak_power_density_W_cm3": "Maximum local power density.",
    "avg_power_density_W_cm3": "Volume-averaged power density.",
    "power_density_peaking_factor": "Peak-to-average power density ratio.",
    "beta_static": "Static effective delayed neutron fraction.",
    "beta_eff": "Circulating-fuel effective delayed neutron fraction.",
    "beta_eff_ratio": "beta_eff / beta_static for the circulating-fuel case.",
    "leakage_fraction": "Approximate total leakage fraction diagnostic.",
    "leakage_top_fraction": "Fraction of estimated leakage through the top boundary.",
    "leakage_side_fraction": "Fraction of estimated leakage through the radial boundary.",
    "leakage_bottom_fraction": "Fraction of estimated leakage through the bottom boundary.",
    "prompt_neutron_lifetime_s": "Prompt neutron lifetime proxy carried from benchmark context.",
}

DATASET_RECORD_EXPLANATIONS = {
    "inputs": "Design variables and operating conditions used for the sample.",
    "geometry": "Derived geometric and buckling quantities, including extrapolated dimensions.",
    "material_properties": "Thermophysical properties used in the sample.",
    "number_densities": "Constituent number densities used to build the XS model.",
    "xs": "Homogenized 2-group macroscopic cross sections and correction factors.",
    "global_labels": "Scalar reactor-response quantities for validation and learning.",
    "thermal_hydraulics": "Loop/state-point estimates derived from the solved design.",
    "mesh": "Spatial grid used for the exported field arrays.",
    "fields": "2D field arrays for flux, fission rate, and power density.",
    "provenance": "Normalization, boundary-condition, and source metadata.",
}


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def plot_temperature_sweep(sweep: dict, out_path: Path) -> None:
    rows = sweep["rows"]
    temps = [row["temperature_K"] for row in rows]
    k_eff = [row["k_eff"] for row in rows]
    alpha = [row["alpha_pcm_per_K"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))
    ax1.plot(temps, k_eff, "o-", color="#005f73", linewidth=2, label="k_eff")
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("k_eff", color="#005f73")
    ax1.tick_params(axis="y", labelcolor="#005f73")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(temps, alpha, "s--", color="#ae2012", linewidth=1.8, label="alpha_T")
    ax2.axhline(ALPHA_TARGET_PCM_PER_K, color="#ca6702", linestyle=":", linewidth=1.5)
    ax2.set_ylabel("alpha_T (pcm/K)", color="#ae2012")
    ax2.tick_params(axis="y", labelcolor="#ae2012")

    ax1.set_title("Upgraded Fuel C Proxy Temperature Reactivity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fields(result: dict, out_path: Path) -> None:
    r = np.array(result["r_centers_cm"])
    z = np.array(result["z_centers_cm"])
    rr, zz = np.meshgrid(r, z, indexing="ij")

    panels = [
        ("phi1_phys_n_cm2_s", "Fast Flux phi1", "inferno"),
        ("phi2_phys_n_cm2_s", "Thermal Flux phi2", "plasma"),
        ("phi_total_phys_n_cm2_s", "Total Flux", "viridis"),
        ("power_density_W_cm3", "Power Density", "magma"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, (key, title, cmap) in zip(axes.flat, panels, strict=True):
        mesh = ax.pcolormesh(rr, zz, np.array(result[key]), shading="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("r (cm)")
        ax.set_ylabel("z (cm)")
        fig.colorbar(mesh, ax=ax, shrink=0.82)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_leu_design_space(out_path: Path) -> dict:
    enrichment_vals = np.linspace(0.03, 0.05, 11)
    radius_vals = np.linspace(50.0, 100.0, 13)
    kmap = np.zeros((len(enrichment_vals), len(radius_vals)), dtype=float)

    for i, enr in enumerate(enrichment_vals):
        for j, radius_cm in enumerate(radius_vals):
            result = evaluate_design_v2(
                enrichment=float(enr),
                uf4_mol_frac=0.04,
                radius_cm=float(radius_cm),
                height_cm=float(2.0 * radius_cm),
                temperature_k=900.0,
                water_vol_frac=0.0,
                nr=18,
                nz=28,
                reactor_mode=LEU_DESIGN_MODE,
            )
            kmap[i, j] = result["k_eff"]

    critical_leu = find_critical_radius_v2(
        k_target=1.0,
        aspect_ratio=2.0,
        enrichment=0.05,
        uf4_mol_frac=0.04,
        temperature_k=900.0,
        water_vol_frac=0.0,
        r_bounds_cm=(60.0, 120.0),
        reactor_mode=LEU_DESIGN_MODE,
    )

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    rr, ee = np.meshgrid(radius_vals, enrichment_vals * 100.0)
    mesh = ax.pcolormesh(rr, ee, kmap, cmap="RdYlGn", shading="gouraud", vmin=0.5, vmax=1.2)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("k_eff")

    contour_levels = [1.0, 1.03]
    try:
        cs = ax.contour(rr, ee, kmap, levels=contour_levels, colors=["white", "#ffd166"], linewidths=[2.2, 1.8])
        ax.clabel(cs, fmt={1.0: "k=1.0", 1.03: "k=1.03"}, fontsize=9)
    except Exception:
        pass

    ax.plot(
        critical_leu["critical_radius_cm"],
        5.0,
        marker="*",
        markersize=18,
        color="#c1121f",
        label="LEU critical point (5.0% enrichment)",
    )
    ax.axhline(5.0, color="#003049", linestyle="--", linewidth=1.5, label="LEU upper limit (5.0%)")
    ax.set_title(
        "LEU Design Space: k_eff(Enrichment, Radius)\n"
        "10 MW(th) | FLiBe + 4% UF4 | H=2R | 900 K",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Core radius (cm)")
    ax.set_ylabel("Enrichment (%)")
    ax.set_ylim(enrichment_vals.min() * 100.0, enrichment_vals.max() * 100.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "radius_cm_values": radius_vals.tolist(),
        "enrichment_percent_values": (enrichment_vals * 100.0).tolist(),
        "k_eff_map": kmap.tolist(),
        "critical_point": {
            "enrichment_percent": 5.0,
            "radius_cm": critical_leu["critical_radius_cm"],
            "height_cm": critical_leu["critical_height_cm"],
            "k_eff": critical_leu["k_eff"],
        },
        "assumptions": {
            "uf4_mol_frac": 0.04,
            "aspect_ratio": 2.0,
            "temperature_K": 900.0,
            "water_vol_frac": 0.0,
        },
    }


def write_report(
    out_path: Path,
    *,
    baseline_legacy: dict,
    upgraded_design: dict,
    fuel_c_result: dict,
    sweep: dict,
    dataset_path: Path,
) -> None:
    legacy_tc = legacy_temperature_coefficient(
        {
            "enrichment": 0.1975,
            "uf4_mol_frac": 0.04,
            "radius": 23.6,
            "height": 47.2,
            "temperature": 900.0,
            "water_vol_frac": 0.0,
        },
        dT=25.0,
    )
    legacy_dn_ref = legacy_delayed_fraction(9.37, 16.45)

    mean_alpha = sweep["mean_alpha_pcm_per_K"]
    alpha_gap = mean_alpha - ALPHA_TARGET_PCM_PER_K
    delayed_ratio = fuel_c_result["delayed_neutrons"]["reduction_factor"]
    delayed_gap = delayed_ratio - DELAYED_RATIO_TARGET
    thermal_flux = fuel_c_result["peak_thermal_flux_n_cm2_s"]
    thermal_flux_ratio = thermal_flux / THERMAL_FLUX_TARGET

    lines = [
        "# MJM Solver Upgrade Report",
        "",
        "## What Was Wrong",
        "",
        "- Legacy power normalization integrated `(Sigma_f1 + Sigma_f2) * (phi1 + phi2)` instead of the group-wise fission source.",
        "- Legacy delayed-neutron logic overwrote its own estimate and collapsed to an unrealistically weak circulating-fuel penalty.",
        "- Legacy solver imposed zero flux at the physical wall rather than an extrapolated-vacuum treatment.",
        "- Legacy data export did not save field labels, so the surrogate pipeline only saw scalar targets.",
        "",
        "## What Changed",
        "",
        "- Added an explicit split between LEU design mode and Fuel C benchmark-anchor mode in `monster_v2_package/solver_v2.py`.",
        "- Added extrapolated-vacuum leakage in the finite-difference operator and explicit buckling outputs `Br^2`, `Bz^2`, and `Bg^2`.",
        "- Added a repaired delayed-neutron model calibrated to the ORNL Fuel C circulating/static ratio using the benchmark group data.",
        "- Added a transport-informed LEU XS proxy path plus a richer dataset exporter that writes scalar labels, XS terms, buckling features, mesh metadata, currents, and full 2D fields.",
        "- Kept Fuel C as a validation anchor only; the active design path now uses the LEU proxy library instead of benchmark-guided defaults.",
        "",
        "## Before / After Snapshot",
        "",
        f"- Legacy MJM design peak flux: `{baseline_legacy['peak_flux']:.3e} n/cm^2/s`",
        f"- Upgraded MJM design peak flux: `{upgraded_design['peak_flux_n_cm2_s']:.3e} n/cm^2/s`",
        f"- Legacy MJM alpha_T near 900 K: `{legacy_tc['alpha_pcm_per_K']:.2f} pcm/K`",
        f"- Upgraded Fuel C proxy mean alpha_T: `{mean_alpha:.2f} pcm/K`",
        f"- ORNL Fuel C target alpha_T: `{ALPHA_TARGET_PCM_PER_K:.3f} pcm/K`",
        f"- Legacy delayed ratio at ORNL residence times: `{legacy_dn_ref['reduction_factor']:.3f}`",
        f"- Upgraded delayed ratio at Fuel C proxy: `{delayed_ratio:.3f}`",
        f"- ORNL delayed ratio target: `{DELAYED_RATIO_TARGET:.3f}`",
        f"- Fuel C proxy peak thermal flux: `{thermal_flux:.3e} n/cm^2/s`",
        f"- ORNL peak thermal flux target: `{THERMAL_FLUX_TARGET:.3e} n/cm^2/s`",
        "",
        "## Benchmark Improvement Summary",
        "",
        f"- Temperature coefficient gap to ORNL: `{alpha_gap:+.2f} pcm/K`",
        f"- Delayed-neutron ratio gap to ORNL: `{delayed_gap:+.3f}`",
        f"- Peak thermal flux / ORNL target: `{thermal_flux_ratio:.3f}`",
        f"- Upgraded `D2`: `{fuel_c_result['xs']['D2']:.3f} cm`",
        f"- Upgraded side-leakage share: `{fuel_c_result['leakage_side_fraction']:.3f}`",
        "- Not every fidelity target is fixed: Fuel C peak thermal flux remains high and the delayed ratio is still above the benchmark target.",
        "",
        "## Exported Artifacts",
        "",
        f"- Dataset: `{dataset_path.relative_to(ROOT)}`",
        "- Flux/power plot: `artifacts/solver_v2/fuel_c_proxy_fields.png`",
        "- Temperature plot: `artifacts/solver_v2/fuel_c_temperature_sweep.png`",
        "- Machine-readable summary: `artifacts/solver_v2/summary.json`",
        "- LEU XS library: `artifacts/leu_transport_xs/leu_transport_proxy_library.json`",
        "",
        "## Remaining Gaps",
        "",
        "- LEU design cross sections are now transport-informed proxies, not full transport-derived GAM/THERMOS or Serpent/SCALE group constants.",
        "- The delayed-neutron model is calibrated to the Fuel C benchmark rather than derived from a full precursor transport solve.",
        "- Flux magnitude remains a major limitation: Fuel C peak thermal flux is still well above the ORNL target even after the LEU cleanup.",
        "- The field surrogate dataset is now training-ready, but higher-fidelity flux magnitudes and kinetics terms will still benefit from real transport-derived XS tables.",
        "",
    ]
    out_path.write_text("\n".join(lines))


def build_solver_bundle(summary: dict) -> dict:
    fuel_c = summary["fuel_c_proxy"]
    upgraded = summary["upgraded_design"]
    bundle = {
        "overview": {
            "report_purpose": "Single-file labeled view of the current upgraded solver outputs.",
            "artifact_root": str(ARTIFACT_DIR.relative_to(ROOT)),
            "dataset_records": summary["dataset_records"],
        },
        "summary_section_definitions": SOLVER_SECTION_EXPLANATIONS,
        "global_label_definitions": GLOBAL_LABEL_EXPLANATIONS,
        "dataset_record_layout": DATASET_RECORD_EXPLANATIONS,
        "summary": summary,
        "artifact_files": {
            "upgrade_report_md": str((ARTIFACT_DIR / "upgrade_report.md").relative_to(ROOT)),
            "summary_json": str((ARTIFACT_DIR / "summary.json").relative_to(ROOT)),
            "comprehensive_report_md": str((ARTIFACT_DIR / "comprehensive_solver_report.md").relative_to(ROOT)),
            "bundle_json": str((ARTIFACT_DIR / "full_solver_bundle.json").relative_to(ROOT)),
            "temperature_plot_png": str((ARTIFACT_DIR / "fuel_c_temperature_sweep.png").relative_to(ROOT)),
            "field_plot_png": str((ARTIFACT_DIR / "fuel_c_proxy_fields.png").relative_to(ROOT)),
            "leu_design_space_png": str((ARTIFACT_DIR / "leu_design_space.png").relative_to(ROOT)),
            "dataset_json": str((ROOT / "artifacts" / "leu_sweeps" / "leu_dataset_v3.json").relative_to(ROOT)),
            "leu_xs_library_json": str((ROOT / "artifacts" / "leu_transport_xs" / "leu_transport_proxy_library.json").relative_to(ROOT)),
        },
        "highlights": {
            "legacy_peak_flux_n_cm2_s": summary["legacy_baseline"]["peak_flux_n_cm2_s"],
            "upgraded_design_k_eff": upgraded["global_labels"]["k_eff"],
            "upgraded_design_radius_cm": upgraded["inputs"]["radius_cm"],
            "fuel_c_alpha_pcm_per_K": summary["temperature_sweep"]["mean_alpha_pcm_per_K"],
            "fuel_c_beta_ratio": fuel_c["global_labels"]["beta_eff_ratio"],
            "fuel_c_peak_thermal_flux_n_cm2_s": fuel_c["global_labels"]["peak_thermal_flux_n_cm2_s"],
        },
    }
    return _to_builtin(bundle)


def write_comprehensive_solver_report(report_path: Path, bundle: dict) -> None:
    summary = bundle["summary"]
    upgraded = summary["upgraded_design"]
    fuel_c = summary["fuel_c_proxy"]
    sweep = summary["temperature_sweep"]
    targets = summary["targets"]

    lines = [
        "# Comprehensive Solver Output Report",
        "",
        "This is the single-file labeled view of the current v2 solver run.",
        "",
        "## Overview",
        "",
        f"- Artifact root: `{bundle['overview']['artifact_root']}`",
        f"- Exported dataset records: {bundle['overview']['dataset_records']}",
        f"- Upgraded MJM critical radius: `{upgraded['inputs']['radius_cm']:.3f} cm`",
        f"- Upgraded MJM critical height: `{upgraded['inputs']['height_cm']:.3f} cm`",
        f"- Upgraded MJM `k_eff`: `{upgraded['global_labels']['k_eff']:.5f}`",
        "",
        "## What Each Top-Level Summary Section Means",
        "",
    ]
    for key, explanation in bundle["summary_section_definitions"].items():
        lines.append(f"- `{key}`: {explanation}")

    lines.extend(
        [
            "",
            "## Explicit Two-Group Energy Definitions",
            "",
            f"- `phi1` fast group: {summary['energy_groups']['group_1_fast']['energy_range_text']}",
            f"- `phi2` thermal group: {summary['energy_groups']['group_2_thermal']['energy_range_text']}",
            f"- Note: {summary['energy_groups']['note']}",
            "",
            "## Exported Dataset Record Layout",
            "",
        ]
    )
    for key, explanation in bundle["dataset_record_layout"].items():
        lines.append(f"- `{key}`: {explanation}")

    lines.extend(
        [
            "",
            "## Global Label Definitions",
            "",
        ]
    )
    for key, explanation in bundle["global_label_definitions"].items():
        lines.append(f"- `{key}`: {explanation}")

    lines.extend(
        [
            "",
            "## Key Solver Results",
            "",
            f"- Legacy peak flux: `{summary['legacy_baseline']['peak_flux_n_cm2_s']:.3e} n/cm^2/s`",
            f"- Upgraded MJM peak flux: `{upgraded['global_labels']['peak_flux_n_cm2_s']:.3e} n/cm^2/s`",
            f"- Fuel C proxy mean `alpha_T`: `{sweep['mean_alpha_pcm_per_K']:.3f} pcm/K`",
            f"- ORNL `alpha_T` target: `{targets['alpha_pcm_per_K']:.3f} pcm/K`",
            f"- Fuel C proxy delayed ratio: `{fuel_c['global_labels']['beta_eff_ratio']:.3f}`",
            f"- ORNL delayed ratio target: `{targets['beta_ratio']:.3f}`",
            f"- Fuel C proxy peak thermal flux: `{fuel_c['global_labels']['peak_thermal_flux_n_cm2_s']:.3e}`",
            f"- ORNL peak thermal flux target: `{targets['peak_thermal_flux_n_cm2_s']:.3e}`",
            f"- Fuel C proxy `D2`: `{fuel_c['xs']['D2_cm']:.3f} cm`",
            f"- Fuel C proxy side leakage fraction: `{fuel_c['global_labels']['leakage_side_fraction']:.3f}`",
            "",
            "## Temperature Sweep",
            "",
            "- The sweep stores temperature, k_eff, k_inf, dk/dT, and alpha_T for each temperature point.",
            f"- Number of sweep points: {len(sweep['rows'])}",
            "",
            "## LEU Design Space",
            "",
            f"- Dedicated LEU map generated at `UF4 = 4 mol%`, `H = 2R`, and `T = 900 K`.",
            f"- LEU critical point at 5.0% enrichment: `R = {summary['leu_design_space']['critical_point']['radius_cm']:.2f} cm`, `H = {summary['leu_design_space']['critical_point']['height_cm']:.2f} cm`.",
            "- This figure is intended for the explicit LEU presentation case and keeps the enrichment axis entirely within the LEU regime.",
            "",
            "## How To Read The Artifacts",
            "",
        ]
    )
    for key, path in bundle["artifact_files"].items():
        lines.append(f"- `{key}`: `{path}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The upgraded solver materially improves temperature-reactivity behavior and LEU design-region consistency relative to v1.",
            "- Fuel C remains a benchmark anchor rather than the LEU design basis, so benchmark comparisons and design-path behavior should be read separately.",
            "- The upgraded solver makes leakage and buckling explicit rather than implicit and exports the actual flux and power fields needed for downstream learning.",
            "- The main remaining gaps are that the LEU design path still uses a transport-informed proxy instead of a full MGXS transport collapse, and flux magnitude is still materially high against ORNL Fuel C.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running legacy baseline...")
    legacy_baseline = legacy_evaluate_design(
        enrichment=0.1975,
        uf4_mol_frac=0.04,
        radius=23.6,
        height=47.2,
        temperature=900.0,
        water_vol_frac=0.0,
        nr=40,
        nz=60,
    )

    print("Running upgraded MJM design search...")
    upgraded_design = find_critical_radius_v2(
        k_target=1.03,
        aspect_ratio=2.0,
        enrichment=0.05,
        uf4_mol_frac=0.04,
        temperature_k=900.0,
        water_vol_frac=0.0,
        reactor_mode=LEU_DESIGN_MODE,
    )

    print("Running Fuel C proxy case...")
    fuel_c_result = evaluate_design_v2(
        enrichment=0.35,
        uf4_mol_frac=0.008,
        radius_cm=70.485,
        height_cm=174.986,
        temperature_k=922.0,
        fuel_volume_fraction=0.225,
        water_vol_frac=0.0,
        nr=36,
        nz=52,
        reactor_mode=FUEL_C_BENCHMARK_MODE,
        xs_model="benchmark_guided",
    )

    print("Running temperature sweep...")
    sweep = compute_temperature_sweep_v2(
        {
            "enrichment": 0.35,
            "uf4_mol_frac": 0.008,
            "radius_cm": 70.485,
            "height_cm": 174.986,
            "fuel_volume_fraction": 0.225,
            "water_vol_frac": 0.0,
            "reactor_mode": FUEL_C_BENCHMARK_MODE,
            "xs_model": "benchmark_guided",
        },
        np.linspace(820.0, 1020.0, 9),
    )

    print("Generating field dataset...")
    dataset_path = ROOT / "artifacts" / "leu_sweeps" / "leu_dataset_v3.json"
    dataset = generate_dataset_v2(dataset_path, n_samples=96, seed=42)

    plot_temperature_sweep(sweep, ARTIFACT_DIR / "fuel_c_temperature_sweep.png")
    plot_fields(fuel_c_result, ARTIFACT_DIR / "fuel_c_proxy_fields.png")
    leu_design_space = plot_leu_design_space(ARTIFACT_DIR / "leu_design_space.png")

    summary = {
        "legacy_baseline": {
            "peak_flux_n_cm2_s": legacy_baseline["peak_flux"],
            "avg_flux_n_cm2_s": legacy_baseline["avg_flux"],
            "peak_power_density_W_cm3": legacy_baseline["peak_power_density_W_cm3"],
            "peaking_factor": legacy_baseline["peaking_factor"],
        },
        "energy_groups": upgraded_design["energy_groups"],
        "upgraded_design": export_sample_record(upgraded_design, sample_id="mjm-critical"),
        "fuel_c_proxy": export_sample_record(fuel_c_result, sample_id="fuel-c-proxy"),
        "temperature_sweep": sweep,
        "leu_design_space": leu_design_space,
        "dataset_records": len(dataset),
        "targets": {
            "alpha_pcm_per_K": ALPHA_TARGET_PCM_PER_K,
            "beta_static": BENCHMARK_STATIC_BETA,
            "beta_eff": BENCHMARK_CIRCULATING_BETA,
            "beta_ratio": DELAYED_RATIO_TARGET,
            "peak_thermal_flux_n_cm2_s": THERMAL_FLUX_TARGET,
        },
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(_to_builtin(summary), indent=2))
    solver_bundle = build_solver_bundle(summary)
    (ARTIFACT_DIR / "full_solver_bundle.json").write_text(json.dumps(solver_bundle, indent=2))
    write_comprehensive_solver_report(ARTIFACT_DIR / "comprehensive_solver_report.md", solver_bundle)

    write_report(
        ARTIFACT_DIR / "upgrade_report.md",
        baseline_legacy=legacy_baseline,
        upgraded_design=upgraded_design,
        fuel_c_result=fuel_c_result,
        sweep=sweep,
        dataset_path=dataset_path,
    )

    print(f"Artifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
