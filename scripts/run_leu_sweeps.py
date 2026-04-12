#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "leu_sweeps"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_DIR / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from monster_v2_package.leu_transport_xs import DEFAULT_LEU_XS_LIBRARY_PATH
from monster_v2_package.solver_v2 import (
    LEU_DESIGN_MODE,
    compute_temperature_sweep_v2,
    evaluate_design_v2,
    find_critical_radius_v2,
    generate_dataset_v2,
)


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


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_near_critical_sweep() -> list[dict]:
    rows = []
    enrichments = np.linspace(0.03, 0.05, 9)
    radius_offsets = [-0.10, -0.05, 0.0, 0.05, 0.10]

    for enrichment in enrichments:
        critical = find_critical_radius_v2(
            k_target=1.0,
            aspect_ratio=2.0,
            enrichment=float(enrichment),
            uf4_mol_frac=0.04,
            temperature_k=900.0,
            water_vol_frac=0.0,
            fuel_volume_fraction=0.225,
            reactor_mode=LEU_DESIGN_MODE,
        )
        for delta in radius_offsets:
            radius_cm = critical["critical_radius_cm"] * (1.0 + delta)
            result = evaluate_design_v2(
                enrichment=float(enrichment),
                uf4_mol_frac=0.04,
                radius_cm=radius_cm,
                height_cm=2.0 * radius_cm,
                temperature_k=900.0,
                water_vol_frac=0.0,
                fuel_volume_fraction=0.225,
                nr=22,
                nz=34,
                reactor_mode=LEU_DESIGN_MODE,
            )
            rows.append(
                {
                    "enrichment": float(enrichment),
                    "critical_radius_cm": critical["critical_radius_cm"],
                    "radius_cm": radius_cm,
                    "radius_offset_fraction": float(delta),
                    "height_cm": 2.0 * radius_cm,
                    "k_eff": result["k_eff"],
                    "k_inf": result["k_inf"],
                    "D2_cm": result["xs"]["D2"],
                    "peak_flux_n_cm2_s": result["peak_flux_n_cm2_s"],
                    "beta_eff_ratio": result["delayed_neutrons"]["reduction_factor"],
                    "leakage_side_fraction": result["leakage_side_fraction"],
                }
            )
    return rows


def build_temperature_sweep() -> list[dict]:
    rows = []
    for enrichment in (0.035, 0.04, 0.045, 0.05):
        critical = find_critical_radius_v2(
            k_target=1.0,
            aspect_ratio=2.0,
            enrichment=enrichment,
            uf4_mol_frac=0.04,
            temperature_k=900.0,
            water_vol_frac=0.0,
            fuel_volume_fraction=0.225,
            reactor_mode=LEU_DESIGN_MODE,
        )
        sweep = compute_temperature_sweep_v2(
            {
                "enrichment": enrichment,
                "uf4_mol_frac": 0.04,
                "radius_cm": critical["critical_radius_cm"],
                "height_cm": critical["critical_height_cm"],
                "temperature_k": 900.0,
                "water_vol_frac": 0.0,
                "fuel_volume_fraction": 0.225,
                "reactor_mode": LEU_DESIGN_MODE,
            },
            np.linspace(800.0, 1050.0, 7),
        )
        for row in sweep["rows"]:
            rows.append(
                {
                    "enrichment": enrichment,
                    "critical_radius_cm": critical["critical_radius_cm"],
                    **row,
                }
            )
    return rows


def build_leakage_sweep() -> list[dict]:
    rows = []
    for enrichment in (0.04, 0.05):
        for aspect_ratio in (1.2, 1.5, 2.0, 2.5, 3.0):
            critical = find_critical_radius_v2(
                k_target=1.0,
                aspect_ratio=aspect_ratio,
                enrichment=enrichment,
                uf4_mol_frac=0.04,
                temperature_k=900.0,
                water_vol_frac=0.0,
                fuel_volume_fraction=0.225,
                reactor_mode=LEU_DESIGN_MODE,
            )
            result = evaluate_design_v2(
                enrichment=enrichment,
                uf4_mol_frac=0.04,
                radius_cm=critical["critical_radius_cm"],
                height_cm=critical["critical_height_cm"],
                temperature_k=900.0,
                water_vol_frac=0.0,
                fuel_volume_fraction=0.225,
                nr=24,
                nz=36,
                reactor_mode=LEU_DESIGN_MODE,
            )
            rows.append(
                {
                    "enrichment": enrichment,
                    "aspect_ratio": aspect_ratio,
                    "critical_radius_cm": critical["critical_radius_cm"],
                    "critical_height_cm": critical["critical_height_cm"],
                    "k_eff": result["k_eff"],
                    "Br2_cm2_inv": result["buckling"]["Br2_cm2_inv"],
                    "Bz2_cm2_inv": result["buckling"]["Bz2_cm2_inv"],
                    "Bg2_cm2_inv": result["buckling"]["Bg2_cm2_inv"],
                    "leakage_fraction": result["leakage_fraction"],
                    "leakage_side_fraction": result["leakage_side_fraction"],
                    "leakage_top_fraction": result["leakage_top_fraction"],
                    "beta_eff_ratio": result["delayed_neutrons"]["reduction_factor"],
                }
            )
    return rows


def build_moderation_sweep() -> list[dict]:
    rows = []
    for enrichment in (0.045, 0.05):
        for uf4_mol_frac in (0.03, 0.04, 0.05):
            for fuel_volume_fraction in (0.18, 0.21, 0.225, 0.24, 0.27):
                critical = find_critical_radius_v2(
                    k_target=1.0,
                    aspect_ratio=2.0,
                    enrichment=enrichment,
                    uf4_mol_frac=uf4_mol_frac,
                    temperature_k=900.0,
                    water_vol_frac=0.0,
                    fuel_volume_fraction=fuel_volume_fraction,
                    reactor_mode=LEU_DESIGN_MODE,
                )
                result = evaluate_design_v2(
                    enrichment=enrichment,
                    uf4_mol_frac=uf4_mol_frac,
                    radius_cm=critical["critical_radius_cm"],
                    height_cm=critical["critical_height_cm"],
                    temperature_k=900.0,
                    water_vol_frac=0.0,
                    fuel_volume_fraction=fuel_volume_fraction,
                    nr=22,
                    nz=34,
                    reactor_mode=LEU_DESIGN_MODE,
                )
                rows.append(
                    {
                        "enrichment": enrichment,
                        "uf4_mol_frac": uf4_mol_frac,
                        "fuel_volume_fraction": fuel_volume_fraction,
                        "graphite_volume_fraction": result["xs"]["volume_fractions"]["graphite"],
                        "critical_radius_cm": critical["critical_radius_cm"],
                        "k_inf": result["k_inf"],
                        "D2_cm": result["xs"]["D2"],
                        "Sigma_a2_cm1": result["xs"]["Sigma_a2"],
                        "nuSigma_f2_cm1": result["xs"]["nu_Sigma_f2"],
                    }
                )
    return rows


def plot_critical_radius(rows: list[dict], out_path: Path) -> None:
    critical_rows = [row for row in rows if abs(row["radius_offset_fraction"]) < 1.0e-12]
    enrich = [row["enrichment"] * 100.0 for row in critical_rows]
    radius = [row["critical_radius_cm"] for row in critical_rows]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(enrich, radius, "o-", linewidth=2, color="#005f73")
    ax.set_xlabel("Enrichment (%)")
    ax.set_ylabel("Critical radius (cm) at k=1.0")
    ax.set_title("LEU Critical Radius vs Enrichment")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_alpha(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for enrichment in sorted({row["enrichment"] for row in rows}):
        subset = [row for row in rows if row["enrichment"] == enrichment]
        ax.plot(
            [row["temperature_K"] for row in subset],
            [row["alpha_pcm_per_K"] for row in subset],
            "o-",
            linewidth=1.8,
            label=f"{enrichment*100:.1f}% U-235",
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("alpha_T (pcm/K)")
    ax.set_title("LEU Temperature Reactivity by Enrichment")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(
    *,
    near_critical_rows: list[dict],
    temperature_rows: list[dict],
    leakage_rows: list[dict],
    moderation_rows: list[dict],
    dataset_rows: list[dict],
    out_path: Path,
) -> None:
    critical_rows = [row for row in near_critical_rows if abs(row["radius_offset_fraction"]) < 1.0e-12]
    side_leak = np.mean([row["leakage_side_fraction"] for row in leakage_rows])
    mean_alpha = np.mean([row["alpha_pcm_per_K"] for row in temperature_rows])
    dataset_keff = np.array([row["global_labels"]["k_eff"] for row in dataset_rows], dtype=float)
    dataset_alpha = np.array([row["global_labels"]["alpha_T_pcm_per_K"] for row in dataset_rows], dtype=float)
    direct_alpha_proxy_count = int(
        sum(
            1
            for row in dataset_rows
            if row.get("provenance", {}).get("alpha_T_details", {}).get("used_direct_transport_proxy")
        )
    )

    lines = [
        "# LEU Sweep Report",
        "",
        "## What Was Added",
        "",
        "- Near-critical LEU enrichment/radius sweeps rooted on k=1.0 critical searches.",
        "- Temperature-reactivity sweeps at multiple LEU enrichments using the transport-informed LEU proxy XS path.",
        "- Leakage-sensitive aspect-ratio sweeps and moderation/UF4 critical-radius sweeps.",
        "- A richer targeted LEU dataset for surrogate/PINN use with explicit provenance and field exports.",
        "- Fuel C remains outside this sweep set as a higher-enrichment benchmark anchor rather than part of the LEU design basis.",
        "",
        "## Sweep Highlights",
        "",
        f"- Critical radius at 3.5% enrichment: `{next(row['critical_radius_cm'] for row in critical_rows if abs(row['enrichment'] - 0.035) < 1.0e-12):.2f} cm`",
        f"- Critical radius at 5.0% enrichment: `{next(row['critical_radius_cm'] for row in critical_rows if abs(row['enrichment'] - 0.05) < 1.0e-12):.2f} cm`",
        f"- Mean LEU alpha_T across temperature sweeps: `{mean_alpha:.2f} pcm/K`",
        f"- Mean side-leakage share across aspect sweeps: `{side_leak:.3f}`",
        f"- Dataset near-critical count (|k-1| <= 0.05): `{int(np.sum(np.abs(dataset_keff - 1.0) <= 0.05))}` / `{len(dataset_rows)}`",
        f"- Exact-zero alpha_T rows in the exported LEU dataset: `{int(np.sum(np.abs(dataset_alpha) <= 1.0e-12))}`",
        f"- alpha_T rows using the direct out-of-grid proxy fallback: `{direct_alpha_proxy_count}`",
        "",
        "## Interpretation Guardrails",
        "",
        "- These sweeps use a transport-informed proxy XS workflow, not a full transport-derived MGXS collapse.",
        "- The targeted LEU dataset is materially richer near criticality and leakage-sensitive cases, but that does not imply every downstream surrogate metric improves automatically.",
        "- A small number of hard-case rows retain the direct proxy fallback for temperature perturbations outside the tabulated XS grid; they are kept explicitly rather than hidden.",
        "- Flux magnitude remains a major solver limitation even after the LEU proxy upgrade.",
        "",
        "## Artifacts",
        "",
        "- `artifacts/leu_sweeps/near_critical_sweep.json`",
        "- `artifacts/leu_sweeps/temperature_sweep.json`",
        "- `artifacts/leu_sweeps/leakage_sweep.json`",
        "- `artifacts/leu_sweeps/moderation_sweep.json`",
        "- `artifacts/leu_sweeps/leu_dataset_v3.json`",
        "- `artifacts/leu_sweeps/leu_dataset_v3_manifest.json`",
        "- `artifacts/leu_sweeps/critical_radius_vs_enrichment.png`",
        "- `artifacts/leu_sweeps/alpha_vs_temperature.png`",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def run_leu_sweep_workflow(*, dataset_samples: int = 96, seed: int = 42) -> dict:
    if not DEFAULT_LEU_XS_LIBRARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing LEU XS library at {DEFAULT_LEU_XS_LIBRARY_PATH}. Run scripts/run_leu_xs_generation.py first."
        )

    near_critical_rows = build_near_critical_sweep()
    temperature_rows = build_temperature_sweep()
    leakage_rows = build_leakage_sweep()
    moderation_rows = build_moderation_sweep()
    dataset_path = ARTIFACT_DIR / "leu_dataset_v3.json"
    dataset_rows = generate_dataset_v2(dataset_path, n_samples=dataset_samples, seed=seed)

    (ARTIFACT_DIR / "near_critical_sweep.json").write_text(json.dumps(_to_builtin(near_critical_rows), indent=2))
    (ARTIFACT_DIR / "temperature_sweep.json").write_text(json.dumps(_to_builtin(temperature_rows), indent=2))
    (ARTIFACT_DIR / "leakage_sweep.json").write_text(json.dumps(_to_builtin(leakage_rows), indent=2))
    (ARTIFACT_DIR / "moderation_sweep.json").write_text(json.dumps(_to_builtin(moderation_rows), indent=2))
    write_csv(near_critical_rows, ARTIFACT_DIR / "near_critical_sweep.csv")
    write_csv(temperature_rows, ARTIFACT_DIR / "temperature_sweep.csv")
    write_csv(leakage_rows, ARTIFACT_DIR / "leakage_sweep.csv")
    write_csv(moderation_rows, ARTIFACT_DIR / "moderation_sweep.csv")
    plot_critical_radius(near_critical_rows, ARTIFACT_DIR / "critical_radius_vs_enrichment.png")
    plot_alpha(temperature_rows, ARTIFACT_DIR / "alpha_vs_temperature.png")
    write_report(
        near_critical_rows=near_critical_rows,
        temperature_rows=temperature_rows,
        leakage_rows=leakage_rows,
        moderation_rows=moderation_rows,
        dataset_rows=dataset_rows,
        out_path=ARTIFACT_DIR / "leu_sweep_report.md",
    )

    return {
        "dataset_path": dataset_path,
        "dataset_rows": dataset_rows,
        "near_critical_rows": near_critical_rows,
        "temperature_rows": temperature_rows,
        "leakage_rows": leakage_rows,
        "moderation_rows": moderation_rows,
    }


def main() -> None:
    run_leu_sweep_workflow(dataset_samples=96, seed=42)
    print(f"Wrote LEU sweeps to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
