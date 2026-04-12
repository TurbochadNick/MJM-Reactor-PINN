#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "leu_transport_xs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monster_v2_package.leu_transport_xs import (
    DEFAULT_LEU_XS_LIBRARY_PATH,
    TRANSPORT_XS_KEYS,
    default_transport_proxy_axes,
    detect_transport_tooling,
    derive_transport_proxy_xs,
    scalar_transport_proxy_row,
    write_transport_proxy_library,
)
from monster_v2_package.solver_v2 import ENERGY_GROUPS, HomogenizedMSRMaterial


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    return value


def build_library_records() -> list[dict]:
    axes = default_transport_proxy_axes()
    records = []
    for enrichment in axes["enrichment"]:
        for uf4_mol_frac in axes["uf4_mol_frac"]:
            for temperature_k in axes["temperature_K"]:
                for fuel_volume_fraction in axes["fuel_volume_fraction"]:
                    for water_vol_frac in axes["water_vol_frac"]:
                        state = {
                            "enrichment": float(enrichment),
                            "uf4_mol_frac": float(uf4_mol_frac),
                            "temperature_K": float(temperature_k),
                            "fuel_volume_fraction": float(fuel_volume_fraction),
                            "water_vol_frac": float(water_vol_frac),
                        }
                        material = HomogenizedMSRMaterial(
                            enrichment=state["enrichment"],
                            uf4_mol_frac=state["uf4_mol_frac"],
                            temperature=state["temperature_K"],
                            water_vol_frac=state["water_vol_frac"],
                            fuel_volume_fraction=state["fuel_volume_fraction"],
                        )
                        base_xs = material.compute_macroscopic_xs(xs_model="benchmark_guided")
                        proxy_xs = derive_transport_proxy_xs(state=state, benchmark_guided_xs=base_xs)
                        record = {
                            "state": state,
                            "xs": {
                                key: float(proxy_xs[key]) for key in (*TRANSPORT_XS_KEYS, "D2_raw")
                            },
                            "metadata": proxy_xs["transport_proxy"],
                        }
                        records.append(record)
    return records


def write_scalar_csv(records: list[dict], out_path: Path) -> None:
    rows = [scalar_transport_proxy_row(record) for record in records]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(records: list[dict], out_path: Path) -> None:
    tooling = detect_transport_tooling()
    axes = default_transport_proxy_axes()
    reference_rows = []
    for record in records:
        state = record["state"]
        if (
            state["water_vol_frac"] == 0.0
            and state["fuel_volume_fraction"] in {0.225, 0.24}
            and state["temperature_K"] in {900.0, 1000.0}
            and state["uf4_mol_frac"] in {0.04, 0.05}
            and state["enrichment"] in {0.035, 0.04, 0.05}
        ):
            reference_rows.append(scalar_transport_proxy_row(record))

    reference_rows = sorted(
        reference_rows,
        key=lambda row: (
            row["enrichment"],
            row["temperature_K"],
            row["uf4_mol_frac"],
            row["fuel_volume_fraction"],
        ),
    )[:8]

    lines = [
        "# LEU Transport-Informed XS Report",
        "",
        "## Tool Detection",
        "",
        f"- Full transport tooling available: `{tooling['full_transport_available']}`",
        f"- Preferred path detected: `{tooling['preferred_path']}`",
        f"- Fallback reason: `{tooling['fallback_reason']}`",
        "",
        "## Method",
        "",
        "- This environment does not provide OpenMC, SCALE/NEWT, or Serpent, so the repo now uses a transport-informed LEU proxy workflow.",
        "- The proxy starts from the benchmark-aware v2 homogenized XS set, then applies explicit moderation, loading, and temperature adjustments tied to the ORNL 22.5/77.5 fuel/graphite reference lattice.",
        "- The resulting few-group constants are tabulated on an explicit LEU grid and written to a reusable machine-readable library.",
        "- This is not full transport-derived MGXS. Fuel C remains a benchmark anchor, while the proxy library supports the LEU design path.",
        "- When a requested LEU state falls outside the tabulated grid, the solver now falls back to the direct proxy formula instead of silently clamping the nearest tabulated state.",
        "",
        "## Group Structure",
        "",
        f"- Fast group: `{ENERGY_GROUPS['group_1_fast']['energy_range_text']}`",
        f"- Thermal group: `{ENERGY_GROUPS['group_2_thermal']['energy_range_text']}`",
        "",
        "## Library Coverage",
        "",
        f"- Total tabulated states: `{len(records)}`",
        "- Enrichment axis: `3.0%`, `3.5%`, `4.0%`, `4.5%`, `5.0%`",
        "- UF4 axis: `2.5`, `3.5`, `4.0`, `5.0 mol%`",
        f"- Temperature axis: `{', '.join(str(int(value)) for value in axes['temperature_K'])} K`",
        "- Fuel-volume axis: `0.18`, `0.21`, `0.225`, `0.24`, `0.27`",
        "- Water axis retained only for continuity edge cases: `0.0`, `0.03`",
        "",
        "## Representative Proxy States",
        "",
    ]
    for row in reference_rows:
        lines.append(
            (
                f"- e={row['enrichment']*100:.1f}% | UF4={row['uf4_mol_frac']*100:.1f} mol% | "
                f"T={row['temperature_K']:.0f} K | fuel={row['fuel_volume_fraction']:.3f} | "
                f"D2={row['D2_cm']:.3f} cm | k_inf={row['k_inf']:.3f} | "
                f"moderation_index={row['moderation_index']:.3f}"
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `artifacts/leu_transport_xs/leu_transport_proxy_library.json`",
            "- `artifacts/leu_transport_xs/leu_transport_proxy_table.csv`",
            "- `artifacts/leu_transport_xs/transport_tooling.json`",
            "- `artifacts/leu_transport_xs/leu_transport_xs_report.md`",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    records = build_library_records()
    axes = default_transport_proxy_axes()

    payload = write_transport_proxy_library(
        output_path=DEFAULT_LEU_XS_LIBRARY_PATH,
        axes=axes,
        group_structure=ENERGY_GROUPS,
        records=records,
    )
    (ARTIFACT_DIR / "transport_tooling.json").write_text(json.dumps(detect_transport_tooling(), indent=2))
    (ARTIFACT_DIR / "leu_transport_xs_summary.json").write_text(json.dumps(_to_builtin(payload), indent=2))
    write_scalar_csv(records, ARTIFACT_DIR / "leu_transport_proxy_table.csv")
    write_report(records, ARTIFACT_DIR / "leu_transport_xs_report.md")

    print(f"Wrote LEU XS library to {DEFAULT_LEU_XS_LIBRARY_PATH}")


if __name__ == "__main__":
    main()
