# Monster V2 Package

This package defines a concrete v2 build target for a finite-cylinder, 2-group diffusion PINN for a molten salt reactor, using `ORNL-TM-730` Fuel C results as the primary benchmark anchor.

The package is organized around four goals:

1. Freeze the benchmark targets we will hold the model against.
2. Define the dataset we actually need to train the model.
3. Specify the field-PINN architecture, losses, and training regimen.
4. Give a phased roadmap from the current v1 solver to a defensible v2.

## Package Contents

- [benchmarks/MSRE_FUEL_C_BENCHMARKS.md](./benchmarks/MSRE_FUEL_C_BENCHMARKS.md)
- [benchmarks/msre_fuel_c_reference.json](./benchmarks/msre_fuel_c_reference.json)
- [docs/DATASET_SPEC.md](./docs/DATASET_SPEC.md)
- [docs/MODEL_AND_TRAINING_SPEC.md](./docs/MODEL_AND_TRAINING_SPEC.md)
- [docs/XS_AND_PHYSICS_STRATEGY.md](./docs/XS_AND_PHYSICS_STRATEGY.md)
- [docs/IMPLEMENTATION_ROADMAP.md](./docs/IMPLEMENTATION_ROADMAP.md)
- [schemas/msr_pinn_v2_sample.schema.json](./schemas/msr_pinn_v2_sample.schema.json)
- [schemas/msr_pinn_v2_manifest.schema.json](./schemas/msr_pinn_v2_manifest.schema.json)

## Recommended V2 Assumptions

- Governing model for MVP: steady-state 2-group neutron diffusion in a finite right-circular cylinder.
- Independent geometric inputs: `radius_cm` and `height_cm`.
- Boundary treatment:
  - Use Marshak/Robin vacuum boundary conditions in the PINN loss.
  - Also store extrapolated dimensions and geometric buckling as explicit features.
- Primary benchmark case: `ORNL-TM-730` Fuel C.
- Delayed neutron correction:
  - Keep as a supervised/global target in v2.
  - Do not couple full precursor transport into the first field model unless time permits.
- Moderation variable:
  - MVP may retain `water_vol_frac` for continuity with v1.
  - Ambitious v2 should replace or augment it with a more physical moderator/fuel-volume description.

## Benchmark Snapshot

Fuel C anchors from `ORNL-TM-730` now captured in this package include:

- Total temperature coefficient: `-6.96e-5 /degF` or about `-12.53 pcm/K`
- Delayed neutron fractions:
  - `beta_eff_static = 0.00666`
  - `beta_eff_circulating = 0.00362`
- Prompt neutron lifetime: `2.40e-4 s`
- Control rod total worth: `5.7% delta-k/k`
- Max thermal flux at `10 MW`: `3.29e13 n/cm^2/s`
- Power coefficient for Fuel C:
  - Constant `T_out`: `-0.006 % delta-k/k per MW`
  - Constant mean of `T_in` and `T_out`: `-0.024 % delta-k/k per MW`

## How To Use This Package

1. Start with [benchmarks/MSRE_FUEL_C_BENCHMARKS.md](./benchmarks/MSRE_FUEL_C_BENCHMARKS.md) to see the reference targets.
2. Use [docs/DATASET_SPEC.md](./docs/DATASET_SPEC.md) and the JSON schemas to build the exporter from the v1 solver.
3. Implement the field/operator model from [docs/MODEL_AND_TRAINING_SPEC.md](./docs/MODEL_AND_TRAINING_SPEC.md).
4. Follow [docs/IMPLEMENTATION_ROADMAP.md](./docs/IMPLEMENTATION_ROADMAP.md) in order.

## Scope Split

- Minimum viable v2:
  - Solver refactor + exporter
  - Full field dataset
  - Operator-style field PINN for `phi1`, `phi2`, `k_eff`, `alpha_T`, and power metrics
  - Buckling-aware losses and benchmark validation
- Ambitious v2:
  - Transport-derived cross sections
  - Multi-region geometry
  - Delayed precursor transport
  - Stronger thermal-hydraulic coupling
  - Depletion/transient extensions

## Current Biggest Remaining Physics Risk

The dominant fidelity gap is still the cross-section model. `ORNL-TM-730` gives us the workflow and benchmark outcomes, but not a clean modern table of homogenized 2-group constants for direct reuse. The safest upgrade path is transport-based preprocessing with `SCALE` or `Serpent`.
