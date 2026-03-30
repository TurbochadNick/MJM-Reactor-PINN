# Implementation Roadmap

This roadmap turns the v1 solver plus the benchmark package into a monster v2.

## Phase 0: Freeze Interfaces

Deliverables:

- adopt the JSON schemas in `schemas/`
- freeze the benchmark set in `benchmarks/`
- freeze the initial design variables:
  - `enrichment`
  - `uf4_mol_frac`
  - `radius_cm`
  - `height_cm`
  - `temperature_K`
  - `water_vol_frac` or its replacement
- freeze the initial BC choice:
  - Marshak/Robin vacuum BC in the PINN
  - extrapolated dimensions stored as features

Exit criteria:

- no ambiguity remains in the sample record structure

## Phase 1: Refactor The Solver Into An Exporter

Goal:

- turn the v1 notebook logic into a reusable module that emits schema-valid sample records

Required outputs:

- scalar labels
- field arrays
- XS terms
- buckling terms
- provenance

Exit criteria:

- one sample can be exported and validated
- a Fuel C-like case reproduces the current v1 outputs

## Phase 2: Generate The V2 Dataset

Goal:

- produce the first full training corpus

Minimum viable target:

- `2,500` scalar records
- `250` field records

Monster target:

- `10,000` scalar records
- `1,000` field records

Must include:

- broad design coverage
- near-critical oversampling
- temperature sweeps
- geometry sweeps

Exit criteria:

- dataset validates against the manifest schema
- benchmark-centered slices exist for Fuel C-like cases

## Phase 3: Train A Strong Scalar Baseline

Goal:

- establish an easy baseline before the field PINN

Predict:

- `k_eff`
- `alpha_T`
- `peaking_factor`
- `peak_flux`
- `peak_power_density`

This is not the final model. It is a quick confidence builder and error detector for the dataset.

Exit criteria:

- global labels are learnable
- feature importance and failure modes are understood

## Phase 4: Train The Field PINN / Operator Net

Goal:

- learn `phi1` and `phi2` as parameterized fields over finite-cylinder geometry

Training plan:

1. field data pretraining
2. add PDE residuals
3. add BC and symmetry
4. add buckling and coefficient losses
5. fine-tune near criticality

Exit criteria:

- held-out field-shape performance is stable
- BC behavior is correct
- benchmark-centered scalar predictions remain accurate

## Phase 5: Benchmark Against ORNL Fuel C

Mandatory checks:

- `alpha_T` near `-12.5 pcm/K`
- `beta_eff_circulating` near `0.00362`
- prompt lifetime order of magnitude correct
- thermal flux scale in the right range
- leakage remains strong and side-dominated

Exit criteria:

- model behavior is credible against the reference case

## Phase 6: Physics Upgrade Path

This phase is optional for minimum viable v2, but strongly recommended for the strongest version.

Priority order:

1. improved XS via transport preprocessing
2. replace or augment `water_vol_frac` with a more physical moderator descriptor
3. add multi-region interfaces
4. add delayed precursor transport or a stronger kinetics head
5. add depletion / poison modules

## Minimum Viable V2 Definition Of Done

- field dataset exists and validates
- finite-cylinder geometry is explicit
- buckling is explicit
- model predicts `phi1`, `phi2`, `k_eff`, `alpha_T`, and power metrics
- Fuel C benchmark behavior is in the right regime
- package is reproducible and handoff-ready

## Ambitious V2 Definition Of Done

- transport-derived XS
- stronger agreement with ORNL benchmark magnitudes
- delayed-neutron behavior supervised and robust
- multi-region or channelized geometry support
- clear path to kinetics/depletion extensions
