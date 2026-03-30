# MJM Reactor PINN

Finite-cylinder, 2-group diffusion PINN package for a molten salt reactor, centered on an `ORNL-TM-730` benchmark-backed v2 roadmap.

The current deliverable in this repo is the v2 design package:

- [monster_v2_package/README.md](./monster_v2_package/README.md)
- [monster_v2_package/benchmarks/MSRE_FUEL_C_BENCHMARKS.md](./monster_v2_package/benchmarks/MSRE_FUEL_C_BENCHMARKS.md)
- [monster_v2_package/docs/DATASET_SPEC.md](./monster_v2_package/docs/DATASET_SPEC.md)
- [monster_v2_package/docs/MODEL_AND_TRAINING_SPEC.md](./monster_v2_package/docs/MODEL_AND_TRAINING_SPEC.md)
- [monster_v2_package/docs/IMPLEMENTATION_ROADMAP.md](./monster_v2_package/docs/IMPLEMENTATION_ROADMAP.md)

## Direction

The near-term build target is:

- steady-state finite-cylinder field PINN
- explicit buckling-aware losses
- benchmarked against `Fuel C` from `ORNL-TM-730`
- designed to grow from Python exporter tooling into heavier `JAX` and `C++` compute paths

## Next Implementation Step

Turn the v1 solver into a reusable exporter that emits:

- global labels
- field arrays
- buckling features
- diffusion/XS terms
- machine-readable records matching the v2 schemas
