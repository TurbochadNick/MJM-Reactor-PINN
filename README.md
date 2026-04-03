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

## Runnable Prototype

The repo now includes a first executable benchmark runner:

- `scripts/run_first_pinn.py`

It loads the original v1 solver from the homework notebook, augments the training data with `alpha_T` and buckling features, and trains a first `PINN-lite` surrogate against the 2-group diffusion solver outputs.

Typical local run:

```bash
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python scripts/run_first_pinn.py
```

The generated outputs are written under `artifacts/first_pinn_run/` and are intentionally gitignored.

## Next Implementation Step

Turn the v1 solver into a reusable exporter that emits:

- global labels
- field arrays
- buckling features
- diffusion/XS terms
- machine-readable records matching the v2 schemas
