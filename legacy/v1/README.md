# Legacy v1 Assets

This folder preserves the original v1 solver artifacts in repo-friendly form:

- `mjm_solver_v1.py`: legacy 2-group finite-cylinder diffusion solver
- `plot_v1.py`: legacy plotting script, patched to use repo-relative paths
- `pinn_training_data.json`: original scalar training dataset

Typical local run:

```bash
MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg python legacy/v1/plot_v1.py
```

Generated figures are written to `artifacts/legacy_v1_plots/`.
