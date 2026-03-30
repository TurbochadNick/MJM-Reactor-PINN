# Model And Training Spec

This is the design spec for the monster v2 finite-cylinder diffusion PINN.

## Problem Definition

We want a parameterized operator that maps:

- design/material inputs
- explicit buckling and geometry features
- spatial coordinates in a finite cylinder

to:

- field outputs `phi1(r,z)` and `phi2(r,z)`
- global reactor-response quantities such as `k_eff`, `alpha_T`, and power metrics

## Governing Equations

Use the steady-state 2-group diffusion eigenvalue problem in cylindrical coordinates with axial symmetry.

Let `g = 1` be the fast group and `g = 2` the thermal group.

### Group 1

```text
- (1/r) d/dr ( r D1 dphi1/dr ) - d/dz ( D1 dphi1/dz ) + Sigma_r1 phi1
= (1/k_eff) [ chi1 nuSigma_f1 phi1 + chi1 nuSigma_f2 phi2 ]
```

### Group 2

```text
- (1/r) d/dr ( r D2 dphi2/dr ) - d/dz ( D2 dphi2/dz ) + Sigma_a2 phi2
= Sigma_s12 phi1 + (1/k_eff) [ chi2 nuSigma_f1 phi1 + chi2 nuSigma_f2 phi2 ]
```

For the current formulation:

- `Sigma_r1 = Sigma_a1 + Sigma_s12`
- no upscatter in MVP
- fast-born fission source:
  - `chi1 ~= 1`
  - `chi2 ~= 0`

## Boundary And Symmetry Conditions

### Axis Symmetry

At `r = 0`:

```text
dphi1/dr = 0
dphi2/dr = 0
```

### Outer Cylinder And Endcaps

Use one of these two equivalent viewpoints:

- physics-native PINN loss:
  - Marshak/Robin vacuum BC
- geometry feature / analytic reduction:
  - zero flux at extrapolated boundaries

Recommended training BC:

```text
phi_g + 2 D_g dphi_g/dn = 0
```

Recommended stored geometric equivalent:

- `Re = R + delta_r`
- `He = H + 2 delta_z`

## Buckling Representation

Buckling must be explicit in both the features and the losses.

### Geometric Buckling

```text
Br^2 = (2.405 / Re)^2
Bz^2 = (pi / He)^2
Bg^2 = Br^2 + Bz^2
```

### How To Use It

Use buckling in three places:

1. input features
2. near-critical regularization
3. leakage-aware diagnostics

### Recommended Buckling Loss

Use a Helmholtz-style field-curvature regularizer on the total flux:

```text
Bg_hat^2(r,z) = - Laplacian(phi_total) / (phi_total + eps)
L_buck = mean( w(r,z) * (Bg_hat^2 - Bg^2)^2 )
```

This is not a replacement for the PDE residual. It is a stabilizing auxiliary constraint near criticality.

## Recommended Architecture

Treat v2 as a parameterized PDE solution operator.

### Inputs

#### Global design/material block

- `enrichment`
- `uf4_mol_frac`
- `radius_cm`
- `height_cm`
- `temperature_K`
- `water_vol_frac`
- `Re`, `He`, `Br^2`, `Bz^2`, `Bg^2`
- macroscopic XS and diffusion coefficients
- optional material-property features

#### Spatial block

- `r / R`
- `z / H`
- signed or normalized distances to:
  - axis
  - radial boundary
  - top boundary
  - bottom boundary

### Network Layout

Recommended:

- global encoder MLP for design/material features
- coordinate encoder for spatial features
- fused residual trunk
- pointwise field heads for:
  - `phi1_shape`
  - `phi2_shape`
- global head for:
  - `k_eff`
  - `k_inf`
  - `alpha_T_pcm_per_K`
  - `peaking_factor`
  - `peak_flux`
  - `avg_flux`
  - `peak_power_density`
  - optional `beta_eff`

This can be implemented either as:

- a coordinate-query PINN/operator net
- or a latent-grid decoder if a fixed mesh is used

For flexibility, the coordinate-query version is preferred.

## Output Strategy

### Minimum Viable V2

Predict:

- `phi1_shape(r,z)`
- `phi2_shape(r,z)`
- `k_eff`
- `alpha_T_pcm_per_K`
- `peaking_factor`
- `peak_flux_n_cm2_s`
- `avg_flux_n_cm2_s`
- `peak_power_density_W_cm3`

### Monster V2

Add:

- physical flux fields
- current fields
- `beta_eff`
- prompt lifetime
- leakage fractions
- thermal-hydraulic outputs
- optional XS derivative heads

## Loss Design

Use staged training with adaptive balancing.

### Total Loss

```text
L =
  lambda_data * L_data
+ lambda_pde * L_pde
+ lambda_bc * L_bc
+ lambda_sym * L_sym
+ lambda_buck * L_buck
+ lambda_global * L_global
+ lambda_power * L_power
+ lambda_pos * L_pos
+ lambda_alpha * L_alpha
+ lambda_beta * L_beta
```

### Data Loss

Supervise:

- `phi1_shape`
- `phi2_shape`
- scalar global labels

### PDE Residual Loss

Compute residuals for both groups using autodiff on the pointwise field outputs.

### Boundary Loss

Apply vacuum/Robin BC on:

- outer radius
- top
- bottom

### Symmetry Loss

Enforce `dphi/dr = 0` at the axis.

### Positivity Loss

Weakly penalize negative flux predictions.

### Power / Normalization Loss

Tie the predicted fields to the stored global power and flux quantities.

### Coefficient Losses

Supervise:

- `alpha_T_pcm_per_K`
- optional `beta_eff`

### Buckling Loss

Use the curvature-to-buckling match near criticality.

## Training Curriculum

### Stage 1

Data-only pretraining:

- field shapes
- scalar outputs

### Stage 2

Turn on:

- PDE residual
- BC loss
- symmetry loss
- power consistency

### Stage 3

Turn up:

- buckling loss
- near-critical weighting
- coefficient losses

### Stage 4

Fine-tune on benchmark-centered sweeps, especially Fuel C-like cases.

## Weighting Strategy

Do not freeze manual weights and hope.

Recommended:

- adaptive loss balancing
- gradient-based rescaling or norm balancing
- near-critical curriculum weighting

Near `k_eff ~= 1`, increase the weights on:

- PDE residual
- BC
- buckling
- `k_eff`

## Validation Targets

### Scalar Targets

- `k_eff` relative error:
  - median `< 0.5%`
  - 95th percentile `< 1.0%`
- `alpha_T`:
  - correct sign across the design space
  - Fuel C magnitude within about `25%` around the ORNL anchor
- `beta_eff` near the Fuel C/1200 gpm case within about `10%`

### Field Targets

- field-shape correlation `> 0.98` on held-out snapshots
- correct edge decay and leakage behavior
- correct axial and radial peak location trends

## Benchmark Anchoring

Fuel C should be used as an explicit validation suite with the ORNL values in this package, especially for:

- total temperature coefficient
- delayed neutron reduction
- prompt lifetime
- thermal flux scale
- leakage fraction
- control worth scale

## Recommended Failure Checks

Flag training runs as suspect if any of these occur:

- negative flux pockets persist
- side leakage collapses unrealistically
- `alpha_T` flips sign near Fuel C reference
- global heads fit while field BC behavior is wrong
- the model interpolates training fields but misses held-out geometry sweeps
