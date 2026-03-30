# V2 Dataset Spec

This is the dataset contract for the monster v2 package.

The core principle is simple: v2 is not just a scalar surrogate. It is a parameterized field solution operator plus a set of global reactor-response heads.

## Design Goals

The dataset must support all of the following:

- finite-cylinder geometry with independent `R` and `H`
- explicit leakage and buckling awareness
- field prediction for `phi1(r,z)` and `phi2(r,z)`
- global prediction for `k_eff`, `alpha_T`, and power metrics
- delayed-neutron and control-worth supervision where available
- future replacement of heuristic cross sections with transport-derived constants

## Recommended Sample Counts

### Minimum Viable V2

- `2,500` scalar/global samples
- `250` field snapshots
- `100` targeted near-critical samples with `0.98 <= k_eff <= 1.05`
- `100` temperature sweeps around the Fuel C-like operating point

### Monster V2

- `10,000` scalar/global samples
- `1,000` field snapshots
- `1,500` near-critical targeted samples
- `500` temperature sweeps
- `500` geometry sweeps with aspect-ratio emphasis
- optional repeated evaluations with alternate XS sets or BC settings

## Sampling Strategy

Use a mixed strategy, not pure random sampling.

- `50%` broad Latin hypercube or Sobol sampling over the full design box
- `20%` near-critical samples found by root-targeted geometry sweeps
- `15%` temperature and density perturbation sweeps around benchmark cases
- `10%` aspect-ratio and boundary-sensitive geometry sweeps
- `5%` hard cases:
  - extreme leakage
  - extreme moderation
  - small-radius or short-height edge cases

## Initial Design Space

Use the v1 ranges as the starting box unless the team narrows them:

| Variable | Initial range |
| --- | --- |
| `enrichment` | `0.03 - 0.25` |
| `uf4_mol_frac` | `0.02 - 0.06` |
| `radius_cm` | `20 - 80` |
| `height_cm` | `40 - 160` |
| `temperature_K` | `773 - 1073` |
| `water_vol_frac` | `0.0 - 0.30` |

Important note:

- `water_vol_frac` is a continuity variable from v1.
- Ambitious v2 should add or replace it with a more physical moderator/fuel-volume or graphite-channel descriptor.

## Required Field Categories

The dataset record should include the following top-level groups.

### 1. Inputs

- `enrichment`
- `uf4_mol_frac`
- `radius_cm`
- `height_cm`
- `temperature_K`
- `water_vol_frac`

### 2. Derived Geometry / Buckling

- `aspect_ratio`
- `core_volume_cm3`
- `surface_area_cm2`
- `delta_r_cm`
- `delta_z_cm`
- `radius_extrap_cm`
- `height_extrap_cm`
- `buckling_radial_cm2_inv`
- `buckling_axial_cm2_inv`
- `buckling_total_cm2_inv`

These should be computed as:

- `Re = R + delta_r`
- `He = H + 2 * delta_z`
- `Br^2 = (2.405 / Re)^2`
- `Bz^2 = (pi / He)^2`
- `Bg^2 = Br^2 + Bz^2`

### 3. Material Properties

- `rho_salt_g_cm3`
- `rho_mix_g_cm3`
- `cp_J_kgK`
- `mu_Pa_s`
- `k_W_mK`
- optional:
  - `thermal_expansion_salt`
  - `thermal_expansion_graphite`
  - `d_rho_dT`
  - `d_mu_dT`

### 4. Number Densities

Store isotopic or constituent number densities explicitly:

- `N_U235_atoms_cm3`
- `N_U238_atoms_cm3`
- `N_Li_atoms_cm3`
- `N_Be_atoms_cm3`
- `N_F_atoms_cm3`
- `N_H_atoms_cm3`
- `N_O_atoms_cm3`

This is important because it makes the dataset interpretable and gives a clean place to upgrade the XS workflow later.

### 5. Macroscopic Cross Sections / Diffusion Coefficients

Required for each sample:

- `D1_cm`
- `D2_cm`
- `Sigma_a1_cm1`
- `Sigma_a2_cm1`
- `Sigma_s12_cm1`
- `Sigma_f1_cm1`
- `Sigma_f2_cm1`
- `nuSigma_f1_cm1`
- `nuSigma_f2_cm1`
- `Sigma_r1_cm1`
- `chi1`
- `chi2`

Recommended extras:

- `dSigma_a1_dT_cm1_per_K`
- `dSigma_a2_dT_cm1_per_K`
- `dnuSigma_f1_dT_cm1_per_K`
- `dnuSigma_f2_dT_cm1_per_K`

### 6. Global Neutronics Labels

Required:

- `k_eff`
- `k_inf`
- `reactivity_pcm`
- `alpha_T_pcm_per_K`
- `peak_flux_n_cm2_s`
- `avg_flux_n_cm2_s`
- `peaking_factor`
- `peak_power_density_W_cm3`
- `avg_power_density_W_cm3`

Recommended:

- `beta_eff`
- `prompt_neutron_lifetime_s`
- `leakage_fraction`
- `leakage_top_fraction`
- `leakage_side_fraction`
- `leakage_bottom_fraction`

### 7. Thermal-Hydraulic Labels

Required if the solver computes them:

- `mass_flow_kg_s`
- `vol_flow_m3_s`
- `flow_velocity_m_s`
- `core_residence_s`
- `loop_transit_s`
- `Re`
- `Pr`
- `T_inlet_K`
- `T_outlet_K`
- `deltaT_K`

### 8. Mesh Description

Store the actual mesh used for the field outputs:

- `nr`
- `nz`
- `r_centers_cm`
- `z_centers_cm`
- optional:
  - `r_edges_cm`
  - `z_edges_cm`

### 9. Field Outputs

Required:

- `phi1_shape`
- `phi2_shape`
- `phi_total_shape`
- `power_density_W_cm3_shape`

Recommended:

- `fission_rate_cm3_s_shape`
- `Jr1_shape`
- `Jz1_shape`
- `Jr2_shape`
- `Jz2_shape`

### 10. Provenance

Every record should say how it was generated:

- `sample_id`
- `solver_version`
- `xs_source`
- `bc_type`
- `normalization_rule`
- `source_document`
- `notes`

## Required Normalization Rules

For the field model, use deterministic normalization, not arbitrary sample-by-sample scaling.

Recommended:

- store shape fields with a fixed integral normalization
- store physical fields separately if power normalization is applied

Example:

- `phi1_shape`, `phi2_shape` are normalized shape fields
- `phi1_phys_n_cm2_s`, `phi2_phys_n_cm2_s` are power-normalized physical fields

This avoids forcing the field model to learn amplitude and shape simultaneously in the first stage.

## Data Splits

Do not use a naive random split alone.

Use three split families:

1. random design holdout
2. geometry holdout:
   - held-out radius bands
   - held-out aspect-ratio bands
3. physics holdout:
   - held-out temperature sweeps
   - held-out near-critical sweeps

This is necessary because v2 is learning an operator over both geometry and materials.

## Acceptance Requirements For The Exporter

The exporter is not done until it can:

- emit machine-readable records that validate against the JSON schema
- reproduce Fuel C benchmark labels within the current solver fidelity
- save both scalar labels and field arrays
- attach explicit BC and XS provenance to every sample
