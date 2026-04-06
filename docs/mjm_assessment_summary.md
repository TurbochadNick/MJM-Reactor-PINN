# MJM Assessment Summary

This is a compact repo-local summary of the uploaded assessment document at `docs/mjm_assessment.docx`.

## High-level scorecard

- Solver numerical structure: `A`
- FLiBe thermophysical model: `A-`
- Flux field spatial shape: `B+`
- Critical dimension search: `B`
- Delayed neutron data source: `B`
- Flux magnitude / power normalisation: `D`
- Temperature coefficient magnitude: `D`
- Delayed neutron reduction ratio: `D`
- Group-2 diffusion coefficient: `D`
- `k_inf` plausibility: `C-`
- Doppler broadening: `C`
- Removal cross section: `C`
- Vacuum boundary conditions: `C`
- PINN flux shape surrogate: `B`
- PINN `k_eff` surrogate: `F`
- PINN physics enforcement: `F`
- Training dataset quality: `C+`
- Solver speed/scalability: `A`

## Priority technical findings

1. Flux magnitude / power normalisation bug
   - The assessment flags an approximately `11x` flux magnitude error due to power normalisation using `(Sigma_f1 + Sigma_f2) * (phi1 + phi2)` instead of `Sigma_f1 * phi1 + Sigma_f2 * phi2`.
   - Benchmark callout: peak thermal flux should be closer to `3.29e13 n/cm^2/s` than the much larger current value.

2. Temperature coefficient is too negative
   - Current solver result is around `-53 pcm/K`.
   - ORNL `Fuel C` benchmark target is about `-12.53 pcm/K`.
   - Suspected causes in the assessment: simplistic Doppler treatment, single global temperature treatment, and an overly large group-2 diffusion coefficient.

3. Delayed neutron reduction is physically wrong
   - Current solver ratio is about `beta_eff / beta_static = 0.982`.
   - ORNL benchmark target is about `0.544`.
   - The assessment points to a formula overwrite / implementation bug.

4. Group-2 diffusion coefficient is too large
   - Assessment target range: roughly `0.30-0.50 cm`.

5. `k_inf` and leakage physics need refinement
   - Improve plausibility of `k_inf`.
   - Replace bare vacuum treatment with extrapolated-vacuum boundary conditions / finite-cylinder leakage treatment.

6. PINN needs real physics and better labels
   - Current scalar surrogate is not enough.
   - Dataset should include flux fields, more near-critical coverage, and explicit PDE / leakage physics terms.

## Practical upgrade targets

- Fix fission-source power normalisation first.
- Produce temperature-reactivity curves and match ORNL `Fuel C` coefficient much more closely.
- Expose and save `phi1(r,z)` and `phi2(r,z)` field outputs.
- Correct circulating-fuel delayed neutron treatment.
- Add extrapolated dimensions / buckling-aware leakage.
- Upgrade the PINN from scalar interpolation toward a field-based physics-guided model.
