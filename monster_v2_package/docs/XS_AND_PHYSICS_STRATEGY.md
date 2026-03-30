# Cross-Section And Physics Strategy

This document isolates the biggest fidelity risk in v2: the cross-section model.

## The Problem

The v1 solver uses a small hard-coded 2-group model with simplified temperature scaling. That is enough for a class prototype, but it is the dominant reason the current temperature coefficient is far more negative than the `ORNL-TM-730` Fuel C benchmark.

The report material gathered for v2 shows that the original MSRE analysis did not rely on a toy two-group scaling law. It used a structured multigroup workflow:

- `GAM-1` for fast-group generation
- `THERMOS` for thermal treatment
- channel-lattice and shielding corrections
- resonance treatment with Dancoff effects

## Recommended Physics Path

### MVP Path

Use the current v1 solver as the data generator, but make it much more transparent.

Required changes:

- export all mixed number densities
- export all macroscopic XS terms
- export buckling features and BC provenance
- export `alpha_T`, `beta_eff`, and leakage metrics

This lets the model learn on the best currently available internal data while preserving a clean place to swap in better XS later.

### Strong V2 Path

Generate homogenized 2-group constants from a transport toolchain.

Best options:

- `SCALE`
- `Serpent`

Target preprocessing setup:

- graphite stringer + salt annulus cell model, consistent with the ORNL lattice description
- temperature grid at least:
  - `800 K`
  - `900 K`
  - `1000 K`
  - optional `1100 K`
- sweep over:
  - uranium enrichment
  - `UF4` loading
  - moderator/fuel fraction proxy

Export per case:

- `D1`, `D2`
- `Sigma_a1`, `Sigma_a2`
- `Sigma_s12`
- `nuSigma_f1`, `nuSigma_f2`
- `Sigma_f1`, `Sigma_f2`
- optional uncertainty or self-shielding descriptors

## What ORNL-TM-730 Gives Us

ORNL gives us very useful workflow constraints and benchmark expectations:

- one-region reference lattice:
  - `22.5 vol% fuel`
  - `77.5 vol% graphite`
- Dancoff/shielding reduces effective surface-to-volume ratio for resonance capture by about `30%`
- fast groups were generated for a `33-group` diffusion method
- `Li-6`, `Li-7`, and `F-19` required special treatment because of library limits

This tells us how to design a modern preprocessing path, even though the report does not hand us a turnkey modern 2-group table.

## Recommended Intermediate Upgrade

If transport preprocessing is not immediately available, do this before anything fancier in the PINN:

1. replace the single `sqrt(T0/T)` thermal scaling with temperature-dependent XS interpolation on a small tabulated grid
2. separate density feedback from spectral feedback
3. treat Fuel C as a calibration anchor for `alpha_T`
4. store all XS terms explicitly in the dataset so later replacement does not break the model interface

## What Not To Do

- Do not hide all the physics inside the PINN and hope it learns around poor XS.
- Do not benchmark only `k_eff` while ignoring `alpha_T` and leakage behavior.
- Do not collapse geometry effects into aspect ratio alone.

## Acceptance Criteria For XS Upgrade

The XS upgrade is doing its job if it materially improves:

- Fuel C temperature coefficient magnitude
- near-critical `k_eff` trend stability
- power coefficient consistency
- leakage-sensitive field shapes
- benchmark agreement without hand-tuned fudge factors
