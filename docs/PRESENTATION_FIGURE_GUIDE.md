# Presentation Figure Guide

This file does not change the presentation. It gives presentation-ready wording
for the main generated figures.

## 1. Field Comparison Figure

Artifact:

- `artifacts/first_pinn_run/field_holdout_comparison.png`

Suggested slide title:

- `Flux Field Prediction: Reference Solver vs Simple Approximation vs Learned Model`

Suggested caption:

- `Comparison of fast- and thermal-neutron flux fields for one holdout reactor case. The left column is the upgraded solver reference, the middle column is a simple analytic approximation, and the right column is the learned field model. Brighter colors indicate higher neutron activity. The learned model tracks the solver field shape much more closely than the simple approximation.`

Plain-English definitions:

- `Fast-neutron flux (phi1)`: distribution of higher-energy neutrons in the core.
- `Thermal-neutron flux (phi2)`: distribution of slowed-down neutrons that strongly support fission.
- `Reference`: the upgraded solver output used as the target for training and comparison.
- `Simple analytic approximation`: a hand-shaped baseline pattern, not a learned solver surrogate.
- `Learned model`: the trained coordinate-query field model.

Why the audience should care:

- Flux shape tells us where neutron activity and power production are concentrated.
- That matters for:
  - power distribution
  - hotspot awareness
  - thermal margin interpretation
  - rapid design screening without rerunning the full solver every time

Speaker notes:

- `This figure shows where neutron activity is concentrated inside the core.`
- `The left column is our upgraded solver reference output.`
- `The middle column is a simple approximation that gets the broad shape but misses the detailed field behavior.`
- `The right column is the learned model, which follows the solver field much more closely.`
- `This matters because the neutron field drives where power is produced and where local thermal and safety margins matter most.`

## 2. Scalar Parity Figure

Artifact:

- `artifacts/first_pinn_run/scalar_parity.png`

Suggested slide title:

- `Prediction Accuracy For Two Key Reactor Metrics`

Suggested caption:

- `Parity plots comparing model predictions against upgraded solver reference values. The dashed line is perfect agreement. The upgraded physics-guided model improves temperature-feedback prediction substantially, while criticality prediction remains more mixed on the small training set.`

Plain-English definitions:

- `k_eff`: reactor criticality level; values near `1` indicate steady critical operation.
- `alpha_T (pcm/K)`: temperature feedback; more negative values generally indicate stronger passive safety feedback.
- `Parity plot`: predicted value versus reference solver value.

Why the audience should care:

- `k_eff` tells us whether the core is near the desired operating criticality.
- `alpha_T` tells us whether the reactor naturally becomes less reactive as it heats up.
- A fast predictive model is only useful if it reproduces these high-level reactor behaviors accurately.

Speaker notes:

- `On the left, we compare model-predicted k_eff with the upgraded solver result.`
- `On the right, we compare predicted temperature feedback, which is one of the most safety-relevant outputs in the project.`
- `The dashed line is perfect prediction; points closer to it are better.`
- `The upgraded model helps noticeably for alpha_T, which is important because negative temperature feedback is a key passive safety characteristic.`
- `k_eff is still more mixed, which tells us the scalar head needs more data and tuning.`

## 3. Short Investor-Friendly Framing

If the audience is mixed technical/non-technical:

- `The upgraded solver gives us a much more credible digital picture of where the reactor is active and how it responds to temperature.`
- `The learned model begins to reproduce those results much faster, which is useful for design exploration and early screening.`
- `The biggest remaining gap is not code structure but nuclear-data fidelity, especially cross sections.`
