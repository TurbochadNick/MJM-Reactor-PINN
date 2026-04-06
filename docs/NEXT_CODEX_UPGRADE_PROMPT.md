# Next Codex Upgrade Prompt

Use the following prompt to start the next Codex thread:

```text
You are working inside the MJM Reactor PINN repository. I want you to autonomously upgrade the legacy molten-salt reactor solver and its PINN stack, not just analyze it.

Repository context:
- Root README: README.md
- Legacy solver assets:
  - legacy/v1/mjm_solver_v1.py
  - legacy/v1/plot_v1.py
  - legacy/v1/pinn_training_data.json
- Assessment rubric:
  - docs/mjm_assessment.docx
  - docs/mjm_assessment_summary.md
- Existing v2 design package:
  - monster_v2_package/
- Existing runnable prototype:
  - scripts/run_first_pinn.py

Mission:
Upgrade the solver so it produces more physically credible neutronics outputs, especially temperature reactivity and flux, then improve the surrogate/PINN accordingly. Use the assessment doc as an acceptance rubric and treat its technical findings as real defects to fix.

Very important operating rules:
1. Do not stop at analysis. Make code changes, run them, verify them, and summarize what improved.
2. Preserve the original legacy files as references, but you may refactor or create new modules/scripts around them.
3. Prefer implementing improvements in-repo rather than writing a speculative plan.
4. When assumptions are necessary, document them clearly in code comments or markdown.
5. Do not revert unrelated user changes.

Top priorities, in order:

Priority 1: Fix flux and power normalization
- Inspect how the solver converts eigenvector flux shapes into physical flux.
- The assessment says the current normalization likely uses the wrong fission source expression:
  - wrong pattern: (Sigma_f1 + Sigma_f2) * (phi1 + phi2)
  - target pattern: Sigma_f1 * phi1 + Sigma_f2 * phi2, with consistent group weighting and energy-per-fission usage
- Correct the normalization and recompute:
  - peak total flux
  - peak thermal flux if available
  - average flux
  - power density map
  - peaking factor
- Save before/after comparisons in a markdown report under artifacts/ or docs/.

Priority 2: Improve reactivity as a function of temperature
- Add or correct a proper temperature sweep workflow that computes:
  - k_eff(T)
  - dk/dT
  - alpha_T in pcm/K
- Use ORNL Fuel C benchmark data as the main validation target.
- The assessment summary says the current solver gives about -53 pcm/K, while the ORNL target is about -12.53 pcm/K.
- Investigate and improve the likely causes:
  - overly crude Doppler treatment
  - single global temperature treatment instead of better salt/graphite treatment
  - overly large group-2 diffusion coefficient
  - simplistic thermal-group treatment
- If full fidelity is not possible in one pass, make the solver directionally more physical and quantify the remaining gap.
- Produce an explicit temperature-reactivity plot and a markdown summary of the result.

Priority 3: Improve flux calculation fidelity
- Expose the actual 2-group field outputs cleanly:
  - phi1(r,z)
  - phi2(r,z)
  - total flux
  - power density map
- Save these fields in a machine-readable format for downstream training.
- Add a reusable exporter script that writes:
  - scalar metadata
  - cross sections / derived leakage features if available
  - 2D field arrays
  - normalization info
- Confirm that radial and axial flux shapes remain reasonable against ORNL-style finite-cylinder expectations.

Priority 4: Fix delayed neutron treatment
- Review the circulating-fuel delayed neutron calculation in the legacy solver.
- The assessment says the current ratio beta_eff / beta_static is about 0.982, but the ORNL target is about 0.544.
- Fix the implementation and verify:
  - beta_static
  - beta_eff
  - beta_eff / beta_static
- If the assessment indicates a formula overwrite bug, locate and correct it.
- Add a small validation output or test for this.

Priority 5: Improve leakage / buckling / boundary conditions
- Replace simplistic bare-vacuum handling with a more defensible finite-cylinder treatment using extrapolated boundaries or equivalent leakage correction.
- Make geometric buckling explicit:
  - Br^2
  - Bz^2
  - Bg^2
- Incorporate this cleanly in the solver outputs and exported data.
- Re-evaluate k_eff and temperature coefficient sensitivity after the boundary/leakage improvement.

Priority 6: Improve transport / group-constant realism where feasible
- Audit the group constants in the legacy solver.
- Focus especially on:
  - group-2 diffusion coefficient D2
  - removal / downscatter treatment
  - thermal-group constants
  - k_inf plausibility
- The assessment summary flags D2 as too large, with a more credible target range around 0.30-0.50 cm.
- If you cannot fully regenerate constants from higher-fidelity tools, implement the best justified approximation possible and document it.

Priority 7: Upgrade the PINN/surrogate beyond the current PINN-lite
- Review scripts/run_first_pinn.py and the current training flow.
- The current model is still mostly a scalar surrogate; it does not yet predict flux fields.
- Upgrade the data pipeline so training data can include:
  - k_eff
  - k_inf
  - alpha_T
  - peaking_factor
  - buckling features
  - flux fields phi1(r,z), phi2(r,z)
  - optional power-density fields
- If feasible in this pass, build a first field-based training prototype that takes design variables plus spatial coordinates and predicts flux.
- If a full field PINN is too much for one pass, at minimum:
  - improve scalar surrogate quality
  - add stronger physics penalties
  - export a training-ready field dataset

Specific deliverables I want from this thread:
1. The upgraded solver code committed in the repo.
2. A reproducible script that runs the improved solver and produces:
   - temperature-reactivity results
   - flux / power plots
   - delayed neutron summary
3. A machine-readable exported dataset for at least a small batch of improved samples.
4. A short report summarizing:
   - what was wrong
   - what was changed
   - what benchmarks improved
   - what still remains
5. If you improve the PINN, include the new training script and quantitative comparison against the current baseline.

Validation targets to track explicitly:
- ORNL Fuel C total temperature coefficient around -12.53 pcm/K
- delayed neutron ratio beta_eff / beta_static around 0.544
- more realistic flux magnitude after normalization fix
- better D2 plausibility
- better physically justified leakage treatment

Suggested execution order:
1. Read docs/mjm_assessment_summary.md and docs/mjm_assessment.docx
2. Inspect legacy/v1/mjm_solver_v1.py in detail
3. Fix flux normalization
4. Fix delayed neutron implementation
5. Improve temperature-reactivity modeling
6. Improve leakage / buckling / boundary treatment
7. Export improved field data
8. Upgrade the PINN pipeline
9. Run verification and summarize results

Do the work directly in the codebase and keep moving until there is a meaningful upgraded result, not just a plan.
```
