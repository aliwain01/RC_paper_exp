# Gate G5 decision (human, 2026-10-25) — local only, not committed (DESIGN §17 D129)

- Intensity grid accepted under the rule: low = grid value whose 2021 median peer percentile is nearest 0.875,
  medium = smallest reaching 0.95 (D98), high = smallest reaching 0.975 (calibration on the extended grid, D128):
  upcoding 0.70 / 0.85 / 1.0; overutilization 1.8 / 3.0 / 4.0; atypical mix 0.01 / 0.05 / 0.10;
  charge inflation 1.5 / 2.0 / 2.6; DME mill 4 / 16 / 32.
- DME mill calibrated against all reference providers (option a; non-referrers have DI1 z = 0, D40).
- patient_mix added to atypical mix's expected spill-over groups.
- D90–D97, D99, D100 and DMEPOS re-suppression on claims (D114) confirmed as written.
- "Add resume first": injection-model training and attribution made resumable (D130) before the long runs.
- Draw 0 built at the old intensities is rebuilt (its build fingerprint changes with the grid).
