# Gate G3 decision (human, 2026-10-23) — local only, not committed (DESIGN §17 D122)

- Branch: DESIGN §10.8 row "No significant difference": H1 not supported (Δ = XGBoost − EBM; every CI includes 0).
  Recommend the glass box; RQ1 reported as a null result; the paper's weight rests on RQ2–RQ5.
  Detection is weak but above chance (ROC-AUC 0.71–0.79, lift ≈ 7–12× at the top 1 %), so the §16
  "near random" branch does not apply.
- Model choice for Phases 4–6: the frozen D74 selection is kept (best GBM `xgb`, best glass box `ebm`,
  best monotone GBM `lgbm_mono`). Other GBM–glass-box pairings are reported as a sensitivity check.
- Caveat to report: the 2023 label window is incomplete (≈ 2.8 of 3 years observed).
- Publication: G3_report.md / .json stay local until the human approves publishing results.

## Re-check with the corrected ranking (D125, 2026-10-23)

G3_report.md regenerated with `ranking: within_fold_year`; the original pooled-margin report is kept as
G3_report_D81_pooled_margin.md/.json. H1 still not supported: Δ (xgb − ebm) PR-AUC +0.00097, CI [−0.00012, +0.00631];
precision@1 % −0.00005, CI [−0.00043, +0.00039]; precision@0.1 % +0.0024, CI [+0.0005, +0.0048], p = 0.03 but Holm 0.12.
ROC-AUC 0.80–0.86 (was 0.71–0.79); recall@1 % 6–15 %. The G3 branch decision stands.
