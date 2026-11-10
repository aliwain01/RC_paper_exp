# Gate G3 report: detection results (end of Phase 3)

Inputs: 28 harmonised CMS files under `data/interim` and 80 registered raw files (`data/raw/manifest.json`).

Generated 2026-10-23T11:46:59.171125+00:00 by `scenario-fraud phase3 report`. HANDOFF §6 G3: detection results with CIs; the human decides which DESIGN §10.8 / §16 branch the paper follows. Ranking: `{ev.get('ranking', 'pooled_margin')}` (DESIGN §17 D125: each provider's within-fold percentile of the margin, so the fold models' differing margin scales do not crowd each other out; `pooled_margin` = raw margins, the original D81). Probabilities are isotonic-calibrated (models.yaml calibration_source). T1 uses the reference model (seed 0) of each family.

- G2 freeze record: recorded 2026-10-20T10:06:36.661046+00:00
- Final fit / calibration: `all_train` / `tuning_model` (DESIGN §17 D59, open)

## Design primary: train 2018–2021, test 2022–2023

Validation average precision (mean over folds of the tuning model on `tune_valid`), used to choose the best models:

| family | mean_valid_ap | chosen_as |
|---|---|---|
| lr | 0.002 |  |
| ebm | 0.018 | glass_box |
| ga2m | 0.013 |  |
| xgb | 0.017 | gbm |
| xgb_mono | 0.007 |  |
| lgbm | 0.015 |  |
| lgbm_mono | 0.013 | gbm_mono |

**T1 (pooled 2022–2023, 95 % NPI-bootstrap CIs, reference models)**

| model | PR-AUC | ROC-AUC | P@0.1% | P@0.5% | P@1.0% | P@2.0% | R@1.0% | Brier | seed SD (PR-AUC) |
|---|---|---|---|---|---|---|---|---|---|
| lr | 0.001 [0.000, 0.001] | 0.798 | 0.001 [0.000, 0.003] | 0.001 [0.000, 0.002] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.001] | 0.119 | 0.00009 | 0.0000 |
| ebm | 0.001 [0.001, 0.001] | 0.835 | 0.001 [0.000, 0.004] | 0.002 [0.001, 0.003] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.002] | 0.134 | 0.00010 | 0.0000 |
| ga2m | 0.001 [0.000, 0.001] | 0.855 | 0.002 [0.000, 0.005] | 0.002 [0.001, 0.003] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.001] | 0.139 | 0.00009 | 0.0000 |
| xgb | 0.002 [0.000, 0.008] | 0.810 | 0.004 [0.001, 0.007] | 0.002 [0.001, 0.003] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.001] | 0.129 | 0.00010 | 0.0003 |
| xgb_mono | 0.001 [0.000, 0.001] | 0.811 | 0.002 [0.000, 0.005] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.001] | 0.124 | 0.00009 | 0.0001 |
| lgbm | 0.001 [0.000, 0.002] | 0.815 | 0.002 [0.000, 0.005] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.001] | 0.149 | 0.00010 | 0.0002 |
| lgbm_mono | 0.001 [0.000, 0.005] | 0.808 | 0.003 [0.000, 0.007] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.001] | 0.001 [0.000, 0.001] | 0.062 | 0.00009 | 0.0003 |

For comparison, the original D81 ranking (raw margins pooled across the fold models):

| model | pr_auc | roc_auc | P@0.1% | P@0.5% | P@1.0% | P@2.0% |
|---|---|---|---|---|---|---|
| lr | 0.00055 | 0.706 | 0.00145 | 0.00116 | 0.00106 | 0.00089 |
| ebm | 0.00052 | 0.785 | 0.00193 | 0.00125 | 0.00087 | 0.00082 |
| ga2m | 0.00046 | 0.772 | 0.00145 | 0.00154 | 0.00106 | 0.00077 |
| xgb | 0.00066 | 0.729 | 0.00241 | 0.00164 | 0.00101 | 0.00084 |
| xgb_mono | 0.00072 | 0.719 | 0.00289 | 0.00116 | 0.00092 | 0.00065 |
| lgbm | 0.00066 | 0.771 | 0.00289 | 0.00164 | 0.00116 | 0.00077 |
| lgbm_mono | 0.00082 | 0.769 | 0.00241 | 0.00096 | 0.00067 | 0.00058 |

**Δ = best GBM (`xgb`) − best glass box (`ebm`)**, paired NPI bootstrap (1000 replicates):

| metric | Δ | CI | p | p (Holm over k) |
|---|---|---|---|---|
| pr_auc | 0.001 | [-0.000, 0.006] | 0.166 | – |
| p@0.001 | 0.002 | [0.000, 0.005] | 0.030 | 0.120 |
| p@0.005 | 0.000 | [-0.000, 0.001] | 0.502 | 1.000 |
| p@0.01 | -0.000 | [-0.000, 0.000] | 1.000 | 1.000 |
| p@0.02 | -0.000 | [-0.000, 0.000] | 0.308 | 0.924 |

- H1 (Δ PR-AUC and Δ precision@1% CIs > 0, Holm across k): **not supported**
- H1 is not supported (a CI includes 0 or lies below it): consistent with the DESIGN §10.8 row "No significant difference" if the gap is not significant, or with a glass-box advantage if Δ < 0.

Per test year:

| model | year | N | positives | base rate | PR-AUC | P@1% |
|---|---|---|---|---|---|---|
| lr | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.002 |
| lr | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| ebm | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.002 |
| ebm | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| ga2m | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.002 |
| ga2m | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| xgb | 2022 | 1027483 | 107 | 0.00010 | 0.002 | 0.001 |
| xgb | 2023 | 1046619 | 87 | 0.00008 | 0.002 | 0.001 |
| xgb_mono | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.002 |
| xgb_mono | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| lgbm | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.002 |
| lgbm | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| lgbm_mono | 2022 | 1027483 | 107 | 0.00010 | 0.002 | 0.001 |
| lgbm_mono | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |

Label horizon (DESIGN §5.4: rows whose W-year window is not fully observed; 2023 has ≈ 2.8 of 3 years with an October 2026 snapshot):

| year | rows | horizon_incomplete | positives |
|---|---|---|---|
| 2022 | 1027483 | 0 | 107 |
| 2023 | 1046619 | 1046619 | 87 |

Calibration within the top 5% (`xgb`):

| bin | n | mean_predicted | observed |
|---|---|---|---|
| 0 | 10370 | 0.0003 | 0.0001 |
| 1 | 10371 | 0.0007 | 0.0007 |
| 2 | 10371 | 0.0008 | 0.0002 |
| 3 | 10370 | 0.0009 | 0.0002 |
| 4 | 10371 | 0.0009 | 0.0006 |
| 5 | 10371 | 0.0009 | 0.0007 |
| 6 | 10370 | 0.0010 | 0.0002 |
| 7 | 10371 | 0.0015 | 0.0012 |
| 8 | 10371 | 0.0015 | 0.0008 |
| 9 | 10370 | 0.0048 | 0.0013 |

Calibration within the top 5% (`ebm`):

| bin | n | mean_predicted | observed |
|---|---|---|---|
| 0 | 10370 | 0.0003 | 0.0010 |
| 1 | 10371 | 0.0004 | 0.0002 |
| 2 | 10371 | 0.0005 | 0.0002 |
| 3 | 10370 | 0.0007 | 0.0004 |
| 4 | 10371 | 0.0013 | 0.0004 |
| 5 | 10371 | 0.0014 | 0.0004 |
| 6 | 10370 | 0.0016 | 0.0012 |
| 7 | 10371 | 0.0018 | 0.0009 |
| 8 | 10371 | 0.0021 | 0.0013 |
| 9 | 10370 | 0.0048 | 0.0015 |

## Design O1: train 2018–2020, test 2021–2021

Validation average precision (mean over folds of the tuning model on `tune_valid`), used to choose the best models:

| family | mean_valid_ap | chosen_as |
|---|---|---|
| lr | 0.002 | glass_box |
| ebm | 0.001 |  |
| ga2m | 0.001 |  |
| xgb | 0.001 |  |
| xgb_mono | 0.001 |  |
| lgbm | 0.001 | gbm |
| lgbm_mono | 0.003 | gbm_mono |

**T1 (2021, reference models)**

| model | PR-AUC | ROC-AUC | P@0.1% | P@0.5% | P@1.0% | P@2.0% | R@1.0% | Brier | seed SD (PR-AUC) |
|---|---|---|---|---|---|---|---|---|---|
| lr | 0.001 | 0.818 | 0.000 | 0.002 | 0.002 | 0.001 | 0.138 | 0.00013 | – |
| ebm | 0.001 | 0.837 | 0.002 | 0.002 | 0.002 | 0.001 | 0.169 | 0.00013 | – |
| ga2m | 0.001 | 0.840 | 0.001 | 0.002 | 0.002 | 0.001 | 0.146 | 0.00013 | – |
| xgb | 0.001 | 0.794 | 0.003 | 0.002 | 0.001 | 0.001 | 0.085 | 0.00013 | – |
| xgb_mono | 0.001 | 0.791 | 0.001 | 0.002 | 0.001 | 0.001 | 0.108 | 0.00013 | – |
| lgbm | 0.001 | 0.828 | 0.003 | 0.002 | 0.002 | 0.001 | 0.138 | 0.00013 | – |
| lgbm_mono | 0.001 | 0.806 | 0.002 | 0.002 | 0.001 | 0.001 | 0.115 | 0.00013 | – |

For comparison, the original D81 ranking (raw margins pooled across the fold models):

| model | pr_auc | roc_auc | P@0.1% | P@0.5% | P@1.0% | P@2.0% |
|---|---|---|---|---|---|---|
| lr | 0.00062 | 0.698 | 0.00000 | 0.00158 | 0.00158 | 0.00124 |
| ebm | 0.00076 | 0.793 | 0.00000 | 0.00178 | 0.00188 | 0.00114 |
| ga2m | 0.00077 | 0.759 | 0.00297 | 0.00178 | 0.00158 | 0.00129 |
| xgb | 0.00052 | 0.697 | 0.00198 | 0.00119 | 0.00129 | 0.00089 |
| xgb_mono | 0.00050 | 0.695 | 0.00000 | 0.00178 | 0.00129 | 0.00104 |
| lgbm | 0.00087 | 0.805 | 0.00297 | 0.00198 | 0.00168 | 0.00129 |
| lgbm_mono | 0.00075 | 0.778 | 0.00198 | 0.00198 | 0.00129 | 0.00129 |

Per test year:

| model | year | N | positives | base rate | PR-AUC | P@1% |
|---|---|---|---|---|---|---|
| lr | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.002 |
| ebm | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.002 |
| ga2m | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.002 |
| xgb | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.001 |
| xgb_mono | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.001 |
| lgbm | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.002 |
| lgbm_mono | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.001 |

Label horizon (DESIGN §5.4: rows whose W-year window is not fully observed; 2023 has ≈ 2.8 of 3 years with an October 2026 snapshot):

| year | rows | horizon_incomplete | positives |
|---|---|---|---|
| 2021 | 1009812 | 0 | 130 |

## Design O2: train 2018–2021, test 2022–2022

Validation average precision (mean over folds of the tuning model on `tune_valid`), used to choose the best models:

| family | mean_valid_ap | chosen_as |
|---|---|---|
| lr | 0.002 |  |
| ebm | 0.018 | glass_box |
| ga2m | 0.013 |  |
| xgb | 0.017 | gbm |
| xgb_mono | 0.007 |  |
| lgbm | 0.015 |  |
| lgbm_mono | 0.013 | gbm_mono |

**T1 (2022, reference models)**

| model | PR-AUC | ROC-AUC | P@0.1% | P@0.5% | P@1.0% | P@2.0% | R@1.0% | Brier | seed SD (PR-AUC) |
|---|---|---|---|---|---|---|---|---|---|
| lr | 0.001 | 0.819 | 0.002 | 0.002 | 0.002 | 0.001 | 0.150 | 0.00010 | – |
| ebm | 0.001 | 0.870 | 0.001 | 0.002 | 0.002 | 0.001 | 0.150 | 0.00011 | – |
| ga2m | 0.001 | 0.881 | 0.003 | 0.002 | 0.002 | 0.001 | 0.168 | 0.00010 | – |
| xgb | 0.002 | 0.825 | 0.004 | 0.003 | 0.001 | 0.001 | 0.121 | 0.00011 | – |
| xgb_mono | 0.001 | 0.833 | 0.002 | 0.002 | 0.002 | 0.001 | 0.150 | 0.00011 | – |
| lgbm | 0.001 | 0.842 | 0.004 | 0.002 | 0.002 | 0.001 | 0.150 | 0.00011 | – |
| lgbm_mono | 0.002 | 0.832 | 0.003 | 0.001 | 0.001 | 0.001 | 0.056 | 0.00010 | – |

For comparison, the original D81 ranking (raw margins pooled across the fold models):

| model | pr_auc | roc_auc | P@0.1% | P@0.5% | P@1.0% | P@2.0% |
|---|---|---|---|---|---|---|
| lr | 0.00085 | 0.699 | 0.00195 | 0.00156 | 0.00117 | 0.00107 |
| ebm | 0.00071 | 0.834 | 0.00195 | 0.00117 | 0.00097 | 0.00117 |
| ga2m | 0.00062 | 0.801 | 0.00097 | 0.00175 | 0.00127 | 0.00102 |
| xgb | 0.00085 | 0.751 | 0.00292 | 0.00156 | 0.00107 | 0.00097 |
| xgb_mono | 0.00087 | 0.745 | 0.00292 | 0.00136 | 0.00097 | 0.00083 |
| lgbm | 0.00105 | 0.793 | 0.00389 | 0.00195 | 0.00136 | 0.00107 |
| lgbm_mono | 0.00116 | 0.786 | 0.00195 | 0.00097 | 0.00068 | 0.00058 |

Per test year:

| model | year | N | positives | base rate | PR-AUC | P@1% |
|---|---|---|---|---|---|---|
| lr | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.002 |
| ebm | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.002 |
| ga2m | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.002 |
| xgb | 2022 | 1027483 | 107 | 0.00010 | 0.002 | 0.001 |
| xgb_mono | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.002 |
| lgbm | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.002 |
| lgbm_mono | 2022 | 1027483 | 107 | 0.00010 | 0.002 | 0.001 |

Label horizon (DESIGN §5.4: rows whose W-year window is not fully observed; 2023 has ≈ 2.8 of 3 years with an October 2026 snapshot):

| year | rows | horizon_incomplete | positives |
|---|---|---|---|
| 2022 | 1027483 | 0 | 107 |

## Design O3: train 2018–2022, test 2023–2023

Validation average precision (mean over folds of the tuning model on `tune_valid`), used to choose the best models:

| family | mean_valid_ap | chosen_as |
|---|---|---|
| lr | 0.001 |  |
| ebm | 0.002 | glass_box |
| ga2m | 0.001 |  |
| xgb | 0.001 |  |
| xgb_mono | 0.001 |  |
| lgbm | 0.001 | gbm |
| lgbm_mono | 0.001 | gbm_mono |

**T1 (2023, reference models)**

| model | PR-AUC | ROC-AUC | P@0.1% | P@0.5% | P@1.0% | P@2.0% | R@1.0% | Brier | seed SD (PR-AUC) |
|---|---|---|---|---|---|---|---|---|---|
| lr | 0.000 | 0.777 | 0.001 | 0.001 | 0.001 | 0.001 | 0.092 | 0.00008 | – |
| ebm | 0.001 | 0.796 | 0.002 | 0.001 | 0.001 | 0.001 | 0.103 | 0.00008 | – |
| ga2m | 0.000 | 0.822 | 0.001 | 0.001 | 0.001 | 0.001 | 0.115 | 0.00008 | – |
| xgb | 0.001 | 0.766 | 0.004 | 0.001 | 0.001 | 0.001 | 0.103 | 0.00008 | – |
| xgb_mono | 0.001 | 0.792 | 0.003 | 0.001 | 0.001 | 0.001 | 0.103 | 0.00008 | – |
| lgbm | 0.001 | 0.796 | 0.002 | 0.002 | 0.001 | 0.001 | 0.149 | 0.00008 | – |
| lgbm_mono | 0.001 | 0.777 | 0.002 | 0.001 | 0.001 | 0.001 | 0.080 | 0.00008 | – |

For comparison, the original D81 ranking (raw margins pooled across the fold models):

| model | pr_auc | roc_auc | P@0.1% | P@0.5% | P@1.0% | P@2.0% |
|---|---|---|---|---|---|---|
| lr | 0.00042 | 0.717 | 0.00096 | 0.00096 | 0.00096 | 0.00076 |
| ebm | 0.00039 | 0.740 | 0.00287 | 0.00115 | 0.00086 | 0.00048 |
| ga2m | 0.00031 | 0.744 | 0.00000 | 0.00115 | 0.00057 | 0.00057 |
| xgb | 0.00048 | 0.697 | 0.00287 | 0.00153 | 0.00086 | 0.00057 |
| xgb_mono | 0.00084 | 0.698 | 0.00191 | 0.00115 | 0.00086 | 0.00057 |
| lgbm | 0.00058 | 0.759 | 0.00191 | 0.00153 | 0.00105 | 0.00057 |
| lgbm_mono | 0.00084 | 0.746 | 0.00191 | 0.00096 | 0.00076 | 0.00067 |

Per test year:

| model | year | N | positives | base rate | PR-AUC | P@1% |
|---|---|---|---|---|---|---|
| lr | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| ebm | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| ga2m | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| xgb | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| xgb_mono | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| lgbm | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| lgbm_mono | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |

Label horizon (DESIGN §5.4: rows whose W-year window is not fully observed; 2023 has ≈ 2.8 of 3 years with an October 2026 snapshot):

| year | rows | horizon_incomplete | positives |
|---|---|---|---|
| 2023 | 1046619 | 1046619 | 87 |

## Showcase

2024: 1073061 provider-years scored by the primary models of their own fold (ebm, ga2m, lgbm, lgbm_mono, lr, xgb, xgb_mono). No accuracy claims (DESIGN §7.6).
