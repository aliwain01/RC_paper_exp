# Gate G3 report: detection results (end of Phase 3)

Inputs: 28 harmonised CMS files under `data/interim` and 80 registered raw files (`data/raw/manifest.json`).

Generated 2026-10-22T20:56:29.841590+00:00 by `scenario-fraud phase3 report`. HANDOFF §6 G3: detection results with CIs; the human decides which DESIGN §10.8 / §16 branch the paper follows. Rankings use raw margins; probabilities are isotonic-calibrated (models.yaml calibration_source). T1 uses the reference model (seed 0) of each family.

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
| lr | 0.001 [0.000, 0.001] | 0.706 | 0.001 [0.000, 0.003] | 0.001 [0.000, 0.002] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.001] | 0.113 | 0.00009 | 0.0000 |
| ebm | 0.001 [0.000, 0.001] | 0.785 | 0.002 [0.000, 0.004] | 0.001 [0.001, 0.002] | 0.001 [0.000, 0.001] | 0.001 [0.001, 0.001] | 0.093 | 0.00010 | 0.0000 |
| ga2m | 0.000 [0.000, 0.001] | 0.772 | 0.001 [0.000, 0.003] | 0.002 [0.001, 0.002] | 0.001 [0.001, 0.002] | 0.001 [0.000, 0.001] | 0.113 | 0.00009 | 0.0000 |
| xgb | 0.001 [0.000, 0.002] | 0.729 | 0.002 [0.000, 0.005] | 0.002 [0.001, 0.003] | 0.001 [0.001, 0.002] | 0.001 [0.001, 0.001] | 0.108 | 0.00010 | 0.0001 |
| xgb_mono | 0.001 [0.000, 0.002] | 0.719 | 0.003 [0.000, 0.007] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.001] | 0.001 [0.000, 0.001] | 0.098 | 0.00009 | 0.0002 |
| lgbm | 0.001 [0.000, 0.001] | 0.771 | 0.003 [0.000, 0.006] | 0.002 [0.001, 0.003] | 0.001 [0.001, 0.002] | 0.001 [0.000, 0.001] | 0.124 | 0.00010 | 0.0001 |
| lgbm_mono | 0.001 [0.000, 0.003] | 0.769 | 0.002 [0.000, 0.005] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.001] | 0.001 [0.000, 0.001] | 0.072 | 0.00009 | 0.0004 |

**Δ = best GBM (`xgb`) − best glass box (`ebm`)**, paired NPI bootstrap (1000 replicates):

| metric | Δ | CI | p | p (Holm over k) |
|---|---|---|---|---|
| pr_auc | 0.000 | [-0.000, 0.001] | 0.288 | – |
| p@0.001 | 0.000 | [-0.001, 0.003] | 0.934 | 1.000 |
| p@0.005 | 0.000 | [-0.000, 0.001] | 0.178 | 0.712 |
| p@0.01 | 0.000 | [-0.000, 0.001] | 0.546 | 1.000 |
| p@0.02 | 0.000 | [-0.000, 0.000] | 0.968 | 1.000 |

- H1 (Δ PR-AUC and Δ precision@1% CIs > 0, Holm across k): **not supported**
- H1 is not supported (a CI includes 0 or lies below it): consistent with the DESIGN §10.8 row "No significant difference" if the gap is not significant, or with a glass-box advantage if Δ < 0.

Per test year:

| model | year | N | positives | base rate | PR-AUC | P@1% |
|---|---|---|---|---|---|---|
| lr | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| lr | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| ebm | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| ebm | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| ga2m | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| ga2m | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| xgb | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| xgb | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| xgb_mono | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| xgb_mono | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| lgbm | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| lgbm | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| lgbm_mono | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| lgbm_mono | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |

Label horizon (DESIGN §5.4: rows whose W-year window is not fully observed; 2023 has ≈ 2.8 of 3 years with an October 2026 snapshot):

| year | rows | horizon_incomplete | positives |
|---|---|---|---|
| 2022 | 1027483 | 0 | 107 |
| 2023 | 1046619 | 1046619 | 87 |

Calibration within the top 5% (`xgb`):

| bin | n | mean_predicted | observed |
|---|---|---|---|
| 0 | 10370 | 0.0003 | 0.0003 |
| 1 | 10371 | 0.0003 | 0.0002 |
| 2 | 10371 | 0.0003 | 0.0000 |
| 3 | 10370 | 0.0004 | 0.0007 |
| 4 | 10371 | 0.0004 | 0.0002 |
| 5 | 10371 | 0.0009 | 0.0008 |
| 6 | 10370 | 0.0009 | 0.0004 |
| 7 | 10371 | 0.0013 | 0.0013 |
| 8 | 10371 | 0.0015 | 0.0010 |
| 9 | 10370 | 0.0048 | 0.0012 |

Calibration within the top 5% (`ebm`):

| bin | n | mean_predicted | observed |
|---|---|---|---|
| 0 | 10370 | 0.0004 | 0.0002 |
| 1 | 10371 | 0.0004 | 0.0004 |
| 2 | 10371 | 0.0004 | 0.0005 |
| 3 | 10370 | 0.0004 | 0.0001 |
| 4 | 10371 | 0.0004 | 0.0002 |
| 5 | 10371 | 0.0004 | 0.0002 |
| 6 | 10370 | 0.0014 | 0.0007 |
| 7 | 10371 | 0.0016 | 0.0012 |
| 8 | 10371 | 0.0019 | 0.0010 |
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
| lr | 0.001 | 0.698 | 0.000 | 0.002 | 0.002 | 0.001 | 0.123 | 0.00013 | – |
| ebm | 0.001 | 0.793 | 0.000 | 0.002 | 0.002 | 0.001 | 0.146 | 0.00013 | – |
| ga2m | 0.001 | 0.759 | 0.003 | 0.002 | 0.002 | 0.001 | 0.123 | 0.00013 | – |
| xgb | 0.001 | 0.697 | 0.002 | 0.001 | 0.001 | 0.001 | 0.100 | 0.00013 | – |
| xgb_mono | 0.000 | 0.695 | 0.000 | 0.002 | 0.001 | 0.001 | 0.100 | 0.00013 | – |
| lgbm | 0.001 | 0.805 | 0.003 | 0.002 | 0.002 | 0.001 | 0.131 | 0.00013 | – |
| lgbm_mono | 0.001 | 0.778 | 0.002 | 0.002 | 0.001 | 0.001 | 0.100 | 0.00013 | – |

Per test year:

| model | year | N | positives | base rate | PR-AUC | P@1% |
|---|---|---|---|---|---|---|
| lr | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.002 |
| ebm | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.002 |
| ga2m | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.002 |
| xgb | 2021 | 1009812 | 130 | 0.00013 | 0.001 | 0.001 |
| xgb_mono | 2021 | 1009812 | 130 | 0.00013 | 0.000 | 0.001 |
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
| lr | 0.001 | 0.699 | 0.002 | 0.002 | 0.001 | 0.001 | 0.112 | 0.00010 | – |
| ebm | 0.001 | 0.834 | 0.002 | 0.001 | 0.001 | 0.001 | 0.093 | 0.00011 | – |
| ga2m | 0.001 | 0.801 | 0.001 | 0.002 | 0.001 | 0.001 | 0.121 | 0.00010 | – |
| xgb | 0.001 | 0.751 | 0.003 | 0.002 | 0.001 | 0.001 | 0.103 | 0.00011 | – |
| xgb_mono | 0.001 | 0.745 | 0.003 | 0.001 | 0.001 | 0.001 | 0.093 | 0.00011 | – |
| lgbm | 0.001 | 0.793 | 0.004 | 0.002 | 0.001 | 0.001 | 0.131 | 0.00011 | – |
| lgbm_mono | 0.001 | 0.786 | 0.002 | 0.001 | 0.001 | 0.001 | 0.065 | 0.00010 | – |

Per test year:

| model | year | N | positives | base rate | PR-AUC | P@1% |
|---|---|---|---|---|---|---|
| lr | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| ebm | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| ga2m | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| xgb | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| xgb_mono | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| lgbm | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |
| lgbm_mono | 2022 | 1027483 | 107 | 0.00010 | 0.001 | 0.001 |

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
| lr | 0.000 | 0.717 | 0.001 | 0.001 | 0.001 | 0.001 | 0.115 | 0.00008 | – |
| ebm | 0.000 | 0.740 | 0.003 | 0.001 | 0.001 | 0.000 | 0.103 | 0.00008 | – |
| ga2m | 0.000 | 0.744 | 0.000 | 0.001 | 0.001 | 0.001 | 0.069 | 0.00008 | – |
| xgb | 0.000 | 0.697 | 0.003 | 0.002 | 0.001 | 0.001 | 0.103 | 0.00008 | – |
| xgb_mono | 0.001 | 0.698 | 0.002 | 0.001 | 0.001 | 0.001 | 0.103 | 0.00008 | – |
| lgbm | 0.001 | 0.759 | 0.002 | 0.002 | 0.001 | 0.001 | 0.126 | 0.00008 | – |
| lgbm_mono | 0.001 | 0.746 | 0.002 | 0.001 | 0.001 | 0.001 | 0.092 | 0.00008 | – |

Per test year:

| model | year | N | positives | base rate | PR-AUC | P@1% |
|---|---|---|---|---|---|---|
| lr | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| ebm | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| ga2m | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| xgb | 2023 | 1046619 | 87 | 0.00008 | 0.000 | 0.001 |
| xgb_mono | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| lgbm | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |
| lgbm_mono | 2023 | 1046619 | 87 | 0.00008 | 0.001 | 0.001 |

Label horizon (DESIGN §5.4: rows whose W-year window is not fully observed; 2023 has ≈ 2.8 of 3 years with an October 2026 snapshot):

| year | rows | horizon_incomplete | positives |
|---|---|---|---|
| 2023 | 1046619 | 1046619 | 87 |

## Showcase

2024: 1073061 provider-years scored by the primary models of their own fold (ebm, ga2m, lgbm, lgbm_mono, lr, xgb, xgb_mono). No accuracy claims (DESIGN §7.6).
