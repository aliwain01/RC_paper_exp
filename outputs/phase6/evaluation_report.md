# Evaluation report (Phase 6)

Inputs: 28 harmonised CMS files under `data/interim` and 80 registered raw files (`data/raw/manifest.json`).

Generated 2026-11-01T01:13:03.283098+00:00 by `scenario-fraud phase6 report`. DESIGN §10, EXPERIMENT_PLAN §5, §7-§9. Explanation vectors are the 7 scheme groups (context excluded); a group counts as found only with a positive contribution; method C ranks by its diagonal (pure effects) (DESIGN §17 D101-D108).

- Best GBM / monotone GBM / glass box (validation, D74): `xgb` / `lgbm_mono` / `ebm`

## T1 Detection (E1)

| model | PR-AUC | ROC-AUC | P@0.1% | P@0.5% | P@1.0% | P@2.0% | Brier |
|---|---|---|---|---|---|---|---|
| lr | 0.001 [0.000, 0.001] | 0.798 | 0.001 | 0.001 | 0.001 | 0.001 | 0.00009 |
| ebm | 0.001 [0.001, 0.001] | 0.835 | 0.001 | 0.002 | 0.001 | 0.001 | 0.00010 |
| ga2m | 0.001 [0.000, 0.001] | 0.855 | 0.002 | 0.002 | 0.001 | 0.001 | 0.00009 |
| xgb | 0.002 [0.000, 0.008] | 0.810 | 0.004 | 0.002 | 0.001 | 0.001 | 0.00010 |
| xgb_mono | 0.001 [0.000, 0.001] | 0.811 | 0.002 | 0.001 | 0.001 | 0.001 | 0.00009 |
| lgbm | 0.001 [0.000, 0.002] | 0.815 | 0.002 | 0.001 | 0.001 | 0.001 | 0.00010 |
| lgbm_mono | 0.001 [0.000, 0.005] | 0.808 | 0.003 | 0.001 | 0.001 | 0.001 | 0.00009 |

From `outputs/phase3/G3_report.json` (CIs and Δ with Holm there).

## T2 Correctness (E4; injection models, use (b))

Rows: single schemes at intensity `medium`; compound columns: the pairs; draws [0, 1, 2, 3, 4]; baseline `peer`.

| model · method | n | Top-1 | Top-3 | MRR | Distractor top-1 | Compound both-in-top-3 | Pair hit |
|---|---|---|---|---|---|---|---|
| GBM (xgb) · A | 7500 | 0.706 | 0.947 | 0.822 | 0.643 | 0.629 | — |
| GBM (xgb) · B | 7500 | 0.705 | 0.947 | 0.822 | 0.641 | 0.634 | — |
| GBM (xgb) · C | 7500 | 0.703 | 0.952 | 0.821 | 0.637 | 0.668 | 0.030 |
| GBM (xgb) · E | 7500 | 0.699 | 0.916 | 0.807 | 0.635 | 0.524 | — |
| GA²M · D | 7500 | 0.673 | 0.913 | 0.789 | 0.607 | 0.645 | 0.049 |
| EBM · D | 7500 | 0.650 | 0.899 | 0.770 | 0.576 | 0.657 | — |
| LR · D | 7500 | 0.563 | 0.859 | 0.706 | 0.465 | 0.498 | — |
| chance (this answer key) |  | 0.171 | 0.486 | 0.403 | 0.171 | 0.171 | 0.060 |

Top-1 by intensity (best GBM, all methods; glass boxes, method D; the rest in `correctness.json`):

| family | method | intensity | n | top-1 | chance |
|---|---|---|---|---|---|
| ebm | D | high | 7500 | 0.608 | 0.171 |
| ebm | D | low | 7497 | 0.637 | 0.171 |
| ebm | D | medium | 7500 | 0.650 | 0.171 |
| ga2m | D | high | 7500 | 0.638 | 0.171 |
| ga2m | D | low | 7497 | 0.657 | 0.171 |
| ga2m | D | medium | 7500 | 0.673 | 0.171 |
| lr | D | high | 7500 | 0.557 | 0.171 |
| lr | D | low | 7497 | 0.524 | 0.171 |
| lr | D | medium | 7500 | 0.563 | 0.171 |
| xgb | A | high | 7500 | 0.701 | 0.171 |
| xgb | A | low | 7497 | 0.671 | 0.171 |
| xgb | A | medium | 7500 | 0.706 | 0.171 |
| xgb | B | high | 7500 | 0.701 | 0.171 |
| xgb | B | low | 7497 | 0.671 | 0.171 |
| xgb | B | medium | 7500 | 0.705 | 0.171 |
| xgb | C | high | 7500 | 0.699 | 0.171 |
| xgb | C | low | 7497 | 0.670 | 0.171 |
| xgb | C | medium | 7500 | 0.703 | 0.171 |
| xgb | E | high | 7500 | 0.694 | 0.171 |
| xgb | E | low | 7497 | 0.664 | 0.171 |
| xgb | E | medium | 7500 | 0.699 | 0.171 |

Top-1 by scheme (medium and other intensities) and spill-over shares are in `correctness.json`.

## E5 Detection stress test (use (a))

| family | schemes_key | intensity | n | median_percentile | in_top_0.001 | in_top_0.005 | in_top_0.01 | in_top_0.02 |
|---|---|---|---|---|---|---|---|---|
| ebm | atypical_mix | high | 1500 | 0.641 | 0.005 | 0.010 | 0.021 | 0.041 |
| ebm | atypical_mix | low | 1500 | 0.539 | 0.001 | 0.005 | 0.011 | 0.022 |
| ebm | atypical_mix | medium | 1500 | 0.620 | 0.001 | 0.005 | 0.013 | 0.034 |
| ebm | charge_inflation | high | 1500 | 0.462 | 0.001 | 0.003 | 0.007 | 0.017 |
| ebm | charge_inflation | low | 1500 | 0.434 | 0.001 | 0.003 | 0.005 | 0.016 |
| ebm | charge_inflation | medium | 1500 | 0.449 | 0.000 | 0.001 | 0.004 | 0.011 |
| ebm | dme_mill | high | 1500 | 0.966 | 0.059 | 0.174 | 0.271 | 0.385 |
| ebm | dme_mill | low | 1500 | 0.941 | 0.024 | 0.109 | 0.187 | 0.287 |
| ebm | dme_mill | medium | 1500 | 0.956 | 0.043 | 0.151 | 0.232 | 0.351 |
| ebm | overutilization | high | 1500 | 0.804 | 0.003 | 0.016 | 0.035 | 0.073 |
| ebm | overutilization | low | 1500 | 0.660 | 0.003 | 0.008 | 0.019 | 0.039 |
| ebm | overutilization | medium | 1500 | 0.781 | 0.003 | 0.013 | 0.025 | 0.051 |
| ebm | overutilization+atypical_mix | medium | 1500 | 0.809 | 0.003 | 0.017 | 0.037 | 0.075 |
| ebm | overutilization+dme_mill | medium | 1500 | 0.990 | 0.139 | 0.369 | 0.499 | 0.634 |
| ebm | upcoding | high | 1500 | 0.620 | 0.000 | 0.010 | 0.024 | 0.044 |
| ebm | upcoding | low | 1497 | 0.486 | 0.001 | 0.003 | 0.011 | 0.021 |
| ebm | upcoding | medium | 1500 | 0.562 | 0.001 | 0.004 | 0.010 | 0.022 |
| ebm | upcoding+charge_inflation | medium | 1497 | 0.467 | 0.000 | 0.003 | 0.006 | 0.015 |
| ebm | upcoding+overutilization | medium | 1500 | 0.782 | 0.003 | 0.017 | 0.040 | 0.073 |
| ga2m | atypical_mix | high | 1500 | 0.659 | 0.006 | 0.013 | 0.022 | 0.044 |
| ga2m | atypical_mix | low | 1500 | 0.537 | 0.001 | 0.005 | 0.011 | 0.023 |
| ga2m | atypical_mix | medium | 1500 | 0.630 | 0.002 | 0.007 | 0.017 | 0.033 |
| ga2m | charge_inflation | high | 1500 | 0.438 | 0.001 | 0.004 | 0.008 | 0.016 |
| ga2m | charge_inflation | low | 1500 | 0.439 | 0.001 | 0.003 | 0.005 | 0.016 |
| ga2m | charge_inflation | medium | 1500 | 0.441 | 0.001 | 0.001 | 0.003 | 0.014 |
| ga2m | dme_mill | high | 1500 | 0.977 | 0.114 | 0.237 | 0.345 | 0.476 |
| ga2m | dme_mill | low | 1500 | 0.939 | 0.033 | 0.121 | 0.187 | 0.287 |
| ga2m | dme_mill | medium | 1500 | 0.962 | 0.073 | 0.191 | 0.282 | 0.404 |
| ga2m | overutilization | high | 1500 | 0.823 | 0.003 | 0.015 | 0.028 | 0.075 |
| ga2m | overutilization | low | 1500 | 0.678 | 0.003 | 0.007 | 0.018 | 0.042 |
| ga2m | overutilization | medium | 1500 | 0.803 | 0.003 | 0.015 | 0.029 | 0.054 |
| ga2m | overutilization+atypical_mix | medium | 1500 | 0.835 | 0.004 | 0.019 | 0.045 | 0.078 |
| ga2m | overutilization+dme_mill | medium | 1500 | 0.990 | 0.170 | 0.391 | 0.503 | 0.649 |
| ga2m | upcoding | high | 1500 | 0.621 | 0.001 | 0.012 | 0.021 | 0.039 |
| ga2m | upcoding | low | 1497 | 0.488 | 0.001 | 0.005 | 0.009 | 0.024 |
| ga2m | upcoding | medium | 1500 | 0.566 | 0.001 | 0.003 | 0.007 | 0.021 |
| ga2m | upcoding+charge_inflation | medium | 1497 | 0.451 | 0.000 | 0.001 | 0.005 | 0.011 |
| ga2m | upcoding+overutilization | medium | 1500 | 0.787 | 0.003 | 0.016 | 0.034 | 0.071 |
| lgbm | atypical_mix | high | 1500 | 0.689 | 0.003 | 0.014 | 0.027 | 0.046 |
| lgbm | atypical_mix | low | 1500 | 0.535 | 0.002 | 0.009 | 0.019 | 0.032 |
| lgbm | atypical_mix | medium | 1500 | 0.650 | 0.002 | 0.008 | 0.013 | 0.027 |
| lgbm | charge_inflation | high | 1500 | 0.474 | 0.001 | 0.005 | 0.009 | 0.017 |
| lgbm | charge_inflation | low | 1500 | 0.456 | 0.001 | 0.003 | 0.005 | 0.013 |
| lgbm | charge_inflation | medium | 1500 | 0.477 | 0.001 | 0.001 | 0.006 | 0.017 |
| lgbm | dme_mill | high | 1500 | 0.989 | 0.151 | 0.369 | 0.479 | 0.617 |
| lgbm | dme_mill | low | 1500 | 0.960 | 0.025 | 0.134 | 0.249 | 0.359 |
| lgbm | dme_mill | medium | 1500 | 0.983 | 0.111 | 0.261 | 0.378 | 0.529 |
| lgbm | overutilization | high | 1500 | 0.884 | 0.001 | 0.019 | 0.043 | 0.103 |
| lgbm | overutilization | low | 1500 | 0.712 | 0.002 | 0.011 | 0.021 | 0.050 |
| lgbm | overutilization | medium | 1500 | 0.854 | 0.003 | 0.011 | 0.029 | 0.074 |
| lgbm | overutilization+atypical_mix | medium | 1500 | 0.868 | 0.005 | 0.021 | 0.038 | 0.087 |
| lgbm | overutilization+dme_mill | medium | 1500 | 0.992 | 0.162 | 0.393 | 0.539 | 0.699 |
| lgbm | upcoding | high | 1500 | 0.566 | 0.000 | 0.013 | 0.028 | 0.041 |
| lgbm | upcoding | low | 1497 | 0.465 | 0.001 | 0.003 | 0.007 | 0.014 |
| lgbm | upcoding | medium | 1500 | 0.506 | 0.001 | 0.005 | 0.012 | 0.025 |
| lgbm | upcoding+charge_inflation | medium | 1497 | 0.464 | 0.001 | 0.002 | 0.006 | 0.015 |
| lgbm | upcoding+overutilization | medium | 1500 | 0.830 | 0.002 | 0.019 | 0.031 | 0.063 |
| lgbm_mono | atypical_mix | high | 1500 | 0.673 | 0.001 | 0.012 | 0.022 | 0.049 |
| lgbm_mono | atypical_mix | low | 1500 | 0.574 | 0.001 | 0.007 | 0.014 | 0.029 |
| lgbm_mono | atypical_mix | medium | 1500 | 0.645 | 0.003 | 0.009 | 0.017 | 0.031 |
| lgbm_mono | charge_inflation | high | 1500 | 0.535 | 0.000 | 0.005 | 0.009 | 0.019 |
| lgbm_mono | charge_inflation | low | 1500 | 0.498 | 0.000 | 0.004 | 0.009 | 0.022 |
| lgbm_mono | charge_inflation | medium | 1500 | 0.532 | 0.001 | 0.005 | 0.009 | 0.025 |
| lgbm_mono | dme_mill | high | 1500 | 0.991 | 0.196 | 0.409 | 0.516 | 0.609 |
| lgbm_mono | dme_mill | low | 1500 | 0.945 | 0.067 | 0.178 | 0.243 | 0.325 |
| lgbm_mono | dme_mill | medium | 1500 | 0.982 | 0.140 | 0.323 | 0.417 | 0.510 |
| lgbm_mono | overutilization | high | 1500 | 0.898 | 0.007 | 0.041 | 0.059 | 0.117 |
| lgbm_mono | overutilization | low | 1500 | 0.747 | 0.003 | 0.014 | 0.022 | 0.051 |
| lgbm_mono | overutilization | medium | 1500 | 0.882 | 0.003 | 0.017 | 0.049 | 0.094 |
| lgbm_mono | overutilization+atypical_mix | medium | 1500 | 0.890 | 0.008 | 0.021 | 0.047 | 0.107 |
| lgbm_mono | overutilization+dme_mill | medium | 1500 | 0.996 | 0.281 | 0.524 | 0.628 | 0.732 |
| lgbm_mono | upcoding | high | 1500 | 0.666 | 0.003 | 0.016 | 0.034 | 0.055 |
| lgbm_mono | upcoding | low | 1497 | 0.524 | 0.000 | 0.005 | 0.013 | 0.026 |
| lgbm_mono | upcoding | medium | 1500 | 0.587 | 0.001 | 0.005 | 0.019 | 0.034 |
| lgbm_mono | upcoding+charge_inflation | medium | 1497 | 0.583 | 0.001 | 0.009 | 0.018 | 0.031 |
| lgbm_mono | upcoding+overutilization | medium | 1500 | 0.879 | 0.007 | 0.029 | 0.066 | 0.113 |
| lr | atypical_mix | high | 1500 | 0.669 | 0.003 | 0.016 | 0.027 | 0.049 |
| lr | atypical_mix | low | 1500 | 0.521 | 0.001 | 0.008 | 0.016 | 0.027 |
| lr | atypical_mix | medium | 1500 | 0.631 | 0.001 | 0.009 | 0.018 | 0.040 |
| lr | charge_inflation | high | 1500 | 0.433 | 0.001 | 0.002 | 0.007 | 0.016 |
| lr | charge_inflation | low | 1500 | 0.459 | 0.001 | 0.005 | 0.009 | 0.018 |
| lr | charge_inflation | medium | 1500 | 0.451 | 0.001 | 0.003 | 0.004 | 0.013 |
| lr | dme_mill | high | 1500 | 0.991 | 0.120 | 0.373 | 0.520 | 0.647 |
| lr | dme_mill | low | 1500 | 0.974 | 0.039 | 0.190 | 0.307 | 0.443 |
| lr | dme_mill | medium | 1500 | 0.985 | 0.089 | 0.297 | 0.427 | 0.567 |
| lr | overutilization | high | 1500 | 0.847 | 0.003 | 0.018 | 0.038 | 0.085 |
| lr | overutilization | low | 1500 | 0.629 | 0.002 | 0.009 | 0.021 | 0.036 |
| lr | overutilization | medium | 1500 | 0.790 | 0.003 | 0.010 | 0.024 | 0.059 |
| lr | overutilization+atypical_mix | medium | 1500 | 0.826 | 0.001 | 0.013 | 0.032 | 0.084 |
| lr | overutilization+dme_mill | medium | 1500 | 0.996 | 0.241 | 0.563 | 0.697 | 0.814 |
| lr | upcoding | high | 1500 | 0.675 | 0.001 | 0.014 | 0.030 | 0.051 |
| lr | upcoding | low | 1497 | 0.576 | 0.001 | 0.006 | 0.011 | 0.029 |
| lr | upcoding | medium | 1500 | 0.629 | 0.001 | 0.004 | 0.011 | 0.025 |
| lr | upcoding+charge_inflation | medium | 1497 | 0.555 | 0.000 | 0.005 | 0.012 | 0.021 |
| lr | upcoding+overutilization | medium | 1500 | 0.839 | 0.004 | 0.022 | 0.048 | 0.104 |
| xgb | atypical_mix | high | 1500 | 0.655 | 0.004 | 0.009 | 0.021 | 0.035 |
| xgb | atypical_mix | low | 1500 | 0.545 | 0.001 | 0.007 | 0.017 | 0.027 |
| xgb | atypical_mix | medium | 1500 | 0.628 | 0.001 | 0.005 | 0.011 | 0.028 |
| xgb | charge_inflation | high | 1500 | 0.469 | 0.000 | 0.005 | 0.009 | 0.019 |
| xgb | charge_inflation | low | 1500 | 0.449 | 0.001 | 0.003 | 0.005 | 0.010 |
| xgb | charge_inflation | medium | 1500 | 0.465 | 0.001 | 0.003 | 0.007 | 0.014 |
| xgb | dme_mill | high | 1500 | 0.956 | 0.070 | 0.189 | 0.269 | 0.361 |
| xgb | dme_mill | low | 1500 | 0.895 | 0.028 | 0.087 | 0.136 | 0.213 |
| xgb | dme_mill | medium | 1500 | 0.938 | 0.054 | 0.150 | 0.212 | 0.292 |
| xgb | overutilization | high | 1500 | 0.825 | 0.004 | 0.022 | 0.044 | 0.079 |
| xgb | overutilization | low | 1500 | 0.690 | 0.001 | 0.007 | 0.016 | 0.037 |
| xgb | overutilization | medium | 1500 | 0.803 | 0.001 | 0.013 | 0.027 | 0.056 |
| xgb | overutilization+atypical_mix | medium | 1500 | 0.812 | 0.005 | 0.023 | 0.040 | 0.073 |
| xgb | overutilization+dme_mill | medium | 1500 | 0.975 | 0.094 | 0.241 | 0.335 | 0.451 |
| xgb | upcoding | high | 1500 | 0.541 | 0.000 | 0.012 | 0.021 | 0.033 |
| xgb | upcoding | low | 1497 | 0.442 | 0.000 | 0.003 | 0.007 | 0.015 |
| xgb | upcoding | medium | 1500 | 0.477 | 0.000 | 0.004 | 0.009 | 0.029 |
| xgb | upcoding+charge_inflation | medium | 1497 | 0.408 | 0.000 | 0.003 | 0.005 | 0.013 |
| xgb | upcoding+overutilization | medium | 1500 | 0.754 | 0.002 | 0.011 | 0.022 | 0.055 |
| xgb_mono | atypical_mix | high | 1500 | 0.667 | 0.004 | 0.015 | 0.024 | 0.049 |
| xgb_mono | atypical_mix | low | 1500 | 0.566 | 0.002 | 0.011 | 0.019 | 0.028 |
| xgb_mono | atypical_mix | medium | 1500 | 0.637 | 0.001 | 0.009 | 0.013 | 0.031 |
| xgb_mono | charge_inflation | high | 1500 | 0.537 | 0.001 | 0.007 | 0.017 | 0.026 |
| xgb_mono | charge_inflation | low | 1500 | 0.500 | 0.000 | 0.003 | 0.010 | 0.017 |
| xgb_mono | charge_inflation | medium | 1500 | 0.539 | 0.001 | 0.007 | 0.012 | 0.021 |
| xgb_mono | dme_mill | high | 1500 | 0.980 | 0.090 | 0.240 | 0.361 | 0.503 |
| xgb_mono | dme_mill | low | 1500 | 0.952 | 0.031 | 0.116 | 0.194 | 0.313 |
| xgb_mono | dme_mill | medium | 1500 | 0.972 | 0.062 | 0.211 | 0.310 | 0.431 |
| xgb_mono | overutilization | high | 1500 | 0.879 | 0.003 | 0.029 | 0.063 | 0.125 |
| xgb_mono | overutilization | low | 1500 | 0.703 | 0.003 | 0.015 | 0.027 | 0.051 |
| xgb_mono | overutilization | medium | 1500 | 0.841 | 0.005 | 0.020 | 0.048 | 0.091 |
| xgb_mono | overutilization+atypical_mix | medium | 1500 | 0.865 | 0.007 | 0.035 | 0.067 | 0.135 |
| xgb_mono | overutilization+dme_mill | medium | 1500 | 0.993 | 0.179 | 0.437 | 0.582 | 0.710 |
| xgb_mono | upcoding | high | 1500 | 0.671 | 0.005 | 0.020 | 0.033 | 0.054 |
| xgb_mono | upcoding | low | 1497 | 0.510 | 0.001 | 0.006 | 0.012 | 0.028 |
| xgb_mono | upcoding | medium | 1500 | 0.586 | 0.001 | 0.009 | 0.018 | 0.035 |
| xgb_mono | upcoding+charge_inflation | medium | 1497 | 0.604 | 0.002 | 0.007 | 0.015 | 0.040 |
| xgb_mono | upcoding+overutilization | medium | 1500 | 0.843 | 0.007 | 0.032 | 0.061 | 0.115 |

## T3 Consistency (E3; best GBM vs EBM / GA²M, both flag at k = 1%)

| GBM method | vs | n | Sign agree | Spearman ρ | Top-3 Jaccard | Share cos | Permutation baseline (cos) | Magnitude ratio |
|---|---|---|---|---|---|---|---|---|
| xgb · A | ebm | 9018 | 1.000 | 0.857 | 0.500 | 0.951 | 0.657 | 1.167 |
| xgb · B | ebm | 9018 | 1.000 | 0.857 | 0.500 | 0.950 | 0.656 | 1.165 |
| xgb · C | ebm | 9018 | 1.000 | 0.857 | 1.000 | 0.956 | 0.668 | 1.221 |
| xgb · E | ebm | 9018 | 0.857 | 0.786 | 0.500 | 0.902 | 0.599 | 1.169 |
| xgb · A | ga2m | 8867 | 1.000 | 0.857 | 0.500 | 0.943 | 0.667 | 1.121 |
| xgb · B | ga2m | 8867 | 1.000 | 0.857 | 0.500 | 0.942 | 0.666 | 1.120 |
| xgb · C | ga2m | 8867 | 1.000 | 0.857 | 0.500 | 0.948 | 0.679 | 1.163 |
| xgb · E | ga2m | 8867 | 0.857 | 0.750 | 0.500 | 0.895 | 0.609 | 1.136 |

All strata (both / GBM only / glass box only / neither) and LR are in `consistency.json`.

## T4 Faithfulness (E6) and stability (E7)

Deletion score = area under the normalised group-deletion curve (lower is more faithful), providers above their baseline, peer baseline. Medians: the normalised area is unbounded when a provider's group contributions have mixed signs and f(x) is close to its baseline, so means are dominated by those providers (means in `faithfulness.json`; DESIGN §17 D104).

| model · method | n | Deletion ↓ (median) | Random order (median) | Reverse order (median) | Share below random | Top-1 agree (seeds) | Kendall τ (seeds) | Top-1 agree (bootstrap) | Kendall τ (bootstrap) | Top-1 agree (perturbation) |
|---|---|---|---|---|---|---|---|---|---|---|
| lr · B | 41501 | 0.006 | 0.505 | 0.994 | 1.000 | 0.991 | 0.992 | 0.743 | 0.718 | 0.970 |
| lr · C | 41501 | 0.006 | 0.505 | 0.994 | 1.000 | 0.991 | 0.992 | 0.743 | 0.718 | 0.970 |
| lr · D | 41501 | 0.006 | 0.505 | 0.994 | 1.000 | 0.991 | 0.992 | 0.733 | 0.702 | 0.970 |
| lr · E | 41501 | 0.006 | 0.505 | 0.994 | 1.000 | 0.991 | 0.992 | 0.743 | 0.718 | 0.970 |
| ebm · B | 44671 | 0.106 | 0.500 | 0.894 | 1.000 | 0.908 | 0.916 | 0.727 | 0.666 | 0.970 |
| ebm · C | 44671 | 0.106 | 0.500 | 0.894 | 1.000 | 0.908 | 0.916 | 0.727 | 0.666 | 0.970 |
| ebm · D | 44671 | 0.106 | 0.500 | 0.894 | 1.000 | 0.907 | 0.894 | 0.726 | 0.652 | 0.970 |
| ebm · E | 44671 | 0.106 | 0.500 | 0.894 | 1.000 | 0.908 | 0.916 | 0.727 | 0.666 | 0.970 |
| ga2m · B | 44549 | 0.111 | 0.506 | 0.899 | 1.000 | 0.886 | 0.890 | 0.685 | 0.616 | 0.976 |
| ga2m · C | 44549 | 0.111 | 0.506 | 0.894 | 1.000 | 0.889 | 0.897 | 0.684 | 0.624 | 0.975 |
| ga2m · D | 44549 | 0.113 | 0.506 | 0.897 | 1.000 | 0.897 | 0.880 | 0.697 | 0.609 | 0.974 |
| ga2m · E | 44549 | 0.115 | 0.506 | 0.898 | 1.000 | 0.868 | 0.868 | 0.659 | 0.584 | 0.974 |
| xgb · A | 45902 | 0.155 | 0.495 | 0.830 | 1.000 | 0.778 | 0.790 | 0.608 | 0.550 | 0.963 |
| xgb · B | 45902 | 0.155 | 0.495 | 0.831 | 1.000 | 0.782 | 0.789 | 0.608 | 0.550 | 0.964 |
| xgb · C | 45902 | 0.160 | 0.495 | 0.803 | 0.999 | 0.811 | 0.825 | 0.629 | 0.591 | 0.969 |
| xgb · E | 45902 | 0.174 | 0.495 | 0.827 | 0.999 | 0.702 | 0.687 | 0.545 | 0.452 | 0.961 |
| xgb_mono · A | 45010 | 0.085 | 0.492 | 0.893 | 1.000 | 0.891 | 0.893 | 0.722 | 0.678 | 0.957 |
| xgb_mono · B | 45010 | 0.085 | 0.492 | 0.893 | 1.000 | 0.890 | 0.892 | 0.720 | 0.678 | 0.957 |
| xgb_mono · C | 45010 | 0.088 | 0.492 | 0.879 | 1.000 | 0.907 | 0.906 | 0.724 | 0.687 | 0.961 |
| xgb_mono · E | 45010 | 0.100 | 0.492 | 0.892 | 1.000 | 0.852 | 0.844 | 0.691 | 0.624 | 0.952 |
| lgbm · A | 44631 | 0.137 | 0.493 | 0.875 | 1.000 | 0.790 | 0.805 | 0.616 | 0.569 | 0.961 |
| lgbm · B | 44634 | 0.137 | 0.493 | 0.876 | 1.000 | 0.797 | 0.806 | 0.623 | 0.570 | 0.965 |
| lgbm · C | 44634 | 0.141 | 0.493 | 0.843 | 0.999 | 0.805 | 0.839 | 0.634 | 0.599 | 0.969 |
| lgbm · E | 44634 | 0.155 | 0.493 | 0.869 | 0.998 | 0.723 | 0.694 | 0.554 | 0.462 | 0.954 |
| lgbm_mono · A | 42160 | 0.085 | 0.495 | 0.954 | 1.000 | 0.794 | 0.801 | 0.678 | 0.639 | 0.962 |
| lgbm_mono · B | 42161 | 0.085 | 0.495 | 0.955 | 1.000 | 0.796 | 0.801 | 0.682 | 0.638 | 0.961 |
| lgbm_mono · C | 42161 | 0.087 | 0.495 | 0.926 | 0.999 | 0.809 | 0.826 | 0.692 | 0.657 | 0.962 |
| lgbm_mono · E | 42161 | 0.098 | 0.495 | 0.952 | 1.000 | 0.742 | 0.718 | 0.645 | 0.569 | 0.959 |

Stability subsample: 2000 provider-years; seeds per family: {'lr': 10, 'ebm': 10, 'ga2m': 10, 'xgb': 10, 'xgb_mono': 10, 'lgbm': 10, 'lgbm_mono': 10}.

## E8 Peer vs global baseline (RQ5)

| family | method | n | context_share_peer | context_share_global | context_top_peer | context_top_global | top1_agreement | spearman_median | cosine_median |
|---|---|---|---|---|---|---|---|---|---|
| xgb | A | 51276 | 0.110 | 0.163 | 0.092 | 0.194 | 0.838 | 0.929 | 0.982 |
| xgb | B | 51276 | 0.110 | 0.164 | 0.092 | 0.196 | 0.837 | 0.929 | 0.982 |
| xgb | E | 51276 | 0.112 | 0.163 | 0.092 | 0.186 | 0.862 | 0.964 | 0.987 |

## E9 Sensitivity suite

| variant | what | PR-AUC (variant / main) | top-1 agree with main (B) | cos vs glass box (B) | deletion median (B) | top-1 injected (B) |
|---|---|---|---|---|---|---|
| S10 | peer cells without region | 0.001 / 0.002 | 0.635 | 0.896 | 0.136 | – |
| S11 | prior-work-compatible label tier | 0.002 / 0.002 | 0.628 | 0.890 | 0.129 | – |
| S12 | negative subsampling 30 % | 0.001 / 0.002 | 0.778 | 0.815 | 0.147 | – |
| S13 | primary tier + unsplit 1128(b)(1) (G1 decision, D117) | 0.002 / 0.002 | 0.773 | 0.906 | 0.146 | – |
| S1_W1 | label window W = 1 (not estimable: primary fold 0: tune_valid has 0 positives (< models.yaml min_valid_positives = 5)) | – / – | – | – | – | – |
| S1_W2 | label window W = 2 | 0.000 / 0.002 | 0.632 | 0.889 | 0.124 | – |
| S1_W5 | label window W = 5 | 0.002 / 0.002 | 0.632 | 0.908 | 0.140 | – |
| S2 | broad label tier | 0.001 / 0.002 | 0.663 | 0.905 | 0.129 | – |
| S3 | individuals only | 0.001 / 0.002 | 0.749 | 0.894 | 0.152 | – |
| S4 | exclude 2020 | 0.001 / 0.002 | 0.711 | 0.899 | 0.142 | – |
| S5 | global vs peer baseline | – / – | – | – | – | – |
| S6 | grouping: correlation clusters (hierarchical on \|rho\|, cut at 7) | 0.002 / 0.002 | 0.203 | 0.984 | 0.065 | 0.576 |
| S7 | grouping: CH2 -> overutilization, PM1 -> context | 0.002 / 0.002 | 0.773 | 0.910 | 0.146 | 0.693 |
| S8 | context ablation | 0.002 / 0.002 | 0.697 | 0.908 | 0.134 | 0.698 |
| S9 | monotone constraints on vs off | 0.002 / – | – | 0.912 | 0.155 | 0.705 |

S5 and S9 are read from the main evaluation (`sensitivity/S5.json`, `S9.json`).

## E10 Fairness profile (descriptive)

Flag rate, precision and flag rate ÷ positive rate at each k, pooled over the test years. LEIE reflects enforcement patterns, which limits interpretation (DESIGN §10.7). Largest groups shown; all in `fairness.json`.

`xgb`, census_division, k = 0.5%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| South Atlantic | 422468 | 25 | 1777 | 0.0042 | 0.001 | 71.08 |
| East North Central | 324719 | 30 | 2096 | 0.0065 | 0.001 | 69.87 |
| Middle Atlantic | 312139 | 27 | 1598 | 0.0051 | 0.001 | 59.19 |
| Pacific | 245587 | 26 | 1608 | 0.0065 | 0.004 | 61.85 |
| West South Central | 206029 | 31 | 1468 | 0.0071 | 0.003 | 47.35 |
| West North Central | 157197 | 20 | 454 | 0.0029 | 0.000 | 22.70 |
| Mountain | 143411 | 11 | 377 | 0.0026 | 0.005 | 34.27 |
| New England | 130066 | 6 | 208 | 0.0016 | 0.000 | 34.67 |
| East South Central | 125786 | 13 | 665 | 0.0053 | 0.003 | 51.15 |
| other | 6700 | 5 | 121 | 0.0181 | 0.000 | 24.20 |

`xgb`, census_division, k = 1.0%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| South Atlantic | 422468 | 25 | 3728 | 0.0088 | 0.001 | 149.12 |
| East North Central | 324719 | 30 | 3957 | 0.0122 | 0.001 | 131.90 |
| Middle Atlantic | 312139 | 27 | 3234 | 0.0104 | 0.001 | 119.78 |
| Pacific | 245587 | 26 | 3031 | 0.0123 | 0.002 | 116.58 |
| West South Central | 206029 | 31 | 2901 | 0.0141 | 0.001 | 93.58 |
| West North Central | 157197 | 20 | 969 | 0.0062 | 0.000 | 48.45 |
| Mountain | 143411 | 11 | 835 | 0.0058 | 0.004 | 75.91 |
| New England | 130066 | 6 | 507 | 0.0039 | 0.000 | 84.50 |
| East South Central | 125786 | 13 | 1370 | 0.0109 | 0.001 | 105.38 |
| other | 6700 | 5 | 210 | 0.0313 | 0.000 | 42.00 |

`xgb`, ent_cd, k = 0.5%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| I | 1955047 | 185 | 9877 | 0.0051 | 0.002 | 53.39 |
| O | 119055 | 9 | 495 | 0.0042 | 0.000 | 55.00 |

`xgb`, ent_cd, k = 1.0%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| I | 1955047 | 185 | 19762 | 0.0101 | 0.001 | 106.82 |
| O | 119055 | 9 | 980 | 0.0082 | 0.000 | 108.89 |

`xgb`, prvdr_type, k = 0.5%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| Nurse Practitioner | 283452 | 9 | 615 | 0.0022 | 0.000 | 68.33 |
| Internal Medicine | 166505 | 35 | 2196 | 0.0132 | 0.001 | 62.74 |
| Physician Assistant | 151258 | 2 | 273 | 0.0018 | 0.000 | 136.50 |
| Family Practice | 144587 | 38 | 1767 | 0.0122 | 0.003 | 46.50 |
| Physical Therapist in Private Practice | 141727 | 7 | 138 | 0.0010 | 0.000 | 19.71 |
| Emergency Medicine | 86685 | 6 | 259 | 0.0030 | 0.008 | 43.17 |
| Chiropractic | 63646 | 11 | 351 | 0.0055 | 0.003 | 31.91 |
| Diagnostic Radiology | 60972 | 5 | 130 | 0.0021 | 0.015 | 26.00 |
| Mass Immunizer Roster Biller | 54483 | 0 | 88 | 0.0016 | 0.000 | – |
| Anesthesiology | 51966 | 4 | 276 | 0.0053 | 0.007 | 69.00 |
| Optometry | 50512 | 0 | 57 | 0.0011 | 0.000 | – |
| Certified Registered Nurse Anesthetist (CRNA) | 41762 | 2 | 23 | 0.0006 | 0.000 | 11.50 |

`xgb`, prvdr_type, k = 1.0%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| Nurse Practitioner | 283452 | 9 | 1503 | 0.0053 | 0.001 | 167.00 |
| Internal Medicine | 166505 | 35 | 4081 | 0.0245 | 0.001 | 116.60 |
| Physician Assistant | 151258 | 2 | 668 | 0.0044 | 0.000 | 334.00 |
| Family Practice | 144587 | 38 | 3549 | 0.0245 | 0.002 | 93.39 |
| Physical Therapist in Private Practice | 141727 | 7 | 349 | 0.0025 | 0.000 | 49.86 |
| Emergency Medicine | 86685 | 6 | 450 | 0.0052 | 0.004 | 75.00 |
| Chiropractic | 63646 | 11 | 702 | 0.0110 | 0.001 | 63.82 |
| Diagnostic Radiology | 60972 | 5 | 243 | 0.0040 | 0.008 | 48.60 |
| Mass Immunizer Roster Biller | 54483 | 0 | 193 | 0.0035 | 0.000 | – |
| Anesthesiology | 51966 | 4 | 472 | 0.0091 | 0.004 | 118.00 |
| Optometry | 50512 | 0 | 133 | 0.0026 | 0.000 | – |
| Certified Registered Nurse Anesthetist (CRNA) | 41762 | 2 | 56 | 0.0013 | 0.000 | 28.00 |

`xgb`, ruca_class, k = 0.5%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| urban | 1837120 | 173 | 9233 | 0.0050 | 0.002 | 53.37 |
| large_rural | 154151 | 14 | 753 | 0.0049 | 0.003 | 53.79 |
| small_rural | 56398 | 4 | 282 | 0.0050 | 0.000 | 70.50 |
| isolated | 22144 | 3 | 92 | 0.0042 | 0.022 | 30.67 |
| unknown | 4289 | 0 | 12 | 0.0028 | 0.000 | – |

`xgb`, ruca_class, k = 1.0%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| urban | 1837120 | 173 | 18398 | 0.0100 | 0.001 | 106.35 |
| large_rural | 154151 | 14 | 1530 | 0.0099 | 0.001 | 109.29 |
| small_rural | 56398 | 4 | 577 | 0.0102 | 0.000 | 144.25 |
| isolated | 22144 | 3 | 220 | 0.0099 | 0.009 | 73.33 |
| unknown | 4289 | 0 | 17 | 0.0040 | 0.000 | – |

`ebm`, census_division, k = 0.5%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| South Atlantic | 422468 | 25 | 2053 | 0.0049 | 0.001 | 82.12 |
| East North Central | 324719 | 30 | 1854 | 0.0057 | 0.002 | 61.80 |
| Middle Atlantic | 312139 | 27 | 1353 | 0.0043 | 0.001 | 50.11 |
| Pacific | 245587 | 26 | 1567 | 0.0064 | 0.003 | 60.27 |
| West South Central | 206029 | 31 | 1786 | 0.0087 | 0.001 | 57.61 |
| West North Central | 157197 | 20 | 390 | 0.0025 | 0.000 | 19.50 |
| Mountain | 143411 | 11 | 313 | 0.0022 | 0.006 | 28.45 |
| New England | 130066 | 6 | 57 | 0.0004 | 0.000 | 9.50 |
| East South Central | 125786 | 13 | 803 | 0.0064 | 0.002 | 61.77 |
| other | 6700 | 5 | 196 | 0.0293 | 0.000 | 39.20 |

`ebm`, census_division, k = 1.0%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| South Atlantic | 422468 | 25 | 4105 | 0.0097 | 0.001 | 164.20 |
| East North Central | 324719 | 30 | 3584 | 0.0110 | 0.002 | 119.47 |
| Middle Atlantic | 312139 | 27 | 2884 | 0.0092 | 0.001 | 106.81 |
| Pacific | 245587 | 26 | 3087 | 0.0126 | 0.002 | 118.73 |
| West South Central | 206029 | 31 | 3408 | 0.0165 | 0.000 | 109.94 |
| West North Central | 157197 | 20 | 833 | 0.0053 | 0.000 | 41.65 |
| Mountain | 143411 | 11 | 651 | 0.0045 | 0.003 | 59.18 |
| New England | 130066 | 6 | 175 | 0.0013 | 0.000 | 29.17 |
| East South Central | 125786 | 13 | 1604 | 0.0128 | 0.002 | 123.38 |
| other | 6700 | 5 | 411 | 0.0613 | 0.000 | 82.20 |

`ebm`, ent_cd, k = 0.5%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| I | 1955047 | 185 | 9865 | 0.0050 | 0.002 | 53.32 |
| O | 119055 | 9 | 507 | 0.0043 | 0.002 | 56.33 |

`ebm`, ent_cd, k = 1.0%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| I | 1955047 | 185 | 19680 | 0.0101 | 0.001 | 106.38 |
| O | 119055 | 9 | 1062 | 0.0089 | 0.002 | 118.00 |

`ebm`, prvdr_type, k = 0.5%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| Nurse Practitioner | 283452 | 9 | 407 | 0.0014 | 0.000 | 45.22 |
| Internal Medicine | 166505 | 35 | 2248 | 0.0135 | 0.001 | 64.23 |
| Physician Assistant | 151258 | 2 | 37 | 0.0002 | 0.000 | 18.50 |
| Family Practice | 144587 | 38 | 2204 | 0.0152 | 0.001 | 58.00 |
| Physical Therapist in Private Practice | 141727 | 7 | 6 | 0.0000 | 0.000 | 0.86 |
| Emergency Medicine | 86685 | 6 | 371 | 0.0043 | 0.005 | 61.83 |
| Chiropractic | 63646 | 11 | 11 | 0.0002 | 0.091 | 1.00 |
| Diagnostic Radiology | 60972 | 5 | 78 | 0.0013 | 0.026 | 15.60 |
| Mass Immunizer Roster Biller | 54483 | 0 | 0 | 0.0000 | – | – |
| Anesthesiology | 51966 | 4 | 274 | 0.0053 | 0.004 | 68.50 |
| Optometry | 50512 | 0 | 2 | 0.0000 | 0.000 | – |
| Certified Registered Nurse Anesthetist (CRNA) | 41762 | 2 | 2 | 0.0000 | 0.000 | 1.00 |

`ebm`, prvdr_type, k = 1.0%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| Nurse Practitioner | 283452 | 9 | 904 | 0.0032 | 0.000 | 100.44 |
| Internal Medicine | 166505 | 35 | 4704 | 0.0283 | 0.001 | 134.40 |
| Physician Assistant | 151258 | 2 | 105 | 0.0007 | 0.000 | 52.50 |
| Family Practice | 144587 | 38 | 4380 | 0.0303 | 0.001 | 115.26 |
| Physical Therapist in Private Practice | 141727 | 7 | 42 | 0.0003 | 0.000 | 6.00 |
| Emergency Medicine | 86685 | 6 | 626 | 0.0072 | 0.003 | 104.33 |
| Chiropractic | 63646 | 11 | 58 | 0.0009 | 0.017 | 5.27 |
| Diagnostic Radiology | 60972 | 5 | 162 | 0.0027 | 0.012 | 32.40 |
| Mass Immunizer Roster Biller | 54483 | 0 | 0 | 0.0000 | – | – |
| Anesthesiology | 51966 | 4 | 528 | 0.0102 | 0.002 | 132.00 |
| Optometry | 50512 | 0 | 2 | 0.0000 | 0.000 | – |
| Certified Registered Nurse Anesthetist (CRNA) | 41762 | 2 | 4 | 0.0001 | 0.000 | 2.00 |

`ebm`, ruca_class, k = 0.5%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| urban | 1837120 | 173 | 9144 | 0.0050 | 0.002 | 52.86 |
| large_rural | 154151 | 14 | 835 | 0.0054 | 0.001 | 59.64 |
| small_rural | 56398 | 4 | 243 | 0.0043 | 0.000 | 60.75 |
| isolated | 22144 | 3 | 146 | 0.0066 | 0.014 | 48.67 |
| unknown | 4289 | 0 | 4 | 0.0009 | 0.000 | – |

`ebm`, ruca_class, k = 1.0%:

| value | N | positives | flagged | flag rate | precision | flag/positive rate |
|---|---|---|---|---|---|---|
| urban | 1837120 | 173 | 18274 | 0.0099 | 0.001 | 105.63 |
| large_rural | 154151 | 14 | 1638 | 0.0106 | 0.002 | 117.00 |
| small_rural | 56398 | 4 | 543 | 0.0096 | 0.000 | 135.75 |
| isolated | 22144 | 3 | 282 | 0.0127 | 0.007 | 94.00 |
| unknown | 4289 | 0 | 5 | 0.0012 | 0.000 | – |

## DESIGN §10.8: which row the numbers are consistent with

- H1 (G3): False; explanations of the best GBM (method B) correct on injected (Wilson 95 % lower bound of top-1 > chance): True; consistent with EBM and GA²M (median share cosine > permutation baseline, both-flag stratum): True.
- Consistent with: "No significant difference": recommend the glass box; the scenario framework and method comparison stand.
- The human decides the branch the paper follows (HANDOFF §8 rule 7).
