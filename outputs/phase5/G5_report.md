# Gate G5 report: scheme injection (Phase 5)

Inputs: 28 harmonised CMS files under `data/interim` and 80 registered raw files (`data/raw/manifest.json`).

Generated 2026-10-27T04:08:06.143863+00:00 by `scenario-fraud phase5 report`. HANDOFF §6 G5: injection intensities actually needed; the human accepts the calibrated grid. Use (a) and use (b) results are evaluated in Phase 6.

## Calibration (2021; target: median peer percentile ≥ 0.95)

**upcoding** (primary feature UP2; current low / medium / high = 0.15 / 0.3 / 0.5; proposed medium = 0.85)

| intensity | median_percentile | share_at_or_above_target | median_z | n |
|---|---|---|---|---|
| 0.05 | 0.672 | 0.11 | 0.1603 | 200 |
| 0.1 | 0.6846 | 0.11 | 0.2114 | 200 |
| 0.15 | 0.6956 | 0.11 | 0.2772 | 200 |
| 0.2 | 0.722 | 0.11 | 0.3576 | 200 |
| 0.3 | 0.7541 | 0.12 | 0.5114 | 200 |
| 0.4 | 0.8099 | 0.17 | 0.6699 | 200 |
| 0.5 | 0.8349 | 0.21 | 0.6763 | 200 |
| 0.7 | 0.901 | 0.295 | 0.9304 | 200 |
| 0.85 | 0.9518 | 0.52 | 1.052 | 200 |
| 1 | 0.9822 | 0.81 | 1.06 | 200 |

**overutilization** (primary feature OU1; current low / medium / high = 1.3 / 1.6 / 2.0; proposed medium = 3.0)

| intensity | median_percentile | share_at_or_above_target | median_z | n |
|---|---|---|---|---|
| 1.1 | 0.5769 | 0.11 | 0.2195 | 200 |
| 1.2 | 0.6622 | 0.12 | 0.4918 | 200 |
| 1.3 | 0.7096 | 0.155 | 0.716 | 200 |
| 1.45 | 0.7785 | 0.21 | 1.137 | 200 |
| 1.6 | 0.819 | 0.26 | 1.499 | 200 |
| 1.8 | 0.8613 | 0.3 | 1.964 | 200 |
| 2 | 0.8904 | 0.36 | 2.432 | 200 |
| 2.5 | 0.9332 | 0.45 | 3.587 | 200 |
| 3 | 0.9579 | 0.54 | 4.762 | 200 |
| 3.5 | 0.9721 | 0.625 | 5.906 | 200 |
| 4 | 0.9821 | 0.715 | 7.005 | 200 |

**atypical_mix** (primary feature SM2; current low / medium / high = 0.1 / 0.2 / 0.35; proposed medium = 0.05)

| intensity | median_percentile | share_at_or_above_target | median_z | n |
|---|---|---|---|---|
| 0.01 | 0.8892 | 0.3 | 0.8598 | 200 |
| 0.02 | 0.9246 | 0.435 | 1.588 | 200 |
| 0.03 | 0.9466 | 0.49 | 2.173 | 200 |
| 0.05 | 0.9733 | 0.57 | 3.276 | 200 |
| 0.1 | 0.9875 | 0.71 | 5.317 | 200 |
| 0.15 | 0.9939 | 0.81 | 7.093 | 200 |
| 0.2 | 0.9958 | 0.825 | 8.164 | 200 |
| 0.25 | 0.997 | 0.855 | 8.91 | 200 |
| 0.35 | 1 | 0.865 | 10 | 200 |
| 0.5 | 1 | 0.905 | 10 | 200 |

**charge_inflation** (primary feature CH1; current low / medium / high = 1.3 / 1.7 / 2.2; proposed medium = 2.0)

| intensity | median_percentile | share_at_or_above_target | median_z | n |
|---|---|---|---|---|
| 1.1 | 0.6367 | 0.085 | 0.3494 | 200 |
| 1.2 | 0.7158 | 0.13 | 0.7299 | 200 |
| 1.3 | 0.7792 | 0.17 | 1.022 | 200 |
| 1.5 | 0.8646 | 0.27 | 1.696 | 200 |
| 1.7 | 0.9178 | 0.37 | 2.294 | 200 |
| 2 | 0.9542 | 0.54 | 3.187 | 200 |
| 2.2 | 0.9657 | 0.575 | 3.755 | 200 |
| 2.6 | 0.9794 | 0.695 | 5.006 | 200 |

**dme_mill** (primary feature DI1; current low / medium / high = 2.0 / 4.0 / 8.0; proposed medium = 24.0)

| intensity | median_percentile | share_at_or_above_target | median_z | n |
|---|---|---|---|---|
| 0.5 | 0.08752 | 0.035 | -0.437 | 200 |
| 1 | 0.182 | 0.04 | -0.224 | 200 |
| 2 | 0.7331 | 0.085 | 0.07629 | 200 |
| 3 | 0.8181 | 0.145 | 0.3854 | 200 |
| 4 | 0.8659 | 0.2 | 0.5169 | 200 |
| 6 | 0.9107 | 0.26 | 0.7118 | 200 |
| 8 | 0.9203 | 0.335 | 0.836 | 200 |
| 12 | 0.9384 | 0.41 | 1.125 | 200 |
| 16 | 0.9497 | 0.5 | 1.486 | 200 |
| 24 | 0.9667 | 0.615 | 2.258 | 200 |
| 32 | 0.9764 | 0.705 | 3.044 | 200 |

## Draw 0

Planned injections (test: per scheme × intensity and per pair; train: ≈ 0.5% of training provider-years split over the same cells):

| part | scheme | intensity | planned | cell_target | distractor_share |
|---|---|---|---|---|---|
| test | atypical_mix | high | 300 | 300 | 0.59 |
| test | atypical_mix | low | 300 | 300 | 0.5433 |
| test | atypical_mix | medium | 300 | 300 | 0.5833 |
| test | charge_inflation | high | 300 | 300 | 0.5967 |
| test | charge_inflation | low | 300 | 300 | 0.59 |
| test | charge_inflation | medium | 300 | 300 | 0.6333 |
| test | dme_mill | high | 300 | 300 | 0.1733 |
| test | dme_mill | low | 300 | 300 | 0.2667 |
| test | dme_mill | medium | 300 | 300 | 0.2767 |
| test | overutilization | high | 300 | 300 | 0.6067 |
| test | overutilization | low | 300 | 300 | 0.61 |
| test | overutilization | medium | 300 | 300 | 0.62 |
| test | overutilization+atypical_mix | medium | 300 | 300 | 0.58 |
| test | overutilization+dme_mill | medium | 300 | 300 | 0.21 |
| test | upcoding | high | 300 | 300 | 0.4367 |
| test | upcoding | low | 300 | 300 | 0.3767 |
| test | upcoding | medium | 300 | 300 | 0.41 |
| test | upcoding+charge_inflation | medium | 300 | 300 | 0.42 |
| test | upcoding+overutilization | medium | 300 | 300 | 0.3867 |
| train | atypical_mix | high | 1033 | 1033 | 0.5876 |
| train | atypical_mix | low | 1033 | 1033 | 0.5712 |
| train | atypical_mix | medium | 1033 | 1033 | 0.5712 |
| train | charge_inflation | high | 1033 | 1033 | 0.6379 |
| train | charge_inflation | low | 1033 | 1033 | 0.5924 |
| train | charge_inflation | medium | 1033 | 1033 | 0.5983 |
| train | dme_mill | high | 1033 | 1033 | 0.3011 |
| train | dme_mill | low | 1033 | 1033 | 0.2798 |
| train | dme_mill | medium | 1033 | 1033 | 0.2469 |
| train | overutilization | high | 1033 | 1033 | 0.575 |
| train | overutilization | low | 1033 | 1033 | 0.6041 |
| train | overutilization | medium | 1033 | 1033 | 0.6021 |
| train | overutilization+atypical_mix | medium | 1033 | 1033 | 0.5586 |
| train | overutilization+dme_mill | medium | 1033 | 1033 | 0.2081 |
| train | upcoding | high | 1033 | 1033 | 0.423 |
| train | upcoding | low | 1033 | 1033 | 0.4221 |
| train | upcoding | medium | 1033 | 1033 | 0.3969 |
| train | upcoding+charge_inflation | medium | 1033 | 1033 | 0.3863 |
| train | upcoding+overutilization | medium | 1033 | 1033 | 0.3756 |

Edits and re-suppression per year:

| year | injected | rows_suppressed_ps | rows_suppressed_dme | dropped_no_visible_rows | featurised |
|---|---|---|---|---|---|
| 2018 | 4795 | 714 | 0 | 1 | 959794 |
| 2019 | 4948 | 722 | 0 | 0 | 988049 |
| 2020 | 4837 | 740 | 0 | 1 | 966438 |
| 2021 | 5047 | 785 | 0 | 0 | 1010238 |
| 2022 | 2815 | 408 | 0 | 1 | 1027753 |
| 2023 | 2885 | 421 | 0 | 1 | 1046783 |

Feature-shift profile of injected test providers (mean change of each group's mean direction-up z; the answer groups should move most, spill-over groups somewhat):

| schemes | n | upcoding | overutilization | atypical_mix | charge_inflation | patient_mix | dme_intensity | dme_concentration |
|---|---|---|---|---|---|---|---|---|
| atypical_mix | 900 | 0.001 | 0.457 | 0.847 | 0.079 | 0.38 | -0.001 | -0.001 |
| charge_inflation | 900 | 0.001 | 0 | 0.008 | 0.794 | 0.006 | -0.002 | 0.001 |
| dme_mill | 900 | 0.001 | 0 | 0.005 | 0 | -0.001 | 4.114 | 7.025 |
| overutilization | 900 | 0 | 1.92 | -0.06 | 0.966 | 2.469 | 0.001 | 0.001 |
| overutilization+atypical_mix | 300 | 0.001 | 2.41 | 0.504 | 1.07 | 2.898 | 0 | 0 |
| overutilization+dme_mill | 300 | 0.001 | 1.952 | -0.04 | 0.997 | 2.341 | 4.592 | 8.031 |
| upcoding | 898 | 1.128 | 0.217 | 0.08 | 0.176 | 0.091 | 0.001 | -0 |
| upcoding+charge_inflation | 300 | 1.042 | -0.017 | 0.097 | 0.888 | 0.071 | 0.001 | 0.002 |
| upcoding+overutilization | 300 | 1.287 | 2.017 | -0.053 | 1.287 | 2.514 | -0.005 | -0.001 |

Leakage checks on the injected data (FEATURE_CATALOG checklist L2, L5):

| id | check | status | detail |
|---|---|---|---|
| L5 | edited cms_ps rows re-suppressed (tot_benes > 10) | pass | 0 rows at or below the threshold |
| L5 | edited dme_s rows re-suppressed (tot_suplr_clms > 10) | pass | 0 rows at or below the threshold |
| L5 | only planned providers' rows edited | pass | 0 edited rows of unplanned NPIs |
| L2 | peer statistics and cell sizes count only non-injected providers | pass | 0 violations over 99876 statistic cells |

Injection models (use b): test rows and injected test rows per family:

| family | test_rows | test_injected |
|---|---|---|
| ebm | 2073910 | 5698 |
| ga2m | 2073910 | 5698 |
| lgbm | 2073910 | 5698 |
| lgbm_mono | 2073910 | 5698 |
| lr | 2073910 | 5698 |
| xgb | 2073910 | 5698 |
| xgb_mono | 2073910 | 5698 |

## Draw 1

Planned injections (test: per scheme × intensity and per pair; train: ≈ 0.5% of training provider-years split over the same cells):

| part | scheme | intensity | planned | cell_target | distractor_share |
|---|---|---|---|---|---|
| test | atypical_mix | high | 300 | 300 | 0.62 |
| test | atypical_mix | low | 300 | 300 | 0.6 |
| test | atypical_mix | medium | 300 | 300 | 0.5333 |
| test | charge_inflation | high | 300 | 300 | 0.6167 |
| test | charge_inflation | low | 300 | 300 | 0.62 |
| test | charge_inflation | medium | 300 | 300 | 0.62 |
| test | dme_mill | high | 300 | 300 | 0.2433 |
| test | dme_mill | low | 300 | 300 | 0.2333 |
| test | dme_mill | medium | 300 | 300 | 0.2533 |
| test | overutilization | high | 300 | 300 | 0.59 |
| test | overutilization | low | 300 | 300 | 0.57 |
| test | overutilization | medium | 300 | 300 | 0.57 |
| test | overutilization+atypical_mix | medium | 300 | 300 | 0.5667 |
| test | overutilization+dme_mill | medium | 300 | 300 | 0.25 |
| test | upcoding | high | 300 | 300 | 0.4267 |
| test | upcoding | low | 300 | 300 | 0.41 |
| test | upcoding | medium | 300 | 300 | 0.4333 |
| test | upcoding+charge_inflation | medium | 300 | 300 | 0.3667 |
| test | upcoding+overutilization | medium | 300 | 300 | 0.34 |
| train | atypical_mix | high | 1033 | 1033 | 0.5663 |
| train | atypical_mix | low | 1033 | 1033 | 0.5915 |
| train | atypical_mix | medium | 1033 | 1033 | 0.5421 |
| train | charge_inflation | high | 1033 | 1033 | 0.575 |
| train | charge_inflation | low | 1033 | 1033 | 0.6012 |
| train | charge_inflation | medium | 1033 | 1033 | 0.605 |
| train | dme_mill | high | 1033 | 1033 | 0.272 |
| train | dme_mill | low | 1033 | 1033 | 0.2691 |
| train | dme_mill | medium | 1033 | 1033 | 0.2391 |
| train | overutilization | high | 1033 | 1033 | 0.607 |
| train | overutilization | low | 1033 | 1033 | 0.607 |
| train | overutilization | medium | 1033 | 1033 | 0.5847 |
| train | overutilization+atypical_mix | medium | 1033 | 1033 | 0.5712 |
| train | overutilization+dme_mill | medium | 1033 | 1033 | 0.2439 |
| train | upcoding | high | 1033 | 1033 | 0.3988 |
| train | upcoding | low | 1033 | 1033 | 0.3766 |
| train | upcoding | medium | 1033 | 1033 | 0.4259 |
| train | upcoding+charge_inflation | medium | 1033 | 1033 | 0.4027 |
| train | upcoding+overutilization | medium | 1033 | 1033 | 0.3824 |

Edits and re-suppression per year:

| year | injected | rows_suppressed_ps | rows_suppressed_dme | dropped_no_visible_rows | featurised |
|---|---|---|---|---|---|
| 2018 | 4764 | 674 | 0 | 0 | 959795 |
| 2019 | 4948 | 735 | 0 | 1 | 988048 |
| 2020 | 4843 | 723 | 0 | 0 | 966439 |
| 2021 | 5072 | 741 | 0 | 4 | 1010234 |
| 2022 | 2829 | 403 | 0 | 2 | 1027752 |
| 2023 | 2871 | 406 | 0 | 0 | 1046784 |

Feature-shift profile of injected test providers (mean change of each group's mean direction-up z; the answer groups should move most, spill-over groups somewhat):

| schemes | n | upcoding | overutilization | atypical_mix | charge_inflation | patient_mix | dme_intensity | dme_concentration |
|---|---|---|---|---|---|---|---|---|
| atypical_mix | 900 | -0.002 | 0.401 | 0.78 | 0.062 | 0.342 | 0 | 0 |
| charge_inflation | 900 | 0.001 | 0.001 | 0.008 | 0.816 | 0.005 | 0.001 | -0 |
| dme_mill | 900 | -0.001 | 0.001 | 0.003 | -0 | -0 | 4.116 | 6.989 |
| overutilization | 900 | 0.001 | 1.932 | -0.048 | 0.973 | 2.478 | 0 | 0 |
| overutilization+atypical_mix | 300 | -0.004 | 2.334 | 0.447 | 1.141 | 2.819 | 0 | 0 |
| overutilization+dme_mill | 300 | -0.003 | 1.865 | -0.058 | 0.951 | 2.366 | 4.407 | 7.73 |
| upcoding | 899 | 1.084 | 0.159 | 0.089 | 0.149 | 0.085 | 0 | 0.001 |
| upcoding+charge_inflation | 299 | 1.162 | -0.006 | 0.066 | 0.934 | 0.109 | 0 | -0 |
| upcoding+overutilization | 300 | 1.148 | 1.987 | -0.047 | 1.272 | 2.452 | -0.007 | 0 |

Leakage checks on the injected data (FEATURE_CATALOG checklist L2, L5):

| id | check | status | detail |
|---|---|---|---|
| L5 | edited cms_ps rows re-suppressed (tot_benes > 10) | pass | 0 rows at or below the threshold |
| L5 | edited dme_s rows re-suppressed (tot_suplr_clms > 10) | pass | 0 rows at or below the threshold |
| L5 | only planned providers' rows edited | pass | 0 edited rows of unplanned NPIs |
| L2 | peer statistics and cell sizes count only non-injected providers | pass | 0 violations over 99818 statistic cells |

Injection models (use b): test rows and injected test rows per family:

| family | test_rows | test_injected |
|---|---|---|
| ebm | 2073910 | 5698 |
| ga2m | 2073910 | 5698 |
| lgbm | 2073910 | 5698 |
| lgbm_mono | 2073910 | 5698 |
| lr | 2073910 | 5698 |
| xgb | 2073910 | 5698 |
| xgb_mono | 2073910 | 5698 |

## Draw 2

Planned injections (test: per scheme × intensity and per pair; train: ≈ 0.5% of training provider-years split over the same cells):

| part | scheme | intensity | planned | cell_target | distractor_share |
|---|---|---|---|---|---|
| test | atypical_mix | high | 300 | 300 | 0.5433 |
| test | atypical_mix | low | 300 | 300 | 0.5667 |
| test | atypical_mix | medium | 300 | 300 | 0.57 |
| test | charge_inflation | high | 300 | 300 | 0.6367 |
| test | charge_inflation | low | 300 | 300 | 0.62 |
| test | charge_inflation | medium | 300 | 300 | 0.6533 |
| test | dme_mill | high | 300 | 300 | 0.3667 |
| test | dme_mill | low | 300 | 300 | 0.24 |
| test | dme_mill | medium | 300 | 300 | 0.2633 |
| test | overutilization | high | 300 | 300 | 0.62 |
| test | overutilization | low | 300 | 300 | 0.61 |
| test | overutilization | medium | 300 | 300 | 0.5967 |
| test | overutilization+atypical_mix | medium | 300 | 300 | 0.5433 |
| test | overutilization+dme_mill | medium | 300 | 300 | 0.2233 |
| test | upcoding | high | 300 | 300 | 0.4433 |
| test | upcoding | low | 300 | 300 | 0.3833 |
| test | upcoding | medium | 300 | 300 | 0.3833 |
| test | upcoding+charge_inflation | medium | 300 | 300 | 0.4167 |
| test | upcoding+overutilization | medium | 300 | 300 | 0.3767 |
| train | atypical_mix | high | 1033 | 1033 | 0.5857 |
| train | atypical_mix | low | 1033 | 1033 | 0.5828 |
| train | atypical_mix | medium | 1033 | 1033 | 0.5828 |
| train | charge_inflation | high | 1033 | 1033 | 0.6031 |
| train | charge_inflation | low | 1033 | 1033 | 0.6225 |
| train | charge_inflation | medium | 1033 | 1033 | 0.5915 |
| train | dme_mill | high | 1033 | 1033 | 0.2565 |
| train | dme_mill | low | 1033 | 1033 | 0.2778 |
| train | dme_mill | medium | 1033 | 1033 | 0.2623 |
| train | overutilization | high | 1033 | 1033 | 0.5992 |
| train | overutilization | low | 1033 | 1033 | 0.5789 |
| train | overutilization | medium | 1033 | 1033 | 0.605 |
| train | overutilization+atypical_mix | medium | 1033 | 1033 | 0.6089 |
| train | overutilization+dme_mill | medium | 1033 | 1033 | 0.2682 |
| train | upcoding | high | 1033 | 1033 | 0.424 |
| train | upcoding | low | 1033 | 1033 | 0.3988 |
| train | upcoding | medium | 1033 | 1033 | 0.3795 |
| train | upcoding+charge_inflation | medium | 1033 | 1033 | 0.4327 |
| train | upcoding+overutilization | medium | 1033 | 1033 | 0.3998 |

Edits and re-suppression per year:

| year | injected | rows_suppressed_ps | rows_suppressed_dme | dropped_no_visible_rows | featurised |
|---|---|---|---|---|---|
| 2018 | 4781 | 701 | 0 | 0 | 959795 |
| 2019 | 4937 | 660 | 0 | 1 | 988048 |
| 2020 | 4829 | 706 | 0 | 2 | 966437 |
| 2021 | 5080 | 764 | 0 | 3 | 1010235 |
| 2022 | 2810 | 470 | 0 | 1 | 1027753 |
| 2023 | 2890 | 439 | 0 | 0 | 1046784 |

Feature-shift profile of injected test providers (mean change of each group's mean direction-up z; the answer groups should move most, spill-over groups somewhat):

| schemes | n | upcoding | overutilization | atypical_mix | charge_inflation | patient_mix | dme_intensity | dme_concentration |
|---|---|---|---|---|---|---|---|---|
| atypical_mix | 900 | -0 | 0.415 | 0.805 | 0.063 | 0.365 | 0 | 0.001 |
| charge_inflation | 900 | 0.001 | 0.001 | 0.014 | 0.796 | 0.002 | 0 | 0 |
| dme_mill | 900 | -0.001 | 0.001 | 0.009 | -0 | 0.001 | 4.137 | 6.871 |
| overutilization | 900 | 0.001 | 1.897 | -0.034 | 0.976 | 2.5 | 0.001 | 0 |
| overutilization+atypical_mix | 300 | 0.002 | 2.279 | 0.424 | 1.028 | 2.779 | -0.006 | -0 |
| overutilization+dme_mill | 300 | 0 | 1.797 | -0.046 | 0.929 | 2.363 | 4.344 | 7.734 |
| upcoding | 900 | 1.033 | 0.185 | 0.121 | 0.145 | 0.062 | -0 | -0 |
| upcoding+charge_inflation | 299 | 1.05 | -0.005 | 0.097 | 0.863 | 0.087 | 0.001 | -0.001 |
| upcoding+overutilization | 300 | 1.069 | 2.074 | -0.028 | 1.22 | 2.436 | -0 | 0.002 |

Leakage checks on the injected data (FEATURE_CATALOG checklist L2, L5):

| id | check | status | detail |
|---|---|---|---|
| L5 | edited cms_ps rows re-suppressed (tot_benes > 10) | pass | 0 rows at or below the threshold |
| L5 | edited dme_s rows re-suppressed (tot_suplr_clms > 10) | pass | 0 rows at or below the threshold |
| L5 | only planned providers' rows edited | pass | 0 edited rows of unplanned NPIs |
| L2 | peer statistics and cell sizes count only non-injected providers | pass | 0 violations over 99847 statistic cells |

Injection models (use b): test rows and injected test rows per family:

| family | test_rows | test_injected |
|---|---|---|
| ebm | 2073911 | 5699 |
| ga2m | 2073911 | 5699 |
| lgbm | 2073911 | 5699 |
| lgbm_mono | 2073911 | 5699 |
| lr | 2073911 | 5699 |
| xgb | 2073911 | 5699 |
| xgb_mono | 2073911 | 5699 |

## Draw 3

Planned injections (test: per scheme × intensity and per pair; train: ≈ 0.5% of training provider-years split over the same cells):

| part | scheme | intensity | planned | cell_target | distractor_share |
|---|---|---|---|---|---|
| test | atypical_mix | high | 300 | 300 | 0.54 |
| test | atypical_mix | low | 300 | 300 | 0.55 |
| test | atypical_mix | medium | 300 | 300 | 0.59 |
| test | charge_inflation | high | 300 | 300 | 0.5867 |
| test | charge_inflation | low | 300 | 300 | 0.6333 |
| test | charge_inflation | medium | 300 | 300 | 0.61 |
| test | dme_mill | high | 300 | 300 | 0.27 |
| test | dme_mill | low | 300 | 300 | 0.2467 |
| test | dme_mill | medium | 300 | 300 | 0.2667 |
| test | overutilization | high | 300 | 300 | 0.61 |
| test | overutilization | low | 300 | 300 | 0.57 |
| test | overutilization | medium | 300 | 300 | 0.6067 |
| test | overutilization+atypical_mix | medium | 300 | 300 | 0.5733 |
| test | overutilization+dme_mill | medium | 300 | 300 | 0.2133 |
| test | upcoding | high | 300 | 300 | 0.4533 |
| test | upcoding | low | 300 | 300 | 0.39 |
| test | upcoding | medium | 300 | 300 | 0.44 |
| test | upcoding+charge_inflation | medium | 300 | 300 | 0.3833 |
| test | upcoding+overutilization | medium | 300 | 300 | 0.3933 |
| train | atypical_mix | high | 1033 | 1033 | 0.5634 |
| train | atypical_mix | low | 1033 | 1033 | 0.5605 |
| train | atypical_mix | medium | 1033 | 1033 | 0.5944 |
| train | charge_inflation | high | 1033 | 1033 | 0.6089 |
| train | charge_inflation | low | 1033 | 1033 | 0.6021 |
| train | charge_inflation | medium | 1033 | 1033 | 0.5992 |
| train | dme_mill | high | 1033 | 1033 | 0.2711 |
| train | dme_mill | low | 1033 | 1033 | 0.2197 |
| train | dme_mill | medium | 1033 | 1033 | 0.2575 |
| train | overutilization | high | 1033 | 1033 | 0.5983 |
| train | overutilization | low | 1033 | 1033 | 0.5954 |
| train | overutilization | medium | 1033 | 1033 | 0.5915 |
| train | overutilization+atypical_mix | medium | 1033 | 1033 | 0.5837 |
| train | overutilization+dme_mill | medium | 1033 | 1033 | 0.2188 |
| train | upcoding | high | 1033 | 1033 | 0.3911 |
| train | upcoding | low | 1033 | 1033 | 0.3717 |
| train | upcoding | medium | 1033 | 1033 | 0.3843 |
| train | upcoding+charge_inflation | medium | 1033 | 1033 | 0.4134 |
| train | upcoding+overutilization | medium | 1033 | 1033 | 0.3959 |

Edits and re-suppression per year:

| year | injected | rows_suppressed_ps | rows_suppressed_dme | dropped_no_visible_rows | featurised |
|---|---|---|---|---|---|
| 2018 | 4849 | 692 | 0 | 2 | 959793 |
| 2019 | 4922 | 743 | 0 | 1 | 988048 |
| 2020 | 4818 | 733 | 0 | 4 | 966435 |
| 2021 | 5038 | 744 | 0 | 1 | 1010237 |
| 2022 | 2810 | 414 | 0 | 0 | 1027754 |
| 2023 | 2890 | 422 | 0 | 1 | 1046783 |

Feature-shift profile of injected test providers (mean change of each group's mean direction-up z; the answer groups should move most, spill-over groups somewhat):

| schemes | n | upcoding | overutilization | atypical_mix | charge_inflation | patient_mix | dme_intensity | dme_concentration |
|---|---|---|---|---|---|---|---|---|
| atypical_mix | 900 | 0.001 | 0.479 | 0.862 | 0.068 | 0.382 | 0 | 0.001 |
| charge_inflation | 900 | -0 | 0.001 | 0.006 | 0.769 | 0.001 | 0.001 | -0.002 |
| dme_mill | 900 | 0 | 0.001 | 0.007 | 0.003 | -0.002 | 4.196 | 7.021 |
| overutilization | 900 | 0.002 | 1.923 | -0.064 | 0.98 | 2.462 | 0 | 0 |
| overutilization+atypical_mix | 300 | 0 | 2.288 | 0.465 | 1.068 | 2.828 | -0 | 0 |
| overutilization+dme_mill | 300 | 0.001 | 1.857 | -0.051 | 0.984 | 2.382 | 4.357 | 7.547 |
| upcoding | 900 | 1.14 | 0.193 | 0.063 | 0.172 | 0.099 | -0.004 | 0 |
| upcoding+charge_inflation | 299 | 1.227 | -0.006 | 0.094 | 0.9 | 0.11 | 0.001 | 0.001 |
| upcoding+overutilization | 300 | 1.12 | 1.99 | -0.023 | 1.277 | 2.496 | -0.006 | -0 |

Leakage checks on the injected data (FEATURE_CATALOG checklist L2, L5):

| id | check | status | detail |
|---|---|---|---|
| L5 | edited cms_ps rows re-suppressed (tot_benes > 10) | pass | 0 rows at or below the threshold |
| L5 | edited dme_s rows re-suppressed (tot_suplr_clms > 10) | pass | 0 rows at or below the threshold |
| L5 | only planned providers' rows edited | pass | 0 edited rows of unplanned NPIs |
| L2 | peer statistics and cell sizes count only non-injected providers | pass | 0 violations over 99876 statistic cells |

Injection models (use b): test rows and injected test rows per family:

| family | test_rows | test_injected |
|---|---|---|
| ebm | 2073911 | 5699 |
| ga2m | 2073911 | 5699 |
| lgbm | 2073911 | 5699 |
| lgbm_mono | 2073911 | 5699 |
| lr | 2073911 | 5699 |
| xgb | 2073911 | 5699 |
| xgb_mono | 2073911 | 5699 |

## Draw 4

Planned injections (test: per scheme × intensity and per pair; train: ≈ 0.5% of training provider-years split over the same cells):

| part | scheme | intensity | planned | cell_target | distractor_share |
|---|---|---|---|---|---|
| test | atypical_mix | high | 300 | 300 | 0.5967 |
| test | atypical_mix | low | 300 | 300 | 0.58 |
| test | atypical_mix | medium | 300 | 300 | 0.5433 |
| test | charge_inflation | high | 300 | 300 | 0.58 |
| test | charge_inflation | low | 300 | 300 | 0.5967 |
| test | charge_inflation | medium | 300 | 300 | 0.6533 |
| test | dme_mill | high | 300 | 300 | 0.2267 |
| test | dme_mill | low | 300 | 300 | 0.2367 |
| test | dme_mill | medium | 300 | 300 | 0.2433 |
| test | overutilization | high | 300 | 300 | 0.6733 |
| test | overutilization | low | 300 | 300 | 0.5967 |
| test | overutilization | medium | 300 | 300 | 0.5667 |
| test | overutilization+atypical_mix | medium | 300 | 300 | 0.5633 |
| test | overutilization+dme_mill | medium | 300 | 300 | 0.2067 |
| test | upcoding | high | 300 | 300 | 0.43 |
| test | upcoding | low | 300 | 300 | 0.43 |
| test | upcoding | medium | 300 | 300 | 0.38 |
| test | upcoding+charge_inflation | medium | 300 | 300 | 0.4067 |
| test | upcoding+overutilization | medium | 300 | 300 | 0.3533 |
| train | atypical_mix | high | 1033 | 1033 | 0.5789 |
| train | atypical_mix | low | 1033 | 1033 | 0.576 |
| train | atypical_mix | medium | 1033 | 1033 | 0.577 |
| train | charge_inflation | high | 1033 | 1033 | 0.5924 |
| train | charge_inflation | low | 1033 | 1033 | 0.6186 |
| train | charge_inflation | medium | 1033 | 1033 | 0.5808 |
| train | dme_mill | high | 1033 | 1033 | 0.2449 |
| train | dme_mill | low | 1033 | 1033 | 0.3166 |
| train | dme_mill | medium | 1033 | 1033 | 0.242 |
| train | overutilization | high | 1033 | 1033 | 0.605 |
| train | overutilization | low | 1033 | 1033 | 0.6128 |
| train | overutilization | medium | 1033 | 1033 | 0.6283 |
| train | overutilization+atypical_mix | medium | 1033 | 1033 | 0.5828 |
| train | overutilization+dme_mill | medium | 1033 | 1033 | 0.2565 |
| train | upcoding | high | 1033 | 1033 | 0.393 |
| train | upcoding | low | 1033 | 1033 | 0.3833 |
| train | upcoding | medium | 1033 | 1033 | 0.3979 |
| train | upcoding+charge_inflation | medium | 1033 | 1033 | 0.3775 |
| train | upcoding+overutilization | medium | 1033 | 1033 | 0.3708 |

Edits and re-suppression per year:

| year | injected | rows_suppressed_ps | rows_suppressed_dme | dropped_no_visible_rows | featurised |
|---|---|---|---|---|---|
| 2018 | 4827 | 634 | 0 | 1 | 959794 |
| 2019 | 4959 | 679 | 0 | 3 | 988046 |
| 2020 | 4807 | 695 | 0 | 2 | 966437 |
| 2021 | 5034 | 721 | 0 | 2 | 1010236 |
| 2022 | 2808 | 421 | 0 | 0 | 1027754 |
| 2023 | 2892 | 423 | 0 | 0 | 1046784 |

Feature-shift profile of injected test providers (mean change of each group's mean direction-up z; the answer groups should move most, spill-over groups somewhat):

| schemes | n | upcoding | overutilization | atypical_mix | charge_inflation | patient_mix | dme_intensity | dme_concentration |
|---|---|---|---|---|---|---|---|---|
| atypical_mix | 900 | -0.001 | 0.473 | 0.858 | 0.062 | 0.377 | 0 | 0.001 |
| charge_inflation | 900 | 0 | 0.001 | 0.002 | 0.799 | -0.002 | 0.001 | 0 |
| dme_mill | 900 | 0 | -0.001 | 0.009 | -0.001 | -0.003 | 4.145 | 6.809 |
| overutilization | 900 | -0.002 | 1.941 | -0.064 | 0.974 | 2.489 | 0 | 0 |
| overutilization+atypical_mix | 300 | -0.004 | 2.329 | 0.472 | 1.08 | 2.775 | 0.001 | -0.003 |
| overutilization+dme_mill | 300 | 0 | 1.737 | -0.037 | 0.893 | 2.339 | 4.466 | 8.011 |
| upcoding | 900 | 1.166 | 0.195 | 0.079 | 0.172 | 0.106 | -0.001 | -0 |
| upcoding+charge_inflation | 300 | 1.072 | -0.008 | 0.12 | 0.889 | 0.072 | 0.001 | 0.002 |
| upcoding+overutilization | 300 | 1.138 | 2.083 | 0.004 | 1.353 | 2.481 | 0.001 | -0 |

Leakage checks on the injected data (FEATURE_CATALOG checklist L2, L5):

| id | check | status | detail |
|---|---|---|---|
| L5 | edited cms_ps rows re-suppressed (tot_benes > 10) | pass | 0 rows at or below the threshold |
| L5 | edited dme_s rows re-suppressed (tot_suplr_clms > 10) | pass | 0 rows at or below the threshold |
| L5 | only planned providers' rows edited | pass | 0 edited rows of unplanned NPIs |
| L2 | peer statistics and cell sizes count only non-injected providers | pass | 0 violations over 99905 statistic cells |

Injection models (use b): test rows and injected test rows per family:

| family | test_rows | test_injected |
|---|---|---|
| ebm | 2073912 | 5700 |
| ga2m | 2073912 | 5700 |
| lgbm | 2073912 | 5700 |
| lgbm_mono | 2073912 | 5700 |
| lr | 2073912 | 5700 |
| xgb | 2073912 | 5700 |
| xgb_mono | 2073912 | 5700 |

## Decisions for the human (G5)

- Accept or change the intensity grid in `configs/injection.yaml` (calibration above).
- The answer key and scheme transforms (DESIGN §17 D90-D97, proposed).
- The DMEPOS re-suppression field (beneficiaries vs claims; DESIGN §18).
