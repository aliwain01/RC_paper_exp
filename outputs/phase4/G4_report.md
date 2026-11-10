# Gate G4 report: attribution timing and store checks (Phase 4)

Inputs: 28 harmonised CMS files under `data/interim` and 80 registered raw files (`data/raw/manifest.json`).

Generated 2026-10-25T01:13:10.120454+00:00 by `scenario-fraud phase4 report`. HANDOFF §6 G4: predictions per second and the projected wall-clock of the coalition tables; the human sets |B| and the explained-set size.

## Explained set

Top 2% by the out-of-fold margin of `xgb` (best GBM on validation) ∪ stratified random 5000 ∪ injected test providers (Phase 5).

| year | explained | top | random | injected |
|---|---|---|---|---|
| 2022 | 25442 | 20553 | 5000 | 0 |
| 2023 | 25834 | 20936 | 5000 | 0 |

## Timing benchmark and projections

51261 explained provider-years; coalition table = 2^8 subsets × |B| predictions per provider. Budget set in attribution.yaml: 24.0 h.

| family | pred/s | hours |B|=100 | hours |B|=200 | hours global | SHAP hours (B=200) |
|---|---|---|---|---|---|
| lr | 87,827 | 4.15 | 8.3 | 41.5 | – |
| ebm | 401,197 | 0.91 | 1.82 | 9.09 | – |
| ga2m | 373,311 | 0.98 | 1.95 | 9.76 | – |
| xgb | 946,189 | 0.39 | 0.77 | 3.85 | 7.02 |
| xgb_mono | 1,572,907 | 0.23 | 0.46 | 2.32 | 0.17 |
| lgbm | 587,468 | 0.62 | 1.24 | 6.2 | 0.18 |
| lgbm_mono | 748,293 | 0.49 | 0.97 | 4.87 | 0.12 |

Totals by baseline choice (all families):

| setting | hours | within budget |
|---|---|---|
| 100 | 7.76 | True |
| 200 | 15.52 | True |
| global | 77.6 | False |

## Attribution store: EXPERIMENT_PLAN §4 additivity checks

Σ Φᴮ, Σ M (method C), Σ Φᴬ and Σ D against v(N) − v(∅) per provider (A tolerance 1e-4, others 1e-6).

| family | baseline | providers | bg_n_min | bg_n_median | max_err_B | max_err_C | max_err_A | A_inexact | max_err_D | additivity |
|---|---|---|---|---|---|---|---|---|---|---|
| lr | peer | 51276 | 50 | 200 | 9.326e-15 | 9.77e-15 |  |  | 7.55e-15 | pass |
| ebm | peer | 51276 | 50 | 200 | 1.066e-14 | 1.066e-14 |  |  | 2.442e-14 | pass |
| ga2m | peer | 51276 | 50 | 200 | 7.994e-15 | 7.994e-15 |  |  | 1.377e-14 | pass |
| xgb | peer | 51276 | 50 | 200 | 4.441e-15 | 5.329e-15 | 2.214e-05 | 0 |  | pass |
| xgb | global | 51276 | 200 | 200 | 3.553e-15 | 3.553e-15 | 2.202e-05 | 0 |  | pass |
| xgb_mono | peer | 51276 | 50 | 200 | 3.553e-15 | 3.553e-15 | 2.53e-05 | 0 |  | pass |
| lgbm | peer | 51276 | 50 | 200 | 2.665e-15 | 2.665e-15 | 0.0003461 | 6 |  | FAIL |
| lgbm_mono | peer | 51276 | 50 | 200 | 3.553e-15 | 3.553e-15 | 0.07619 | 1 |  | FAIL |

Background levels used (peer baseline falls back up the hierarchy below `min_background`, D82):

| family | baseline | cell_spec_ent | global | peer_cell |
|---|---|---|---|---|
| lr | peer | 102 |  | 51174 |
| ebm | peer | 102 |  | 51174 |
| ga2m | peer | 102 |  | 51174 |
| xgb | peer | 102 |  | 51174 |
| xgb | global |  | 51276 |  |
| xgb_mono | peer | 102 |  | 51174 |
| lgbm | peer | 102 |  | 51174 |
| lgbm_mono | peer | 102 |  | 51174 |

## Path-dependent diagnostics (DESIGN §9.4)

| family | fold | n | max_abs_A_path_minus_A | mean_abs_A_path_minus_A | corr_offdiag_IV_vs_C |
|---|---|---|---|---|---|
| xgb | 0 | 365 | 2.045 | 0.1325 | 0.8328 |
| xgb | 3 | 400 | 1.909 | 0.2995 | 0.8021 |
| xgb | 1 | 405 | 0.9748 | 0.136 | 0.8772 |
| xgb | 4 | 387 | 1.197 | 0.1741 | 0.8802 |
| xgb | 2 | 443 | 3.614 | 0.4912 | 0.7852 |
| xgb_mono | 0 | 365 | 2.296 | 0.1859 | 0.8425 |
| xgb_mono | 3 | 400 | 3.545 | 0.2352 | 0.7636 |
| xgb_mono | 1 | 405 | 0.8968 | 0.1053 | 0.8652 |
| xgb_mono | 4 | 387 | 1.217 | 0.1924 | 0.8041 |
| xgb_mono | 2 | 443 | 3.513 | 0.2953 | 0.6873 |
| lgbm | 0 | 365 | 2.193 | 0.118 | 0.851 |
| lgbm | 3 | 400 | 2.13 | 0.1407 | 0.8188 |
| lgbm | 1 | 405 | 1.052 | 0.06676 | 0.8749 |
| lgbm | 4 | 387 | 1.59 | 0.07433 | 0.8869 |
| lgbm | 2 | 443 | 2.396 | 0.09694 | 0.7758 |
| lgbm_mono | 3 | 400 | 2.458 | 0.1495 | 0.6377 |
| lgbm_mono | 0 | 365 | 2.896 | 0.1183 | 0.8052 |
| lgbm_mono | 4 | 387 | 1.612 | 0.08935 | 0.8438 |
| lgbm_mono | 1 | 405 | 1.982 | 0.1522 | 0.7559 |
| lgbm_mono | 2 | 443 | 1.779 | 0.03141 | 0.8104 |

## Decisions for the human (G4)

- |B| (attribution.yaml `background.B`) and the explained-set size (`explained.top_frac`, `random_n`).
- Which families get coalition tables (`coalition_families`; E6 asks for every model × method pair).
- `background.min_background` (D82).
