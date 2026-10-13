# Pipeline contracts (implementation)

How the components in DESIGN §8 hand data to each other: file locations, columns and function
signatures. The design documents in `design/` remain the specification of *what* is computed; this file
pins *where and in which shape*. When a contract changes, update this file in the same commit.

## 0. Conventions (all components)

- Package `src/scenario_fraud/`, one module per component, named as in HANDOFF §4.
- Paths only via `Paths` (`paths.py`); parameters only via YAML in `configs/` (`config.load_config`).
  No magic numbers in code: a threshold the design names lives in a config file with a comment citing
  the design section.
- Polars for tables, processed **one year at a time** (real Part B is ≈ 10 M service rows per year;
  the container has 15 GB RAM). Parquet for every persisted table.
- Fail loudly (raise a module-specific `…Error`) on violated assumptions; never coerce silently.
- Every CLI command goes through `cli.py` and `runlog.run_log` (phase name = `phase2`, `phase3`, …).
- Determinism: every random draw takes an explicit seed from `configs/experiment.yaml`; derive
  sub-seeds with `numpy.random.SeedSequence` / hashing, never global state.
- Tests in `tests/`, using the `proj` fixture (a throw-away project root). Synthetic data come only from
  `tests/synth.py` and only ever land in pytest temp dirs (`make_synthetic` refuses the checkout and any root that
  holds non-synthetic data, and flags every file it writes). Run `.venv/bin/python -m pytest -W error`.
- **Never** read or write `data/` from tests, never import the legacy prototype files, never commit.
- Any choice the design leaves open, or any deviation, is recorded as a row in DESIGN §17 (next free
  number; mark it "proposed, confirm at G2" when the human has not decided it) and in the affected
  design file. Code and docs never disagree silently (HANDOFF §8 rule 1).

## 1. Phase 1 outputs (exist)

| Artefact | Columns / content |
|---|---|
| `data/interim/{cms_ps,cms_prov,dme_s,dme_p}/{year}.parquet` | harmonised names from `configs/column_maps/_schema.yaml`, plus `year` (Int32) |
| `data/processed/exclusions.parquet` | one row per exclusion event: `npi`, `authority`, `excl_date`, `excl_year`, `rein_date`, `in_current`, … |
| `data/processed/first_exclusions.parquet` | `npi`, `tier` (`primary`/`prior_work`/`broad`/`any`), `excl_date`, `authority`, `excl_year` |
| `data/processed/labels_{tier}_W{W}.parquet` | `npi`, `year`, `y` (1/0/null), `drop_reason`, `label_tier`, `excl_type`, `excl_year`, `label_horizon_complete`, `label_horizon_years`, `W` |

## 2. Phase 2: cohort (C4), features (C5), folds (C7)

### 2.1 Cohort and peer cells: `cohort.py` (C4) → `data/processed/cohort.parquet`

One row per (npi, year) present in **either** the harmonised by-Provider file or the service-level file
(DESIGN §17 D21); sorted by year, npi.

| Column | Meaning |
|---|---|
| `npi` (str), `year` (Int32) | key |
| `ent_cd`, `prvdr_type`, `state`, `ruca`, `mdcr_prtcptg_ind` | provider attributes: first non-null in `cms_ps` grain order (`hcpcs_cd`, `place_of_srvc`), else the by-Provider file's (D23) |
| `census_division` | from `state` (stripped, upper-cased) via `configs/census_divisions.yaml` (9 divisions; territories, military, foreign and null → `other`) |
| `ruca_class` | integer part of `ruca` in `experiment.cohort.ruca_classes` ranges (`urban` 1–3, `large_rural` 4–6, `small_rural` 7–9, `isolated` 10), else `unknown` (D25) |
| `nondrug_srvcs` | Σ `tot_srvcs` over visible `cms_ps` rows with `hcpcs_drug_ind != "Y"` (0 without service rows) |
| `prov_tot_benes` | by-Provider `tot_benes` (null without a by-Provider row) |
| `in_cohort` (bool), `cohort_exclusion_reason` | `in_cohort` iff in both files, `ent_cd` and `prvdr_type` non-null and `nondrug_srvcs ≥ experiment.cohort.min_nondrug_services`; else the first of `not_in_service_file`, `not_in_provider_file`, `missing_peer_attributes`, `below_min_nondrug_services` |
| `cell_ent`, `cell_spec_ent`, `cell_spec_ent_div` | ids `"{year}|{ent}"`, `"{year}|{ent}|{spec}"`, `"{year}|{ent}|{spec}|{div}"` (null if a part is null) |
| `n_cell_ent`, `n_cell_spec_ent`, `n_cell_spec_ent_div` (Int64) | **counted** providers per cell: `in_cohort` and, when the optional `injected` column exists, not injected (`cohort.counted`) |
| `peer_cell`, `peer_level` | `spec_ent_div` if `n_cell_spec_ent_div ≥ peer.refine_min` (200; the division cell's own count, D20), else `spec_ent` if `n_cell_spec_ent ≥ peer.min_cell` (100, D19), else `ent`; also set for out-of-cohort and injected rows, whose presence never counts (D22) |
| `injected` (bool, optional; Phase 5) | injected provider (C6); never counted, never in a fit |

`build_cohort(paths, years=None) -> pl.DataFrame`: every harmonised year (rewrites the file), or the given years, which
replace those years and keep the others only when `cohort.meta.json` records the same cohort config, else
`CohortError` (D27). Side file `data/processed/cohort.meta.json`: `years`, `config` (thresholds, RUCA classes,
`census_divisions_sha256`), `synthetic_inputs`, `per_year` {rows, in_cohort, exclusion_reasons, peer_level_mix,
n_peer_cells, smallest_peer_cell, undersized_ent_cells, attribute_conflicts, census_division_other, ruca_unknown}.
Pure helpers: `cohort_year(ps, prov, year, cfg) -> (frame, summary)`, `assign_peer_cells(df, cfg)` (re-runnable on a
cohort frame, e.g. after C6 appends injected rows), `census_division_expr`, `ruca_class_expr`, `counted(df)`.
Fails loudly (`CohortError`) on the conditions of D26.

### 2.2 Features: `features.py` (C5)

Split so that injected data (C6, Phase 5) can be featurised against peer statistics fitted on real data:

```python
refs = fit_peer_reference(frames, cohort, year, cfg=None)        # label-free code reference
raw  = raw_statistics(frames, cohort, refs, year, cfg=None)      # r per npi-year (any provider, injected or not)
ps   = fit_peer_stats(raw, cohort, year, cfg=None)               # medians, scales, beta-binomial priors, PM fits
feat = transform(raw, cohort, ps, refs, year, cfg=None)          # model features
build_features(paths, years=None) -> {year: meta}                # orchestrates, persists everything below
```

`frames` is a dict `{"cms_ps", "cms_prov", "dme_s", "dme_p"}` of that year's harmonised frames (`dme_*` may be
empty frames with the schema columns, `empty_frame`), plus an optional `"nppes"` frame (`npi`,
`enumeration_date`) that must give **every featurised NPI exactly one non-null date** (D44). `cohort` is that year's
cohort rows (CONTRACTS §2.1 columns; optional `injected`). `fit_*` use only **reference providers** (`in_cohort` and
not `injected`), whatever rows they are given (D45); `raw_statistics` / `transform` featurise every row given whose
peer cell has fitted statistics. A peer cell's members are all reference providers whose cell at the peer level is
that cell (D32). `raw` also carries `PMx__y`, the `bene_cc_*` covariates, `{ID}__k`/`{ID}__n` of shrunk rates and,
for the reference-dependent features (`REF_DEPENDENT`: SM1, SM2, CH4, PM2, PM3), `SM1@{level}`, `SM2__k@{level}`,
`CH4@{level}`: each row's value against the reference of its cell at that level (D33). `build_features` featurises
in-cohort rows only, one year at a time, never reads labels or LEIE, and is bit-identical whatever the input row
order (D47).

Persisted:

| Artefact | Content |
|---|---|
| `data/processed/features/{year}.parquet` | `npi`, `year`, `peer_cell`, model feature columns, `raw__{ID}` for every robust_z feature and `raw__{ID}__k`/`__n` for shrunk rates (D46) |
| `data/processed/features/{year}.meta.json` | rows, reference providers, peer cells, inputs present, `not_computable` (e.g. CX7), per-feature undefined / clipped counts, stats-level and scale-method mix, α ranges, PM covariates and fit levels, structural zeros, CX8 quantiles, `synthetic_inputs` |
| `data/processed/peer/{year}/codes.parquet` | `level`, `peer_cell`, `hcpcs_cd`, `n_cell`, `srv`, `srv_share`, `n_providers`, `provider_share`, `rare`, `med_allowed_f`/`n_allowed_f`, `med_allowed_o`/`n_allowed_o`: non-drug codes per used peer cell, plus national rows (`level` `national`, `peer_cell` = year) |
| `data/processed/peer/{year}/stats.parquet` | per peer cell × robust_z feature: `level`, `stat_cell`, `n`, `median`, `scale`, `scale_method` (`mad`/`iqr`/`std`/`parent_*`/`unit`), `alpha`, `prior_mean` |
| `data/processed/peer/{year}/pm_models.parquet` | PM2/PM3 per peer cell: `target`, `level`, `fit_cell`, `n`, `term`, `coef`, `impute` (long format) |

**Model feature columns** are named by FEATURE_CATALOG ID: `UP0`–`UP6`, `OU1`–`OU5`, `SM1`–`SM4`,
`CH1`–`CH4`, `PM1`–`PM4` (+ `PM4_missing`), `DI0`–`DI4`, `DC1`, `DC2`, `CX5`, `CX8` (+ `CX7` if NPPES
exists); one-hot context columns `CX1__<value>`, `CX2__<value>`, `CX3__<value>`, `CX4__<value>`,
`CX6__<value>` with values lower-cased and non-alphanumerics replaced by `_` (XGBoost-safe names; null → `na`;
vocabulary = that year's reference providers, D42). Indicators and one-hots are Int8, all others Float64, no nulls.
Robust z is clipped to ±10; an undefined r gives z = 0 (D34); with no DME-P row, z = 0 on DI1, DI3, DI4, DC1, DC2
(D40). Helpers: `load_features(paths, years)` (years stacked, absent one-hots 0), `load_frames`, `model_columns`,
`prune_candidates`, `high_risk_expr`, `alpha_expr`.

`configs/features.yaml` lists every feature: `id`, `name`, `group`, `direction` (`up`/`none`),
`suppression` (`V`/`T`/`VT`), `kind` (`robust_z` | `indicator` | `context_raw` | `one_hot`),
`shrink` (bool, count-based rates), `monotone` (+1 for `up` robust_z features, else 0), `fit_on` (`defined` |
`dme_referrers`), `source` (one-hot cohort column), `optional`, plus `transform` (z clip ±10, MAD constant 1.4826,
scale fallbacks), `peer_stats.min_defined`, `shrinkage`, `atypical_mix`, `charge_index`, `patient_mix`, `prune`,
`code_sets`, `dme_high_risk` (status proposed until G2) and `pruned: []` (filled at G2). `configs/groups.yaml` gives
the 8 groups in fixed order `G0 context, G1 upcoding, G2 overutilization, G3 atypical_mix, G4 charge_inflation,
G5 patient_mix, G6 dme_intensity, G7 dme_concentration`, each with its feature IDs (context also by prefix), and
`variants` (S7). `groups.feature_groups(columns, variant=None) -> dict[column, group]` must map every model column to
exactly one group.

### 2.3 Folds and splits: `splits.py` (C7)

Parameters: `experiment.yaml: splits` (`n_folds` 5, `seed`, `valid_frac` 0.2, `designs` `primary`
2018–21 → 2022–23 (must equal `years.train/test`), `O1` 2018–20 → 2021, `O2` 2018–21 → 2022, `O3`
2018–22 → 2023, `showcase` {year 2024, design primary}, `negative_subsample` {rate 0.15,
sensitivity_rates [0.30], seed}). `SplitConfig.load(paths)` validates them (every train year before every test
year, ≥ 2 train years). Sub-seeds: `numpy.random.SeedSequence([seed, purpose, fold])`; NPI order within a
stratum = splitmix64 hash of the NPI (stable across library versions and input row order).

- `assign_folds(paths) -> pl.DataFrame` → `data/processed/folds.parquet` (+ `folds.meta.json`: config,
  `synthetic_inputs`, per-fold NPIs and ever-positive NPIs, largest per-stratum spread): one row per NPI with ≥ 1 in-cohort year,
  `npi`, `fold` (Int32, 0..4), `strat_specialty` (modal `prvdr_type` over the NPI's in-cohort years; a tie goes
  to the specialty of the latest tied year), `ever_positive` (NPI has an exclusion in `exclusions.parquet`
  under the authorities of any tier in `labels.yaml`, any date). Stratified by strat_specialty ×
  ever_positive: strata laid end to end (ever-positive first), NPIs hashed within a stratum, position *i* →
  fold *i* mod 5, so fold sizes, ever-positive NPIs per fold and each stratum's count per fold differ by ≤ 1.
  `make_folds(cohort, positive_npis, cfg)` is the pure core.
- `holdout_npis(folds, fold, cfg) -> pl.Series` = H_k, the validation hold-out of fold k: exactly
  ⌊0.2 × N⌋ of the N NPIs not in fold k, stratified the same way (each stratum within one of its 20 %),
  deterministic, different per fold, the same for every design.
- `split_rows(frame, fold, design, paths, *, exclude_years=()) -> frame + role` (pure core `assign_roles(frame,
  folds, fold, design, cfg, holdout=None, *, check=True, exclude_years=())`); `frame` needs `npi`, `year`; row order kept; `role` is `pl.Enum(ROLES)`; an NPI
  without a fold raises `SplitError`. For design train years [a, b], test years [c, e]:

  | role | rows |
  |---|---|
  | `test` | NPI in fold k, year in [c, e] |
  | `tune_valid` | NPI ∉ k and ∈ H_k, year b (last train year) |
  | `tune_train` | NPI ∉ k and ∉ H_k, years [a, b − 1] |
  | `train` | NPI ∉ k, years [a, b], neither of the above (H_k's earlier rows; the others' year-b rows) |
  | `none` | everything else (fold-k NPIs in train years, other NPIs in test years, other years) |

  `TRAIN_ROLES` = (`train`, `tune_train`, `tune_valid`) is fold k's training set; tuning fits on `tune_train` and
  validates on `tune_valid`. Which of these rows the final model and its isotonic calibrator use is **open** (DESIGN
  §17 D59) and is decided in Phase 3. `exclude_years` (sensitivity S4): rows of those years get `none`, and
  `tune_valid` moves to the last remaining train year, which needs ≥ 2 remaining train years (D60).
- `design="showcase"`: trains like `primary`; `test` = fold-k NPIs in 2024 (scored by the primary-design
  model of their fold; no accuracy claims).
- `subsample_negatives(frame, cfg, rate=None, label="y") -> frame + weight` (Phase 3 training rows): every
  positive (weight 1) and each negative (npi, year) kept iff its seeded hash < rate (weight 1 / rate). The
  sample is identical for every model family, fold and design, and the 15 % sample is nested in the 30 % one.

### 2.4 QA and gate G2

- `provenance.py`: `file_provenance(path)` → `synthetic` (Parquet key-value metadata `scenario_fraud.synthetic` =
  `"true"`, written by `tests/synth.py`, or a sidecar `synthetic: true`), `harmonised` (sidecar without the flag) or
  `unknown` (neither); `inputs_synthetic(files)` → True / None (some unknown) / False, recorded as `synthetic_inputs`
  in `cohort.meta.json`, `features/{year}.meta.json`, `folds.meta.json` and every phase2 run log.
- `qa.py`: `phase2_qa(paths, rebuild=None, rebuild_year=None)` → `outputs/phase2/QA_report.md` (+ `.json`): input
  provenance (a **synthetic-data banner only when** an interim file is flagged synthetic or a manifest entry's notes
  say synthetic; a **provenance-unknown banner** when a harmonised file has neither sidecar nor flag), §1 cohort counts
  and exclusion reasons per year, §2 peer level mix, §3 raw-statistic quantiles per feature and year
  (`experiment.yaml: qa.quantiles`), §4 undefined / structural-zero rates, §5 CX8 coverage, §6 near-duplicate pairs
  (`features.prune_candidates` per year), §7 Census-division refinement evidence for DESIGN §19 Q2 (`region_section`:
  per feature, |division-cell median − specialty × entity median| in robust-z units and the scale ratio;
  `qa.region_shift_threshold`), §8 fold balance and per design × fold role counts with positives
  (`labels_{primary}_W{W}`), and §9 the FEATURE_CATALOG leakage checklist (`id` L1–L6 in catalog order; status
  `pass` / `fail` / `partial` / `pending` / `not_run`): L1 static scan of `cohort.py`, `features.py`, `groups.py`,
  `provenance.py` + one feature year rebuilt in a scratch project holding only that year's interim files and cohort
  rows, with `SCENARIO_FRAUD_ROOT` pointed at it (`qa.leakage_rebuild`, `leakage_rebuild_year`), compared bit for
  bit, any rebuild exception = fail; L2 every `stats.parquet` `n`, `codes.parquet` `n_cell` (and `n_providers ≤
  n_cell`), `pm_models.parquet` `n` and cohort `n_cell_*` / `peer_cell` equal the counts of non-injected in-cohort
  providers of the cell (with a defined value); L3 roles respect folds and years (model fits pending, Phase 3); L4 no
  NPI on both sides of any design × fold, every test row scored by one fold; L5 pending (Phase 5); L6 sha256 of cohort
  threshold, high-risk DME list and grouping (`protocol_hashes`), pass if no `outputs/phase3..7` file exists or the
  hashes equal the **freeze record** `outputs/phase2/G2_freeze.json`.
- `g2_report.py`: `build_report(paths)` → `outputs/phase2/G2_report.md` (+ `.json`): final feature list after
  `pruned`, features that could not be computed (or were structurally empty in a year) and why, near-duplicate
  candidates, the protocol diff (every DESIGN §17 row after `experiment.yaml: g2.baseline_last_decision` = D18,
  parameters to freeze, `protocol_hashes` and config sha256), the freeze status, the leakage checklist and the
  decisions needed (incl. rows marked proposed or open, peer granularity and the open DESIGN §19 questions). While no
  `outputs/phase3..7` file exists it (re)writes `G2_freeze.json` {recorded_utc, protocol_hashes, config_sha256};
  afterwards it never rewrites it (D67).
- CLI: `scenario-fraud phase2 cohort [--year Y …] | features [--year Y …] | folds | qa [--no-rebuild]
  [--rebuild-year Y] | report`, each inside `run_log(paths, "phase2", …)`.

## 3. Phase 3: models (C8) and detection metrics

Parameters: `configs/models.yaml` (families, search spaces, seeds, final fit, calibration source, per-design tuning)
and `experiment.yaml: evaluation` (k values, bootstrap, Holm level, reliability bins). DESIGN §17 D73–D81.

### 3.1 Model matrix: `modelmatrix.py`

`load_design(paths, design, *, label_tier=None, W=None, years=None) -> DesignFrame` reads the design's feature years
once (`features.load_features`), joins `labels_{tier}_W{W}.parquet` (rows with `y` null dropped; defaults from
`labels.yaml`) and `folds.parquet`. `DesignFrame.frame` has `npi`, `year`, `peer_cell`, `fold`, `y`,
`label_horizon_complete` and the model columns in fixed order (`column_order`: groups.yaml order, features.yaml order,
one-hots last and sorted); `columns`, `groups` (column → group), `folds`. `roles(df, fold, scfg, design=None)` gives
the C7 roles; `matrix(frame, columns, rows=None)` the float32 matrix (absent one-hot columns = 0, any other missing
column or non-finite value fails). `column_kinds(columns)` maps columns to features.yaml kinds.

### 3.2 Model families: `models.py`

| Key | Model |
|---|---|
| `lr` | pipeline: cubic B-spline basis (linear extrapolation, quantile knots) for `lr.spline_features` + standardised continuous columns + binary columns passed through → `LogisticRegression` (saga, `l1_ratio`, C) |
| `ebm` / `ga2m` | `ExplainableBoostingClassifier`, `interactions` 0 / tuned ≤ `ga2m.max_interactions` (FAST) |
| `xgb`, `xgb_mono` | `xgboost.train` (hist); `_mono` uses `features.yaml: monotone` in column order |
| `lgbm`, `lgbm_mono` | `lightgbm.train` (deterministic); likewise |

`fit(family, X, y, w, params, seed, columns, cfg) -> Fitted` (`params` includes `pos_weight`, which multiplies the
positives' weights); `margin(fitted, X, batch)` = raw log-odds; `save(fitted, dir)` / `load(dir)` (`model.json` /
`model.txt` native, else `model.joblib`, plus `fitted.json`: family, columns, params, seed, extra);
`lr_design(fitted, X) -> (Z, names, groups)` with `Z @ coef + intercept` = the LR margin and spline basis columns
named `{feature}__spl{j}` in their feature's group (method D, C12); `Fitted.extra`: `pair_terms` (EBM/GA²M),
`converged` (LR).

### 3.3 Trainer: `train.py` (C8)

`train_design(paths, design, *, families, folds, seeds, n_trials)`, per fold k and family:

1. training rows = TRAIN_ROLES, negatives subsampled by `splits.subsample_negatives` (identical for every family);
   `tune_train` rows from that sample; `tune_valid` = all its rows (not subsampled), ≥ `min_valid_positives`;
2. tuning (designs.<d>.tune): Optuna TPE, objective = average precision on `tune_valid`; otherwise the primary
   design's `tuning.json` best parameters;
3. tuning model (best params on `tune_train`) → `valid_ap` and, with `calibration_source: tuning_model`, the isotonic
   calibrator;
4. final models per seed on the final rows (`final_fit`); `calibration_source: final_model` fits the calibrator on
   each final model's `tune_valid` margins (only with `exclude_holdout_npis`);
5. test rows scored.

Persisted: `data/processed/models/{design}/{family}/fold{k}/` with `seed{s}/` (model + `fitted.json` +
`calibrator.joblib`), `columns.json`, `tuning.json` (best params, every trial, `valid_ap`, row and positive counts,
`params_source`, `pair_terms`, seconds); `data/processed/scores/{design}/{family}/fold{k}.parquet` with `npi`, `year`,
`fold`, `y`, `peer_cell`, `label_horizon_complete`, `seed`, `margin`, `prob`, `label_tier`, `W` (test rows only).
`score_showcase(paths)` writes `scores/showcase/{family}/fold{k}.parquet` (showcase year, each NPI's own fold, every
seed). `load_scores`, `load_tuning` read them back. `check_frozen(paths)` is the G2 guard (D73).

### 3.4 Detection metrics: `metrics.py` (shared with C13)

`detection_metrics(df, k_fracs)` → pooled and per-year N, positives, base rate, PR-AUC, ROC-AUC, precision / recall /
lift@k, Brier, `reliability_top` (quantile bins of calibrated probability in each year's top 5 %).
`bootstrap_ci(df, …)` and `paired_bootstrap(a, b, …)` (paired NPI bootstrap: row weight = its NPI's draw count;
metrics computed on presorted arrays) return percentile CIs, Δ and two-sided bootstrap p-values; `holm(pvalues,
alpha)`. Ranking: margin descending, ties by a fixed (npi, year) hash; pooled top-k = per-year counts summed (D81).

### 3.5 Gate G3: `g3_report.py` → `outputs/phase3/G3_report.md` (+ `.json`)

Per design with scores: validation average precision per family and the chosen best models; T1 (primary: with
NPI-bootstrap CIs and seed spread); Δ best GBM − best glass box with CI, p and Holm; the H1 check; per-year results;
label-horizon counts; calibration in the top 5 %; the showcase row count; the synthetic / unknown-provenance banner
(`qa.provenance_lines`). It states which DESIGN §10.8 row the detection numbers are consistent with; the human decides.
CLI: `scenario-fraud phase3 train [--design D] [--family F …] [--fold K …] [--seeds N] [--trials N] |
score-showcase [--family F …] | report`, each inside `run_log(paths, "phase3", …)`.

## 4. Phase 4: attribution (C9–C12)

Parameters: `configs/attribution.yaml`. DESIGN §17 D82–D89. Groups are the 8 of `groups.GROUP_ORDER` (context = bit 0).

- `explained_set.py`: `build_explained_set(paths, injected=None)` → `data/processed/attributions/explained_set.parquet`
  (`npi`, `year`, `fold`, `top`, `random`, `injected`, `score_family`): per test year of `explained.design`, top
  `top_frac` by the best GBM's seed-0 margin, ∪ `random_n` stratified by specialty, ∪ injected rows (Phase 5).
- `attribution/background.py` (C9): `peer_backgrounds(explained, candidates, cfg, fold, year)` → (assignment `npi`,
  `bg_key` = `"{level}={cell}"`, `bg_level` ∈ peer_cell / cell_spec_ent / cell_ent / year, `bg_n`; members `bg_key`,
  `npi`), `global_background(candidates, cfg, fold, year)` → NPIs. Candidates = in-cohort feature rows of NPIs not in
  fold k in that year.
- `attribution/coalition.py` (C10): `coalition_values(predict, X, Xb, col_group, m, batch_rows)` → V (n × 2^m; bit g
  set = group g from the provider), `shapley(V)` (B), `interaction_matrix(V, phi)` (C), `counterfactual(V)` (E: log-odds,
  probability), `deletion_curve`, `deletion_score`, `order_by`, `random_deletion_score`.
- `attribution/summed_shap.py` (C11): `feature_shap(fitted, X, Xb)` (interventional, raw margin, `Independent` masker
  of size |B|), `group_sums`, `path_dependent(fitted, X, col_group, m)` (diagnostics).
- `attribution/glassbox.py` (C12): `lr_contrib`, `lr_feature_contrib`, `ebm_contrib` (group vector with ½-splits,
  pair matrix in method C's convention, per-column contributions).
- `attribute.py`: `explain(paths, *, families, baselines, folds, limit)` → `data/processed/attributions/{family}/
  {baseline}/fold{k}.parquet` with `npi`, `year`, `fold`, `bg_key`, `bg_level`, `bg_n`, list columns `v` (256), `B`,
  `E`, `E_prob`, `A`, `D` (8), `C`, `D_pair` (64, row-major m × m), `A_feat`, `D_feat` (per model column, order in
  `{family}/columns.json` with `col_group` and `groups`), scalars `margin` (= v(N) = f(x)), `base` (= v(∅)),
  `del_{B,A,D,E}`, `del_{…}_reverse`, `del_random`; memberships in `attributions/backgrounds/{year}_fold{k}.parquet`
  (`bg_key`, `npi`, `year`, `fold`). `load_attributions(paths, family, baseline)`. `diagnostics(paths)` →
  `outputs/phase4/diagnostics.json` (path-dependent A vs interventional A; feature-level interactions vs C).
- `bench.py`: `benchmark(paths, families)` → `outputs/phase4/bench.json` (predictions/s, SHAP s/provider, projected
  hours per |B| option and global baseline, budget).
- `g4_report.py`: `build_report(paths)` → `outputs/phase4/G4_report.md` (+ `.json`): explained set, benchmark and
  projections, background levels, EXPERIMENT_PLAN §4 additivity on the store (B, C, D 1e-6; A 1e-4), diagnostics.
- CLI: `scenario-fraud phase4 explained-set | explain [--family F …] [--baseline peer|global] [--fold K …]
  [--limit N] | diagnostics | bench [--family F …] | report`. `explain` refuses after a protocol change (D73).

## 5. Phase 5: scheme injection (C6) and injection models

Parameters: `configs/injection.yaml`. DESIGN §17 D90–D100. Everything per draw d lives under
`data/processed/injection/draw{d}/`.

- `injection.py` (C6): `InjectionConfig`; `eligibility(paths, years, icfg)` (in-cohort, never-positive provider-years
  with `ok_<scheme>` flags); `plan(paths, draw)` → `plan.parquet` (`npi`, `year`, `fold`, `strat_specialty`,
  `peer_cell`, `part` train/test, `schemes` list, `intensity` label, `pair`, `cell`, `cell_target`, `x_<scheme>`
  intensity values, `answer_groups`, `distractor`, `distractor_groups`, `draw`); transforms `upcoding`,
  `overutilization`, `atypical_mix`, `charge_inflation`, `dme_mill`; `apply_year(frames, who, coh, refs, fcfg, icfg,
  draw, year)` → edited frames + edit log; `build(paths, draw)` → `edited/{dataset}/{year}.parquet` (edited rows of
  planned providers only), `features/{year}.parquet` (whole year re-featurised, `injected` column),
  `peer/{year}/{stats,cohort}.parquet`, `edit_log.parquet`, `summary.json`; `load_injected_features`.
- `inject_models.py`: `train_models(paths, draw)` → `models/{family}/fold{k}/seed0/`, `scores/{family}/fold{k}.parquet`
  (`npi`, `year`, `fold`, `y` = injected, `peer_cell`, `margin`, plan columns); `stress_test` → `use_a.parquet`
  (`family`, `margin`, `percentile` among clean test rows); `explain(paths, draw)` →
  `attributions/{family}/{baseline}/fold{k}.parquet` (Phase 4 store columns + `schemes`, `intensity`, `pair`,
  `answer_groups`, `distractor`, `distractor_groups`); `load_injected_attributions(paths, family, baseline, draws)`
  (adds `draw`); `calibrate(paths)` → `outputs/phase5/calibration.json`.
- `g5_report.py`: `outputs/phase5/G5_report.md` (+ `.json`): calibration, plan vs targets, edits and re-suppression,
  feature-shift profile by group, leakage checks L2/L5, injection-model row counts.
- CLI: `scenario-fraud phase5 calibrate | build [--draw D …] | train [--draw D …] [--family F …] | stress … |
  explain … [--limit N] | report`. Training and explaining refuse after a protocol change (D73).
