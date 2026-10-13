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
  `tests/synth.py` and only ever land in pytest temp dirs. Run `.venv/bin/python -m pytest -W error`.
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

One row per (npi, year) present in **both** the harmonised by-Provider file and the service-level file.

| Column | Meaning |
|---|---|
| `npi` (str), `year` (Int32) | key |
| `ent_cd`, `prvdr_type`, `state`, `ruca`, `mdcr_prtcptg_ind` | provider attributes from `cms_ps` (constant within npi-year; first non-null) |
| `census_division` | from `state` via `configs/census_divisions.yaml` (9 divisions; territories, military and unknown → `other`) |
| `ruca_class` | `urban` (primary RUCA 1–3), `large_rural` (4–6), `small_rural` (7–9), `isolated` (10), else `unknown` |
| `nondrug_srvcs` | Σ `tot_srvcs` over visible `cms_ps` rows with `hcpcs_drug_ind != "Y"` |
| `prov_tot_benes` | by-Provider `tot_benes` |
| `in_cohort` (bool), `cohort_exclusion_reason` | DESIGN §7.2: `in_cohort` iff `nondrug_srvcs ≥ experiment.cohort.min_nondrug_services` |
| `cell_ent`, `cell_spec_ent`, `cell_spec_ent_div` | ids `"{year}|{ent}"`, `"{year}|{ent}|{spec}"`, `"{year}|{ent}|{spec}|{div}"` |
| `n_cell_ent`, `n_cell_spec_ent`, `n_cell_spec_ent_div` | in-cohort provider counts per cell |
| `peer_cell`, `peer_level` | chosen cell: `spec_ent_div` if `n_cell_spec_ent_div ≥ peer.refine_min` (200), else `spec_ent` if `n_cell_spec_ent ≥ peer.min_cell` , else `ent` |

`build_cohort(paths, years=None) -> pl.DataFrame` writes the file. Cell sizes count in-cohort providers.
`peer.min_cell` is not set by the design: it is an implementation choice (config + §17 row).

### 2.2 Features: `features.py` (C5)

Split so that injected data (C6, Phase 5) can be featurised against peer statistics fitted on real data:

```python
refs = fit_peer_reference(frames, cohort, year)        # label-free; non-injected in-cohort providers
raw  = raw_statistics(frames, cohort, refs, year)      # r_* per npi-year (any provider, injected or not)
ps   = fit_peer_stats(raw, cohort, year)               # medians, scales, beta-binomial alphas, PM regressions
feat = transform(raw, cohort, ps, refs, year)          # model features
build_features(paths, years=None)                      # orchestrates, persists everything below
```

`frames` is a dict `{"cms_ps", "cms_prov", "dme_s", "dme_p"}` of that year's harmonised frames
(`dme_*` may be empty frames with the schema columns). `fit_*` functions receive only rows of
non-injected, in-cohort providers.

Persisted:

| Artefact | Content |
|---|---|
| `data/processed/features/{year}.parquet` | `npi`, `year`, `peer_cell`, model feature columns, `raw__{ID}` raw statistics |
| `data/processed/peer/{year}/codes.parquet` | per peer cell × HCPCS: pooled service share, share of cell providers billing it, `rare` flag, median allowed; national per-code medians |
| `data/processed/peer/{year}/stats.parquet` | per peer cell × feature ID: `median`, `scale`, `n`, `alpha` (rates) |
| `data/processed/peer/{year}/pm_models.parquet` | PM2/PM3 regression coefficients per cell (with the level actually used) |

**Model feature columns** are named by FEATURE_CATALOG ID: `UP0`–`UP6`, `OU1`–`OU5`, `SM1`–`SM4`,
`CH1`–`CH4`, `PM1`–`PM4` (+ `PM4_missing`), `DI0`–`DI4`, `DC1`, `DC2`, `CX5`, `CX8` (+ `CX7` if NPPES
exists); one-hot context columns `CX1__<value>`, `CX2__<value>`, `CX3__<value>`, `CX4__<value>`,
`CX6__<value>` with values lower-cased and non-alphanumerics replaced by `_` (XGBoost-safe names).

`configs/features.yaml` lists every feature: `id`, `name`, `group`, `direction` (`up`/`none`),
`suppression` (`V`/`T`/`VT`), `kind` (`robust_z` | `indicator` | `context_raw` | `one_hot`),
`shrink` (bool, count-based rates), `monotone` (+1 for `up` scheme features, else 0), plus
`code_sets`, `dme_high_risk`, `transform` parameters (z clip ±10, MAD constant 1.4826, scale
fallbacks), and `pruned: []` (filled at G2). `configs/groups.yaml` gives the 8 groups in fixed order
`G0 context, G1 upcoding, G2 overutilization, G3 atypical_mix, G4 charge_inflation, G5 patient_mix,
G6 dme_intensity, G7 dme_concentration`, each with its feature IDs (context also by prefix).
`groups.feature_groups(columns) -> dict[column, group]` must map every model column to exactly one group.

### 2.3 Folds and splits: `splits.py` (C7)

- `assign_folds(paths) -> pl.DataFrame` → `data/processed/folds.parquet`: `npi`, `fold` (0..4),
  `strat_specialty` (modal `prvdr_type` over the NPI's cohort years), `ever_positive` (NPI has an
  exclusion under any tier's authorities). Stratified by specialty × ever_positive, fixed seed.
- `split_rows(frame, fold, design, paths) -> frame + role` with role ∈ {`train`, `test`, `tune_train`,
  `tune_valid`, `none`} for a design in `experiment.yaml: splits.designs` (`primary`, `O1`, `O2`, `O3`):
  test = NPIs in `fold`, test years; train = other NPIs, train years; inside train, a deterministic
  stratified 20 % NPI hold-out (`splits.valid_frac`) gives `tune_valid` = their rows in the last
  train year, `tune_train` = the other 80 % NPIs' rows in the earlier train years.
- Showcase year 2024 is scored by the primary-design models (no role needed beyond `test`-like scoring).

### 2.4 QA and gate G2

- `qa.py`: `phase2_qa(paths)` → `outputs/phase2/QA_report.md` (+ JSON): cohort counts per year, peer
  level mix, feature quantiles per year, undefined/structural-zero rates, CX8 coverage distribution,
  near-duplicate pairs (|ρ| > 0.95 within a group, Spearman), fold balance (positives per fold), and the
  FEATURE_CATALOG leakage checklist evaluated programmatically where possible.
- `g2_report.py`: `outputs/phase2/G2_report.md`: final feature list after `pruned`, features that could
  not be computed and why, and the protocol diff (every §17 row added since v1.1).
- CLI: `scenario-fraud phase2 cohort | features | folds | qa | report`.
