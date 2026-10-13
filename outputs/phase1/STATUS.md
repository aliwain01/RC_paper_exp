# Status: Phases 0-7 built and tested on synthetic data; data acquisition blocked

12 October 2026 · follows [HANDOFF.md](../../HANDOFF.md) §9 quick start. Gate **G1 not reached**: no data could be downloaded from this environment.

## Phase 7 (13 October 2026): code built, tested on synthetic data only

Built: `render.py` (C14: decision record per `configs/decision_record.schema.json`, analyst, provider, beneficiary and oversight views, unit-manager queue summary), `reports.py` (C15: paper tables in Markdown, CSV and LaTeX; figures F2–F8; draft outline; reproducibility package), and `scenario-fraud phase7 render | render-sample | queue | tables | figures | draft | package | report`.

- **Notices are mock-ups.** Provider and beneficiary notices appear only after a decision, and every decision here is a mock decision with a MOCK-UP banner. The renderer refuses a notice that shows the score, uses an audience's forbidden words or gives context as a reason (DESIGN §11.3).
- **The draft is an outline.** It inserts the generated tables and leaves the narrative, the abstract and the §10.8 branch to the authors.
- **Decisions at the end of Phase 7:** D109–D112 (proposed).

## Phase 6 (13 October 2026): code built, tested on synthetic data only

Built: `evaluate.py` (C13: correctness, consistency, faithfulness, peer vs global baseline, use (a) stress test, fairness), `stability.py` (seeds, NPI-bootstrap retrains, binomial perturbations on a 2,000-provider subsample), `sensitivity.py` (S1–S12, each in its own scratch project), `eval_report.py` (table shells T1–T4 plus E5, E8, E9, E10), and `scenario-fraud phase6 evaluate | stability | sensitivity | report`.

- **Checks on synthetic data.** Chance levels are exact per answer key. For additive models (LR, EBM without pairs) B, C, D and E agree exactly, as they should. Peer cells without region (S10) reproduce the main run when no Census-division cell exists.
- **Deletion metric caveat (D104).** The normalised deletion area (D86) has no bound when a provider's contributions have mixed signs and its score sits close to its baseline. On synthetic data the means were strongly negative, so tables report medians.
- **Decisions before Phase 6 runs on real data:** D101–D108 (proposed). The report names the DESIGN §10.8 row the numbers are consistent with (rule D108); the human decides the branch.

## Phase 5 (13 October 2026): code built, tested on synthetic data only

Built: `injection.py` (C6: plan, five scheme transforms on harmonised rows, summary-total updates, re-suppression, re-featurising each injected copy with injected providers excluded from peer statistics), `inject_models.py` (injection models (use b), stress test (use a), attribution of injected providers, intensity calibration), `g5_report.py`, and `scenario-fraud phase5 calibrate | build | train | stress | explain | report`.

- **Checks pass on synthetic data.** The G5 report checks re-suppression, that only planned rows are edited, and that peer statistics and cell sizes exclude injected providers (checklist L2/L5).
- **Calibration matters.** On synthetic data the plan's starting intensities did not all land near the 95th percentile, and DME mill plateaued below it. Real-data calibration decides the grid at G5.
- **Decisions for G5:** the intensity grid, the answer key (DME mill counts for either DME group) and D90–D100 (proposed).

## Phase 4 (13 October 2026): code built, tested on synthetic data only

Built: `attribution/background.py` (C9), `attribution/coalition.py` (C10: methods B, C, E and deletion curves from one exact 256-subset coalition table), `attribution/summed_shap.py` (C11, method A), `attribution/glassbox.py` (C12, method D), `explained_set.py`, `attribute.py`, `bench.py`, `g4_report.py`, and `scenario-fraud phase4 explained-set | explain | diagnostics | bench | report`.

- **EXPERIMENT_PLAN §4 unit tests pass.** Additivity holds to 1e-6. A planted pairwise interaction is recovered exactly. B equals D for LR and EBM, and A equals B for depth-1 trees. The G4 report re-checks additivity on the stored attributions.
- **Library catch (D87).** shap's default masker quietly subsamples the background to 100 rows, which breaks method A's additivity. The code passes a full-size masker.
- **Decisions for G4:** |B| and the explained-set size from the benchmark, plus D82–D89 (proposed).
- **Tests only, no results.** Benchmark numbers in tests describe synthetic data and small models.

## Phase 3 (13 October 2026): code built, tested on synthetic data only

Built: `modelmatrix.py`, `models.py` (LR with splines, EBM, GA²M, XGBoost and LightGBM, each GBM with and without monotone constraints), `train.py` (tuning, final fits, isotonic calibration, out-of-fold scores, showcase scoring), `metrics.py` (precision/recall/lift@k, PR-AUC, ROC-AUC, Brier, reliability, NPI bootstrap, Holm), `g3_report.py`, and `scenario-fraud phase3 train | score-showcase | report`.

- **Tests only, no results.** All models were trained on synthetic data in pytest temp directories; no number describes Medicare providers.
- **Training refuses to start before the G2 freeze** (D73): G2 must be decided on real data first.
- **Decisions for G2:** D73–D81 (proposed), and D59 (final fit vs calibration; default implemented as D79).
- **Compute is unmeasured at real scale.** Tuning is 60 trials × 7 families × 5 folds on ≈ 0.5–0.7 M subsampled rows; EBM and GA²M dominate. Time one fold before the full run.

## Phase 2 (13 October 2026): code built, tested on synthetic data only

The following are built, reviewed and fixed: C4 `cohort.py`, C5 `features.py` + `groups.py`, C7 `splits.py`, the QA report (`qa.py`, including the leakage checklist), the G2 report (`g2_report.py`), and `scenario-fraud phase2 cohort | features | folds | qa | report`.

- **Tests only, no results.** Every test runs on synthetic files from `tests/synth.py` inside pytest temp directories. No number from them describes Medicare providers.
- **No real artefacts yet.** The real cohort, features and folds can only be built after Phase 1 data exist.
- **Gate G2 not reached.** It needs real data and the human decisions listed by `phase2 report`:
  - the §17 rows D19–D70 marked proposed or open;
  - the high-risk DME list;
  - pruning;
  - peer granularity (§19 Q2);
  - the final-fit and calibration rows (D59);
  - CX7 / NPPES;
  - the open §19 questions.

## What exists now

| Item | Where | State |
|---|---|---|
| Layout (HANDOFF §4), `.gitignore` excluding `data/` | repo root | done |
| Environment | `.venv`, `pyproject.toml`, `requirements.lock` (136 pinned packages) | Python 3.13; numpy, pandas, pyarrow, polars, duckdb, scikit-learn, xgboost, lightgbm, interpret-core (EBM), shap, optuna, scipy and matplotlib all install and import |
| C1 acquisition & manifest | `src/scenario_fraud/acquire.py`, `configs/sources.yaml` | register manual files, fetch URLs, data.gov listing (route A), Wayback CDX + `id_` fetch, SHA-256 manifest, verify, header dumps |
| C2 schema harmoniser | `src/scenario_fraud/harmonise.py`, `configs/column_maps/_schema.yaml`, `_templates.yaml` | strict maps tied to a file checksum; fails on unmapped, stale, TODO, missing-required, uncastable values and bad NPIs; drafts maps from real headers |
| C3 LEIE labeller | `src/scenario_fraud/labels.py`, `configs/labels.yaml` | union of current + supplements + snapshots, authority tiers, label rule (§5.4), horizon flags, coverage gaps, reinstatement-loss profile |
| C16 CLI + run logs | `src/scenario_fraud/cli.py`, `runlog.py` | `scenario-fraud phase1 …`; every run logs git commit, config hashes and package versions to `outputs/phase1/runs/` |
| G1 report generator | `src/scenario_fraud/g1_report.py` | writes `outputs/phase1/G1_report.md` + `.json`; each section says when its inputs are missing |
| Tests | `tests/` | 27 pass: C1 (9), C2 (8), C3 (8), end-to-end CLI (2). Fixtures are tiny synthetic CSVs that only exercise the code |

Run the tests with `.venv/bin/python -m pytest`.

## What was executed, and what happened

| Step | Command | Result |
|---|---|---|
| Route A (data.gov catalogue) | `phase1 datagov-list medicare-physician-other-practitioners-by-provider-and-service` | **Blocked**: the sandbox proxy refuses `catalog.data.gov` (403 on CONNECT) |
| LEIE full file | `phase1 fetch … --url https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv` | **Blocked**: `oig.hhs.gov` refused by the proxy |
| Internet Archive listing | `phase1 wayback-list` | **Blocked**: `web.archive.org` refused by the proxy |
| data.cms.gov (route B) | probe only | **Blocked**: `data.cms.gov` refused by the proxy |
| G1 report | `phase1 report` | Generated; every data section reads "not available" |

These 403s come from **this cloud environment's network policy**, not from the sites' bot protection (DESIGN §5.2). Nothing was downloaded, and no data was mocked in place of real files.

## Not done yet, and why

- **No raw files, header dumps, column maps or labels.** C2 maps must be written from the real CSV headers (HANDOFF §7). The `bene_cc_*` names and DMEPOS 2024 stay unverified.
- **LEIE layout and `EXCLTYPE` encoding are not verified.** `configs/labels.yaml` holds best-knowledge values marked `verified: false`. The labeller stops on any `EXCLTYPE` it doesn't know, so the real file forces a check against the OIG record-layout PDF.
- **DESIGN §17 and §18 are unchanged.** No claim could be verified or falsified, and there were no design deviations.

## To unblock: pick one

1. **Allow the hosts in this environment's network settings:** `oig.hhs.gov`, `www.oig.hhs.gov`, `catalog.data.gov`, `data.cms.gov`, `archive.org`, `web.archive.org`, plus `download.cms.gov` if NPPES is wanted. Then run the runbook below here.
   Caveats: the container has **≈ 30 GB of free disk, 4 CPUs and 15 GB RAM, and it is ephemeral**. `data/` is git-ignored, so downloads disappear when the session ends. Seven years of Part B by-Provider-and-Service CSVs ("several GB per year") may not fit alongside their Parquet copies.
2. **Run Phase 1 on a workstation** (recommended for the full download). Clone, create the venv from `requirements.lock`, then follow the runbook. Files downloaded in a browser are registered with `phase1 register` (route C).

## Runbook once access exists

```bash
S=.venv/bin/scenario-fraud
# LEIE: current file, record-layout PDF, monthly supplements, archive snapshots (Q5 approval first)
$S phase1 fetch --source leie --year 2026 --kind current --route oig --as-of <today> --url <link from download page>
$S phase1 register --source leie --year 2026 --kind record_layout <layout.pdf>
$S phase1 register --source leie --year 2025 --kind supplement_rein --as-of 2025-01 <2501REIN.csv>   # each month
$S phase1 wayback-list && $S phase1 wayback-fetch <timestamp> ...
# CMS: list per-year distributions, then fetch (or register browser downloads); data + dictionary + methodology
$S phase1 datagov-list <slug from configs/sources.yaml>
$S phase1 fetch --source cms_ps --year 2023 --kind data --route datagov --url <distribution url>
$S phase1 verify && $S phase1 headers
$S phase1 draft-colmap --dataset cms_ps --year 2023   # then review every line and set reviewed: true
$S phase1 harmonise && $S phase1 labels && $S phase1 report   # -> outputs/phase1/G1_report.md, then stop at G1
```

## HANDOFF §5 questions

| # | Status |
|---|---|
| Q1 git + layout | Taken as yes from the request to apply the handoff. It was already a git repo; layout and `.gitignore` are created. |
| Q2 Python | This container: Python 3.13 venv, all packages installed. Confirm the version for the machine that will run the full pipeline. |
| Q3 compute | Measured here: 4 CPU, 15 GB RAM, ~30 GB free disk, ephemeral. Budget for EBM and coalition tables still open. |
| Q4 CMS download route | **Open.** All routes blocked from this container (see above). |
| Q5 Internet Archive snapshots | **Open.** Approval needed, and the host is blocked here. |
| Q6 years 2018–2024, Part D out | Assumed as in DESIGN D12/D17 (`configs/sources.yaml`). Please confirm. |
| Q7 DESIGN §19 open questions | **Open.** Needed before the protocol freeze at G2, not for Phase 1. |

## Interpretations to confirm at G1

1. **Excluded under a non-tier authority → rows dropped**, not negatives (`other_excluded_policy: drop`). This follows §5.4: "never-excluded providers are unlabelled negatives". A `negative` option exists for a sensitivity run.
2. **`1128(b)(1)` not split into (A)/(B)** in the file: an unsplit code is held out of the primary tier until decided (`labels.yaml` comment).
3. **G1 counts are pre-cohort** (all by-Provider NPI-years). A second column applies the §7.2 ≥ 50 non-drug-services rule wherever `cms_ps` is harmonised. The cohort and peer cells (C4) are Phase 2.
