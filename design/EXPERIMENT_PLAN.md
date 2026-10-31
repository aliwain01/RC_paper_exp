# Experiment Plan and Evaluation Protocol — v1.1

Companion to [DESIGN_DOCUMENT.md](DESIGN_DOCUMENT.md) §7–10 · [Fig. 7](figures/fig07_scheme_injection.png) · [Fig. 8](figures/fig08_evaluation_framework.png).

Freeze this protocol before any test result is seen. Changes after Phase 3 starts are logged with their reason.

---

## 1. Experiment matrix

| ID | Experiment | Models | Data | Answers | Components |
|---|---|---|---|---|---|
| E1 | Detection benchmark | LR, EBM, GA²M, GBM (± monotone) | Real; cross-fitted; primary + rolling origins | RQ1 | C7, C8, C13 |
| E2 | Attribution computation | all | Real explained set; peer and global baselines | input to E3–E8 | C9–C12 |
| E3 | Consistency | GBM (A, B, C, E) vs glass box (D) | Real explained set | RQ2 | C13 |
| E4 | Correctness | injection models (one per family) | Injected, use (b) | RQ3 | C6, C8, C10–C13 |
| E5 | Detection stress test | LEIE-trained models | Injected, use (a) | context for RQ1/RQ3 | C6, C13 |
| E6 | Faithfulness | all model × method pairs | Real explained set (from coalition tables) | RQ4 | C10, C13 |
| E7 | Stability | all model × method pairs | Real; 10 seeds + 10 bootstraps; 2,000-provider subsample for B/C/E | RQ4 | C8–C13 |
| E8 | Baseline comparison | GBM; A, B, E | Real explained set | RQ5 | C9–C11 |
| E9 | Sensitivity suite | best GBM + best glass box | Real (+ injected where relevant) | robustness | all |
| E10 | Fairness profile | best GBM + best glass box | Real test years | limitations / oversight | C13 |

---

## 2. Splits (component C7)

- **Provider folds:** NPIs partitioned once into 5 folds, stratified by specialty and ever-positive status (any tier), fixed seed.
- **Primary design:** for each fold *k*, train on NPIs ∉ *k*, years 2018–2021, and score NPIs ∈ *k*, years 2022–2023. Concatenating the 5 test folds gives out-of-fold scores for every provider-year in 2022–2023.
- **Tuning / calibration:** inside each training set, hold out 20 % of training NPIs; train on 2018–2020, validate on their 2021 rows.
- **Rolling origins:** O1 (2018–20 → 2021), O2 (2018–21 → 2022), O3 (2018–22 → 2023), each cross-fitted the same way.
- **Showcase:** 2024, scored by the primary models (no accuracy claims).
- **Negative subsampling (training only):** all positives + a random 15 % of negatives (fixed seed, identical for every model family), weighted 1/0.15. Test data are never subsampled.

### Implementation notes (Phase 2, C7)

How `src/scenario_fraud/splits.py` and `configs/experiment.yaml: splits` resolve what this section leaves open. Each choice is a DESIGN §17 row; D54–D58 and D60–D63 are proposed (confirm at G2), and D59 is open. Built and tested on synthetic data only.

- **Folds** (D54–D56). Folds cover the NPIs with at least one in-cohort year. The specialty stratum is the NPI's modal `prvdr_type` over those years; a tie goes to the latest tied year. "Ever-positive" means an exclusion under any `labels.yaml` tier, at any date. Strata are laid end to end and NPIs are ordered by a seeded hash, so fold sizes, positives per fold and every stratum's count per fold differ by at most 1.
- **Tuning hold-out** (D57). H_k is exactly ⌊0.2 N⌋ of the N NPIs outside fold k, stratified the same way. It differs per fold and is the same for every design.
- **Roles** (D58), for train years [a, b] and test years [c, e]:
  - `test`: fold-k NPIs in years [c, e].
  - `tune_valid`: H_k NPIs in year b ("validate on their 2021 rows").
  - `tune_train`: the other non-k NPIs in years [a, b − 1] ("train on 2018–2020").
  - `train`: the remaining non-k rows of years [a, b].
- **Final model and calibration (open, D59).** The training set is every non-k row of the train years. This section says the final model trains on all of it, while DESIGN §7.8 says "calibration on the validation slice". The two cannot both hold without calibrating on the final model's own training rows. Phase 3 must choose one of these options, and the choice is listed as a G2 decision:
  - (i) the final model on all training rows, with a calibrator fitted on the tuning model's held-out scores or on cross-fitted scores;
  - (ii) the final model on `train` + `tune_train`, with the calibrator fitted on its `tune_valid` scores.
- **Rolling origins and 2020** (D60). O1's last train year is 2020, so O1 validates on the flagged COVID year. For sensitivity S4, `split_rows(..., exclude_years=[2020])` gives 2020 rows the role `none` and validates on the last remaining train year (O1: 2019).
- **Showcase** (D61). Fold-k NPIs' 2024 rows are scored by their own fold's primary-design model.
- **Negative subsampling** (D63). Each negative (npi, year) is kept independently when its seeded hash falls below the rate, so the sample is about 15 % rather than an exact count. Kept negatives get weight 1/rate. The sample is the same for every model family, fold and design, and the 15 % sample is nested in the 30 % one (S12).

---

## 3. Detection metrics (E1)

For a test year with *N* scored provider-years:

- **precision@k** = positives in top ⌈k·N⌉ / ⌈k·N⌉, for k ∈ {0.1, 0.5, 1, 2} %.
- **recall@k** = positives in top ⌈k·N⌉ / total positives.
- **lift@k** = precision@k / base rate.
- **PR-AUC** (primary), **ROC-AUC** (secondary).
- **Calibration:** Brier score; reliability curve over 10 quantile bins within the top 5 %.

**Comparison.**

- Δ = GBM − best glass box, with 95 % CI from a paired bootstrap over NPIs (resample NPIs with all their rows; 1,000 replicates).
- Holm correction across k.
- H1 is supported if both Δ PR-AUC and Δ precision@1 % have CIs > 0.
- **Matched tuning budget:** 30 Optuna TPE trials per family (60 planned; reduced at the Phase 3 timing run, DESIGN §17 D120); objective = validation PR-AUC.

**Power note.** With prevalence plausibly ≈ 0.02–0.05 % and ≈ 1.1 M providers per test year, there are on the order of 200–550 positive provider-years per year. These numbers are assumptions until Phase 1. Precision@0.1 % rests on ≈ 1,100 flagged providers per year. If Phase 1 counts make CIs uninformative, RQ1 is reported descriptively (design doc §16).

### Implementation notes (Phase 3, C8 and metrics)

DESIGN §17 D73–D81 hold the reasons; parameters live in `configs/models.yaml` and `experiment.yaml: evaluation`.

- **Families:** `lr`, `ebm`, `ga2m`, `xgb`, `xgb_mono`, `lgbm`, `lgbm_mono`. "GBM" in T1 and Δ is the better of XGBoost and LightGBM by mean validation average precision; "GBM (monotone)" likewise among the monotone variants; the best glass box among LR, EBM and GA²M (D74).
- **Tuning:** 30 Optuna TPE trials per family and fold on the primary design (D120); rolling origins reuse those parameters with one seed (D75). Tuning stops if `tune_valid` has fewer than 5 positives (D80).
- **Final fit and calibration:** all training rows; isotonic calibrator from the tuning model's validation margins (D79; the D59 choice is still open).
- **Metrics:** providers are ranked by their within-fold percentile of the margin (D125), so the five fold models' different margin scales cannot crowd each other out; pooled precision/recall@k sum per-year counts; ties broken by a fixed hash; Holm adjusts two-sided bootstrap p-values; T1 CIs are NPI-bootstrap percentile intervals of the seed-0 models (D81).
- **Guard:** training refuses to start before the G2 freeze record exists, or if the frozen parameters changed (D73).
- **Resume** (D120): a (family, fold) whose `tuning.json` records the same trial budget and seed count is skipped; an interrupted tuning replays its logged trials (`progress/trials.jsonl`) through the same seeded sampler, so the result equals an uninterrupted run; saved seed models are loaded. All three reuse work only under the same settings fingerprint (the family's models.yaml section, the pos_weight space, tuning seed, final fit, calibration source and design columns); a changed setting retrains. `phase3 train --no-resume` retrains from scratch. Progress lines go to `outputs/phase3/train_progress.log`.

---

## 4. Attribution protocol (E2)

| Setting | Value |
|---|---|
| Output scale | raw margin (log-odds), before calibration |
| Peer background *B* | up to 200 providers from the **same year and peer cell, training folds**; fall back up the peer hierarchy if < 200 |
| Global background | 1,000 training-fold providers of the same year, stratified by specialty |
| Explained set (per test year) | top 2 % by out-of-fold GBM score ∪ stratified random 5,000 ∪ all injected test providers |
| **Coalition table (C10)** | for each provider: 256 subsets *S* of the 8 groups; for each *S*, build \|*B*\| hybrid rows (groups in *S* from the provider, others from each background row), predict, average → *v*(*S*) |
| Method B | exact Shapley over 8 players from *v* |
| Method C | pairwise Shapley interaction index *I*_GH from *v*; M_GH = *I*_GH/2; M_GG = Φᴮ_G − Σ_{H≠G} M_GH |
| Method E | Δ_G = *v*(*N*) − *v*(*N*∖{G}) (log-odds and probability) |
| Deletion curves | read from *v* along each method's ranking |
| Method A (C11) | `shap.TreeExplainer(model, data=B, feature_perturbation="interventional", model_output="raw")`, run per peer cell; group sums. Path-dependent variant on a 2,000-provider subsample for diagnostics |
| Method D (C12) | LR: β·(x − mean_B x) per term; EBM: term scores minus their mean over *B*; cross-group pair terms stored as cells and as ½-splits |
| Feature-level interaction cross-check | TreeSHAP interaction values (path-dependent) on 2,000 providers |
| Timing benchmark | Phase 4, before full runs: measure predictions/s; set \|*B*\| ∈ {100, 200} and explained-set size to fit the compute budget |

**Unit tests (C10):**

1. **Additivity:** \|Σ_G Φᴮ_G − (*v*(*N*) − *v*(∅))\| < 1e-6, and Σ M = the same.
2. On a synthetic model *f* = additive + one known pairwise term, B and C recover the planted values exactly.
3. On a linear model, Φᴮ equals method D's LR group sums (up to background averaging).

**Unit test (C11):** \|Σ φ − (*f*(*x*) − mean_B *f*)\| < 1e-4.

### Implementation notes (Phase 4, C9–C12)

DESIGN §17 D82–D89; parameters in `configs/attribution.yaml`.

- **Backgrounds:** a peer cell is used while it has ≥ 50 training-fold providers; up to 200 are drawn in a seeded order, else the next level up (D82–D83). The global baseline is 1,000 providers stratified by specialty (D84); at G4 it was cut to 200 and run for the best GBM only (D124). Memberships are stored for provenance. Attribution is computed in resumable chunks of 1,000 providers (D123).
- **Explained set:** top 2 % of the best GBM's seed-0 margins ∪ a stratified random 5,000 per test year ∪ injected test providers (D85).
- **Methods:** coalition tables for every family (B, C, E, deletion), A for tree families, D for glass boxes, both baselines (D89). Method A needs a full-size `Independent` masker (D87).
- **Unit tests:** all three tests above pass to 1e-6 or better (`tests/test_attribution_engine.py`). D equals B exactly for LR and EBM without pairs, and A equals B for depth-1 trees. `phase4 report` re-checks additivity on the stored attributions.
- **Deletion:** groups removed by decreasing signed attribution; trapezoid area / 8; defined for providers above their baseline (D86).

---

## 5. Explanation metrics

aᵢ = 7-vector of scheme-group attributions for provider *i* (context excluded). Rankings use positive contributions unless stated.

### 5.1 Correctness (E4)

- **Top-1** = mean 1[argmax aᵢ = injected scheme]. **Top-3** = mean 1[scheme ∈ top-3]. **MRR** = mean 1/rank.
- **Compound:** both-in-top-3 rate. For method C, **pair hit** = 1[argmax over pairs of (M_GH + M_HG) = injected pair].
- **Spill-over profile:** mean attribution share on the expected secondary group (descriptive).
- **Distractors:** injected providers who already sit above the peer 95th percentile in a *different* group. Top-1 is reported separately for them.
- **Chance:** top-1 1/7, top-3 3/7, pair hit 1/21.
- Breakdown: scheme × intensity × specialty × method × model family.

### 5.2 Consistency (E3)

For GBM method *M* (vector *a*) and glass-box D (vector *b*):

- **sign agreement** over groups with max(|aᵍ|, |bᵍ|) > 0.05;
- **Spearman ρ**;
- **top-3 Jaccard**;
- **share cosine** cos(a/‖a‖₁, b/‖b‖₁);
- **magnitude ratio** ‖a‖₁/‖b‖₁ (descriptive).

Report medians and IQRs by flag agreement at k = 1 % (both / GBM only / glass box only / neither). **Permutation baseline:** the same metrics with *b* taken from a random other provider in the same stratum.

### 5.3 Faithfulness (E6)

For method *M*'s order π: f₀ = *v*(*N*), f₁ = *v*(*N*∖{π₁}), …, f₈ = *v*(∅). Score = area under (fⱼ − f₈)/(f₀ − f₈) over j; **lower is more faithful.** Baselines: mean of 50 random orders, and the reverse order.

### 5.4 Stability (E7)

10 seeds and 10 NPI-bootstrap retrains:

- top-1 agreement with the reference run;
- mean pairwise Kendall τ;
- **perturbation:** binomial resampling of rate numerators given denominators, recompute features → top-1 agreement.

### Implementation notes (Phase 6, C13: §5)

DESIGN §17 D101–D105; parameters in `configs/evaluation.yaml`; code in `evaluate.py` and `stability.py`. Built and tested on synthetic data only.

- **Vectors and ranks** (D101): the 7 scheme groups; a group is found only with a positive contribution; ties by group order. Method C ranks by its diagonal and its pair hit uses M_GH + M_HG; GA²M's D uses its pair cells. Chance levels are exact for each answer key, so DME mill has chance top-1 2/7 and top-3 5/7.
- **T2** (D102): single schemes at medium intensity; distractor top-1 over medium single-scheme distractors; compound columns over the pairs.
- **Consistency** (D103): flags at k = 1 % from each family's seed-0 out-of-fold margins; the permutation partner is another provider of the same year and flag stratum.
- **Faithfulness** (D104): stored deletion scores plus method C's (diagonal order), over all explained providers above their baseline and over the top 2 %. Reported as medians: the normalised area is unbounded when contributions have mixed signs and f(x) is close to its baseline.
- **Stability** (D105): one 2,000-provider subsample explained in every run (10 seeds, 10 NPI-bootstrap retrains, 10 perturbations) with the same backgrounds. Perturbation redraws the numerators of the count-based (shrunk) rates and recomputes their z from the persisted peer statistics.

---

## 6. Scheme injection (E4, E5; component C6)

**Pools.**

- Clean providers = never excluded under any tier, in the relevant folds, for whom the scheme is applicable:
  - Upcoding: ≥ 50 visible established-visit services;
  - DME: has DME-S rows, or belongs to a specialty with ≥ 20 % DME referrers.
- Injected providers are excluded from per-year peer-statistic computation.

**Use (b) injection models.**

- Labels: injected = 1, clean non-injected = 0; LEIE positives (any tier) removed.
- Training: injected training-fold providers, 2018–2021, at ≈ 0.5 % of training provider-years, split evenly across schemes and pairs.
- Test: injected test-fold providers, 2022–2023.
- One injection model per family (LR, EBM, GA²M, GBM), same hyperparameters as the LEIE models.

**Use (a):** LEIE-trained models score the same injected test providers; report their percentile ranks.

| Parameter | Values |
|---|---|
| Single schemes | Upcoding · Overutilization · Atypical services · Charge inflation · DME mill referrals |
| Compound pairs | Upcoding+Overutilization · Overutilization+Atypical · Upcoding+Charge · DME mill+Overutilization |
| Intensity | low / medium / high |
| Test injections | 300 per scheme × intensity; 300 per pair (medium) |
| Seeds | 5 independent draws |

**Transforms (raw rows; summary totals updated consistently; suppression re-applied):**

| Scheme | P&S / DME-S row edit | Summary-file update | Low / Med / High |
|---|---|---|---|
| Upcoding | move share *p* of 99212–13 services (and proportional beneficiary counts, capped by Prov `Tot_Benes`) to 99214–15; new rows take the provider's per-code averages, else peer-cell medians | Prov totals of allowed / paid / standardised amounts recomputed | p = 0.15 / 0.30 / 0.50 |
| Overutilization | multiply `Tot_Srvcs` and `Tot_Bene_Day_Srvcs` by *m* on existing non-drug rows (beneficiaries unchanged) | Prov `Tot_Srvcs` and amount totals | m = 1.3 / 1.6 / 2.0 |
| Atypical services | add rows for codes billed by < 1 % of the peer cell, worth share *q* of payment, at those codes' national median prices; beneficiaries ≥ 11 so rows survive suppression | Prov `Tot_HCPCS_Cds`, `Tot_Srvcs`, totals | q = 0.10 / 0.20 / 0.35 |
| Charge inflation | multiply `Avg_Sbmtd_Chrg` by *c*; allowed and paid unchanged | Prov submitted-charge total | c = 1.3 / 1.7 / 2.2 |
| DME mill referrals | add DME-S rows in high-risk categories worth *d* × the peer-cell median DME spend, with `Tot_Suplrs` = 1–2 | DME-P beneficiary and payment totals | d = 2 / 4 / 8 |

Intensities are starting values, calibrated in Phase 5 so that "medium" places the primary feature near the peer 95th percentile. The calibration is reported. **Accepted at G5 (DESIGN §17 D129):** upcoding 0.70 / 0.85 / 1.0; overutilization 1.8 / 3.0 / 4.0; atypical services 0.01 / 0.05 / 0.10; charge inflation 1.5 / 2.0 / 2.6; DME mill 4 / 16 / 32.

### Implementation notes (Phase 5, C6)

DESIGN §17 D90–D100; parameters in `configs/injection.yaml`.

- **Answer key:** each scheme maps to its group. DME mill counts as correct for either DME group, so its chance top-1 is 2/7. Expected spill-over groups are listed per scheme (D90).
- **Transforms:** upcoding splits moved services ½/½ between 99214 and 99215 (D91). Atypical services add 3 rare codes at national median prices (D93). DME mill adds 2 high-risk codes with 1–2 suppliers (D94). Summary totals change by the visible change; edited rows are re-suppressed (D92).
- **Featurising:** each draw's injected copy is re-featurised with injected providers excluded from peer cells and statistics. The G5 report checks this against checklist L2 and L5 (D95).
- **Plan and distractors:** each provider-year is injected at most once per draw, stratified by specialty (D96). Distractors are defined on the clean features (D97).
- **Calibration:** `phase5 calibrate` reports the peer percentile of each scheme's primary feature over an intensity grid, and the human accepts the grid at G5 (D98).
- **Resume of injection models and their attributions** (D130): `phase5 train` skips a (family, fold) whose `done.json` records the same fingerprint (the draw's build, the reused parameters, the model and split settings); `phase5 explain` writes chunked pieces as in Phase 4 (D123) and skips pieces already written under the same settings, injection model and providers. `--no-resume` redoes both. Progress lines go to `outputs/phase5/progress.log`.
- **Unobservable injections** (D131): an injected provider-year left without visible service-level rows after re-suppression is dropped from the injected copy and counted in the G5 report.
- **Memory and resume** (D127): the year's files are scanned lazily; `phase5 build` skips years finished under the same settings and plan, takes `--year`, and `scripts/run_phase5_build.sh` builds each year in its own process.
- **Models:** injection models reuse the LEIE models' tuned parameters, and LEIE positives are removed. Use (a) is a percentile rank among clean test rows (D99). Injected providers are explained with the Phase 4 engine (D100).

---

## 7. Sensitivity suite (E9)

| ID | Variant |
|---|---|
| S1 | W ∈ {1, 2, 5} |
| S2 | Broad label tier |
| S3 | Individuals only |
| S4 | Exclude 2020 |
| S5 | Global vs peer baseline (also E8) |
| S6 | Grouping: correlation clusters (hierarchical on \|ρ\|, cut at 7) |
| S7 | Grouping: CH2 → Overutilization; PM1 → context |
| S8 | Context ablation |
| S9 | Monotone constraints on vs off |
| S10 | Peer cells with vs without region |
| S11 | **Prior-work-compatible label tier** (a1, a2, a3, b4, b7, c3G), for comparability with FAU results |
| S12 | Negative subsampling 15 % vs 30 % (GBM and LR only) |
| S13 | Primary tier + the unsplit 1128(b)(1) (G1 decision, DESIGN §17 D117) |

### Implementation notes (Phase 6, C13: §7)

DESIGN §17 D106–D107; `configs/evaluation.yaml: sensitivity`; code in `sensitivity.py`.

- **Resume** (D132): stability reuses runs written under the same fingerprint; a finished sensitivity variant is skipped and an interrupted one continues in its scratch project. Progress lines go to `outputs/phase6/progress.log`.
- Each variant runs in a scratch project (`data/processed/sensitivity/{id}/`) holding the main configs with its overrides, the main data linked in place, and a freeze record derived from the main G2 freeze. S1 is run as three variants (W = 1, 2, 5).
- Retrained variants reuse the main tuned hyperparameters with one seed. They explain a seeded 2,000-provider subsample of the main explained set, with the best GBM and the best glass box (S12: the best GBM and LR).
- Injected providers are re-explained where a variant changes the explanations (S6, S7, S8; draw 0). S5 and S9 are read from the main evaluation.
- S6 clusters the scheme-group features on 1 − |Spearman ρ| (average linkage, 7 clusters). Each cluster is named after the scheme group it overlaps most, so the answer key still applies.
- Reported per variant: detection of the variant's models next to the same main models; consistency; faithfulness; top-1 agreement and Kendall τ with the main explanations of the same providers; correctness where re-explained.

---

## 8. Fairness profile (E10)

At k ∈ {0.5, 1} %, by specialty, entity type, Census division and RUCA class: flag rate, precision, and flag rate ÷ positive rate. Descriptive only, interpreted in light of LEIE enforcement patterns.

**Implementation notes (Phase 6).** Best GBM and best glass box, seed 0. Flags are the top ⌈k·N⌉ of each test year, pooled over the test years (D108). `phase6 report` fills the §9 shells and names the DESIGN §10.8 row the numbers are consistent with, using the D108 rule; the human decides the branch.

---

## 9. Result table shells

**T1 Detection (pooled 2022–23, cross-fitted, 95 % CI)**

| Model | PR-AUC | ROC-AUC | P@0.1 % | P@0.5 % | P@1 % | P@2 % | R@1 % | Brier |
|---|---|---|---|---|---|---|---|---|
| LR | | | | | | | | |
| EBM | | | | | | | | |
| GA²M | | | | | | | | |
| GBM | | | | | | | | |
| GBM (monotone) | | | | | | | | |
| Δ GBM − best glass box | | | | | | | | |

**T2 Correctness (medium intensity; injection models)**

| Model · method | Top-1 | Top-3 | MRR | Distractor top-1 | Compound both-in-top-3 | Pair hit (C) |
|---|---|---|---|---|---|---|
| GBM · A | | | | | | — |
| GBM · B | | | | | | — |
| GBM · C | | | | | | |
| GBM · E | | | | | | — |
| GA²M · D | | | | | | |
| EBM · D | | | | | | — |
| LR · D | | | | | | — |
| chance | 0.14 | 0.43 | | 0.14 | | 0.05 |

**T3 Consistency (GBM vs EBM/GA²M, both flag at k = 1 %)**

| GBM method | Sign agree | Spearman ρ | Top-3 Jaccard | Share cos | Permutation baseline (cos) |
|---|---|---|---|---|---|

**T4 Faithfulness and stability**

| Model · method | Deletion ↓ | Random order | Top-1 agree (seeds) | Kendall τ (bootstrap) |
|---|---|---|---|---|

**Implementation notes (Phase 6–7).** `phase6 report` fills these shells (`outputs/phase6/evaluation_report.md`). `phase7 tables` writes T1–T4, the sensitivity table and the fairness table as Markdown, CSV and LaTeX; `phase7 figures` draws the DESIGN §12 figures (DESIGN §17 D112). T4 reports median deletion scores (D104).

---

## 10. Compute and tooling

- **Libraries:** DuckDB / Polars / PyArrow; scikit-learn; InterpretML; XGBoost; LightGBM; Optuna; `shap`; SciPy; Matplotlib.
- **Training:** ≈ 3–4 M training rows before subsampling, ≈ 0.5–0.7 M after; 4 families × 5 folds × 10 seeds plus tuning. EBM/GA²M dominate wall-clock time.
- **Coalition tables:** ≈ 30–40 k explained providers × 256 × \|B\| ≤ 200 ≈ 1.5–2 billion predictions per model per test year. Sized by the Phase 4 timing benchmark; reduce \|B\| or the explained set if needed. Stability runs (×20) use a 2,000-provider subsample.
- **Reproducibility:** manifest (route, date, size, SHA-256); YAML configs for column maps, groups, features and parameters; fixed seeds; environment lockfile; run logs.
