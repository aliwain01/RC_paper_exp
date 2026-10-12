# Scenario-Level Explanations for Health-Care Fraud Detection on Public Medicare Data

**Design document · v1.1 · 11 October 2026**
Status: proposal, verified against primary sources where possible (see §18). Nothing has been built yet; this is the specification the build will follow.
Companion files: [FEATURE_CATALOG.md](FEATURE_CATALOG.md) · [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) · figures in [figures/](figures/)

**Changes from v1.0**, after a verification pass against CMS data dictionaries, OIG pages and the cited literature:

- Peer statistics are now **label-free and computed per year** (v1.0 contradicted itself by saying "fit on training years" and "year-specific").
- The single 80/20 provider split is replaced by **5-fold provider cross-fitting**. With the low positive rates reported in the literature, a 20 % test pool would leave too few positives.
- Method C (the interaction matrix) is now computed from the **same exact group-level calculation as method B**, at the group level. This removes its dependence on path-dependent TreeSHAP and makes it consistent with the peer baseline.
- The **reinstatement recovery plan is corrected.** OIG archives only about the last 12 months of supplement files, so historical recovery relies on Internet Archive snapshots, which have gaps.
- DMEPOS 2024 now appears to be published. The DMEPOS **by Referring Provider** summary file is added for distinct-beneficiary counts.
- The effects of **CMS small-cell suppression** on each feature are made explicit. One feature that can't be computed from public data was removed.
- Medicare fee-for-service actors and terminology are corrected (UPIC, ADR, MSN, Fraud Prevention System), and the policy framing is narrowed.
- New: §8 **Components to build**, §16 **Threats to validity and research value**, §18 **Claim verification status**.

---

## Contents

1. [Summary](#1-summary)
2. [Terminology](#2-terminology)
3. [Background and motivation](#3-background-and-motivation)
4. [Problem statement, hypotheses and contributions](#4-problem-statement-hypotheses-and-contributions)
5. [Data](#5-data)
6. [Architecture overview](#6-architecture-overview)
7. [Stage-by-stage design](#7-stage-by-stage-design)
8. [Components to build](#8-components-to-build)
9. [Scenario-level attribution](#9-scenario-level-attribution)
10. [Evaluation](#10-evaluation)
11. [Audiences, outputs and the decision lifecycle](#11-audiences-outputs-and-the-decision-lifecycle)
12. [Expected outcomes and deliverables](#12-expected-outcomes-and-deliverables)
13. [Limitations](#13-limitations)
14. [Risks and mitigations](#14-risks-and-mitigations)
15. [Work plan](#15-work-plan)
16. [Threats to validity and research value](#16-threats-to-validity-and-research-value)
17. [Decision log](#17-decision-log)
18. [Claim verification status](#18-claim-verification-status)
19. [Open questions](#19-open-questions)
20. [References](#20-references)

---

## 1. Summary

Fraud models in health care decide whom to audit and, downstream, whose payments to hold. Every such decision has to be explainable to the analyst acting on it, to the provider it affects, and to whoever audits the process later. Feature-level explanations are not usable by these readers. They need explanations in terms of **how fraud is committed**: upcoding, overutilization, unusual referral patterns.

This project builds and evaluates **scenario-level attribution**. Model inputs are grouped into domain-defined **scheme groups**, and each prediction is explained by each scheme's contribution. The output is a short list of **scenario indicators** per provider.

Design constraints:

1. **Real, public data only.** CMS Medicare fee-for-service provider-level data (Part B practitioner services and DMEPOS referrals), labelled with the HHS-OIG List of Excluded Individuals and Entities (LEIE). No claim-level real data is available to us.
2. **Explanations must be tested, not asserted.** A semi-synthetic experiment injects documented schemes into real provider profiles, so explanations can be scored against a known answer.
3. **The black box has to earn its place.** Gradient-boosted models (GBMs) are compared with strong glass-box models (logistic regression, Explainable Boosting Machine). Their scenario attributions are compared for direction and magnitude.

Five attribution methods are compared on four criteria (correctness, consistency, faithfulness, stability). Three of them (group Shapley, the group interaction matrix and the peer counterfactual) are computed from one shared table of exact group-coalition values. Outputs are designed per audience and linked by an auditable decision record.

---

## 2. Terminology

The internal term "reason codes" is **not** used. In US health care it also refers to standardised claim-adjustment codes (CARC/RARC).

| Term | Meaning |
|---|---|
| **Scheme group** | A fixed set of features describing one way fraud is committed (e.g. Upcoding). Seven groups plus a context group. |
| **Scenario-level attribution** | Any method assigning part of a prediction to each scheme group. |
| **Scenario indicator** | What a reader sees: a scheme group, its contribution and supporting facts. |
| **Compound scheme** | Two scheme groups raising risk together by more than their separate effects (a group-level interaction). |
| **Context group** | Provider traits (specialty, entity type, region, urban–rural, size, tenure, data coverage). Model inputs; never shown as reasons. |
| **Peer group** | Providers with the same specialty and entity type in the same year (refined by region when large enough). Used for feature normalisation and as the attribution baseline. |
| **Coalition table** | For one provider, the model's expected output for all 2⁸ = 256 subsets of groups "kept" (others replaced by peer values). Methods B, C and E and the deletion curves are all computed from it. |
| **Glass box** | A model whose explanation is exact by construction (logistic regression, EBM). |
| **Provider-year** | Unit of analysis: one NPI in one calendar year. |
| **LEIE** | List of Excluded Individuals and Entities (HHS Office of Inspector General). |
| **UPIC / MAC** | Unified Program Integrity Contractor (investigates Medicare FFS fraud for CMS) / Medicare Administrative Contractor (processes claims). |
| **ADR / MSN** | Additional Documentation Request (sent to providers) / Medicare Summary Notice (sent to beneficiaries). |

---

## 3. Background and motivation

### 3.1 Why explainability is not optional

- **Decisions affect livelihoods.** A flag can lead to an audit, a documentation request or prepayment review, and false positives fall unevenly across specialties and regions.
- **Governance expectations exist.** For **commercial insurers**, the NAIC *Model Bulletin on the Use of AI Systems by Insurers* (Dec 2023) sets governance and documentation expectations; adoption varies by state. **Medicare fee-for-service** program integrity is run by CMS and its contractors (MACs, UPICs). CMS has operated a predictive-analytics *Fraud Prevention System* on FFS claims since 2011. Providers have formal appeal rights. The paper should frame explainability as needed for **due process, auditability and investigator usefulness**, not as satisfying one specific law. The authors should check applicability before citing any regulation.
- **Analysts need to know what to look for.** "High risk" gives them nothing to act on. "Upcoding: 78 % of office visits at the highest level vs 22 % peer median" does.

### 3.2 Why scheme-level grouping, and what is new

Feature-level SHAP is precise but unreadable, and correlated features split credit unpredictably. Grouping by scheme matches how fraud is investigated and makes explanations more stable.

**What is not new:** grouping attributions is established. Credit scoring uses adverse-action reasons. The literature has Owen values and groupShapley (Jullum, Redelmeier & Aas, 2021). Explainable ML on Medicare + LEIE data has also been published, e.g. Hancock, Bauder, Wang & Khoshgoftaar (2023), and Shekhar, Leder-Luis & Akoglu (2023) on unsupervised explainable fraud detection.

**What this project contributes:**

- a scheme taxonomy grounded in public data,
- a systematic comparison of group-attribution methods, including an exact group-interaction method,
- **measured** correctness with injected ground truth,
- an explicit glass-box comparison of performance *and* explanations, and
- an audience-aware lifecycle design.

### 3.3 What we learned from the first prototype

The prototype (`synthetic_health_claims (1).csv`, `reason_codes_health_final (2).ipynb`) used a synthetic Kaggle dataset. Its labels turned out to be generated by about five two-variable AND rules: a 17-term logistic scorecard reached AUC 0.873 vs XGBoost 0.878, both measured in this project. Its scenarios therefore couldn't be validated. It also used explanation diversity as a design target, and it had leakage and encoding issues.

Its one transferable finding was that about 41 % of positive attribution on fraud cases came from cross-group interactions. That motivates the interaction-aware method C (§9.2).

---

## 4. Problem statement, hypotheses and contributions

### 4.1 Problem statement

> Given public, provider-level Medicare FFS billing data and exclusion-based labels, can we (a) rank providers for audit with useful precision at realistic audit capacity, and (b) explain each ranking in terms of documented fraud schemes, with explanations that are **correct** (name the true scheme when it is known), **consistent** (agree with an inherently interpretable model), **faithful** (reflect what the model uses) and **stable**? Can those explanations also be delivered to each audience in the decision lifecycle without exposing detection logic or blaming providers for who they are?

### 4.2 Research questions and hypotheses

| # | Research question | Hypothesis / answer form |
|---|---|---|
| RQ1 | Does a GBM outperform the best glass box at audit-relevant capacity? | **H1:** GBM > best of {LR, EBM, GA²M} on PR-AUC and precision@1 %, with paired-bootstrap CIs excluding 0. A null result is reported as such. |
| RQ2 | Do GBM scenario attributions agree with the glass box's exact ones? | **H2a:** among providers flagged by both, median sign agreement ≥ 0.8 and top-3 overlap above the permutation baseline. |
| RQ3 | Do scenario attributions name the true scheme when it is known? | **H2b:** top-1 recovery on injected providers well above chance (1/7). Compound schemes are named by method C more often than they are recovered by additive methods. |
| RQ4 | Which attribution method is most faithful and stable? | Rank methods on deletion curves and retraining stability; no prior hypothesis. |
| RQ5 | Does the peer baseline change explanations versus the global baseline? | **H3:** context attribution falls to near zero, and scheme rankings change for a measurable share of flagged providers. |
| RQ6 | Can one internal record serve all audiences with traceable, audience-safe explanations? | Design contribution with worked examples (§11); not empirically validated. |

### 4.3 Contributions

1. A scheme taxonomy and peer-relative feature set for public Part B + DMEPOS data, with every feature's suppression behaviour documented.
2. A careful LEIE labelling protocol: fraud-related authorities, look-back window, label horizon, reinstatement recovery with quantified gaps.
3. An exact **coalition-table** attribution engine giving group Shapley values, group interaction matrices, peer counterfactuals and deletion curves from one computation.
4. Ground-truth evaluation of explanation correctness via scheme injection into real provider profiles.
5. A glass-box vs black-box comparison of performance and explanations.
6. An audience-aware explanation lifecycle with a provenance chain.

### 4.4 Non-goals

- Claim-level prediction (no public claim-level labels).
- Establishing that any provider committed fraud. Outputs are audit priorities.
- Part D prescribing (deferred).
- Part A institutional claims, Medicare Advantage, Medicaid, commercial insurance (not in these public files).
- Laboratory/ordering-provider schemes other than DME (the Part B practitioner file records the **rendering** provider; referrals are only public for DMEPOS).

---

## 5. Data

### 5.1 Provider-level, not claim-level

CMS does not publish individual claims. It publishes **aggregates**: one row per provider × HCPCS code × place of service × year, with counts and average amounts. The LEIE labels a **person or entity** on a date, not claims. The prediction question is therefore:

> Does this provider-year's billing pattern resemble those of providers later excluded under fraud-related authorities?

Fraud units do target providers, and many schemes show as distribution shifts (e.g. visit-level mix), not single bills. What is lost: encounter-level diagnoses, patients, dates and claim-level explanations (§13).

### 5.2 Sources

| Source | Grain | Years* | Key fields (verified, see §18) | Role |
|---|---|---|---|---|
| **Medicare Physician & Other Practitioners — by Provider and Service** (CMS) | NPI × HCPCS × place of service (F/O) × year | 2013–**2024** | `Rndrng_NPI`, `Rndrng_Prvdr_Ent_Cd`, `Rndrng_Prvdr_State_Abrvtn`, `Rndrng_Prvdr_RUCA`, `Rndrng_Prvdr_Type`, `Rndrng_Prvdr_Mdcr_Prtcptg_Ind`, `HCPCS_Cd`, `HCPCS_Drug_Ind`, `Place_Of_Srvc`, `Tot_Benes`, `Tot_Srvcs`, `Tot_Bene_Day_Srvcs`, `Avg_Sbmtd_Chrg`, `Avg_Mdcr_Alowd_Amt`, `Avg_Mdcr_Pymt_Amt`, `Avg_Mdcr_Stdzd_Amt` | Core scheme features |
| **Medicare Physician & Other Practitioners — by Provider** (CMS) | NPI × year | 2013–2024 | `Tot_HCPCS_Cds`, `Tot_Benes`, `Tot_Srvcs`, totals of charges/payments, `Bene_Avg_Risk_Scre` (average HCC risk score), `Bene_Dual_Cnt`, chronic-condition percentages (`Bene_CC_*`; names vary by year) | Distinct-beneficiary totals, patient mix |
| **Medicare DMEPOS — by Referring Provider and Service** (CMS) | Referring NPI × HCPCS × rental indicator × year | 2013/14–**2024**† | `Rfrg_NPI`, `RBCS_Lvl`, `RBCS_Id`, `HCPCS_Cd`, `Suplr_Rentl_Ind`, `Tot_Suplrs`, `Tot_Suplr_Benes`, `Tot_Suplr_Clms`, `Tot_Suplr_Srvcs`, `Avg_Suplr_Sbmtd_Chrg`, `Avg_Suplr_Mdcr_Alowd_Amt`, `Avg_Suplr_Mdcr_Pymt_Amt`, `Avg_Suplr_Mdcr_Stdzd_Amt` | Equipment-referral features |
| **Medicare DMEPOS — by Referring Provider** (CMS) | Referring NPI × year | as above | `Tot_Suplr_Benes` (distinct), `DME_Tot_Suplr_Benes` and category-level totals | Distinct DME beneficiaries (avoids double counting across HCPCS rows) |
| **LEIE downloadable database** (HHS-OIG) | Excluded individual / entity | current file, refreshed monthly | `NPI`, `EXCLTYPE`, `EXCLDATE`, `REINDATE`, name and address fields | Labels |
| **LEIE monthly supplements** (HHS-OIG) | Monthly exclusions / reinstatements | only ≈ last 12 months archived (earliest listed: Jan 2025) | same layout | Recent reinstatements |
| **Internet Archive snapshots of the LEIE file** | Historical full files | snapshots confirmed for Jan 2017 and Jan 2019; none returned near Jan 2022 or Jan 2024 | same layout | Recover exclusions later removed by reinstatement |
| **NPPES** (CMS), optional | NPI | current | enumeration date, taxonomy, entity type | Tenure (context) |

\*As checked in October 2026: the CMS dataset page lists 2024 as the latest Part B year (updated May 2026). †The data.gov catalogue for DMEPOS by referring provider lists coverage through 2024-12-31 (updated July 2026), and the CMS API page lists 2024 as the latest year. Earlier catalogue entries showed 2023, so this is re-checked in Phase 1.

**Sizes.** The by-Provider file for 2024 has 1,296,739 rows and 81 columns (CMS API page). The by-Provider-and-Service file is expected at ≈ 9–10 M rows/year. DMEPOS files are smaller. LEIE has tens of thousands of rows. Exact sizes are recorded in Phase 1.

**Access caveat.** Automated requests to data.cms.gov from this environment returned HTTP 403 (bot protection) during verification. Phase 1 must confirm a working download route: the CMS download links in a normal browser, the data.cms.gov API, or the data.gov catalogue's distribution links. Files are then registered in the manifest by checksum regardless of route.

**Small-cell suppression (CMS):** aggregated records derived from **10 or fewer beneficiaries are excluded**. Beneficiary sub-group counts below 11 are suppressed in the summary files. Consequences:

- a provider's rare codes (e.g. a 99215 billed for 8 patients) are **invisible** in the service-level file;
- per-provider sums over service rows understate true totals;
- DME referral rows for small item volumes vanish.

Every feature in [FEATURE_CATALOG.md](FEATURE_CATALOG.md) states how it behaves under suppression. A **data-coverage** feature (visible service-level services ÷ summary-file services) is added to the context group, so the model, and readers, know how much of a provider is visible.

### 5.3 Labels: exclusion authorities

All authority codes below are from the OIG exclusion-authorities page (verified). We use three tiers:

| Tier | Authorities (SSA §1128 / §1156) | Use |
|---|---|---|
| **Primary (fraud-related)** | 1128(a)(1) program-related crimes · 1128(a)(3) felony health-care fraud · 1128(b)(1)(A) misdemeanour health-care fraud · 1128(b)(6) excessive charges / unnecessary services · 1128(b)(7) fraud, kickbacks, other prohibited activities · 1128(b)(16) false statements | Main results |
| **Prior-work-compatible** | 1128(a)(1), (a)(2), (a)(3), (b)(4), (b)(7), (c)(3)(G)(i–ii), the set used in the FAU (Khoshgoftaar group) Medicare papers | Comparability sensitivity |
| **Broad** | Primary + (b)(2), (b)(5), (b)(8), (b)(15), §1156, (a)(4), (b)(3) | Sensitivity |
| **Removed** (in primary analysis) | (a)(2) patient abuse/neglect · (b)(4) licence revocation · (b)(14) loan default · (b)(9)–(13) disclosure/access failures | Neither positive nor negative |

The `EXCLTYPE` string format is confirmed against the OIG record-layout PDF in Phase 1.

**Expected prevalence.** Published work using broader label sets reports positive rates of roughly 0.03–0.07 % of records (Johnson & Khoshgoftaar 2019; Tajrobehkar et al. 2024 preprint). Our primary tier is narrower, so prevalence will likely be lower. The true count is the **Phase 1 go/no-go gate** (§15).

### 5.4 Label rule

For a provider whose first primary-tier exclusion is in year *y*ᴱ, with look-back window *W* (primary *W* = 3):

- (NPI, *t*) is **positive** if *t* < *y*ᴱ ≤ *t* + *W*;
- rows with *t* ≥ *y*ᴱ are **dropped** (barred or partial year);
- rows with *t* < *y*ᴱ − *W* are **dropped** (onset unknown);
- never-excluded providers are **unlabelled negatives** (positive-unlabelled setting).

*W* = 3 is an **assumption**. Fraud onset is not observable in public data. Sensitivity runs use *W* ∈ {1, 2, 5}.

**Label horizon.** Year *t* needs exclusions observed through the end of *t* + *W*. With an October 2026 snapshot:

- years ≤ 2022 are complete,
- 2023 has ≈ 2.8 of 3 years,
- 2024 has ≈ 1.8 years.

Accuracy is claimed for 2022–2023 (2023's shortfall reported); 2024 is a showcase only. See Fig. 3.

**Reinstatements.** The full LEIE file lists only exclusions **currently in effect**; reinstated parties are removed. OIG archives monthly supplements for only about 12 months. Recovery plan:

1. Take the union of the current file, all available monthly reinstatement supplements, and every Internet Archive snapshot of the LEIE file that can be retrieved. Snapshots are confirmed for 2017 and 2019; availability for other years is checked in Phase 1.
2. Report the number of exclusions recovered and the **uncovered intervals**.
3. Note that mandatory exclusions carry a 5-year minimum, so exclusions starting 2019 or later can't have been reinstated before 2024. The loss mainly affects permissive authorities (e.g. (b)(7) has no minimum period). Estimate the loss rate by authority from the 2025–2026 reinstatement supplements.

**NPI coverage.** OIG added NPIs from 2008 onward, and not all excluded parties have one. Matching is NPI-only; unmatched exclusions are counted and reported.

---

## 6. Architecture overview

![Fig. 1 — End-to-end architecture](figures/fig01_system_architecture.png)

1. **Public sources:** five CMS/OIG datasets plus Internet Archive LEIE snapshots (NPPES optional).
2. **Preparation:** ingest, harmonise columns per year, define the cohort, label, compute label-free peer-relative features, assign groups.
3. **Analysis datasets:** the provider-year table, cross-fitting folds with a temporal split, and a scheme-injected copy.
4. **Models:** LR, EBM, GA²M, GBM.
5. **Attribution:** a coalition-table engine (methods B, C, E), summed SHAP (A) and glass-box native contributions (D), all against peer baselines.
6. **Evaluation:** detection plus four explanation tests.
7. **Outputs:** analyst view, decision record, provider/beneficiary notices (mock-ups), oversight reports, paper results.

![Fig. 2 — Data model](figures/fig02_data_model.png)

---

## 7. Stage-by-stage design

### 7.1 Ingestion and harmonisation

- Raw CSV → Parquet, read-only. The manifest records URL or route, download date, size and SHA-256.
- A per-year **column map** (YAML), written by hand from each year's CMS data dictionary, maps raw columns to a harmonised schema. Column names have changed across releases (e.g. chronic-condition fields, DMEPOS payment fields). Unmapped columns fail loudly.

### 7.2 Cohort

- **Unit:** Part B rendering NPI × year. DMEPOS rows join on `Rfrg_NPI` = `Rndrng_NPI`; referrers with no Part B row are out of scope.
- **Entity type:** individuals and organisations both kept (entity type in context). Individuals-only is a sensitivity run.
- **Minimum volume:** the by-Provider file already omits providers with ≤ 10 beneficiaries. We additionally require ≥ 50 non-drug services in the year. This threshold is fixed before modelling.
- **Peer groups:** specialty (`Rndrng_Prvdr_Type`) × entity type × year, refined by Census division when the cell has ≥ 200 providers. Fallback order: specialty × entity type × year → entity type × year.

### 7.3 Labelling

As in §5.3–5.4. Per row: `y`, `label_tier`, `excl_type`, `excl_year`, `label_horizon_complete`.

### 7.4 Feature engineering

- **Peer statistics are label-free and computed per year from all non-injected providers in that year.** That covers medians, MADs, percentiles, reference code distributions, shrinkage priors and acuity-adjustment regressions. They use no labels, so including test-fold providers is not label leakage, and it mirrors deployment, where the full year's billing is available at scoring time. This also absorbs national coding shifts such as the 2021 office-visit and 2023 ED/hospital E&M guideline changes.
- **Anything fitted using labels** (models, calibration, thresholds) uses training folds and training years only.
- Transform: robust z = (r − median_P) / (1.4826 · MAD_P), clipped to [−10, 10]; percentile as an alternative.
- Rates with small denominators are shrunk toward the peer median (beta-binomial empirical Bayes, fitted per peer cell and year).
- Structural zeros (e.g. no E&M services, no DME referrals) get explicit indicators.
- Full list: [FEATURE_CATALOG.md](FEATURE_CATALOG.md).

### 7.5 Scheme groups

![Fig. 4 — Scenario taxonomy](figures/fig04_scenario_taxonomy.png)

| # | Scheme group | Captures | Source |
|---|---|---|---|
| 1 | Upcoding | higher-paying codes than delivered | Part B service-level |
| 2 | Overutilization | more services than patients need | Part B service-level + summary |
| 3 | Atypical service mix | codes the specialty rarely bills | Part B service-level |
| 4 | Charge inflation | prices out of line with peers | Part B service-level + summary |
| 5 | Patient-mix mismatch | intensity not explained by patient acuity | Part B summary |
| 6 | DME referral intensity | more equipment ordered than peers, esp. high-risk categories | DMEPOS (both files) |
| 7 | DME referral concentration | orders filled by few suppliers | DMEPOS service-level |
| — | **Context** (input, never a reason) | specialty, entity type, region, urban–rural, size, tenure, data coverage | Part B, NPPES |

Groups are fixed before modelling; one group per feature. Explanation diversity is **not** an objective. Grouping variants are tested (§10.6).

**Why context exists if it is never a reason:**

1. It sets the comparison.
2. It stops scheme features absorbing structural differences.
3. It enables the fairness audit.
4. *Data coverage* tells the model how much of a provider survives suppression.

A context **ablation** tests whether it is needed.

### 7.6 Splits: temporal × provider cross-fitting

Most providers bill every year, and the same positive provider can appear in several years. So splitting must be by provider *and* by time.

- NPIs are partitioned once into **5 folds**, stratified by specialty and ever-positive status.
- For fold *k*: train on NPIs not in *k*, years 2018–2021. Test on NPIs in *k*, years 2022–2023. Repeating over all 5 folds gives **out-of-fold scores for every provider in the test years**, so every positive counts in the evaluation.
- **Tuning and calibration** happen inside the training folds: train 2018–2020, validate 2021 on a held-out 20 % of training NPIs.
- **Rolling origins** (robustness): (2018–20 → 2021), (2018–21 → 2022), (2018–22 → 2023), each with the same fold scheme.
- **Showcase:** 2024, scored by models trained on 2018–2021 (no accuracy claims).
- **2020** is kept but flagged (COVID utilisation shock); results excluding 2020 are reported as a sensitivity check.

### 7.7 Scheme injection (ground truth)

![Fig. 7 — Scheme injection](figures/fig07_scheme_injection.png)

**Purpose.** LEIE never says *which* scheme a provider used. To measure correctness we need cases where the scheme is known.

**Procedure** (component C6, §8):

1. Draw clean providers (never excluded under any tier), stratified by specialty, among those for whom the scheme is applicable (e.g. bills E&M codes for upcoding).
2. Assign a scheme (or a pair) and an intensity.
3. Edit the **raw service-level rows** and update the matching **summary-file totals** consistently (both files for Part B; both files for DMEPOS).
4. **Re-apply CMS suppression** (drop service-level rows with ≤ 10 beneficiaries), so injected data look like public data.
5. Recompute features with the same per-year peer statistics. Injected providers are **excluded** from peer-statistic computation so that they don't shift the reference.

**Two uses:**

- **(a) Detection stress test:** LEIE-trained models score injected test providers. This is informative only, since LEIE fraud may look different.
- **(b) Explanation correctness (primary):** a separate **injection model** per model family, with labels = injected (1) vs clean non-injected (0); LEIE positives removed. It is trained on injected training-fold providers and evaluated on injected test-fold providers. The injected scheme is the answer key.

**Interpretation limit.** Use (b) is a **necessary, not sufficient** test. A model trained on our own injections will learn them, so single-scheme recovery may be easy. Methods are differentiated mainly by **compound injections**, **spill-over** (e.g. upcoding raises payment per service, touching Charge inflation) and **distractor cases** (injection into providers who already have a natural outlier in another group). Parameters: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) §6.

### 7.8 Models

| Model | Configuration | Explanation |
|---|---|---|
| Logistic regression | L1/L2, on peer-relative features + natural-spline terms for key features | Exact |
| EBM (main effects) | InterpretML `ExplainableBoostingClassifier`, interactions = 0 | Exact |
| EBM GA²M | same, with automatically selected pairwise terms (InterpretML FAST) | Exact incl. pairs |
| GBM | XGBoost and LightGBM; with and without monotone constraints on scheme features | Post hoc (§9) |

- **Categoricals:** one-hot encoding for all models in the primary configuration. This keeps every attribution tool compatible. Native categorical handling is a variant, used only if attribution tooling handles it, checked in Phase 3.
- **Training size:** ≈ 0.8 × ~1.1 M providers × 4 years ≈ 3–4 M rows. To keep EBM/GA²M tractable and comparisons fair, **all** model families train on the same data: all positives plus a 10–20 % random sample of negatives, weighted by the inverse sampling rate. Ranking metrics are unaffected in expectation; probabilities are recalibrated.
- **Imbalance:** inverse-sampling weights plus class weights; no SMOTE.
- **Calibration:** isotonic, on the validation slice.
- **Seeds:** 10 per configuration.

---

## 8. Components to build

Everything below is **written by us** unless marked *library*. Libraries are existing, maintained open-source packages: DuckDB, Polars, PyArrow, scikit-learn, InterpretML, XGBoost, LightGBM, `shap`, Optuna, SciPy, NumPy, Matplotlib. No component assumes a capability that hasn't been verified to exist in these libraries.

| ID | Component | Inputs → outputs | Key logic | Depends on |
|---|---|---|---|---|
| C1 | **Acquisition & manifest** | download routes → raw files + `manifest.json` | download (or register manually downloaded files), checksum, record provenance | — |
| C2 | **Schema harmoniser** | raw CSV + per-year YAML column maps → harmonised Parquet | rename, type, validate; fail on unmapped/missing columns | C1 |
| C3 | **LEIE labeller** | LEIE current + supplements + archive snapshots → `labels.parquet` | union of sources, de-duplicate, authority tiers, NPI match, label rule, horizon flags, gap report | C1 |
| C4 | **Cohort & peer builder** | harmonised Part B → cohort + peer-cell assignment | volume threshold, peer hierarchy with fallback | C2 |
| C5 | **Feature builder** | harmonised Part B + DMEPOS + peer cells → feature table | per-year label-free peer statistics, shrinkage, divergences, acuity residuals, indicators, coverage | C2, C4 |
| C6 | **Injection engine** | raw rows of selected providers + scheme spec → edited raw rows (both files) + injection metadata | scheme transforms, consistent total updates, suppression re-application | C2, C4 |
| C7 | **Fold & split assigner** | cohort + labels → fold IDs, split tags | stratified 5-fold NPI partition, temporal tags, validation slice | C3, C4 |
| C8 | **Model trainer** | features + folds → fitted models, OOF scores, calibrators | negative subsampling, Optuna tuning (matched budget), seeds; *library*: scikit-learn, InterpretML, XGBoost, LightGBM, Optuna | C5, C7 |
| C9 | **Background sampler** | features + peer cells → background sets per provider-year | same-year, same-peer-cell, training-fold providers (≤ 200); global variant | C5, C7 |
| C10 | **Coalition-table engine** (methods B, C, E, deletion) | model + explained providers + backgrounds → 256 coalition values per provider | vectorised: build 256 × \|B\| hybrid rows per provider, predict in batches, average; then compute group Shapley (B), group interaction matrix (C), counterfactuals (E), deletion curves | C8, C9 |
| C11 | **Summed-SHAP wrapper** (method A) | model + backgrounds → feature SHAP → group sums | *library*: `shap.TreeExplainer` interventional, run per peer cell; path-dependent variant for diagnostics | C8, C9 |
| C12 | **Glass-box extractor** (method D) | LR / EBM + backgrounds → group contributions | LR: β·(x − E_B x); EBM: term scores centred on E_B; pair terms → group or cell | C8, C9 |
| C13 | **Evaluation suite** | scores, labels, attributions, injection metadata → metric tables + CIs | detection metrics, NPI-level paired bootstrap, correctness, consistency, faithfulness, stability, fairness | C8, C10–C12 |
| C14 | **Indicator & view renderer** | one provider's record → analyst view, provider/beneficiary notice mock-ups (text/HTML) | ranking rules (§9.6), supporting-fact selection, templates, view rules (§11.3) | C10–C12 |
| C15 | **Report & figure generator** | metric tables → paper tables and figures | *library*: Matplotlib | C13 |
| C16 | **Config & orchestration** | YAML configs (groups, features, parameters, seeds) → pipeline runs | one entry point per phase; deterministic seeds; run log | all |

The **decision record and provenance store** (§11) are specified as a JSON schema and exercised by C14 on mock decisions. No live case-management system is built.

---

## 9. Scenario-level attribution

![Fig. 5 — Attribution methods](figures/fig05_attribution_methods.png)

### 9.1 Notation

- *f*(*x*): model output in **log-odds** (raw margin, before calibration).
- *N* = {*G*₀ (context), *G*₁ … *G*₇}, |*N*| = *m* = 8 groups.
- *B*(*x*): background set = up to 200 **same-year, same-peer-cell, training-fold** providers (default "peer baseline"); the global variant samples 1,000 providers across peer cells.
- **Coalition value:** for *S* ⊆ *N*, *v*(*S*) = (1/|*B*|) Σ_{z∈B} *f*(*x*_S, *z*_{N∖S}). Groups in *S* keep the provider's values; the others take the background provider's values (interventional). *v*(∅) = E_B[*f*], *v*(*N*) = *f*(*x*).

### 9.2 Methods

**A. Summed SHAP (baseline; prototype method).**
Φᴬ_G = Σ_{j∈G} φⱼ, with φ from interventional TreeSHAP against *B*. Exactly additive. Cross-group interactions are split half-and-half and never named.

**B. Group Shapley (quotient game).**
Φᴮ_G = Σ_{S⊆N∖{G}} [|S|!(m−|S|−1)!/m!] · [*v*(S∪{G}) − *v*(S)].
Exact from the coalition table. It equals the group totals of the Owen value (quotient-game property). It differs from A when groups interact.

**C. Group interaction matrix.**
Pairwise Shapley interaction index (Grabisch & Roubens 1999) at the group level:

*I*_GH = Σ_{S⊆N∖{G,H}} [|S|!(m−|S|−2)!/(m−1)!] · [*v*(S∪{G,H}) − *v*(S∪{G}) − *v*(S∪{H}) + *v*(S)].

Following the SHAP-interaction convention (Lundberg et al. 2020):

- off-diagonal M_GH = *I*_GH / 2,
- diagonal M_GG = Φᴮ_G − Σ_{H≠G} M_GH.

All cells sum to *f*(*x*) − E_B[*f*], and row sums equal Φᴮ. The diagonal is the **pure** scheme effect; M_GH + M_HG is the **compound** effect of the pair. It is computed from the **same coalition table** as B, against the **peer** baseline. Feature-level TreeSHAP interaction values (path-dependent) are computed on a 2,000-provider subsample only as a cross-check.

**D. Glass-box native.**
- LR: *c*ⱼ = βⱼ(*x*ⱼ − E_B[*x*ⱼ]) (per basis term for splines), summed per group.
- EBM: *c*ⱼ = *f*ⱼ(*x*ⱼ) − E_B[*f*ⱼ(*z*ⱼ)]. Within-group pair terms go to the group. Cross-group pair terms are stored both as a matrix cell (comparable to C) and split ½/½ (comparable to A and B).

**E. Peer counterfactual.**
Δ_G = *v*(*N*) − *v*(*N*∖{*G*}): the drop in risk if group *G* alone took peer values. Also read from the coalition table, and reported in probability units. Not additive.

### 9.3 Baseline

![Fig. 6 — Peer baseline](figures/fig06_peer_baseline.png)

- **Global baseline:** "why is this provider riskier than the average provider?" Specialty becomes a "reason", which can't be defended.
- **Peer baseline:** "why riskier than same-specialty, same-entity, same-region peers in the same year?"

Context features matched by the peer cell (specialty, entity type, region, year) equal the background values, so their contribution is **exactly zero**. Unmatched context features (urban–rural, size, tenure, coverage) may keep small attributions. These are shown to analysts as "peer-group adjustment", never as reasons. Both baselines are computed (RQ5).

### 9.4 Interventional vs path-dependent

Methods A (primary), B, C, D and E all use the interventional peer background. Path-dependent TreeSHAP appears only in diagnostics (A-path and feature-level interactions), where we report how far its group totals diverge from the interventional ones.

### 9.5 Explained set and compute

- **Explained set per test year:** top 2 % by out-of-fold GBM score + a stratified random sample of 5,000 + all injected test providers. Roughly 20–30 k + 5 k + ~4.5 k ≈ 30–40 k provider-years with cross-fitting over ≈ 1.1 M providers.
- **Coalition table cost:** 256 × |*B*| (≤ 200) rows per provider ≈ 51 k predictions, so ≈ 1.5–2 billion predictions per model per test year. That is feasible with batched XGBoost/LightGBM inference on a multi-core workstation over hours, or it can be reduced with |*B*| = 100 or a smaller random sample. The exact budget is set after a timing benchmark in Phase 4.
- Method A and D run on the full test set; they are cheap.
- **Stability retrains (×20):** coalition-table methods run on a 2,000-provider subsample only.

### 9.6 From attribution to scenario indicator

1. Rank scheme groups by the chosen primary method (selected after §10 results; default B).
2. Report up to 3 groups with **positive** contribution ≥ 0.1 log-odds. Negative contributions are shown separately as *mitigating* factors.
3. Flag compound pairs whose M_GH + M_HG ≥ 0.1 log-odds.
4. Attach supporting facts: the 1–2 features in the group whose per-feature contribution (method A within group) is largest, stated as raw values vs the peer median.
5. Never output the context group as a reason.

---

## 10. Evaluation

![Fig. 8 — Evaluation framework](figures/fig08_evaluation_framework.png)

Protocol and formulas: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

### 10.1 Detection

- precision@k and recall@k with k ∈ {0.1, 0.5, 1, 2} % of scored providers per test year;
- PR-AUC (primary), ROC-AUC, lift;
- calibration (Brier, reliability within the top 5 %);
- paired NPI-level bootstrap (1,000 replicates) for GBM − best glass box; Holm correction over k;
- per rolling origin and pooled.

### 10.2 Correctness (injected)

Top-1, top-3, MRR versus chance (1/7, 3/7); compound both-in-top-3; method-C pair hit versus chance (1/21); spill-over profile; distractor cases.

### 10.3 Consistency (GBM vs glass box)

Sign agreement (|value| > 0.05), Spearman ρ, top-3 Jaccard, share-vector cosine; stratified by flag agreement at k = 1 %; compared with a permutation baseline. Agreement indicates plausibility, not correctness (Krishna et al. 2022), hence §10.2 and §10.4.

### 10.4 Faithfulness

Group-deletion curves read from the coalition table (*v*(*N*), *v*(*N*∖{π₁}), *v*(*N*∖{π₁,π₂}), …). Area under the normalised curve; compared with random and reverse orders.

### 10.5 Stability

10 seeds + 10 bootstrap retrains: top-1 agreement, Kendall τ; binomial-resampling perturbation of rate features.

### 10.6 Sensitivity

*W*; label tiers (primary, prior-work-compatible, broad); individuals only; without 2020; global vs peer baseline; grouping variants (correlation clusters; moved features); context ablation; monotone constraints; peer granularity.

### 10.7 Fairness (descriptive)

Flag rate, precision and flag/label ratio by specialty, entity type, region and urban–rural class at k ∈ {0.5, 1} %. LEIE reflects enforcement patterns, which limits interpretation.

### 10.8 Results → conclusions

| Result | Reported conclusion |
|---|---|
| GBM significantly better; explanations consistent and correct | Performance gain without losing explanation validity |
| GBM better; disagrees with glass box but correct on injected | Glass box misses structure; GBM explanations usable with caution |
| GBM better; incorrect on injected | Extra performance not explainable at scheme level |
| No significant difference | Recommend glass box; scenario framework and method comparison stand |

---

## 11. Audiences, outputs and the decision lifecycle

![Fig. 9 — Audiences and lifecycle](figures/fig09_audiences_lifecycle.png)

### 11.1 Deployment context (for framing)

In Medicare FFS, claims are processed by MACs, fraud is investigated by UPICs for CMS, and OIG and DOJ handle enforcement. Providers receive ADRs and can be placed on prepayment review; beneficiaries receive MSNs. For a commercial insurer the same roles sit in its special investigations unit (SIU), claims operations and member services. The design is written for the generic roles (**analyst/investigator, provider, beneficiary/member, oversight**). The method transfers to any payer that can compute the same provider-level aggregates from its own claims.

### 11.2 One record, several views

One internal record per scored provider: score, attributions (all methods), supporting facts, model and data versions, peer cell. A human decision is appended later. **Provenance chain:** external notice → decision record → attribution snapshot → model and data version.

| Audience | Goal | Sees | Never sees | Built from |
|---|---|---|---|---|
| Analyst / investigator | Triage and evidence | Rank, calibrated probability, top indicators with facts, compound pairs, peer stats, counterfactuals, drill-down | — | C, E (+ A, B) |
| Unit manager / compliance | Capacity and quality | Queue summaries, global scheme importance, precision at capacity, drift | individual drill-down unless needed | aggregates |
| Provider | Understand and respond | The action (e.g. ADR), a policy-based reason, a factual peer comparison of their own billing | score, model internals, context as a reason, thresholds | decision record + selected facts |
| Beneficiary / member | Claim status | Status and policy reason, appeal route | fraud language, investigation details | decision record |
| Oversight / auditor | Verify governance | model card, fairness audit, sampled provenance chains | — | audit trail |

**Precedent for the provider view:** CMS's **Comparative Billing Reports (CBRs)** are confidential, educational peer-comparison reports sent to outlier providers. CMS states they are not a precursor to audit. Distribution was paused from January 2024 pending redesign. The provider view borrows the CBR's *format* (own metric vs state/national peers), not its educational framing.

### 11.3 Rules

1. A human makes every adverse decision.
2. Context is never a reason.
3. No score or detection logic leaves the investigation unit (it would teach evasion and imply an accusation).
4. External reasons map to the decision record, not to raw attributions.
5. Beneficiaries receive status and policy reasons only.

### 11.4 Illustrative renderings (mock-ups, not results)

**Analyst view**

> **Rank 37 of 1.1 M scored providers · P(fraud-related exclusion within 3 yrs) = 4.7 %** · peer group: Internal Medicine, individual, West South Central, 2023 · data coverage 94 %
> 1. **Upcoding** (+1.10): 78 % of established-patient visits at level 5 (peer median 22 %); allowed $/visit 1.6× peers
> 2. **Overutilization** (+0.62): 9.4 non-drug services per patient (peer median 3.1)
>    ↳ **Compound: Upcoding × Overutilization** (+0.88)
> 3. **DME referral concentration** (+0.21): 41 DME claims per supplier (peer median 6)
> *Mitigating:* Patient-mix (−0.12), average risk score above peers
> *If coding levels matched peers:* risk 4.7 % → 1.2 %

**Provider notice (ADR-style)**

> We are reviewing a sample of your 2023 Medicare claims and request supporting documentation for the services listed. For context, your billing differs from peers in your specialty and region: level-5 established-patient visits were 78 % of your visits (peer median 22 %). This comparison is informational and is not a finding. [Response and appeal instructions.]

**Beneficiary notice**

> Payment for your 14 March 2024 visit is under routine review. You do not need to do anything. [Appeal rights if a claim is later denied.]

### 11.5 Validated vs proposed

Public data validates scoring, attribution and the analyst view's content. The provider, beneficiary and oversight views are a **design proposal** with worked examples. There is no claimant-side data and no user study.

---

## 12. Expected outcomes and deliverables

**Paper:**

- **Tables:** detection with CIs (LR / EBM / GA²M / GBM); explanation metrics by method; sensitivity results.
- **Figures:** architecture; global scheme importance; per-scheme recovery; consistency distributions; deletion curves; an example group interaction matrix; global vs peer baseline; fairness profile.
- **Narrative:** the §10.8 branch the data supports.

**Engineering:** the pipeline C1–C16 with manifest, configs and seeds; the coalition-table engine as a reusable module; the renderer and mock-up views.

**Users of the results:**

- researchers: a reproducible protocol with ground-truth explanation testing on public data;
- program-integrity practitioners: evidence on whether black-box gains survive explanation scrutiny, and an audience-safe explanation pattern;
- auditors: a concrete provenance example.

---

## 13. Limitations

### 13.1 Data

| Limitation | Effect | Handling |
|---|---|---|
| Aggregated provider-level data | No claim-level prediction or explanation | Framed as audit prioritisation |
| **Suppression (≤ 10 beneficiaries)** | Rare codes and small DME volumes invisible; shares biased for small providers | Volume threshold; coverage feature; per-feature notes; injected data re-suppressed |
| Exclusion ≠ fraud | Legal outcome; many frauds never excluded; plea deals | Fraud-related tiers; sensitivity tiers; wording "fraud-related exclusion" |
| LEIE skews to flagrant cases | Models learn obvious outliers, less subtle fraud (Tajrobehkar et al. 2024) | Stated; injection tests subtler intensities |
| Positive-unlabelled labels | Measured precision is a lower bound | Stated; optional PU sensitivity |
| Very low prevalence (≈ 0.03–0.07 % in prior work, likely lower here) | Wide CIs | Cross-fitting; pooled years; precision@k; Phase 1 gate |
| Label lag | Recent years look clean | Horizon rule; 2024 showcase only |
| Reinstatements removed; supplements archived ≈ 12 months; Internet Archive snapshots incomplete | Missing historical positives | Union of sources; gap and loss-rate report |
| Missing NPIs in LEIE | Unmatched positives | Counted and reported |
| W (onset) unobservable | Label noise near window edges | W sensitivity |
| Rendering-provider view only (Part B) | Ordering/referral schemes except DME invisible (e.g. lab orders) | Stated scope |
| Medicare FFS only | No MA (≈ half of beneficiaries), Medicaid, commercial | Stated scope |
| Self-reported specialty | Peer mis-assignment | Fallback hierarchy; peer-granularity sensitivity |
| Coding changes (2021, 2023), COVID 2020 | Drift | Per-year peer statistics; 2020 flag; rolling origins |
| Automated access to data.cms.gov blocked from some clients | Acquisition friction | Manual or API routes; manifest |

### 13.2 Attribution

| Limitation | Effect | Handling |
|---|---|---|
| Attributions explain the model, not fraud | Not causal; not evidence | Explicit wording in all views |
| Baseline dependence | Different questions answered | Peer baseline justified; global reported |
| Interventional sampling creates off-manifold hybrids (Kumar et al. 2020) | Coalition values may query unrealistic inputs | Peer backgrounds reduce the gap; acknowledged |
| Grouping is a modelling choice | Different explanations | A-priori groups; grouping variants |
| Agreement ≠ correctness | Shared bias possible | Separate correctness and faithfulness tests |
| Injection circularity | Recovery easier than real life | Necessary-not-sufficient framing; compound, spill-over, distractor cases |
| Coalition-table cost | Can't explain everyone with B/C/E | Explained set |

### 13.3 Modelling and evaluation

- Conclusions depend on glass-box strength (tuned EBM/GA²M, spline LR, matched tuning budgets).
- Negative subsampling for tractability (applied equally to all model families).
- Single country, payer and data type: limited external validity.

### 13.4 Lifecycle and ethics

- Non-analyst views are not empirically validated.
- False positives carry real cost, so outputs are audit priorities with a human decision.
- Disparities may reflect enforcement patterns rather than fraud prevalence; the audit reports them, it can't fix them.
- Gaming risk is addressed by the view rules.

---

## 14. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Too few primary-tier positives | Medium–High | High | **Phase 1 gate**: if < ~300 unique positive NPIs over 2018–2023, switch primary to prior-work-compatible or broad tier and/or extend training back to 2015 |
| Weak signal | High | Medium | Expected; the focus is explanation quality; report lift@k |
| GBM shows no advantage | Medium | Low | Planned branch (§10.8) |
| data.cms.gov download blocked | Medium | Low | Manual browser / API / data.gov routes |
| Column schema drift | High | Low | Per-year YAML maps; fail loudly |
| Internet Archive LEIE gaps | High | Medium | Quantify; rely on mandatory-exclusion minimum periods; report bias |
| Coalition-table compute | Medium | Medium | Timing benchmark; smaller background; batching |
| EBM training time | Medium | Medium | Negative subsampling; limit pairwise terms |
| Prior-work overlap (Hancock et al. 2023; Shekhar et al. 2023) | Certain | Medium | Phase 0 review; differentiate on scheme grouping, exact group interactions, ground-truth correctness, glass-box comparison, lifecycle |
| Injection realism critique | Medium | Medium | Transforms tied to documented scheme types; spill-over; intensity sweep; limitations |

---

## 15. Work plan

| Phase | Content | Exit criterion |
|---|---|---|
| 0. Literature & protocol | Read prior Medicare/LEIE and grouped-attribution work; freeze RQs, metrics, groups | Protocol frozen |
| 1. Data acquisition (C1–C3) | Download routes, manifest, column maps; LEIE + supplements + archive snapshots; NPI match | **Go/no-go:** positive NPI counts per year and tier; archive gap report; DMEPOS 2024 confirmed or not |
| 2. Cohort & features (C4, C5, C7) | Cohort, peer cells, features, folds | Feature table + QA report (distributions, suppression coverage, leakage checklist) |
| 3. Models (C8) | LR, EBM, GA²M, GBM; tuning; calibration; seeds | Detection results (RQ1) |
| 4. Attribution (C9–C12) | Timing benchmark; methods A–E; both baselines | Attribution store |
| 5. Injection (C6) | Calibrate intensities; injection models | Correctness results (RQ3) |
| 6. Evaluation (C13) | Consistency, faithfulness, stability, sensitivity, fairness | RQ2, RQ4, RQ5 |
| 7. Outputs & writing (C14, C15) | Renderer, mock-ups, figures, draft | Draft + reproducibility package |

---

## 16. Threats to validity and research value

**Does the research problem make sense?** Yes, with the framing in this document. The problem (explaining provider-level fraud risk in scheme terms to several audiences) is real and under-studied in terms of *measured* explanation quality. The combination of real public data, injected ground truth and an exact group-interaction method is defensible and, to our knowledge, not the standard setup in prior Medicare-LEIE work. Phase 0 must confirm this.

**Construct validity.** The label measures "excluded under fraud-related authority within *W* years", not "committed fraud". All claims in the paper use that construct.

**Internal validity.** Leakage is controlled by label-free peer statistics, label-dependent fitting only on training folds and years, provider cross-fitting, and a pre-registered protocol.

**External validity.** Medicare FFS, US, provider-level aggregates only.

**Conclusion validity.** Low prevalence makes detection CIs wide; cross-fitting and pooling are essential. If the Phase 1 count is too low for RQ1 to be powered, RQ1 is reported descriptively, and the paper's weight shifts to RQ2–RQ5, which use explained sets and injected data where sample size is controlled.

**What could make the outcome uninteresting, and the guard against it:**

- Injection recovery is trivially perfect for every method → compound/spill-over/distractor cases are the differentiating tests.
- All methods agree → that is still a finding (method choice doesn't matter at group level), and it is reported.
- Detection is near random → the paper becomes a methods paper on injected data plus a negative real-data result. Still publishable in applied venues, but weaker. This is decided at the Phase 3 checkpoint.

---

## 17. Decision log

| # | Decision | Reason |
|---|---|---|
| D1 | Drop Kaggle synthetic data | Rule-generated labels |
| D2 | CMS Part B + DMEPOS with LEIE labels | Only real, public, labelled option |
| D3 | Provider-year unit | Labels and data are provider-level |
| D4 | "Scenario indicators / scenario-level attribution" | Avoid conflict with internal and CARC/RARC usage |
| D5 | Strong glass boxes | Performance gap must be shown |
| D6 | Compare explanations whatever the performance result | Every outcome is informative |
| D7 | No diversity objective | Faithfulness over balance |
| D8 | Context = inputs, never reasons; ablation-tested | Peer-setting, deconfounding, fairness, coverage |
| D9 | Same-year peer-cell baseline | Investigator question; context attribution exactly zero for matched traits |
| D10 | Coalition-table engine for B, C, E, deletion | Exact, consistent, peer-baseline interactions; one computation |
| D11 | Injection on raw rows with re-suppression | Ground truth with realistic side effects |
| D12 | Train 2018–21, test 2022–23, showcase 2024 | Label horizon |
| D13 | 5-fold provider cross-fitting | Every positive evaluated; no provider leakage |
| D14 | Label-free per-year peer statistics | Deployment-realistic; absorbs coding changes; no label leakage |
| D15 | One-hot categoricals | Attribution-tool compatibility |
| D16 | Equal negative subsampling for all models | EBM tractability; fair comparison |
| D17 | Part D deferred | Scope |
| D18 | One record, audience views, provenance chain | Lifecycle explainability without exposing detection logic |

---

## 18. Claim verification status

| Claim in this document | Status | Source / plan |
|---|---|---|
| Part B by Provider & Service fields listed in §5.2 | **Verified** (column list, 2013–2024 coverage) | Third-party catalogue mirror of the CMS file; CMS data dictionary (RY25) title confirmed |
| Part B latest year = 2024 | **Verified** | CMS dataset page and look-up tool (updated May 2026) |
| By-Provider has `Tot_HCPCS_Cds`, `Bene_Avg_Risk_Scre` (avg HCC score), `Bene_Dual_Cnt`; 1,296,739 rows (2024) | **Verified** | CMS by-Provider data dictionary (RY21); CMS API page |
| `Bene_CC_*` exact names per year | To verify | Phase 1 column maps |
| Suppression of records from ≤ 10 beneficiaries | **Verified** | CMS methodology (Part B, DMEPOS) |
| DMEPOS by Referring Provider & Service fields incl. `RBCS_Id`, `Suplr_Rentl_Ind`, `Tot_Suplrs`, payment averages | **Verified** | CMS data dictionaries / API pages (via search) |
| DMEPOS latest year = 2024 | Likely; re-check | data.gov catalogue (Jul 2026), CMS API page; earlier entries said 2023 |
| DMEPOS by Referring Provider has distinct `Tot_Suplr_Benes`, `DME_Tot_Suplr_Benes` | **Verified** | CMS referring-provider data dictionary (via search) |
| LEIE full file = exclusions in effect; reinstated removed | **Verified** | OIG LEIE download page |
| Monthly supplements archived ≈ 12 months (earliest listed Jan 2025) | **Verified** | OIG supplement archive page |
| Internet Archive LEIE snapshots exist for 2017, 2019; none returned near 2022, 2024 | **Verified (partial)** | Wayback availability API; full coverage checked in Phase 1 |
| LEIE NPIs from 2008, not universal; pre-May-2013 supplements lack NPI | **Verified** | OIG LEIE pages |
| Exclusion authority codes and minimum periods | **Verified** | OIG exclusion-authorities page |
| FAU label set (a1, a2, a3, b4, b7, c3G) | **Verified** (as reported) | Bauder & Khoshgoftaar; Johnson & Khoshgoftaar 2019 |
| Prior-work positive rate ≈ 0.03–0.07 % | **Verified** (as reported) | Johnson & Khoshgoftaar 2019; Tajrobehkar et al. 2024 |
| CBRs: confidential, educational, not audit precursors; paused Jan 2024 | **Verified** | CMS/association sources |
| CMS Fraud Prevention System since 2011 | Background knowledge; cite primary source in Phase 0 | CMS / Small Business Jobs Act 2010 |
| NAIC Model Bulletin on AI (Dec 2023) | Background knowledge; cite primary source in Phase 0 | NAIC |
| 2021 office E&M and 2023 ED/hospital E&M guideline changes; 99201 deleted 2021 | Background knowledge; cite AMA/CMS in Phase 0 | AMA CPT |
| groupShapley = arXiv 2106.12228 (Jullum, Redelmeier, Aas) | **Verified** | arXiv |
| Hancock, Bauder, Wang, Khoshgoftaar (2023) J. Big Data 10 | **Verified** | DOAJ / journal |
| Shekhar, Leder-Luis, Akoglu (2023) NBER w30946 | **Verified** | NBER |
| Tajrobehkar et al. (2024) medRxiv LEIE critique | **Verified** (preprint) | medRxiv |
| `shap.TreeExplainer` interventional mode; interaction values path-dependent only | Library behaviour; confirmed by test in Phase 4 | `shap` docs |
| EBM pairwise detection (FAST) in InterpretML | Library behaviour; confirmed in Phase 3 | InterpretML docs |
| Data-access 403 from automated clients | **Observed** during verification | — |
| All numbers in Figs. 5–6 and §11.4 | **Illustrative**, not results | — |

---

## 19. Open questions

1. Audit-capacity range: are 0.1–2 % of providers right for the framing?
2. Peer granularity: is region needed? (Decided empirically in Phase 2.)
3. Venue: will reviewers expect even a small expert review of the analyst view?
4. Part D: extension here or follow-up?
5. Positive-unlabelled learning: include as a variant or leave as a limitation?

---

## 20. References

**Data sources**
- CMS. Medicare Physician & Other Practitioners — by Provider and Service. https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service
- CMS. Medicare Physician & Other Practitioners — by Provider (API docs). https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider/api-docs
- CMS. Medicare Physician & Other Practitioners — by Provider Data Dictionary (RY21). https://data.cms.gov/sites/default/files/2021-10/5d9da495-6798-4428-8260-ce362d8fc5a5/MUP_PHY_RY21_20211021_DD_PRV.pdf
- CMS. Medicare Physician & Other Practitioners Methodology (RY25). https://data.cms.gov/sites/default/files/2025-04/MUP_PHY_RY25_202350312_Methodology_508.pdf
- CMS. DMEPOS — by Referring Provider and Service (API docs). https://data.cms.gov/provider-summary-by-type-of-service/medicare-durable-medical-equipment-devices-supplies/medicare-durable-medical-equipment-devices-supplies-by-referring-provider-and-service/api-docs
- CMS. DMEPOS — by Referring Provider (catalogue). https://catalog.data.gov/dataset/medicare-durable-medical-equipment-devices-supplies-by-referring-provider
- CMS. DMEPOS Methodology (RY25). https://data.cms.gov/sites/default/files/2025-06/a837dcf7-3d06-4c1f-8d4a-82ee737b7932/MUP_DME_RY25_Methodology_20250617_508.pdf
- HHS-OIG. LEIE Downloadable Databases. https://oig.hhs.gov/exclusions/exclusions_list.asp
- HHS-OIG. LEIE Supplement Archive. https://www.oig.hhs.gov/exclusions/supplement_archive.asp
- HHS-OIG. Exclusion Authorities. https://oig.hhs.gov/exclusions/authorities.asp

**Medicare fraud detection**
- Herland, M., Khoshgoftaar, T. M., & Bauder, R. A. (2018). Big Data fraud detection using multiple Medicare data sources. *Journal of Big Data*, 5. https://journalofbigdata.springeropen.com/articles/10.1186/s40537-018-0138-3
- Johnson, J. M., & Khoshgoftaar, T. M. (2019). Medicare fraud detection using neural networks. *Journal of Big Data*, 6. https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0225-0
- Hancock, J. T., Bauder, R. A., Wang, H., & Khoshgoftaar, T. M. (2023). Explainable machine learning models for Medicare fraud detection. *Journal of Big Data*, 10. https://doi.org/10.1186/s40537-023-00821-5
- Shekhar, S., Leder-Luis, J., & Akoglu, L. (2023). Unsupervised machine learning for explainable health care fraud detection. NBER Working Paper 30946. https://www.nber.org/papers/w30946
- Tajrobehkar, et al. (2024). Utilization analysis and fraud detection in Medicare via machine learning. medRxiv 2024.12.30.24319784 (preprint). https://www.medrxiv.org/content/10.1101/2024.12.30.24319784v1

**Explainability**
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
- Owen, G. (1977). Values of games with a priori unions. In *Mathematical Economics and Game Theory*.
- Grabisch, M., & Roubens, M. (1999). An axiomatic approach to the concept of interaction among players in cooperative games. *International Journal of Game Theory*, 28, 547–565.
- Jullum, M., Redelmeier, A., & Aas, K. (2021). groupShapley: Efficient prediction explanation with Shapley values for feature groups. arXiv:2106.12228.
- Au, Q., Herbinger, J., Stachl, C., Bischl, B., & Casalicchio, G. (2022). Grouped feature importance and combined features effect plot. *Data Mining and Knowledge Discovery*.
- Chen, H., Janizek, J. D., Lundberg, S., & Lee, S.-I. (2020). True to the model or true to the data? arXiv:2006.16234.
- Kumar, I. E., Venkatasubramanian, S., Scheidegger, C., & Friedler, S. (2020). Problems with Shapley-value-based explanations as feature importance measures. *ICML*.
- Krishna, S., et al. (2022). The disagreement problem in explainable machine learning: A practitioner's perspective. arXiv:2202.01602.
- Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. *KDD*.
- Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. arXiv:1909.09223.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206–215.

**Policy and program context** (primary sources to be added in Phase 0)
- NAIC (2023). Model Bulletin: Use of Artificial Intelligence Systems by Insurers.
- CMS Comparative Billing Reports (CBR) program; pause announced for 2024.
- CMS Fraud Prevention System (Small Business Jobs Act of 2010).
