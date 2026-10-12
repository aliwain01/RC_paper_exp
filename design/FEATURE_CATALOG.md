# Feature Catalog — v1.1

Companion to [DESIGN_DOCUMENT.md](DESIGN_DOCUMENT.md) §7.4–7.5 · [Fig. 4](figures/fig04_scenario_taxonomy.png). Built by component **C5** (feature builder).

## Conventions

- One row = one Part B rendering NPI × year (*t*).
- Source abbreviations:
  - **P&S** = Physician & Other Practitioners by Provider and Service
  - **Prov** = Physician & Other Practitioners by Provider
  - **DME-S** = DMEPOS by Referring Provider and Service
  - **DME-P** = DMEPOS by Referring Provider
- Field names below were verified against CMS data dictionaries / catalogue mirrors (design doc §18), **except** those marked †, which are confirmed in Phase 1 column maps. Names can differ by year; the per-year YAML map (C2) resolves them.
- Every scheme feature is a **raw statistic *r*** transformed to a peer-relative value. The default is a robust z-score within the peer cell; percentiles are an alternative.
- **Peer statistics are label-free and computed per year from all non-injected providers in that year** (design doc §7.4). Nothing in this file uses labels.
- **Shrinkage** for rates: `(k + α·m) / (n + α)`, where *m* is the peer-cell median rate and α comes from a beta-binomial fit per peer cell and year.
- **Dir.** = expected direction (↑ = higher is more suspicious). ↑ features are monotone-constraint candidates.
- **Suppression note** = how CMS's exclusion of rows derived from ≤ 10 beneficiaries affects the feature:
  - *V* = computed over **visible** rows only;
  - *T* = uses a summary-file **total**, which is not affected row by row.
- Target ≈ 35–50 features. Near-duplicates (|ρ| > 0.95 within a group) are pruned in Phase 2 and the pruning is reported.

### Code sets

| Set | Codes | Note |
|---|---|---|
| Established office visit | 99211–99215 | level 5 = 99215 |
| New office visit | 99202–99205 | 99201 deleted from 2021 |
| ED visit | 99281–99285 | level 5 = 99285; ED guideline revision 2023 |
| Drug lines | `HCPCS_Drug_Ind` = Y | units, not visits; excluded from service-count features |
| DME high-risk categories | selected `RBCS_Id` families and HCPCS ranges: orthotic braces (L-codes), power mobility devices (K0800–K0899) | Exact list frozen in Phase 2, citing OIG/DOJ scheme reports |

Office and ED E&M guideline changes shift national distributions. Per-year peer statistics absorb them.

---

## 1 · Upcoding

| ID | Feature | Raw statistic *r* | Fields | Dir. | Suppr. | Rationale |
|---|---|---|---|---|---|---|
| UP1 | Level-5 established share | Σ srv(99215) / Σ srv(99211–15) | P&S `HCPCS_Cd`, `Tot_Srvcs` | ↑ | V | Classic E&M upcoding |
| UP2 | High-level established share | Σ srv(99214–15) / Σ srv(99211–15) | P&S | ↑ | V | Broader variant; less affected by suppression of the level-5 row |
| UP3 | Level-5 new-patient share | Σ srv(99205) / Σ srv(99202–05) | P&S | ↑ | V | Same scheme, new patients |
| UP4 | Level-5 ED share | Σ srv(99285) / Σ srv(99281–85) | P&S | ↑ | V | ED upcoding |
| UP5 | Mean established E&M level | service-weighted mean of level 1–5 | P&S | ↑ | V | Continuous summary |
| UP6 | Allowed $ per E&M service | Σ(srv × `Avg_Mdcr_Alowd_Amt`) / Σ srv over E&M codes | P&S | ↑ | V | Financial footprint |
| UP0 | E&M indicator | 1 if ≥ 1 visible E&M row in the set | P&S | — | V | Structural zero; UP1–6 set to peer median when 0 |

*Suppression caveat:* a provider whose 99215 row covers ≤ 10 beneficiaries shows 0 % level-5 visits. Shares are therefore biased **downward for small providers**. The ≥ 50-service cohort rule and the coverage feature (CX8) mitigate this. UP2 and UP5 are less sensitive than UP1.

## 2 · Overutilization

| ID | Feature | Raw statistic *r* | Fields | Dir. | Suppr. | Rationale |
|---|---|---|---|---|---|---|
| OU1 | Non-drug services per patient | Σ srv (drug = N) / Prov `Tot_Benes` | P&S, Prov | ↑ | V/T | Core intensity; drug units excluded |
| OU2 | Beneficiary-days per patient | Σ `Tot_Bene_Day_Srvcs` (drug = N) / Prov `Tot_Benes` | P&S, Prov | ↑ | V/T | Repeated encounters |
| OU3 | Units per beneficiary-day | Σ srv / Σ `Tot_Bene_Day_Srvcs` (drug = N) | P&S | ↑ | V | Multiple units per patient-day (excess units / unbundling) |
| OU4 | Distinct codes per patient | Prov `Tot_HCPCS_Cds` / Prov `Tot_Benes` | Prov | ↑ | T | Breadth per patient |
| OU5 | Volume (log) | log(1 + Σ srv, drug = N) | P&S | ↑ | V | Volume outlier |

## 3 · Atypical service mix

| ID | Feature | Raw statistic *r* | Fields | Dir. | Suppr. | Rationale |
|---|---|---|---|---|---|---|
| SM1 | Mix divergence | Jensen–Shannon divergence between the provider's non-drug HCPCS service distribution and the peer cell's pooled distribution (same year) | P&S | ↑ | V | Billing outside specialty profile |
| SM2 | Rare-code share | share of services in codes billed by < 1 % of the peer cell (same year) | P&S | ↑ | V | Unusual services (understated by suppression) |
| SM3 | Billing concentration | Herfindahl index of payment across HCPCS | P&S | ↑ | V | Single-code mills |
| SM4 | Drug payment share | Σ pay (drug = Y) / Σ pay | P&S | ↑ | V | Part B drug billing out of line |

*Note:* SM1 and SM3 may be collinear; prune if |ρ| > 0.95.

## 4 · Charge inflation

| ID | Feature | Raw statistic *r* | Fields | Dir. | Suppr. | Rationale |
|---|---|---|---|---|---|---|
| CH1 | Submitted / allowed ratio | Σ(srv × `Avg_Sbmtd_Chrg`) / Σ(srv × `Avg_Mdcr_Alowd_Amt`) | P&S | ↑ | V | Charge mark-up |
| CH2 | Standardised payment per patient | Σ(srv × `Avg_Mdcr_Stdzd_Amt`) / Prov `Tot_Benes` | P&S, Prov | ↑ | V/T | Spend per patient, geography-neutral |
| CH3 | Standardised payment per service | Σ(srv × `Avg_Mdcr_Stdzd_Amt`) / Σ srv (drug = N) | P&S | ↑ | V | Spend per unit |
| CH4 | Code-adjusted price index | service-weighted mean over codes of (provider allowed / peer-cell median allowed for that code) | P&S | ↑ | V | Price holding mix fixed |

*Note:* CH2 overlaps Overutilization; moving it is grouping variant S7.

## 5 · Patient-mix mismatch

| ID | Feature | Raw statistic *r* | Fields | Dir. | Suppr. | Rationale |
|---|---|---|---|---|---|---|
| PM1 | Average risk score | `Bene_Avg_Risk_Scre` (average HCC score) | Prov | none | T | Acuity context for intensity |
| PM2 | Intensity residual (services) | residual of log(OU1) regressed on PM1 and chronic-condition percentages within the peer cell, same year (unsupervised) | Prov, P&S | ↑ | V/T | More services than acuity predicts |
| PM3 | Intensity residual (payment) | residual of log(CH2) on the same covariates | Prov, P&S | ↑ | V/T | More spend than acuity predicts |
| PM4 | Dual-eligible share | `Bene_Dual_Cnt` / `Tot_Benes` | Prov | none | T* | Some schemes target dual-eligibles; direction left to model |

\*Sub-group counts < 11 are suppressed in Prov. Missing values get an indicator. `Bene_CC_*`† field names vary by year.

## 6 · DME referral intensity

| ID | Feature | Raw statistic *r* | Fields | Dir. | Suppr. | Rationale |
|---|---|---|---|---|---|---|
| DI1 | DME paid per DME patient | Σ(`Tot_Suplr_Srvcs` × `Avg_Suplr_Mdcr_Pymt_Amt`) (DME-S) / `Tot_Suplr_Benes` (DME-P) | DME-S, DME-P | ↑ | V/T | Spend per referred patient |
| DI2 | Share of panel referred for DME | `Tot_Suplr_Benes` (DME-P) / `Tot_Benes` (Prov) | DME-P, Prov | ↑ | T | Proportion of patients receiving equipment |
| DI3 | High-risk category share | DME payment in high-risk categories / total DME payment | DME-S | ↑ | V | Brace / power-mobility schemes |
| DI4 | DME claims per DME patient | Σ `Tot_Suplr_Clms` (DME-S) / `Tot_Suplr_Benes` (DME-P) | DME-S, DME-P | ↑ | V/T | Repeated orders |
| DI0 | No-DME indicator | 1 if no DME-P row for the NPI-year | DME-P | — | T | Structural zero; DI1–4 = 0 |

## 7 · DME referral concentration

| ID | Feature | Raw statistic *r* | Fields | Dir. | Suppr. | Rationale |
|---|---|---|---|---|---|---|
| DC1 | Claims per supplier | service-weighted mean over HCPCS rows of `Tot_Suplr_Clms` / `Tot_Suplrs` | DME-S | ↑ | V | Few suppliers filling many orders (steering / kickback signal) |
| DC2 | Rental share | services with `Suplr_Rentl_Ind` = Y / total DME services | DME-S | none | V | Rental-heavy schemes; direction left to model |

*Not computable from public data (removed in v1.1):* dominant-supplier share. DME-S reports **counts** of suppliers per referring provider × HCPCS, not supplier identities. DC1 is therefore an approximate concentration measure, and this is stated as a limitation.

---

## Context group — inputs, never reasons

| ID | Feature | Source | Purpose |
|---|---|---|---|
| CX1 | Specialty (`Rndrng_Prvdr_Type`), one-hot | P&S | Peer cell; deconfounding |
| CX2 | Entity type (`Rndrng_Prvdr_Ent_Cd`), one-hot | P&S | Peer cell |
| CX3 | Census division (from `Rndrng_Prvdr_State_Abrvtn`), one-hot | P&S | Peer refinement |
| CX4 | Urban–rural (`Rndrng_Prvdr_RUCA`, grouped into 4 classes) | P&S | Practice environment |
| CX5 | Practice size: log `Tot_Benes` | Prov | Volume context |
| CX6 | Medicare participation (`Rndrng_Prvdr_Mdcr_Prtcptg_Ind`) | P&S | Billing arrangement |
| CX7 | Tenure: years since NPI enumeration (optional) | NPPES | New-entity risk; fairness-sensitive, so kept in context |
| CX8 | **Data coverage**: Σ visible srv (P&S) / `Tot_Srvcs` (Prov) | P&S, Prov | How much of the provider survives suppression |

Under the peer baseline, CX1–CX3 (and year) equal the background, so their attribution is exactly zero. CX4–CX8 may keep small attributions, reported as "peer-group adjustment".

---

## Leakage checklist (Phase 2 QA)

- [ ] No feature uses labels, LEIE data or information from years after *t*.
- [ ] Peer statistics per year use non-injected providers only.
- [ ] Every label-dependent fit (models, calibration, thresholds, tuning) uses training folds and training years only.
- [ ] Provider folds verified disjoint across train/test for every fold.
- [ ] Injected providers are edited on raw rows, re-suppressed, then featurised.
- [ ] Cohort threshold, high-risk DME list and grouping are fixed before any test result is seen.
