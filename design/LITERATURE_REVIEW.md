# Phase 0: literature review and claim verification

13 October 2026 · DESIGN §15 Phase 0 ("read prior work; freeze RQs, metrics, groups"). Produced by four research agents, each followed by an independent agent that tried to refute its claims. Statuses below are the **post-check** statuses.

**Access limits.** From the build environment, OIG, CMS, NAIC, AMA, GAO, journal sites, arXiv and PMC were all unreachable (DNS failure or proxy 403). The only reachable sources were GitHub and web-search results. So:

- every publisher fact here comes from a **search-result snippet** of the publisher's page, or from **third-party copies** of the data files on GitHub;
- no full paper text was read;
- "authoritative" means the publisher's own URL appeared in the results, not that the document was opened.

Items marked ⚠ must be checked by hand before they are cited in the paper.

The protocol freeze (research questions, metrics, groups) stays with the human at gate G2. This review only feeds it.

---

## 1. LEIE file format (DESIGN §5.3; HANDOFF §7)

| Finding | Status | Consequence |
|---|---|---|
| `UPDATED.csv` header is exactly `LASTNAME,FIRSTNAME,MIDNAME,BUSNAME,GENERAL,SPECIALTY,UPIN,NPI,DOB,ADDRESS,CITY,STATE,ZIP,EXCLTYPE,EXCLDATE,REINDATE,WAIVERDATE,WVRSTATE` | confirmed (third-party captures of the live file, Aug 2026) | `configs/labels.yaml` layout is correct |
| The OIG record-layout PDF lists the same 18 fields with lengths (EXCLTYPE 9 chars) and **does not enumerate EXCLTYPE values** | partially confirmed | the meaning of codes comes from the OIG authorities page; values are checked against the real file by C3 |
| EXCLTYPE values seen in real files: 1128a1–a4, 1128b1–b8, 1128b11, 1128b12, 1128b14, 1128b15, 1128b16, 1128Aa, 1156, **1160**, `BRCH CIA`, `BRCH SA` | confirmed (several independent copies) | `1160` added to the map |
| **1128(b)(1) is a single code `1128b1`** in the file, not split into (A)/(B) | confirmed | **G1 decision**: the primary tier names (b)(1)(A) only; the unsplit code also covers (b)(1)(B) |
| No encoding for 1128(c)(3)(G)(i)/(ii) was found in any copy | not found | the prior-work tier's (c)(3)(G) entries may never match |
| Some older full files pad EXCLTYPE to 9 characters (`1128a1   `) | confirmed | handled: values are stripped on read |
| Dates are `YYYYMMDD`; empty reinstatement/waiver dates are `00000000`; missing NPI is `0000000000` (~90 % of rows); empty DOB is blank | confirmed | layout config is correct |
| The full file holds only exclusions in effect: REINDATE is `00000000` everywhere; reinstatement dates appear only in REIN supplements | confirmed | as assumed in DESIGN §5.4 |
| Supplements: `https://oig.hhs.gov/exclusions/downloadables/{YYYY}/{YYMM}excl.csv` and `…rein.csv` (2024–2026); same layout from June 2013 on; files from May 2013 or earlier lack NPI and waiver fields | partially confirmed | `configs/sources.yaml` updated; old supplements are rejected by the reader (they cannot be NPI-matched anyway) |
| Only ~12 months of supplements are online; not every month has a file; one 2026 "supplement" was reportedly the full file | partially confirmed | C1/C3: inspect each supplement's row count before use |
| `UPDATED.csv` URL still live (Aug 2026) | confirmed (third-party) | runbook URL kept |

## 2. CMS file columns (DESIGN §5.2, §18 "To verify")

| Finding | Status | Consequence |
|---|---|---|
| Part B by Provider and Service: same 28-column header for data years 2013–2024 in the current release; every template name matches | confirmed (third-party schema comparison) | template correct |
| Part B by Provider: 81 columns, all template names present, header identical 2019–2024 (2018 likely) | confirmed for 2019–2024 | template correct |
| **`Bene_CC_*` names:** 25 columns in the current release, e.g. `Bene_CC_BH_Depress_V1_Pct`, `Bene_CC_PH_Diabetes_V2_Pct`; the V1/V2 suffix is fixed per condition; unchanged 2019–2024. Older releases used 16 names such as `Bene_CC_Dbts_Pct` | confirmed (current); partially confirmed (older) | §18 row resolved for the current release; the prefix rule handles both |
| By-Provider splits drug and medical totals (`Drug_*`, `Med_*`, `*_Sprsn_Ind`) | confirmed | dropped by template prefix (unused) |
| Suppressed numeric values are blank; `*`/`#` occur in the suppression-indicator columns | partially confirmed (numeric columns only; `#` also occurs in addresses) | default `null_values: [""]` is right |
| Provider gender is absent from the current release (present in older ones) | partially confirmed | kept in drop lists for old vintages |
| **DMEPOS by Referring Provider and Service spells `HCPCS_CD`** (capital CD); other template names match; 33 columns | confirmed | template fixed |
| DMEPOS by Referring Provider: 97 columns incl. `Rfrg_NPI`, `Tot_Suplr_Benes`, `DME_Tot_Suplr_Benes`, category totals with `DME_`/`POS_`/`Drug_` prefixes | confirmed | template maps the three used columns, drops the rest by prefix |
| DMEPOS service-level 2018–2020 layout in the current release | not found | Phase 1 header dumps decide |
| DMEPOS by Referring Provider and Service **2024 is published** | confirmed (CMS api-docs snippet) | G1 still records it from the real download |
| DMEPOS by Referring Provider **2024** | partially confirmed (search summary only) | G1 |
| ⚠ **DMEPOS suppression:** CMS documents mix "10 or fewer beneficiaries" with "10 or fewer claims" | partially confirmed | DESIGN §5.2 states beneficiaries for both sources. Phase 5 re-suppression of injected DME rows must follow the methodology PDF; the threshold field is configurable |

## 3. Background claims (DESIGN §18 rows "cite primary source in Phase 0")

| Claim | Status | Citation |
|---|---|---|
| CMS Fraud Prevention System applies predictive analytics to Medicare FFS claims before payment since **30 June 2011** (statute: 1 July 2011 start in 10 states; GAO: July 2011) | confirmed | CMS, *Report to Congress: Fraud Prevention System, Second Implementation Year* (2014); Small Business Jobs Act of 2010, P.L. 111-240 §4241, 42 U.S.C. 1320a-7m |
| NAIC Model Bulletin *Use of Artificial Intelligence Systems by Insurers* adopted **4 Dec 2023** | confirmed | NAIC press release, 4 Dec 2023 |
| Adoption: "over half of all states have adopted this or similar guidance" (NAIC, 16 Dec 2025) | confirmed; ⚠ exact current count not established | NAIC statement; NAIC implementation map |
| CPT 2021: office E&M 99202–99215 selected by MDM or time; **99201 deleted** 1 Jan 2021 | confirmed | AMA, *CPT E/M Office … Code and Guideline Changes* |
| CPT 2023: framework extended to ED 99281–99285 (MDM only, no time) and hospital inpatient/observation, 1 Jan 2023 | confirmed | AMA, *2023 E/M Descriptors and Guidelines* |
| CBRs are educational peer-comparison reports, "not an indication or precursor to an audit", **but "selected providers may be referred for additional review and education"** | partially confirmed | CMS FFS compliance programs page; CBR sample report (RELI Group) |
| ⚠ CBRs are confidential | not verified (secondary sources only) | — |
| CBR (and PEPPER) distribution paused from January 2024; CMS RFI with responses due 19 Aug 2024 | partially confirmed (pause date from trade associations; RFI from CMS) | CMS CBR/PEPPER RFI (July 2024) |
| CBR distribution resumed by Oct 2026 | not found (PEPPER relaunched Dec 2025) | — |
| Part B suppression: records from ≤ 10 beneficiaries excluded; summary sub-group counts < 11 suppressed | confirmed | CMS Medicare Physician & Other Practitioners Methodology |

## 4. Prior work and novelty (DESIGN §3.2, §4.3)

### 4.1 Closest prior work

| Work | What it does | Overlap |
|---|---|---|
| Hancock, Bauder, Wang & Khoshgoftaar (2023), *J. Big Data* 10:154 | ensemble feature selection from tree importances on Part B/Part D with LEIE labels | Medicare + LEIE explainability; ⚠ whether it uses SHAP is unresolved |
| FAU line (Bauder, Herland, Johnson, Khoshgoftaar) | LEIE labels from a fraud-indicative subset of authorities; records before the exclusion end marked fraudulent | **direct precedent for fraud-related authority subsets** |
| Johnson & Khoshgoftaar (2023), *SN Comput. Sci.* 4(4) | Part B/D/DMEPOS 2013–2019; ⚠ reportedly a "cumulative exclusion list" from yearly archived LEIE files | possible precedent for reinstatement recovery (check by hand) |
| Herland, Khoshgoftaar & Bauder (2018), *J. Big Data* 5 | Part B, Part D, DMEPOS + LEIE, three learners | setting and performance comparison; no explanation comparison seen |
| Shekhar, Leder-Luis & Akoglu (2023), NBER w30946 (journal version in *JPAM*, "Can Machine Learning Target Health Care Fraud?") | unsupervised multi-view detector on inpatient claims, each view a "reason", peer-based DRG view, validated on DOJ cases | reason-level explanations, peer comparison |
| Tajrobehkar et al. (medRxiv, posted 1 Jan 2025) | LEIE label critique; physician-labelled ophthalmology data; feature SHAP | label quality; SHAP on Part B |
| Liang, Bauder & Khoshgoftaar (ICTAI 2024); Hancock, Khoshgoftaar & Liang (2025) | SHAP-driven feature selection on Medicare | SHAP on Medicare (for selection) |
| RonCom/medicare-fwa (GitHub, Sep 2026, not peer-reviewed) | specialty×year peers, robust z + empirical Bayes, scheme-like metrics, "top three drivers", LEIE 3-year window with fraud subset and reinstatement recovery from snapshots | **closest overlap with contributions 1–2**; no group Shapley, no injection |
| Park & Brociner (2026), arXiv 2609.28477 | descriptive provider behaviour profiles for FWA review | peer/composition features; review framing |
| MDPI *MAKE* 8(9):279 (2026), SHAP-guided RAG audit triage | XGBoost + summed group SHAP on the Kaggle provider-fraud set, "risk archetypes", LLM narratives | grouped health-fraud explanations and audit output; post hoc sums, no LEIE, no injection |
| Jullum, Redelmeier & Aas (2021), groupShapley | Shapley values with feature groups as players | the group Shapley method itself |
| Grabisch & Roubens (1999); SHAP-IQ (NeurIPS 2023); shapiq (NeurIPS 2024 D&B) | Shapley interaction index; any-order interaction tooling | the group interaction method itself |
| Owen values (`shap` PartitionExplainer/ExactExplainer); O-Shap (2026) | hierarchical attribution; group definitions matter | method comparison |
| Au et al. (2022) *DMKD*; Bordt & von Luxburg (2023) AISTATS | grouped importance; Shapley ↔ GAM equivalence | consistency logic (GBM vs GA²M) |
| Group SHAP in financial fraud (*ESWA* 2022; IEEE BigComp 2026) | grouped attributions, non-health | grouping in fraud |
| Zhou, Booth, Ribeiro & Shah (2022), AAAI | "semi-natural" ground truth by modifying real data to test attributions (images, text) | **methodological precedent for injection** |
| Tritscher et al. (2022, 2024) | simulated ERP fraud data with ground-truth explanations | fraud-domain precedent for explanation ground truth (simulated) |
| XAI-Bench, OpenXAI, ADBench | synthetic tabular ground truth; anomaly injection for detection | injection exists, not for Medicare explanations |
| Merrick & Taly (2020); Albini et al. (2022) | reference distributions, counterfactual baselines | peer baseline / counterfactuals |
| Bhatt et al. (2020); Jesus et al. (2021); Amarasinghe et al. (2024) | explanations in deployment; fraud analysts' use of explanations | audience/lifecycle framing |
| CMS "Crushing Fraud Chili Cook-Off" (2025; Milliman winner, "glass box") | explainable ML challenge on Medicare FFS claims | deployment context |

### 4.2 Novelty verdicts per contribution (DESIGN §4.3)

| # | Contribution | Verdict |
|---|---|---|
| 1 | Scheme taxonomy + peer-relative features for public Part B + DMEPOS, suppression documented | no peer-reviewed precedent found; close non-peer-reviewed overlap (RonCom/medicare-fwa) |
| 2a | Fraud-related LEIE authorities; reinstatement recovery | **not new** (FAU authority subsets; reinstatement recovery possibly in Johnson & Khoshgoftaar 2023 ⚠) |
| 2b | Explicit look-back window W, label horizon, quantified recovery gaps | no peer-reviewed precedent found |
| 3a | Group Shapley values and group interaction indices as methods | **not new** (groupShapley; Grabisch & Roubens; shapiq) |
| 3b | One exact coalition-table computation giving B, C, E and deletion curves, applied to fraud with a peer baseline | no precedent found in fraud/Medicare; note that R `shapr` also has a "coalition table" |
| 4 | Explanation correctness by injecting schemes into real provider profiles | no precedent found on Medicare/LEIE (an absence claim from snippet-limited search) |
| 5 | Glass box (EBM/GA²M/LR) vs GBM on performance **and** explanations, Medicare/LEIE | no precedent found |
| 6 | Audience-aware lifecycle with provenance chain | no precedent found in Medicare fraud literature |

**Suggested framing for §3.2:** the contribution is the *combination* of
- exact group-level attributions (including interactions) against a peer baseline,
- **measured** correctness on injected real profiles, and
- a glass-box comparison of explanations,

on public Medicare data. The individual methods and the fraud-authority labelling are prior art and should be cited as such.

## 5. Proposed design changes (for the human at G2; logged in DESIGN §17)

1. §3.2: restate "what is not new" with the works above, and narrow contributions 2 and 3 as in §4.2.
2. §5.3: decide whether the unsplit `1128(b)(1)` joins the primary tier (G1).
3. §5.2: DMEPOS suppression threshold (beneficiaries vs claims) to be confirmed from the methodology PDF.
4. §11.2 / §18: CBR wording keeps the referral caveat; confidentiality unverified.
5. §20: add the works in §4.1.

## 6. To check by hand before citing

1. Johnson & Khoshgoftaar (2023): whether the cumulative exclusion list recovers reinstated providers.
2. ResearchGate 389215397 "Interpretable Machine Learning Models for Medicare Fraud Detection": unidentified; could compete with contribution 5.
3. The source of a fully synthetic 60,000-claim Medicare/Medicaid study with injected fraud + SHAP.
4. Whether Hancock et al. (2023) uses SHAP.
5. Whether Shekhar et al. tested their explanations with injection.
6. Exact NAIC adoption count from the NAIC map PDF.
7. DMEPOS methodology suppression rule.
