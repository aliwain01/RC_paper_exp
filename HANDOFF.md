# CLAUDE.md: Execution handoff

You are picking up a research project that has been **designed but not built**. This file tells you where things are, what order to read and build them in, and which questions to put to the human before and during execution. It does **not** restate the design; the design documents are the source of truth.

---

## 1. Project map

```
<project root>/
├── CLAUDE.md                          ← this file (navigation + execution protocol)
├── design/                            ← THE SPECIFICATION. Build from here.
│   ├── README.md                      index of the design package
│   ├── DESIGN_DOCUMENT.md             what to build and why (v1.1)
│   ├── FEATURE_CATALOG.md             every feature: formula, CMS fields, suppression behaviour
│   ├── EXPERIMENT_PLAN.md             splits, metrics, parameters, unit tests, table shells
│   └── figures/
│       ├── fig01…fig09_*.png          diagrams of the design (read alongside the doc)
│       └── _src/                      matplotlib scripts that draw the figures
│           ├── vizlib.py              shared drawing helpers
│           ├── fig_arch.py            figs 1–3
│           └── fig_method.py          figs 4–9
├── features_health_final.py           ┐
├── rc_final.py                        │ LEGACY PROTOTYPE (synthetic Kaggle data).
├── rc_reasoning.py                    │ Superseded. Do not build on, import from,
├── reason_codes_health_final (2).ipynb│ or "fix" these files. Keep them only as the
└── synthetic_health_claims (1).csv    ┘ record of why the design changed (DESIGN_DOCUMENT §3.3).
```

Notes on the tree:
- Paths contain **spaces and parentheses**. Always quote them.
- The project is **not a git repository** (see §5, Q1).
- No pipeline code, data folder, environment file or config exists yet. Everything under §4's proposed layout is yours to create.

---

## 2. What to read, in what order

Read the documents in this order before writing code. Section numbers refer to DESIGN_DOCUMENT.md unless stated.

| Step | Read | Why |
|---|---|---|
| 1 | §1 Summary, §2 Terminology | Vocabulary used everywhere. Note that the term "reason codes" must **not** be used anywhere. |
| 2 | **§18 Claim verification status** | Tells you which facts are verified and which you must confirm. Treat anything not marked *Verified* as unconfirmed. |
| 3 | §6 + Fig. 1, §8 **Components to build** | The build inventory (C1–C16): inputs, outputs, dependencies. This is your task list. |
| 4 | §15 Work plan | Phases, exit criteria and the **Phase 1 go/no-go gate**. |
| 5 | §17 Decision log | Decisions already made with the human. Don't reopen them without asking. |
| 6 | Per phase, see the map in §3 below | Read the relevant sections just before building each component. |

Two rules about the figures:
- The figures restate the doc visually. If a figure and the doc disagree, **the doc wins**; regenerate the figure afterwards (§6).
- Numbers inside Figs. 5 and 6 and DESIGN_DOCUMENT §11.4 are illustrative mock-ups, not targets.

---

## 3. Phase → where the spec lives

| Phase (DESIGN §15) | Components (§8) | Read before building | Figures |
|---|---|---|---|
| 0 Literature & protocol | — | §3.2, §16, §19, §20; EXPERIMENT_PLAN header ("freeze before results") | — |
| 1 Data acquisition | C1, C2, C3 | §5.2 sources + access caveat, §5.3 label tiers, §5.4 label rule + reinstatements; §18 rows marked "To verify" | Fig. 2, Fig. 3 |
| 2 Cohort & features | C4, C5, C7 | §7.2, §7.4, §7.5, §7.6; **all of FEATURE_CATALOG.md** (incl. leakage checklist); EXPERIMENT_PLAN §2 | Fig. 4 |
| 3 Models | C8 | §7.8; EXPERIMENT_PLAN §2 (subsampling), §3 | Fig. 8 |
| 4 Attribution | C9, C10, C11, C12 | §9 (all); EXPERIMENT_PLAN §4 incl. **unit tests and timing benchmark** | Fig. 5, Fig. 6 |
| 5 Injection | C6 | §7.7; EXPERIMENT_PLAN §6 | Fig. 7 |
| 6 Evaluation | C13 | §10; EXPERIMENT_PLAN §5, §7, §8, §9 (table shells) | Fig. 8 |
| 7 Outputs & writing | C14, C15 | §11, §12; §13 limitations for the write-up | Fig. 9 |
| Cross-cutting | C16 | §8 (C16 row); EXPERIMENT_PLAN §10 | — |

---

## 4. Proposed code layout (create it; nothing exists yet)

This layout is a suggestion that keeps component IDs traceable. Confirm it with the human (§5, Q1) before creating it.

```
<project root>/
├── pyproject.toml / requirements.txt + lockfile
├── configs/
│   ├── column_maps/{dataset}_{year}.yaml    C2: written by hand from CMS data dictionaries / file headers
│   ├── groups.yaml                          scheme groups + context (FEATURE_CATALOG)
│   ├── features.yaml                        feature definitions
│   ├── labels.yaml                          authority tiers, W, horizon
│   └── experiment.yaml                      folds, seeds, k values, injection grid
├── data/            (gitignored)
│   ├── raw/{source}/{year}/                 untouched downloads + manifest.json (C1)
│   ├── interim/                             harmonised Parquet (C2)
│   └── processed/                           cohort, labels, features, folds
├── src/<package>/
│   ├── acquire.py        C1     ├── injection.py   C6     ├── attribution/
│   ├── harmonise.py      C2     ├── splits.py      C7     │   ├── background.py  C9
│   ├── labels.py         C3     ├── train.py       C8     │   ├── coalition.py   C10
│   ├── cohort.py         C4     ├── evaluate.py    C13    │   ├── summed_shap.py C11
│   ├── features.py       C5     ├── render.py      C14    │   └── glassbox.py    C12
│   └── reports.py        C15    └── cli.py         C16 (one entry point per phase)
├── tests/                         unit tests from EXPERIMENT_PLAN §4 + leakage checks
├── outputs/{run_id}/              metrics, attribution stores, tables, figures, run log
└── design/                        (existing; update when the design changes)
```

---

## 5. Ask the human before starting

These are open decisions or resources the design depends on. Don't assume answers.

| # | Question | Why it blocks |
|---|---|---|
| Q1 | May I `git init`, create the layout in §4, and add a `.gitignore` excluding `data/`? | No version control exists; reproducibility (C16) assumes it. |
| Q2 | Which Python environment should I use? | The machine that authored the design had Python 3.9 at `D:\Python` **without** matplotlib, xgboost, shap or interpret. Expect to create a fresh venv. Confirm the Python version (3.10+ recommended) and permission to install packages. |
| Q3 | How much disk, RAM and CPU is available, and is there a time budget for compute? | Phase 1 downloads several years of multi-GB CSVs. EBM training and coalition tables (DESIGN §9.5, EXPERIMENT_PLAN §10) are hours-scale. Sets \|B\|, explained-set size and seed counts. |
| Q4 | How will CMS files be downloaded? | data.cms.gov returned **HTTP 403 to automated requests** during design. The human may need to download in a browser and drop the files into `data/raw/`. C1 must support "register manually downloaded file". |
| Q5 | Is retrieving Internet Archive snapshots of the LEIE file acceptable, and who retrieves them? | Needed for reinstatement recovery (DESIGN §5.4). Snapshot availability beyond 2017 and 2019 is unknown. |
| Q6 | Confirm the year range (2018–2024) and that Part D stays out of scope. | DESIGN §17 D12, D17; §19 Q4. |
| Q7 | Answers to DESIGN §19 open questions (audit-capacity range, peer granularity, expert review, PU learning). | Some change the protocol, which must be frozen before Phase 3. |

---

## 6. Human decision gates during execution

Stop and report to the human at each gate. Don't proceed on your own judgement.

| Gate | When | What to report | Decision the human makes |
|---|---|---|---|
| G1 | End of Phase 1 | Positive NPI counts per year × label tier; NPI-match losses; Internet Archive gap intervals; whether DMEPOS 2024 exists; actual file sizes and row counts | Go / switch primary label tier / extend years (DESIGN §14 first row) |
| G2 | Before Phase 3 | Final feature list after pruning; any feature that couldn't be computed; protocol diff vs EXPERIMENT_PLAN | **Freeze the protocol** |
| G3 | End of Phase 3 | Detection results with CIs | Which DESIGN §10.8 / §16 branch the paper follows |
| G4 | Phase 4 timing benchmark | Predictions/s, projected wall-clock for coalition tables | \|B\| and explained-set size |
| G5 | Phase 5 calibration | Injection intensities actually needed | Accept the calibrated grid |

---

## 7. Known code and technical limitations

These are facts to check at build time, not design content.

| Area | Limitation / risk | What to do |
|---|---|---|
| CMS column names | Change across years. Field lists in the design came partly from catalogue mirrors, not the raw headers. | Build C2 column maps from the **actual CSV headers** of each downloaded year; fail loudly on unmapped columns. Update FEATURE_CATALOG if a field is missing or renamed. |
| `Bene_CC_*` fields | Exact names per year unverified. | Resolve in C2; record the mapping. |
| LEIE `EXCLTYPE` format | String encoding of authority codes unconfirmed. | Read the OIG record-layout PDF / file before writing C3. |
| `shap` | Interaction values exist only for path-dependent TreeSHAP. Interventional TreeSHAP with categorical splits may be unsupported in some versions. | The design avoids both (method C uses the coalition table; categoricals are one-hot). Still pin the `shap` version and run EXPERIMENT_PLAN §4 unit tests. |
| InterpretML EBM | Slow on millions of rows, especially with pairwise terms. | Use the shared negative subsample (EXPERIMENT_PLAN §2); cap pair terms; time it first. |
| Coalition table (C10) | 256 × \|B\| predictions per provider; memory blow-up if built naively. | Build hybrid rows in chunks; batch predictions; benchmark before the full run (gate G4). |
| Suppression | CMS drops rows from ≤ 10 beneficiaries; injected data must be re-suppressed. | Follow FEATURE_CATALOG suppression notes and EXPERIMENT_PLAN §6. |
| Windows paths | The project path has spaces; the shell may be PowerShell or Git Bash. | Quote all paths; use `pathlib`; avoid shell-specific scripts. |
| Figures | `_src` scripts need `matplotlib` + `numpy`. They write into `design/figures/`. | After any design change that affects a figure, edit the script and re-run (commands in `design/README.md`). |

---

## 8. Working rules for this project

1. **The design docs are the specification.** If execution forces a deviation (missing field, infeasible step, different result), do three things: (a) implement the smallest change, (b) add a row to DESIGN_DOCUMENT §17 Decision log with the reason, and (c) update the affected doc, feature catalog or experiment-plan section and the figure. Never let the code and the docs disagree silently.
2. **Update §18** when you verify or falsify a claim marked "To verify".
3. **Never look at test-year results before gate G2 freezes the protocol.**
4. **Label-dependent fitting** happens on training folds and years only. Peer statistics stay label-free (FEATURE_CATALOG conventions). Run the leakage checklist at the end of Phase 2.
5. **Don't touch or reuse the legacy prototype files**; they model a different dataset.
6. **No outward publication** of data, results or drafts without the human's explicit approval.
7. Report outcomes faithfully at each gate, including failed or underpowered results; the design has planned branches for them (DESIGN §10.8, §16).

---

## 9. Quick start (after §5 answers)

1. Create the environment and layout (Q1, Q2).
2. Phase 1: implement C1 → acquire or register files using the runbook in §10 → C2 column maps from real headers → C3 labels → produce the **G1 report**.
3. Stop at G1 and wait for the human's decision before Phase 2.

---

## 10. Data acquisition runbook (component C1)

Each source has a status. **Checked** = the page or endpoint was reached or confirmed during design (Oct 2026). **Unconfirmed** = believed correct, verify first. Never guess a direct file URL or a dataset UUID. Take them from the landing page or catalogue entry.

### 10.0 Before downloading anything

- Tell the human the **file names, sources and approximate sizes** and get approval. Part B by Provider & Service is several GB per year as CSV.
- Check free disk space.
- Save every file under `data/raw/{source}/{year}/` and record it in `manifest.json`: source page, file URL or "manual", download date, size, SHA-256.

### 10.1 CMS files (Part B ×2, DMEPOS ×2)

| Dataset | Landing page | Status |
|---|---|---|
| Physician & Other Practitioners — by Provider and Service | https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service | Checked (via search; 403 to automated fetch) |
| Physician & Other Practitioners — by Provider | https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider | Checked (API docs page via search) |
| DMEPOS — by Referring Provider and Service | https://data.cms.gov/provider-summary-by-type-of-service/medicare-durable-medical-equipment-devices-supplies/medicare-durable-medical-equipment-devices-supplies-by-referring-provider-and-service | Checked (API docs page via search) |
| DMEPOS — by Referring Provider | https://data.cms.gov/provider-summary-by-type-of-service/medicare-durable-medical-equipment-devices-supplies/medicare-durable-medical-equipment-devices-supplies-by-referring-provider | Checked (API docs page via search) |
| Mirrors / per-year file lists | data.gov catalogue, e.g. https://catalog.data.gov/dataset/medicare-physician-other-practitioners-by-provider-and-service and https://catalog.data.gov/dataset/medicare-durable-medical-equipment-devices-supplies-by-referring-provider | Checked |

**Route A (preferred if it works): data.gov catalogue.**
Each catalogue entry lists per-year "distributions" with download URLs. These may point at data.cms.gov too. data.gov also offers dataset metadata via its CKAN API (`https://catalog.data.gov/api/3/action/package_show?id=<dataset-slug>`); the endpoint pattern is unconfirmed for these slugs. Read the per-year CSV links from the `resources` list.

**Route B: data.cms.gov API.**
The landing page's "API" tab shows a dataset-version UUID per year. The API returns paged JSON, which is too slow for ~10 M-row files. Use it only for spot checks (e.g. column names), not bulk download.

**Route C (fallback): manual download.**
data.cms.gov's bot protection returned **HTTP 403** to automated clients during design. If routes A and B fail, ask the human to download each year's CSV from the landing page's "Download" option in a normal browser and place it in `data/raw/…`. C1 then registers the files (checksum + manifest) without fetching.

**Years:** 2018–2024 for all four files. If time allows, 2015–2017 too, for the G1 fallback of extending training years. Also save each year's **data dictionary PDF** and **methodology PDF**, linked from the landing pages. C2 column maps are built from the actual CSV headers, with the dictionaries as reference.

### 10.2 OIG LEIE (labels)

| Item | Location | Status |
|---|---|---|
| Download page | https://oig.hhs.gov/exclusions/exclusions_list.asp | Checked |
| Full current file | `https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv`. This URL was archived with HTTP 200 in 2017 and 2019. Confirm it is still the current link on the download page. | Unconfirmed for 2026 |
| Record-layout PDF | linked from the download page | Read before writing C3 (defines `EXCLTYPE` format) |
| Monthly supplements (exclusions + reinstatements) | https://www.oig.hhs.gov/exclusions/supplement_archive.asp (≈ last 12 months; earliest listed Jan 2025) | Checked |
| Authority codes | https://oig.hhs.gov/exclusions/authorities.asp | Checked |

Download the full file **and** every available reinstatement supplement. Don't merge them blindly: OIG says to use one strategy for keeping a list current. Here, the supplements serve only to recover reinstated parties.

### 10.3 Internet Archive snapshots of the LEIE file (reinstatement recovery)

- **List snapshots.** Availability API (checked, works): `https://archive.org/wayback/available?url=oig.hhs.gov/exclusions/downloadables/UPDATED.csv&timestamp=YYYYMMDD`. It returns the closest snapshot. Snapshots near 2017-01-25 and 2019-01-10 were confirmed; none were returned near 2022-01 or 2024-01. For a full listing use the CDX API, `https://web.archive.org/cdx/search/cdx?url=oig.hhs.gov/exclusions/downloadables/UPDATED.csv&output=json`. It couldn't be reached from the design environment, so it is unconfirmed.
- **Fetch the raw bytes.** `https://web.archive.org/web/<timestamp>id_/https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv`. The `id_` suffix returns the original file without the Wayback toolbar.
- Also try older or alternative OIG paths if the CDX listing shows the file moved. Record every snapshot used and the uncovered intervals for the G1 report.
- Ask the human first (Q5).

### 10.4 NPPES (optional; tenure feature CX7)

- Monthly full-replacement files are listed at https://download.cms.gov/nppes/NPI_Files.html (unconfirmed during design). The file is very large, several GB zipped.
- Only enumeration date, entity type and taxonomy are needed. Ask before downloading. If it is skipped, drop CX7 and log it in DESIGN §17.

### 10.5 After acquisition (end of C1)

- `manifest.json` complete; checksums verified.
- One header dump per file and year (column names + row count) saved to `outputs/phase1/headers/`. These are inputs to C2 and to the G1 report.
- Any source that couldn't be obtained is listed in the G1 report together with its effect on the design.
