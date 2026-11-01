# CLAUDE.md: Execution handoff

This research project was designed in `design/` and has now been **built and run end to end on the real data** (Phases 0–7, gates G1–G5, plus the end-of-Phase-6 and end-of-Phase-7 decisions). This file tells you:
- where things are;
- what was decided;
- how to resume or reproduce a run;
- what is left (writing and publication, both the human's).

It does **not** restate the design or the results. The design documents are the source of truth. The results are in `outputs/` (published with the human's approval, D140).

Updated 2026-11-01.

---

## 0. Current status

| Phase | Status | Gate / decision (DESIGN §17) |
|---|---|---|
| 0 Literature & protocol | Done | — |
| 1 Data acquisition (C1–C3) | Done: 28 CMS files 2018–2024, LEIE full file + 42 monthly supplements, 80 files in `data/raw/manifest.json` | G1 decided 2026-10-20 (D117) |
| 2 Cohort, features, folds (C4, C5, C7) | Done; leakage checklist run | G2 decided 2026-10-20, **protocol frozen** (D119; `outputs/phase2/G2_freeze.json`) |
| 3 Models (C8) | Done: 7 families × 5 folds × 10 seeds; budget D120, LR solver D121 | G3 decided 2026-10-23 (D122, D125) |
| 4 Attribution (C9–C12) | Done: methods A–E, peer baseline for all 7 families, global baseline for the best GBM | G4 decided 2026-10-23 (D124) |
| 5 Injection (C6) | Done: 5 draws, injection models and explanations | G5 decided 2026-10-25 (D129) |
| 6 Evaluation (C13) | Done: correctness, consistency, faithfulness, baselines, stress, fairness, stability (10/10/10), sensitivity S1–S13 (S1_W1 not estimable, D134) | §10.8 branch chosen 2026-11-01 (D137) |
| 7 Outputs (C14, C15) | Done: tables T1–T6, figures F2–F8, sample views, queue, draft outline, reproducibility package | Formats confirmed 2026-11-01 (D139) |

**What is left (human work):**
1. **Writing the paper.** `outputs/phase7/draft.md` has the outline with the tables inserted. The abstract, narrative and interpretation are left to the authors, following the branch chosen in D137.
2. **Publication of the paper itself.** The Phase 3–7 results are in the repository (D140). The paper and any further release are the human's call (§8 rule 6).
3. **Keeping the local data.** `data/` (raw, interim, processed: ≈ 25 GB) and the per-provider sample views are not in git. Archive them before the working environment is discarded if they are wanted. Rebuilding from raw data takes days of compute (§9).

Optional: draw 4's injected feature tables were deleted to free disk space. Its models, scores and explanations are intact. `scenario-fraud phase5 build` recreates them if a complete Phase 5 data set is needed.

---

## 1. Project map

```
<project root>/                         git repo; GitHub aliwain01/RC_paper_exp (PUBLIC)
├── HANDOFF.md                          ← this file (navigation, status, execution protocol)
├── README.md                           setup, code layout table, CLI entry points
├── design/                             THE SPECIFICATION (updated as the build deviated; §17 D19–D139)
│   ├── DESIGN_DOCUMENT.md · FEATURE_CATALOG.md · EXPERIMENT_PLAN.md · README.md
│   └── figures/ (+ _src/ drawing scripts)
├── docs/CONTRACTS.md                   interfaces between components (file schemas, CLI contracts)
├── configs/                            every parameter (sources, column maps, labels, experiment, features,
│                                       groups, models, attribution, injection, evaluation, views, record schema)
├── src/scenario_fraud/                 the pipeline, package `scenario_fraud` (component map in README)
├── scripts/run_phase{3,4,5,6,7}.sh     resumable detached runners for the long phases (+ run_phase5_build.sh)
├── tests/                              unit + end-to-end tests on synthetic data (tests/synth.py)
├── pyproject.toml · requirements.lock  environment (Python 3.13, pinned)
├── data/        (git-ignored)          raw/ (+ committed manifest.json), interim/, processed/
├── outputs/
│   ├── phase1/  STATUS.md, G1_report, headers/   ← committed
│   ├── phase2/  G2_freeze.json, G2_report, QA_report  ← committed
│   └── phase3–7/  gate reports, evaluation report, tables, figures, queue, draft, package  ← committed (D140)
│                  phase7/records/ and phase7/views/<record>/ (real NPIs, mock decision)  ← LOCAL ONLY
└── features_health_final.py, rc_final.py, rc_reasoning.py,
    reason_codes_health_final (2).ipynb, synthetic_health_claims (1).csv
                                        LEGACY PROTOTYPE: record behind DESIGN §3.3; never imported or edited
```

Notes:
- Paths contain **spaces and parentheses**. Always quote them.
- Development branch: `claude/vibrant-ride-8m6ihv` (PR aliwain01/RC_paper_exp#1). Commits carry no model identifiers.
- `outputs/phase1/STATUS.md` is the progress log up to G2. This file is the current status.

---

## 2. What to read, in what order

For a new session picking up the finished pipeline:

| Step | Read | Why |
|---|---|---|
| 1 | §0 of this file | Status and what is left. |
| 2 | DESIGN §1 Summary, §2 Terminology | Vocabulary. The term "reason codes" must **not** be used anywhere. |
| 3 | **DESIGN §17 Decision log, D113–D139** | Everything decided or changed during execution, including the gate decisions. Don't reopen a decision without asking. |
| 4 | README.md | Code layout and CLI entry points. |
| 5 | `outputs/phase6/evaluation_report.md`, `outputs/phase7/` | The results (committed, D140). |
| 6 | Per phase, the map in §3 below | The spec behind each component, before changing it. |

Two rules about the figures:
- The design figures restate the doc visually. If a figure and the doc disagree, **the doc wins**; regenerate the figure afterwards.
- Numbers inside design Figs. 5 and 6 and DESIGN §11.4 are illustrative mock-ups, not targets or results.

---

## 3. Phase → where the spec lives → how to run it

| Phase (DESIGN §15) | Components (§8) | Spec | Run (CLI `scenario-fraud`, or script) | Design figures |
|---|---|---|---|---|
| 0 Literature & protocol | — | §3.2, §16, §19, §20; EXPERIMENT_PLAN header | — | — |
| 1 Data acquisition | C1, C2, C3 | §5.2–5.4; §18 | `phase1 …` (verify, harmonise, labels, report) | Fig. 2, 3 |
| 2 Cohort & features | C4, C5, C7 | §7.2, §7.4–7.6; FEATURE_CATALOG (incl. leakage checklist); EXPERIMENT_PLAN §2 | `phase2 cohort / features / folds / qa / report` | Fig. 4 |
| 3 Models | C8 | §7.8; EXPERIMENT_PLAN §2–3 | `scripts/run_phase3.sh` | Fig. 8 |
| 4 Attribution | C9–C12 | §9; EXPERIMENT_PLAN §4 (unit tests, benchmark) | `scripts/run_phase4.sh` | Fig. 5, 6 |
| 5 Injection | C6 | §7.7; EXPERIMENT_PLAN §6 | `scripts/run_phase5_build.sh`, `scripts/run_phase5.sh` | Fig. 7 |
| 6 Evaluation | C13 | §10; EXPERIMENT_PLAN §5, §7–9 | `scripts/run_phase6.sh` | Fig. 8 |
| 7 Outputs & writing | C14, C15 | §11, §12, §13 | `scripts/run_phase7.sh` | Fig. 9 |
| Cross-cutting | C16 | §8 (C16 row); EXPERIMENT_PLAN §10 | run logs in `outputs/phase*/runs/`; `phase7 package` | — |

The full command sequence is in `outputs/phase7/REPRODUCE.md` (generated by `phase7 package`). That file records the git commit, package versions, config hashes, seeds and the G2 freeze.

---

## 4. Code layout (as built)

The layout proposed in the original handoff was created as planned, with one change: the package is `src/scenario_fraud/`. README.md has the component → module table (C1 `acquire` … C16 `cli`/`runlog`/`config`), the config files and the output folders. Interfaces between components are in `docs/CONTRACTS.md`.

---

## 5. Pre-start questions (all answered)

| # | Question | Answer |
|---|---|---|
| Q1 | Git and layout | Git repository on GitHub (aliwain01/RC_paper_exp, public); layout created; `data/` and most of `outputs/` git-ignored (D116). |
| Q2 | Python environment | Python 3.13 venv via `uv`, versions pinned in `requirements.lock` (setup in README). |
| Q3 | Compute | Cloud container, 4 CPUs, 15 GB RAM (processes are killed at ≈ 14 GB) and a fixed writable-disk allowance (≈ 39 GB used when full). The container restarts every few hours, so all long runs are resumable (§7). |
| Q4 | CMS downloads | Automated, from the data.cms.gov catalogue once the human allowed the CMS and OIG hosts in the environment's network settings (D115). Raw CSVs are stored gzip-compressed (D113). |
| Q5 | Internet Archive LEIE snapshots | Not reachable from the environment; proceed without them and report the missing pre-2025 reinstatements as a limitation (D117). |
| Q6 | Years and Part D | 2018–2024; Part D left to follow-up work (D119). |
| Q7 | DESIGN §19 open questions | Answered at G2 (D119): audit capacity 0.1–2 %, Census-division peer cells kept (S10 without), expert review and PU learning stated as limitations. |

---

## 6. Human decision gates (all passed)

Stop and report to the human at each gate. Don't proceed on your own judgement. Each gate report is in `outputs/phase{N}/G{N}_report.md`.

| Gate | When | What was reported | Decision (DESIGN §17) |
|---|---|---|---|
| G1 | End of Phase 1 | Positive NPI counts per year × tier, NPI-match losses, reinstatement gaps, file sizes and row counts | D117: primary tier; 1128(b)(1) only in sensitivity S13; no Internet Archive snapshots |
| G2 | Before Phase 3 | Feature list, protocol diff | D119: protocol frozen; CX7 dropped; no pruning |
| G3 | End of Phase 3 | Detection results with CIs | D122, D125: D74 model selection kept; branch recorded with the G3 report; fold-model ranking rule |
| G4 | Phase 4 benchmark | Throughput, projected wall-clock | D124: \|B\| = 200; explained set as planned for all 7 families; global baseline for the best GBM |
| G5 | Phase 5 calibration | Intensities needed | D129: calibrated grid under a fixed rule; D90–D100 confirmed |
| End of Phase 6 | Evaluation report | §10.8 row the numbers are consistent with (D108 rule) | D137: §10.8 branch chosen; S1_W1 kept not estimable |
| End of Phase 7 | Tables, figures, views | Formats and view content | D139: D112 formats confirmed; views unchanged. D140: results published, per-provider views kept out |

---

## 7. Known technical limitations and how they were handled

| Area | What happened | Handling |
|---|---|---|
| CMS column names | Identical across 2018–2024 in the real headers | Column maps built from the real headers (`configs/column_maps/`); C2 fails loudly on unmapped columns. |
| Suppression | DMEPOS suppresses on claims; blank beneficiary counts | D114, D118 (DI2_missing); injected rows re-suppressed; unobservable injections dropped (D131). |
| LEIE | Few records carry an NPI; reinstatements before 2025 unrecoverable | G1 report; limitation (D117). |
| Memory (≈ 14 GB kill) | Out-of-memory kills in calibration, the injection build, injection training, stability bootstraps, the single-process sensitivity suite and entity-filtered training | Lazy scans and per-year build processes (D127); lean loading of model columns as Float32 (D131, D133, D136; fits unchanged); one process per sensitivity variant (D135). |
| Disk allowance | Filled once in Phase 6 (sensitivity S10 rebuilds cohort and features in a scratch project) | Each injection draw's feature tables take ≈ 1.6 GB, each sensitivity scratch project 0.1–2 GB. Delete `data/processed/sensitivity/{id}/` once `outputs/phase6/sensitivity/{id}.json` exists (finished variants are skipped before setup), and pytest temp dirs. Draw 4's injected features were deleted (§0). |
| Container restarts | Several restarts during the long runs | Every long step is resumable under fingerprints: training (D120), attribution chunks (D123, D130), per-year build (D127), stability runs and sensitivity variants (D132), per-draw and evaluate stamps in the runner scripts. Relaunch the same script; it skips finished work. Changing a fingerprinted config or code invalidates the matching pieces, so don't edit them mid-run. |
| Long LR fits | Elastic-net LR (saga) can take ≈ 1 h per fit on some folds with no log line | Expected (D121); check the process is using CPU before assuming a hang. |
| shap additivity | Method A failed shap's additivity check on a few rows | Kept and flagged; inexact rows excluded in Phase 6 (D126). |
| Running detached | — | `setsid nohup scripts/run_phaseN.sh >/dev/null 2>&1 < /dev/null & disown`. Check liveness with `pgrep -f "^bash scripts/run_phaseN.sh"` (a plain pattern matches the checking shell). Logs: `outputs/phaseN/run_phaseN.log` and `progress.log`. |
| Windows paths | The project path has spaces | Quote all paths; `pathlib` throughout. |
| Figures | Design `_src` scripts write into `design/figures/`; paper figures come from `phase7 figures` | After a design change that affects a figure, edit its script and re-run it. |

---

## 8. Working rules for this project

1. **The design docs are the specification.** If execution forces a deviation (missing field, infeasible step, different result), do three things: (a) implement the smallest change, (b) add a row to DESIGN_DOCUMENT §17 Decision log with the reason, and (c) update the affected doc, feature catalog or experiment-plan section and the figure. Never let the code and the docs disagree silently.
2. **Update §18** when you verify or falsify a claim marked "To verify".
3. **Never look at test-year results before gate G2 freezes the protocol.** (Frozen at D119. Training refuses to run if the frozen hashes change, D73.)
4. **Label-dependent fitting** happens on training folds and years only. Peer statistics stay label-free (FEATURE_CATALOG conventions).
5. **Don't touch or reuse the legacy prototype files**; they model a different dataset.
6. **No outward publication** of data, results or drafts without the human's explicit approval. The repository is **public**. Raw and processed data (including the LEIE file with names and addresses) stay out of git. Nothing that names an individual provider is committed: no NPIs, and no sample decision records or views. The Phase 3–7 results were published with approval (D140); new results need the same approval before they are committed.
7. Report outcomes faithfully at each gate, including failed or underpowered results; the design has planned branches for them (DESIGN §10.8, §16).
8. Run the tests before every push: `.venv/bin/python -m pytest -W error -p no:cacheprovider`.

---

## 9. Resuming or reproducing

1. **Environment:** follow README Setup, then run the test suite.
2. **Data:** if `data/raw/` is empty, re-acquire with `phase1` using `configs/sources.yaml` and the runbook in §10. `phase1 verify` checks the files against `data/raw/manifest.json`. CMS files may have been revised since the recorded download; a checksum mismatch is reported, not ignored.
3. **Pipeline:** run the phases in order with the commands in §3 / `outputs/phase7/REPRODUCE.md`. The G2 freeze record must exist before Phase 3. The gate decisions in §6 are already recorded in the configs, so a rerun does not stop at the gates.
4. **Compute:** expect days in total on a 4-CPU, 15 GB machine. Phase 3 training (LR and EBM), Phase 4 coalition tables, Phase 5 explanations and Phase 6 stability and sensitivity each take many hours. Phase 7 takes minutes.

---

## 10. Data acquisition runbook (component C1), with outcomes

Each source keeps its design-time status. The outcome of the October 2026 acquisition is added. Never guess a direct file URL or a dataset UUID. Take them from the landing page or catalogue entry.

### 10.0 Before downloading anything

- Tell the human the **file names, sources and approximate sizes** and get approval. Part B by Provider & Service is several GB per year as CSV.
- Check free disk space (the environment has a fixed allowance, §7).
- Save every file under `data/raw/{source}/{year}/` and record it in `manifest.json`: source page, file URL or "manual", download date, size, SHA-256.

### 10.1 CMS files (Part B ×2, DMEPOS ×2)

| Dataset | Landing page |
|---|---|
| Physician & Other Practitioners — by Provider and Service | https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service |
| Physician & Other Practitioners — by Provider | https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider |
| DMEPOS — by Referring Provider and Service | https://data.cms.gov/provider-summary-by-type-of-service/medicare-durable-medical-equipment-devices-supplies/medicare-durable-medical-equipment-devices-supplies-by-referring-provider-and-service |
| DMEPOS — by Referring Provider | https://data.cms.gov/provider-summary-by-type-of-service/medicare-durable-medical-equipment-devices-supplies/medicare-durable-medical-equipment-devices-supplies-by-referring-provider |

**Outcome:** downloaded automatically from the data.cms.gov catalogue (`https://data.cms.gov/data.json`), one distribution per dataset and year with a direct CSV link (D115). This needed the CMS hosts allowed in the environment's network settings. 2018–2024 for all four files (28 files, 21.5 GB as CSV, 5.2 GB gzip-compressed, D113), with each year's data dictionary and methodology PDF. The design's fallbacks remain if the catalogue fails:
- the data.gov CKAN catalogue;
- the data.cms.gov API for spot checks only;
- manual browser download registered by C1.

### 10.2 OIG LEIE (labels)

| Item | Location |
|---|---|
| Download page | https://oig.hhs.gov/exclusions/exclusions_list.asp |
| Full current file | `https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv` |
| Record-layout PDF | linked from the download page |
| Monthly supplements | https://www.oig.hhs.gov/exclusions/supplement_archive.asp |
| Authority codes | https://oig.hhs.gov/exclusions/authorities.asp |

**Outcome:**
- Downloaded the full file (09-2026, on 2026-10-19), the record-layout PDF and all 42 supplements online (2025-01 to 2026-09).
- 1128(b)(1) is unsplit in the file (D71).
- The supplements are used only to recover reinstatements.
- The LEIE file contains names, dates of birth and addresses: **never commit it**.

### 10.3 Internet Archive snapshots of the LEIE file

**Outcome:** web.archive.org reset connections from the environment, so no snapshot was fetched. Exclusions that started and ended before 2025 are invisible; this is a reported limitation (D117). Retry only with the human's approval and from an environment that can reach the archive:
- Availability API: `https://archive.org/wayback/available?url=oig.hhs.gov/exclusions/downloadables/UPDATED.csv&timestamp=YYYYMMDD`
- Raw bytes: `https://web.archive.org/web/<timestamp>id_/https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv`

### 10.4 NPPES (optional; tenure feature CX7)

**Outcome:** skipped. CX7 dropped at G2 (D119).

### 10.5 After acquisition (end of C1)

- `manifest.json` complete; checksums verified (`phase1 verify`: 80 files pass).
- One header dump per file and year in `outputs/phase1/headers/` (committed).
