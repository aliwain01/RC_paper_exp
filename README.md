# Scenario-level explanations for health-care fraud detection

Research pipeline specified in [design/](design/README.md) (DESIGN_DOCUMENT v1.1). [HANDOFF.md](HANDOFF.md) is the execution protocol. Current progress is in [outputs/phase1/STATUS.md](outputs/phase1/STATUS.md).

The legacy prototype files in the root (`features_health_final.py`, `rc_final.py`, `rc_reasoning.py`, the notebook and the synthetic CSV) are kept only as the record behind DESIGN §3.3. Nothing imports them.

## Setup

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python -e . --no-deps
.venv/bin/python -m pytest
```

## Layout

| Path | Contents |
|---|---|
| `configs/` | `sources.yaml` (C1), `column_maps/` (C2), `labels.yaml` (C3), `experiment.yaml` (cohort, splits, QA, G2), `census_divisions.yaml` (C4), `features.yaml` + `groups.yaml` (C5) |
| `src/scenario_fraud/` | `acquire` C1 · `harmonise` C2 · `labels` C3 · `g1_report` (gate G1) · `cohort` C4 · `features` + `groups` C5 · `splits` C7 · `qa` + `g2_report` (Phase 2 QA, gate G2) · `provenance` (synthetic / unknown input flags) · `cli`/`runlog`/`config` C16 |
| `data/` | raw downloads + `manifest.json`, interim Parquet, processed tables (git-ignored) |
| `outputs/phase1/` | header dumps, run logs, G1 report |
| `outputs/phase2/` | run logs, QA report (incl. leakage checklist), G2 report, `G2_freeze.json` (protocol hashes recorded before any Phase 3 output) |
| `tests/` | unit and end-to-end tests; `synth.py` writes synthetic harmonised files for tests only (refuses the checkout and projects with real data) |

Entry points: `scenario-fraud phase1 --help`, `scenario-fraud phase2 --help` (cohort → features → folds → qa → report).
