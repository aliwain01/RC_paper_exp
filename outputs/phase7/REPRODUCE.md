# Reproduce

Environment: `uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python -r requirements.lock && uv pip install --python .venv/bin/python -e . --no-deps`.

Then, with the data registered in `data/raw/manifest.json` (HANDOFF §10):

```bash
S=.venv/bin/scenario-fraud
$S phase1 verify && $S phase1 harmonise && $S phase1 labels && $S phase1 report      # gate G1
$S phase2 cohort && $S phase2 features && $S phase2 folds && $S phase2 qa && $S phase2 report  # gate G2 (freeze)
$S phase3 train && for d in O1 O2 O3; do $S phase3 train --design $d; done && $S phase3 score-showcase && $S phase3 report  # G3
$S phase4 explained-set && $S phase4 bench && $S phase4 report                      # gate G4: |B|, explained set
$S phase4 explain && $S phase4 diagnostics && $S phase4 report
$S phase5 calibrate && $S phase5 report                                            # gate G5: intensity grid
$S phase5 build && $S phase5 train && $S phase5 stress && $S phase5 explain && $S phase5 report
$S phase6 evaluate && $S phase6 stability && $S phase6 sensitivity && $S phase6 report
$S phase7 tables && $S phase7 figures && $S phase7 render-sample && $S phase7 queue && $S phase7 draft && $S phase7 package
```

Recorded state: git 2d53bab1880ddba18e74be2e3cffbb1e1fcdea06, 384 run logs, 12 config files hashed in `reproducibility.json`.
