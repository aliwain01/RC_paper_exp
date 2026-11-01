#!/usr/bin/env bash
# Full Phase 3 run (DESIGN §17 D120): the primary design fold by fold, the rolling origins O1-O3, the 2024 showcase
# scores and the G3 report. Every step resumes: re-running this script after an interruption skips finished
# (family, fold) pieces and replays logged tuning trials. Progress: outputs/phase3/train_progress.log.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=outputs/phase3/run_phase3.log
mkdir -p outputs/phase3

step() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) STEP $*" | tee -a "$LOG"
  if ! "$PY" -m scenario_fraud.cli "$@" >>"$LOG" 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED $*" | tee -a "$LOG"
    exit 1
  fi
}

for k in 0 1 2 3 4; do
  step phase3 train --design primary --fold "$k"
done
for d in O1 O2 O3; do
  step phase3 train --design "$d"
done
step phase3 score-showcase
step phase3 report
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ALL DONE" | tee -a "$LOG"
