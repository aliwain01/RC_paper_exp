#!/usr/bin/env bash
# Phase 5 on real data after G5 (DESIGN §17 D127-D131): for each draw, build (one process per year), train the
# injection models (use b), score injected providers with the LEIE models (use a) and explain them; then the G5
# report. Resumable: every step skips work finished under the same settings, so re-running this script after an
# interruption continues where it stopped. Progress: outputs/phase5/run_phase5.log and outputs/phase5/progress.log.
# Usage: scripts/run_phase5.sh [draw ...]   (default: every draw in configs/injection.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=outputs/phase5/run_phase5.log
mkdir -p outputs/phase5

step() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) STEP $*" | tee -a "$LOG"
  if ! "$PY" -m scenario_fraud.cli "$@" >>"$LOG" 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED $*" | tee -a "$LOG"
    exit 1
  fi
}

if [ $# -gt 0 ]; then DRAWS="$*"; else
  DRAWS=$("$PY" -c "import yaml; print(' '.join(str(d) for d in range(yaml.safe_load(open('configs/injection.yaml'))['draws'])))")
fi
for d in $DRAWS; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) STEP build draw $d" | tee -a "$LOG"
  if ! scripts/run_phase5_build.sh "$d" >>"$LOG" 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED build draw $d" | tee -a "$LOG"
    exit 1
  fi
  step phase5 train --draw "$d"
  step phase5 stress --draw "$d"
  step phase5 explain --draw "$d"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DRAW $d DONE" | tee -a "$LOG"
done
step phase5 report
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ALL DONE" | tee -a "$LOG"
