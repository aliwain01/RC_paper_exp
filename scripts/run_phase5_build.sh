#!/usr/bin/env bash
# Phase 5 build on real data (DESIGN §17 D127): each planned year of each draw in its own process, so a year's
# service-level rows are freed before the next is read. Resumable: finished years are skipped.
# Usage: scripts/run_phase5_build.sh [draw ...]   (default: draw 0)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=outputs/phase5/run_phase5_build.log
mkdir -p outputs/phase5
for d in "${@:-0}"; do
  for y in 2018 2019 2020 2021 2022 2023; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) STEP build draw $d year $y" | tee -a "$LOG"
    if ! "$PY" -m scenario_fraud.cli phase5 build --draw "$d" --year "$y" >>"$LOG" 2>&1; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED build draw $d year $y" | tee -a "$LOG"
      exit 1
    fi
  done
  "$PY" -m scenario_fraud.cli phase5 build --draw "$d" >>"$LOG" 2>&1   # all years done: writes the draw summary
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ALL DONE" | tee -a "$LOG"
