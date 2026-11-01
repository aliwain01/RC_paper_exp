#!/usr/bin/env bash
# Phase 4 on real data (DESIGN §17 D123-D124): methods A-E for the explained set under the peer baseline for every
# family and the global baseline for `global_families`, the path-dependent diagnostics, then the attribution report.
# Resumable: re-running this script after an interruption skips the chunks already written.
# Progress: outputs/phase4/explain_progress.log.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=outputs/phase4/run_phase4.log
mkdir -p outputs/phase4

step() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) STEP $*" | tee -a "$LOG"
  if ! "$PY" -m scenario_fraud.cli "$@" >>"$LOG" 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED $*" | tee -a "$LOG"
    exit 1
  fi
}

step phase4 explain
step phase4 diagnostics
step phase4 report
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ALL DONE" | tee -a "$LOG"
