#!/usr/bin/env bash
# Phase 7 on real data (DESIGN §11, §12; D112): paper tables and figures, sample analyst / oversight views (mock ADR
# decision), the unit-manager queue, the draft outline and the reproducibility package, one step per process so a
# failure names its step. Outputs stay under outputs/phase7/ (git-ignored; HANDOFF §8 rule 6). Progress:
# outputs/phase7/run_phase7.log. Usage: scripts/run_phase7.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=outputs/phase7/run_phase7.log
mkdir -p outputs/phase7

step() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) STEP $*" | tee -a "$LOG"
  if ! "$PY" -m scenario_fraud.cli "$@" >>"$LOG" 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED $*" | tee -a "$LOG"
    exit 1
  fi
}

step phase7 tables
step phase7 figures
step phase7 render-sample --n 3 --action adr
step phase7 queue
step phase7 draft
step phase7 package
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ALL DONE" | tee -a "$LOG"
