#!/usr/bin/env bash
# Phase 6 on real data (DESIGN §17 D132, D135): evaluation parts, stability (E7), sensitivity variants (E9), then the
# evaluation report. Resumable: `evaluate` is skipped when it finished under the same configs and code; stability runs
# and sensitivity variants already finished are reused. Progress: outputs/phase6/run_phase6.log and
# outputs/phase6/progress.log. Usage: scripts/run_phase6.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=outputs/phase6/run_phase6.log
mkdir -p outputs/phase6

step() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) STEP $*" | tee -a "$LOG"
  if ! "$PY" -m scenario_fraud.cli "$@" >>"$LOG" 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED $*" | tee -a "$LOG"
    exit 1
  fi
}

STAMP=$(cat configs/*.yaml src/scenario_fraud/*.py src/scenario_fraud/*/*.py | sha256sum | cut -c1-16)
MARK=outputs/phase6/evaluate_done
if [ -f "$MARK" ] && [ "$(cat "$MARK")" = "$STAMP" ]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) evaluate already done, skipped" | tee -a "$LOG"
else
  step phase6 evaluate
  echo "$STAMP" > "$MARK"
fi
step phase6 stability
# one process per variant: a variant's design and models are freed before the next one loads its own (D135)
VARIANTS=$("$PY" -c "import yaml; print(' '.join(yaml.safe_load(open('configs/evaluation.yaml'))['sensitivity']['variants']))")
for v in $VARIANTS; do
  step phase6 sensitivity --variant "$v"
done
step phase6 sensitivity   # every variant done: reloads the results and lists them
step phase6 report
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ALL DONE" | tee -a "$LOG"
