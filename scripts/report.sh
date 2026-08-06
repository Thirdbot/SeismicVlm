#!/usr/bin/env bash
# Assemble the benchmark logs into the paper tables (markdown) → $RUN_DIR/report.md.
#   scripts/report.sh                         # all runs/bench_*.log
#   scripts/report.sh runs/bench_a.log ...    # specific logs
source "$(dirname "$0")/config.sh"
LOGS=("$@"); [ ${#LOGS[@]} -eq 0 ] && LOGS=("$RUN_DIR"/bench_*.log)
"$PY" -m hybrid.eval.report "${LOGS[@]}" | tee "$RUN_DIR/report.md"
echo "[report] → $RUN_DIR/report.md" >&2
