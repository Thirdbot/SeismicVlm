#!/usr/bin/env bash
# COMPREHENSIVE per-dataset benchmark of ONE checkpoint → the paper report (mask + attribute axes, constant
# baselines, n). Per-dataset, NEVER pooled. Uncapped held-out by default. Writes a log under $RUN_DIR.
#
#   CKPT=hybrid/checkpoints/reader_joint_full.pt DATASETS=thebe,cracks,smeaheia scripts/benchmark.sh
#   CKPT=hybrid/checkpoints/reader.pt            DATASETS=synthetic             scripts/benchmark.sh
source "$(dirname "$0")/config.sh"

CKPT="${CKPT:?set CKPT=path/to/checkpoint.pt}"
DATASETS="${DATASETS:-thebe,cracks,smeaheia}"
TAG="${TAG:-$(basename "${CKPT%.pt}")}"
LOG="$RUN_DIR/bench_${TAG}.log"

echo "[benchmark] $CKPT · datasets=$DATASETS · N_TEST=$N_TEST (uncapped) → $LOG"
CKPT="$CKPT" DATASETS="$DATASETS" N_TEST="$N_TEST" "$PY" -m hybrid.eval.benchmark | tee "$LOG"
