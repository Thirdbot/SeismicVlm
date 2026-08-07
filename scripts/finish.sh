#!/usr/bin/env bash
# EVAL-ONLY FINISH — the joints (4:3:3 + 8:1:1) and the {alone} baselines are already trained and banked.
# This runs ONLY the uncapped benchmarks + report + malform (no training). Safe to re-run: each checkpoint's
# benchmark is a separate process writing its own runs/bench_*.log, so a re-cut only loses the in-flight one.
# The benchmark loop is now @torch.no_grad (hybrid/eval/benchmark.py) → no autograd-graph RAM creep on the
# 12628-scene Thebe pass (the thing that froze the uncapped run). N_TEST=100000 (config.sh) = fully uncapped.
set -u
cd "$(dirname "$0")/.."
source scripts/config.sh

echo "===== [3] uncapped benchmarks (real surveys; synthetic images deleted → skipped) $(date) ====="
CKPT="$CKPT_DIR/reader_joint_full.pt"       DATASETS=thebe,cracks,smeaheia bash scripts/benchmark.sh
CKPT="$CKPT_DIR/reader_joint_full_811.pt"   DATASETS=thebe,cracks,smeaheia bash scripts/benchmark.sh
CKPT="$CKPT_DIR/reader_alone_thebe_full.pt" DATASETS=thebe                 bash scripts/benchmark.sh
CKPT="$CKPT_DIR/reader_alone_cracks.pt"     DATASETS=cracks                bash scripts/benchmark.sh
CKPT="$CKPT_DIR/reader_alone_smeaheia.pt"   DATASETS=smeaheia              bash scripts/benchmark.sh

echo "===== [4] report + narration/malform $(date) ====="
bash scripts/report.sh "$RUN_DIR"/bench_*.log
READER="$CKPT_DIR/reader_joint_full.pt" N=8 bash scripts/inference.sh
echo "FINISH_DONE $(date)"
