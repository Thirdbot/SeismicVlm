#!/usr/bin/env bash
# POWER-CUT RECOVERY — the {alone} baselines already finished (Stage 1 banked). This resumes from Stage 2:
# warm-resume the partial 4:3:3 joint, run the fresh 8:1:1, then the uncapped benchmarks + report + malform.
# Checkpointed every 10k (a re-cut loses <=1h; re-run this script to resume). Logs to runs/ (survives reboot).
set -u
cd "$(dirname "$0")/.."
source scripts/config.sh

echo "===== [1] warm-resume 4:3:3 (top-up on the ~68% partial) $(date) ====="
RESUME=1 WEIGHTS=thebe:4,cracks:3,smeaheia:3 TOTAL_STEPS=40000 CKPT_EVERY=10000 \
  SAVE="$CKPT_DIR/reader_joint_full.pt" bash scripts/joint.sh

echo "===== [2] fresh 8:1:1 $(date) ====="
WEIGHTS=thebe:8,cracks:1,smeaheia:1 TOTAL_STEPS=48000 CKPT_EVERY=10000 \
  SAVE="$CKPT_DIR/reader_joint_full_811.pt" bash scripts/joint.sh

echo "===== [3] uncapped benchmarks (real surveys; synthetic images deleted → skipped) $(date) ====="
CKPT="$CKPT_DIR/reader_joint_full.pt"       DATASETS=thebe,cracks,smeaheia bash scripts/benchmark.sh
CKPT="$CKPT_DIR/reader_joint_full_811.pt"   DATASETS=thebe,cracks,smeaheia bash scripts/benchmark.sh
CKPT="$CKPT_DIR/reader_alone_thebe_full.pt" DATASETS=thebe                 bash scripts/benchmark.sh
CKPT="$CKPT_DIR/reader_alone_cracks.pt"     DATASETS=cracks                bash scripts/benchmark.sh
CKPT="$CKPT_DIR/reader_alone_smeaheia.pt"   DATASETS=smeaheia              bash scripts/benchmark.sh

echo "===== [4] report + narration/malform $(date) ====="
bash scripts/report.sh "$RUN_DIR"/bench_*.log
READER="$CKPT_DIR/reader_joint_full.pt" N=8 bash scripts/inference.sh
echo "RECOVER_DONE $(date)"
