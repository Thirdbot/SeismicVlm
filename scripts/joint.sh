#!/usr/bin/env bash
# COMPLEMENTARY-JOINT real-field finetune via weighted round-robin (the deployed training strategy).
# One shared reader learns all surveys; small surveys borrow what they can't train alone. Checkpointed.
#
#   WEIGHTS=thebe:4,cracks:3,smeaheia:3 TOTAL_STEPS=100000 SAVE=hybrid/checkpoints/reader_joint_full.pt \
#     scripts/joint.sh
#
# Full Thebe coverage requires TOTAL_STEPS >= thebe_pool / thebe_weight_fraction (e.g. 4:3:3 -> >=94640).
source "$(dirname "$0")/config.sh"

WEIGHTS="${WEIGHTS:-thebe:4,cracks:3,smeaheia:3}"
# Train (and quick-benchmark) exactly the surveys named in WEIGHTS: derive DATASETS from its keys so
# WEIGHTS=thebe:1 is a true ALONE run, not a 1:1:1 joint. Override DATASETS to benchmark on more surveys.
DATASETS="${DATASETS:-$(echo "$WEIGHTS" | tr ',' '\n' | cut -d: -f1 | paste -sd,)}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
SAVE="${SAVE:-$CKPT_DIR/reader_joint_full.pt}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
BENCH_N_TEST="${BENCH_N_TEST:-300}"              # run_joint_rr's built-in quick benchmark (full report via scripts/benchmark.sh)

echo "[joint] weights=$WEIGHTS datasets=$DATASETS steps=$TOTAL_STEPS save=$SAVE loss=Tversky($TVERSKY)/pw$POS_WEIGHT_MAX heads[c=$TRAIN_CLASS m=$TRAIN_MEASURE d=$TRAIN_DERIVED]"
WEIGHTS="$WEIGHTS" DATASETS="$DATASETS" TOTAL_STEPS="$TOTAL_STEPS" JOINT_SAVE="$SAVE" CKPT_EVERY="$CKPT_EVERY" JOINT_EPOCHS=1 \
  N_TEST="$BENCH_N_TEST" TVERSKY="$TVERSKY" POS_WEIGHT_MAX="$POS_WEIGHT_MAX" \
  TRAIN_CLASS="$TRAIN_CLASS" TRAIN_MEASURE="$TRAIN_MEASURE" TRAIN_DERIVED="$TRAIN_DERIVED" \
  "$PY" -m hybrid.eval.run_joint_rr
