#!/usr/bin/env bash
# SINGLE-SURVEY {alone} baseline — the ablation control for the complementarity claim: train ONE survey by
# itself, at a chosen exposure, with the SAME loss/toggles/gating as the joint (only difference = other
# surveys absent). Compare its held-out to the {joint} number for that survey.
#
#   SURVEY=smeaheia STEPS=1000 scripts/alone.sh                 # small survey (STEPS>=pool -> full data)
#   SURVEY=thebe    STEPS=40000 scripts/alone.sh                # match the joint's thebe exposure (40k)
source "$(dirname "$0")/config.sh"

SURVEY="${SURVEY:?set SURVEY=thebe|cracks|smeaheia}"
STEPS="${STEPS:-1000}"
SAVE="${SAVE:-$CKPT_DIR/reader_alone_${SURVEY}.pt}"
CKPT_EVERY="${CKPT_EVERY:-10000}"

echo "[alone] survey=$SURVEY steps=$STEPS save=$SAVE (same loss/toggles/gating as joint)"
DATASETS="$SURVEY" WEIGHTS="$SURVEY:1" TOTAL_STEPS="$STEPS" JOINT_SAVE="$SAVE" CKPT_EVERY="$CKPT_EVERY" \
  JOINT_EPOCHS=1 N_TEST=300 TVERSKY="$TVERSKY" POS_WEIGHT_MAX="$POS_WEIGHT_MAX" \
  TRAIN_CLASS="$TRAIN_CLASS" TRAIN_MEASURE="$TRAIN_MEASURE" TRAIN_DERIVED="$TRAIN_DERIVED" \
  "$PY" -m hybrid.eval.run_joint_rr
