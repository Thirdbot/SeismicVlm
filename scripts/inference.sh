#!/usr/bin/env bash
# NARRATION-ON-REAL report + MALFORM tally — the degenerate-language check after real-field finetune.
# Runs the deployed pipeline (reader measures facts+masks → frozen LM narrates) on held-out scenes: prints
# the LM chains, saves mask overlays, and reports the malformation tally (unclosed · key-leak · confab ·
# truncated — driven toward 0 by derive-off + train-measure). Captured to runs/inference_<survey>.txt.
#
#   READER=hybrid/checkpoints/reader_joint_full.pt scripts/inference.sh      # all real surveys, deploy checkpoint
#   DATASETS="smeaheia" READER=… scripts/inference.sh                        # one survey
source "$(dirname "$0")/config.sh"

READER="${READER:-hybrid/checkpoints/reader_joint_full.pt}"
N="${N:-8}"
read -ra DS <<< "${DATASETS:-thebe cracks smeaheia}"

for d in "${DS[@]}"; do
  OUT_TXT="$RUN_DIR/inference_${d}.txt"
  echo "[inference] $d · reader=$READER → $OUT_TXT" >&2
  DATASET="$d" READER="$READER" N="$N" "$PY" -m hybrid.eval.inference 2>&1 | tee "$OUT_TXT"
done
echo "[inference] chains + malform tally in $RUN_DIR/inference_*.txt" >&2
