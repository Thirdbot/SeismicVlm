#!/usr/bin/env bash
# INTERNAL / LANGUAGE evaluation → captured to runs/eval_report.txt (the saved-file counterpart to the
# vision runs/report.md). Prints: schema check · copy fidelity (GT + reader yardsticks) · reader mechanism
# (count/class/mask-dice) · reader attrs (dip/throw/area) · boxes (mAP/GIoU) · CHAIR faithfulness.
# (Narration-overlap BLEU/METEOR/CIDEr were dropped — free narration != templated answer; see components.py.)
# For JUST the language half (copy + CHAIR, no vision/box metrics) use scripts/eval_language.sh.
#
#   scripts/eval.sh                          # synthetic component benchmark (full language report; needs synthetic data)
#   DATASET=smeaheia scripts/eval.sh         # real held-out (vision parts; real sets are vision-only)
#   REAL_TRANSFER=1 scripts/eval.sh          # real-field before/after (adapter isolation)
#   READER=hybrid/checkpoints/reader_joint_full.pt CKPT=stage3_answer.pt scripts/eval.sh   # pick weights
#
# NOTE: the language metrics (copy/CHAIR) require the SYNTHETIC data + narrator; real datasets are
# vision-only. run_eval reads DATASET/READER/CKPT/SCENES/REAL_TRANSFER from the env (pass-through).
source "$(dirname "$0")/config.sh"

OUT="${OUT:-$RUN_DIR/eval_report.txt}"
echo "[eval] running hybrid.run_eval → $OUT" >&2
"$PY" -m hybrid.run_eval 2>&1 | tee "$OUT"
echo "[eval] saved $OUT" >&2
