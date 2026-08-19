#!/usr/bin/env bash
# Standalone LANGUAGE re-eval (copy fidelity + faithfulness/CHAIR). The narrator is trained
# independently of the vision reader, so this re-scores JUST the language half on saved weights —
# no retraining. Also reports BLEU/ROUGE-L/CIDEr-D/METEOR as a labeled OVERLAP lower-bound (comparability
# only, NOT the faithfulness axis); CHAIR backs the FULL marker set.
#
#   scripts/eval_language.sh                                              # synthetic held-out, stage3_answer.pt
#   CKPT=stage3_narrator.pt scripts/eval_language.sh                      # a different narrator ckpt
#   DATASET=smeaheia READER=hybrid/checkpoints/run_all/B_joint.pt scripts/eval_language.sh
#
# Env: CKPT (narrator) · READER (vision reader) · DATASET (synthetic|smeaheia) · SCENES (cap).
set -euo pipefail
source "$(dirname "$0")/config.sh"
export CKPT="${CKPT:-stage3_answer.pt}"
export READER="${READER:-$CKPT_DIR/reader.pt}"
"$PY" -m hybrid.eval.language
