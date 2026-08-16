#!/usr/bin/env bash
# Causal SWAP re-eval (seam faithfulness) — the reproducer for tab:faith's flip/baseline rows.
# Intervenes on each injected value (dip/throw/centroid), regenerates, and measures how often the
# narration FOLLOWS the swapped value (flip) vs states the un-swapped value (baseline, a control).
# Read-only over saved weights — trains nothing, edits nothing.
#
#   scripts/eval_swap.sh                                                  # synthetic held-out, GT-injected
#   CKPT=stage3_narrator.pt scripts/eval_swap.sh                          # the deployed narrator ckpt
#   USE_READER=1 scripts/eval_swap.sh                                     # swap reader-measured facts, not GT
#   DATASET=smeaheia READER=hybrid/checkpoints/ab_experiment/B_joint.pt scripts/eval_swap.sh
#
# Env: CKPT (narrator) · READER (vision reader) · DATASET (synthetic|smeaheia) · USE_READER (0/1) · SCENES (cap).
set -euo pipefail
source "$(dirname "$0")/config.sh"
export CKPT="${CKPT:-stage3_answer.pt}"
export READER="${READER:-$CKPT_DIR/reader.pt}"
"$PY" -m hybrid.eval.swap
