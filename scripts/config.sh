#!/usr/bin/env bash
# Central, reproducible configuration for the seismic-VLM runs. `source` this from every run script.
# EVERYTHING is env-overridable; the defaults encode the CURRENT VALIDATED methodology (do not change a
# default without re-validating — the mask loss and head toggles were fixed by sweeps/ablations).
set -u

# REPO defaults to the repo root (this script lives in <repo>/scripts/); override with REPO=… if needed.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$REPO/.venv/bin/python}"
CKPT_DIR="${CKPT_DIR:-hybrid/checkpoints}"
RUN_DIR="${RUN_DIR:-$REPO/runs}"                 # where logs land
mkdir -p "$RUN_DIR"

export SFM_CKPT="${SFM_CKPT:-$CKPT_DIR/SFM-Base-512.pth}"   # frozen encoder (REQUIRED — no silent NCS fallback)

# --- validated mask loss (fixed by the Tversky/pos_weight sweep) ---
export TVERSKY="${TVERSKY:-0.4,0.6,1.0}"         # additive Dice + Focal-Tversky(alpha,beta,gamma); beta>alpha penalizes over-prediction
export POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-15}"    # BCE positive-weight upper clamp (the over-prediction control)

# --- real-field head toggles (derive OFF: relations are reasoned from geometry, never asserted) ---
export TRAIN_CLASS="${TRAIN_CLASS:-1}"
export TRAIN_MEASURE="${TRAIN_MEASURE:-1}"
export TRAIN_DERIVED="${TRAIN_DERIVED:-0}"

# --- evaluation policy (ALWAYS uncapped held-out; per-dataset, never pooled) ---
export N_TEST="${N_TEST:-100000}"                # >= every test split → full held-out (uncapped)

cd "$REPO"
