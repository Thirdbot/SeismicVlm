#!/usr/bin/env bash
# ============================================================================================
# CROSS-EVAL — the "why joint" diagnostic + the segmentation-transfer test in ONE 3x3 matrix.
# Reuses the ALONE checkpoints run_all.sh already trained (alone_<survey>.pt): each single-survey
# model is benchmarked on EVERY survey — diagonal = in-domain, off-diagonal = UNSEEN transfer.
#
# Reads BOTH mask metrics per cell so the transfer question is one glance:
#   cIoU  (pooled IoU)  — pixel-exact; punishes width/convention mismatch → expected to NOT transfer
#   tolF1 (@2px)        — centerline agreement; forgives width → tests if STRUCTURE transfers
#   detF1               — detection (location) → expected to transfer (motivates joint)
# PURE masks (DILATE_R=0). No training — read-only over the alone ckpts.
#
# RUN:   scripts/cross_eval.sh                       # after run_all.sh (uses $CKPT_DIR/run_all/alone_*.pt)
#        CKPTS=hybrid/checkpoints/run_all DET_THRESH=0.9 scripts/cross_eval.sh
# KNOBS: DATASETS · CKPTS (dir holding alone_<ds>.pt) · DET_THRESH · OUT
# ============================================================================================
source "$(dirname "$0")/config.sh"

DATASETS="${DATASETS:-thebe,cracks,smeaheia}"
CKPTS="${CKPTS:-$CKPT_DIR/run_all}"                 # dir with alone_<survey>.pt from run_all.sh
DET_THRESH="${DET_THRESH:-0.9}"
OUT="${OUT:-$CKPT_DIR/cross_eval}"; mkdir -p "$OUT"
export ACTIVE_CLASSES="${ACTIVE_CLASSES:-fault}" DILATE_R=0
export N_TEST="${N_TEST:-100000}"
IFS=',' read -ra SURV <<< "$DATASETS"

echo "############ CROSS-EVAL · train-one → eval-all · thresh=$DET_THRESH · PURE masks ############"
for SRC in "${SURV[@]}"; do
  CK="$CKPTS/alone_$SRC.pt"
  if [ ! -f "$CK" ]; then echo "  !! missing $CK — run run_all.sh (or alone.sh) for $SRC first"; continue; fi
  for TGT in "${SURV[@]}"; do
    echo "==== train=$SRC  →  eval=$TGT ===="
    DILATE_R=0 CKPT="$CK" DATASETS="$TGT" DET_THRESH="$DET_THRESH" N_TEST="$N_TEST" \
      "$PY" -m hybrid.eval.benchmark 2>&1 | tee "$OUT/${SRC}__${TGT}.log"
  done
done

echo "############ CROSS-EVAL MATRIX (rows=train, cols=eval · diagonal=in-domain) ############"
"$PY" - "$OUT" "$DATASETS" <<'PY'
import json, os, sys
run, surv = sys.argv[1], sys.argv[2].split(",")
def cell(src, tgt):
    p = os.path.join(run, f"{src}__{tgt}.log")
    if not os.path.exists(p): return None
    for line in open(p):
        if line.startswith("[METRICS] "):
            try: return json.loads(line[10:])
            except Exception: pass
    return None
def grid(key, label, fmt="{:.3f}"):
    print(f"\n  {label}  (rows=train, cols=eval)")
    print("           " + "".join(f"{t:>12}" for t in surv))
    for s in surv:
        row = f"    {s:>7}"
        for t in surv:
            d = cell(s, t); v = d.get(key) if d else None
            row += f"{(fmt.format(v) if isinstance(v,(int,float)) and v==v else '  —'):>12}"
        print(row)
grid("ciou",  "cIoU  (pixel-exact — width-sensitive; off-diagonal = does WIDTH transfer)")
grid("tolf1", "tolF1 (@2px centerline — off-diagonal = does STRUCTURE transfer)")
grid("detF1", "detF1 (detection/location — off-diagonal = does LOCATION transfer → motivates joint)")
print("\n  READ: off-diagonal cIoU low + tolF1 high  ⇒  structure transfers, only width is survey-specific.")
print("        off-diagonal detF1 high                ⇒  location transfers → the case FOR round-robin joint.")
PY
echo "CROSS_EVAL_DONE · logs in $OUT"
