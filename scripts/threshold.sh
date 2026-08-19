#!/usr/bin/env bash
# ============================================================================================
# THRESHOLD SWEEP — pick the detection operating point (DET_THRESH) on an EXISTING checkpoint, no
# training. detF1 is killed by over-detection (FP), so the default 0.9 may not be F1-optimal. Scores on
# PURE masks (DILATE_R=0). Also compares builds: run once per THEBE_SOURCE (patches vs volume).
#
# RUN:   scripts/threshold.sh
#        THEBE_SOURCE=volume scripts/threshold.sh                 # against the Dataverse volume build
# KNOBS: CKPT · DATASETS · THRESHOLDS · THEBE_SOURCE
# ============================================================================================
source "$(dirname "$0")/config.sh"

CKPT="${CKPT:-$CKPT_DIR/run_all/B_joint.pt}"
DATASETS="${DATASETS:-thebe,cracks,smeaheia}"
THRESHOLDS="${THRESHOLDS:-0.5 0.7 0.8 0.9 0.95 0.98}"
export THEBE_SOURCE="${THEBE_SOURCE:-volume}"    # match run_all + thebe/__init__ (volume is the methodology); patches = opt-in compare
export ACTIVE_CLASSES="${ACTIVE_CLASSES:-fault}"
export DILATE_R=0                                              # PURE scoring — never dilate the GT
export N_TEST="${N_TEST:-100000}"

echo "############ PHASE 0 · DET_THRESH sweep · ckpt=$(basename "$CKPT") · source=$THEBE_SOURCE · PURE masks ############"
for T in $THRESHOLDS; do
  echo "==== DET_THRESH=$T ===="
  DET_THRESH="$T" CKPT="$CKPT" DATASETS="$DATASETS" N_TEST="$N_TEST" \
    "$PY" -m hybrid.eval.benchmark 2>&1 | tee "$RUN_DIR/phase0_t${T}.log"
done

echo "############ PHASE 0 SUMMARY — detF1 (P/R) · pooled IoU · tol-F1 per threshold ############"
"$PY" - "$RUN_DIR" $THRESHOLDS <<'PY'
import json, sys, os
run, thrs = sys.argv[1], sys.argv[2:]
best = {}
for t in thrs:
    p = os.path.join(run, f"phase0_t{t}.log")
    if not os.path.exists(p): continue
    for line in open(p):
        if not line.startswith("[METRICS] "): continue
        try: d = json.loads(line[10:])
        except Exception: continue
        ds = d.get("dataset")
        f1 = d.get("detF1")
        print(f"  thr {t:>5} · {ds:9} detF1 {d.get('detF1',0):.3f} (P {d.get('detP',0):.2f}/R {d.get('detR',0):.2f}) "
              f"· pooled_iou {d.get('pooled_iou',0):.3f} · tol-F1 {d.get('tolf1',0):.3f}")
        if f1 is not None and f1 == f1 and f1 > best.get(ds, (-1, None))[0]:
            best[ds] = (f1, t)
    print()
print("  F1-OPTIMAL threshold per survey:")
for ds, (f1, t) in best.items():
    print(f"    {ds:9} best detF1 {f1:.3f} @ DET_THRESH={t}")
if best:
    avg = sum(float(t) for _, (_, t) in best.items()) / len(best)
    print(f"  AVERAGE optimal threshold across surveys = {avg:.3f}")
    print(f"  → run the joint ratio-selection + A/B with  DET_THRESH={avg:.3f}  (per-survey values above for deployment)")
PY
echo "PHASE0_DONE · pick the F1-optimal DET_THRESH; re-run with THEBE_SOURCE=volume to compare builds"
