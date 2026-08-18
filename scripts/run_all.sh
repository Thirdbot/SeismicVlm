#!/usr/bin/env bash
# ============================================================================================
# RUN_ALL — the full real-field pipeline in ONE sequence, on the ungated volume + all-pos/neg data
# (loss/dilation untouched). Order: alone → per-survey threshold → cross-eval (transfer diagnostic) →
# ratio-selection (joint) → A/B → final benchmark. Deployment (joint vs per-survey alone) is DECIDED
# AFTER, from the printed table.
#
# Chains the existing primitives (run_joint_rr + benchmark) — no new training logic. Every run scores on
# PURE masks with the common metrics (cIoU/gIoU/Pr@X · det AP · formal dip/throw). ratio-selection is
# joint-only; threshold is an eval knob swept per survey; A/B is the TRAIN_MEASURE toggle.
#
# RUN:   scripts/run_all.sh
# KNOBS: DATASETS · RATIOS · THRESHOLDS · TOTAL_STEPS · ALONE_STEPS · OUT · CROSS_EVAL(1/0)
# ============================================================================================
source "$(dirname "$0")/config.sh"

DATASETS="${DATASETS:-thebe,cracks,smeaheia}"
RATIOS="${RATIOS:-1:1:1 8:1:1 4:3:3}"                       # candidate JOINT mixes (survey order = DATASETS order)
THRESHOLDS="${THRESHOLDS:-0.5 0.7 0.8 0.9 0.95}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
# ALONE steps are COVERAGE-MATCHED per survey: pools differ 100x (Thebe ~38k train panels vs CRACKS ~300),
# so one fixed step count is 0.8 epochs for Thebe but 100+ for CRACKS — an unfair, mis-calibrated baseline.
# steps = ALONE_EPOCHS × (that survey's train pool), floored at ALONE_MIN_STEPS so the tiny surveys still
# converge. Set ALONE_STEPS to force a uniform override (skips the per-survey calc).
ALONE_EPOCHS="${ALONE_EPOCHS:-3}"
ALONE_MIN_STEPS="${ALONE_MIN_STEPS:-5000}"
OUT="${OUT:-$CKPT_DIR/run_all}"; mkdir -p "$OUT"
export ACTIVE_CLASSES="${ACTIVE_CLASSES:-fault}" DILATE_R=0
export N_TEST="${N_TEST:-100000}" REAL_CAP="${REAL_CAP:-1000000}"
IFS=',' read -ra SURV <<< "$DATASETS"

bench () {  # $1=ckpt $2=datasets $3=det_thresh $4=logtag
  DILATE_R=0 CKPT="$1" DATASETS="$2" DET_THRESH="$3" N_TEST="$N_TEST" \
    "$PY" -m hybrid.eval.benchmark 2>&1 | tee "$OUT/$4.log"
}

echo "############ RUN_ALL · data=$DATASETS (ungated volume, all pos+neg) · ratios=[$RATIOS] ############"

# 1) ALONE per survey (fixed single-survey WEIGHTS — no ratio) → per-survey ceilings + their thresholds.
# Steps are coverage-matched: ALONE_EPOCHS × the survey's own train pool (from the SAME scenes() training
# uses), so Thebe (~38k) and CRACKS (~300) each get the same number of PASSES, not the same step count.
for DS in "${SURV[@]}"; do
  if [ -n "${ALONE_STEPS:-}" ]; then
    ST="$ALONE_STEPS"                                          # uniform override (explicit opt-out)
  else
    POOL=$("$PY" - "$DS" 2>/dev/null <<'PY' | sed -n 's/^POOLCOUNT //p'
import importlib, sys
tr = importlib.import_module(f"hybrid.data.{sys.argv[1]}").scenes()[1]   # (all, TRAIN, test) → train pool
print(f"POOLCOUNT {len(tr)}")
PY
)
    POOL="${POOL:-0}"
    ST=$(( ALONE_EPOCHS * POOL ))
    [ "$ST" -lt "$ALONE_MIN_STEPS" ] && ST="$ALONE_MIN_STEPS"  # floor so tiny surveys converge
    echo "  [$DS] train-pool $POOL × ${ALONE_EPOCHS}ep → $ST steps (floor $ALONE_MIN_STEPS)"
  fi
  echo "==== alone · $DS ($ST steps) ===="
  DATASETS="$DS" WEIGHTS="$DS:1" TOTAL_STEPS="$ST" JOINT_EPOCHS=1 TRAIN_CLASS=1 TRAIN_MEASURE=1 \
    JOINT_SAVE="$OUT/alone_$DS.pt" "$PY" -m hybrid.eval.run_joint_rr 2>&1 | tee "$OUT/train_alone_$DS.log"
  for T in $THRESHOLDS; do bench "$OUT/alone_$DS.pt" "$DS" "$T" "thr_${DS}_$T"; done
done

# 2) THRESHOLD: per-survey F1-optimal (from its alone) + AVERAGE (for the joint eval)
AVG=$("$PY" - "$OUT" "$DATASETS" $THRESHOLDS <<'PY'
import json, sys, os, glob
run, datasets = sys.argv[1], sys.argv[2].split(","); thrs = sys.argv[3:]
best = {}
for ds in datasets:
    for t in thrs:
        p = os.path.join(run, f"thr_{ds}_{t}.log")
        for line in (open(p) if os.path.exists(p) else []):
            if line.startswith("[METRICS] "):
                try: d = json.loads(line[10:])
                except Exception: continue
                if d.get("dataset") == ds and d.get("detF1") == d.get("detF1"):
                    if d["detF1"] > best.get(ds, (-1, None))[0]: best[ds] = (d["detF1"], float(t))
for ds, (f1, t) in best.items():
    print(f"  {ds:9} threshold {t} (detF1 {f1:.3f})", file=sys.stderr)
avg = sum(t for _, t in best.values()) / len(best) if best else 0.9
print(f"  AVERAGE threshold = {avg:.3f}", file=sys.stderr)
print(f"{avg:.3f}")
PY
)
echo "  → joint ratio-selection + A/B use DET_THRESH=$AVG"

# 2b) CROSS-EVAL (the "why joint" + segmentation-transfer diagnostic): reuse the alone_*.pt checkpoints,
# benchmark each single-survey model on EVERY survey → 3x3 matrix of cIoU / tolF1 / detF1. Off-diagonal
# cIoU-low/tolF1-high = structure transfers (width doesn't); detF1-high = location transfers → the case
# FOR joint. Toggle off with CROSS_EVAL=0. No training — read-only over the alone ckpts.
if [ "${CROSS_EVAL:-1}" != "0" ]; then
  echo "==== cross-eval (train-one → eval-all, at DET_THRESH=$AVG) ===="
  CKPTS="$OUT" DATASETS="$DATASETS" DET_THRESH="$AVG" OUT="$OUT/cross" N_TEST="$N_TEST" \
    "$(dirname "$0")/cross_eval.sh" 2>&1 | tee "$OUT/cross_eval.log"
fi

# 3) RATIO-SELECTION (joint only): train each ratio, benchmark ALL surveys at the average threshold
for R in $RATIOS; do
  W=$(paste -d, <(printf '%s\n' "${SURV[@]}") <(echo "$R" | tr ':' '\n') | sed 's/,/:/' | paste -sd,)
  tag=$(echo "$R" | tr ':' '')
  echo "==== ratio $R  (WEIGHTS=$W) ===="
  WEIGHTS="$W" TOTAL_STEPS="$TOTAL_STEPS" JOINT_EPOCHS=1 TRAIN_CLASS=1 TRAIN_MEASURE=0 \
    JOINT_SAVE="$OUT/ratio_$tag.pt" "$PY" -m hybrid.eval.run_joint_rr 2>&1 | tee "$OUT/train_ratio_$tag.log"
  bench "$OUT/ratio_$tag.pt" "$DATASETS" "$AVG" "bench_ratio_$tag"
done
BEST=$("$PY" - "$OUT" $RATIOS <<'PY'
import json, sys, os
run, ratios = sys.argv[1], sys.argv[2:]
scores = {}
for R in ratios:
    tag = R.replace(":", ""); f1s = []
    p = os.path.join(run, f"bench_ratio_{tag}.log")
    for line in (open(p) if os.path.exists(p) else []):
        if line.startswith("[METRICS] "):
            try: f1s.append(json.loads(line[10:])["detF1"])
            except Exception: pass
    f1s = [x for x in f1s if x == x]
    if f1s: scores[R] = sum(f1s) / len(f1s)
for R, s in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  ratio {R:9} mean detF1 {s:.3f}", file=sys.stderr)
print(max(scores, key=scores.get) if scores else ratios[0])
PY
)
echo "  → BEST ratio = $BEST"
BW=$(paste -d, <(printf '%s\n' "${SURV[@]}") <(echo "$BEST" | tr ':' '\n') | sed 's/,/:/' | paste -sd,)

# 4) A vs B on the winning ratio (attribute toggle) — A reuses the ratio ckpt, B retrains with measure on
btag=$(echo "$BEST" | tr ':' '')
cp "$OUT/ratio_$btag.pt" "$OUT/A_joint.pt"
WEIGHTS="$BW" TOTAL_STEPS="$TOTAL_STEPS" JOINT_EPOCHS=1 TRAIN_CLASS=1 TRAIN_MEASURE=1 \
  JOINT_SAVE="$OUT/B_joint.pt" "$PY" -m hybrid.eval.run_joint_rr 2>&1 | tee "$OUT/train_B.log"

# 5) FINAL benchmark: alone + A + B, at the average threshold (per-survey thresholds are in step 2 logs)
echo "############ RUN_ALL FINAL (DET_THRESH=$AVG · pure masks · common metrics) ############"
for DS in "${SURV[@]}"; do bench "$OUT/alone_$DS.pt" "$DS" "$AVG" "final_alone_$DS"; done
bench "$OUT/A_joint.pt" "$DATASETS" "$AVG" "final_A"
bench "$OUT/B_joint.pt" "$DATASETS" "$AVG" "final_B"
echo "RUN_ALL_DONE · ckpts in $OUT · best ratio $BEST · avg threshold $AVG · per-survey thresholds in step-2 logs"
echo "  → decide joint-vs-alone from final_* (each survey's alone ceiling vs A/B joint); A≈B ⇒ real attrs not needed"
