#!/usr/bin/env bash
# ============================================================================================
# DILATION SWEEP — does fattening thin fault masks help FINDING and/or SEGMENTING?
#
# Holds the ratio FIXED at the already-selected best (default 1:1:1) and varies ONLY the real-field
# mask DILATE_R on the RR-joint real adapter, trained from the FROZEN synthetic base (reader.pt).
# For each DILATE_R: train the adapter, then benchmark TWICE —
#   · self  : eval at the SAME dilation it trained on (self-consistent)
#   · pure  : eval at DILATE_R=0 (honest same-target seg comparison; skipped for DR=0)
#
# HONESTY (read the summary this way):
#   · detF1 is DILATION-INDEPENDENT (centroid-matched) → the clean cross-DR metric for FINDING.
#   · pooled IoU / Dice RISE with dilation trivially (fat target ↔ fat prediction) → only compare
#     segmentation on the `pure_*` rows (eval @ DR=0) and tol-F1, NOT on the self_* pooled IoU.
#   A dilated-trained model predicts FAT, so on pure eval it over-predicts (precision drops); a real
#   seg gain is one that survives the pure eval.
#
# NOTE: real masks already carry a built-in width (Smeaheia sticks are fat polylines; CRACKS/Thebe
# have their own thickness). DILATE_R adds ON TOP, so the sweet spot is usually LOW (2-3); DR=5 may
# over-fatten Smeaheia. DR=0 is the current baseline (== A_joint / ratio3).
#
# If a dilation wins, it REPLACES the deployed checkpoint + the mask tables MUST be re-stated at that
# DILATE_R (pooled IoU at DR>0 is NOT comparable to the pure-DR=0 numbers in the current tables) —
# same two-track caution as old/new Thebe.
#
# RUN:   scripts/dilate_sweep.sh
# KNOBS: DILATE_VALUES · WEIGHTS · TOTAL_STEPS · DATASETS · DET_THRESH · OUT
# ============================================================================================
source "$(dirname "$0")/config.sh"

DILATE_VALUES="${DILATE_VALUES:-0 2 3 5}"                 # the sweep (0 = pure baseline)
WEIGHTS="${WEIGHTS:-thebe:1,cracks:1,smeaheia:1}"        # FIXED best ratio (selected on pure dilation)
TOTAL_STEPS="${TOTAL_STEPS:-100000}"                     # rr-joint steps (1 round-robin epoch)
DATASETS="${DATASETS:-thebe,cracks,smeaheia}"
DET_THRESH="${DET_THRESH:-0.9}"                          # detection operating point (match the tables)
OUT="${OUT:-$CKPT_DIR/dilate_sweep}"                    # where sweep checkpoints land
export DATASETS DET_THRESH
export N_TEST="${N_TEST:-100000}"                        # uncapped eval
export REAL_CAP="${REAL_CAP:-1000000}"                  # load all built scenes
export ACTIVE_CLASSES="${ACTIVE_CLASSES:-fault}"         # fault-scoped (matches the deployment)
mkdir -p "$OUT"

# benchmark one checkpoint at a chosen EVAL dilation → its own tagged [METRICS] log
bench () {   # $1 = ckpt path   $2 = tag (self_DR | pure_DR)   $3 = eval DILATE_R
  echo "==== bench $2  (ckpt=$(basename "$1"), eval DILATE_R=$3, DET_THRESH=$DET_THRESH) ===="
  DILATE_R="$3" CKPT="$1" DATASETS="$DATASETS" DET_THRESH="$DET_THRESH" N_TEST="$N_TEST" \
    "$PY" -m hybrid.eval.benchmark 2>&1 | tee "$RUN_DIR/dilate_bench_$2.log"
}

for DR in $DILATE_VALUES; do
  echo "############ DILATE_R=$DR — train real adapter @ WEIGHTS=$WEIGHTS (base = reader.pt, frozen) ############"
  SAVE="$OUT/dilate_${DR}.pt"
  DILATE_R="$DR" WEIGHTS="$WEIGHTS" TOTAL_STEPS="$TOTAL_STEPS" JOINT_EPOCHS=1 \
    TRAIN_CLASS=1 TRAIN_MEASURE=0 JOINT_SAVE="$SAVE" \
    "$PY" -m hybrid.eval.run_joint_rr 2>&1 | tee "$RUN_DIR/dilate_train_${DR}.log"
  bench "$SAVE" "self_${DR}" "$DR"                        # self-consistent (train DR == eval DR)
  [ "$DR" != "0" ] && bench "$SAVE" "pure_${DR}" "0"      # honest same-target seg comparison (eval @ DR=0)
done

echo "############ SUMMARY — detF1 · tol-F1 · pooled IoU across DILATE_R ############"
"$PY" - "$RUN_DIR" <<'PY'
import json, sys, glob, os, collections
run = sys.argv[1]
rows = collections.defaultdict(dict)                       # tag -> {dataset: metrics}
for f in sorted(glob.glob(os.path.join(run, "dilate_bench_*.log"))):
    tag = os.path.basename(f)[len("dilate_bench_"):-len(".log")]   # self_3 / pure_3 / self_0
    for line in open(f):
        if line.startswith("[METRICS] "):
            try: d = json.loads(line[10:])
            except Exception: continue
            rows[tag][d.get("dataset")] = d
DS = ["thebe", "cracks", "smeaheia"]
def cell(d, k):
    v = (d or {}).get(k)
    return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "  -  "
def mean(vals):
    n = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return sum(n)/len(n) if n else float("nan")
def order(t):                                              # DR-major, self before pure
    m, dr = (t.split("_") + ["0"])[:2]
    return (int(dr), 0 if m == "self" else 1)
print(f"\n{'tag':9}{'metric':11}" + "".join(f"{d:>10}" for d in DS) + f"{'MEAN':>9}")
print("-" * 68)
for tag in sorted(rows, key=order):
    for k in ("detF1", "tolf1", "pooled_iou"):
        vals = [rows[tag].get(d, {}).get(k) for d in DS]
        print(f"{tag:9}{k:11}" + "".join(f"{cell(rows[tag].get(d, {}), k):>10}" for d in DS)
              + f"{mean(vals):>9.3f}")
    print()
print("READ: detF1 is dilation-INDEPENDENT — compare it across DR to judge FINDING (higher = dilation")
print("helps the model find faults). pooled IoU rises with DR trivially on self_* rows; judge SEGMENTING")
print("on the pure_* rows (eval @ DR=0) + tol-F1. A win must show up on detF1 and/or pure_* seg, not")
print("just self_* pooled IoU.")
PY
echo "DILATE_SWEEP_DONE · weights in $OUT · logs in $RUN_DIR/dilate_*.log"
