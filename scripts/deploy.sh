#!/usr/bin/env bash
# ============================================================================================
# DEPLOY — the FINAL packaged model. One measure-ON RR-joint run at the SETTLED config
# (best ratio + attributes un-gated + chosen dilation), benchmarked with the same eval metrics,
# saved as deploy.pt, and (optionally) rendered into qualitative inference panels.
#
# This is the CONSOLIDATION of the experiments — NOT exploratory (that's ab_experiment.sh):
#   · WEIGHTS       = the best ratio from selection
#   · TRAIN_MEASURE = 1  → attributes UN-GATED (head on; still DATA-gated to Smeaheia, the only real dip GT)
#   · DILATE_R      = the chosen real-field mask dilation (0 = pure; 3 = the dilation-sweep winner)
#   trained on top of the FROZEN synthetic base (reader.pt).
#
# ⚠ GROUNDING: WEIGHTS must be the ratio selected ON THE BASE+DILATION YOU DEPLOY. If DILATE_R>0 and you
# did NOT re-select the ratio at that dilation (via `DILATE_R=N SYNTH_FULL=1 ab_experiment.sh`), then the
# ratio you pass is grounded on the PURE base — say so, or re-ground first. Do not silently carry 1:1:1
# onto a dilated base.
# ⚠ TABLES: if DILATE_R>0, the mask numbers are at that dilation and are NOT comparable to the pure-DR=0
# tables — re-state them at this DILATE_R.
#
# RUN:   DILATE_R=3 WEIGHTS=thebe:1,cracks:1,smeaheia:1 scripts/deploy.sh
# KNOBS: DILATE_R · WEIGHTS · TOTAL_STEPS · DATASETS · DET_THRESH · DEPLOY_CKPT · RUN_INFER(0/1)
# ============================================================================================
source "$(dirname "$0")/config.sh"

DILATE_R="${DILATE_R:-3}"                                  # chosen dilation (0=pure, 3=sweep winner)
WEIGHTS="${WEIGHTS:-thebe:1,cracks:1,smeaheia:1}"         # best ratio — MUST be grounded on this base+dilation
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
DATASETS="${DATASETS:-thebe,cracks,smeaheia}"
DET_THRESH="${DET_THRESH:-0.9}"                           # match the tables' operating point
DEPLOY_CKPT="${DEPLOY_CKPT:-$CKPT_DIR/deploy.pt}"
NARRATOR="${NARRATOR:-$CKPT_DIR/stage3_narrator.pt}"      # deployed language narrator (for RUN_INFER)
RUN_INFER="${RUN_INFER:-0}"                               # 1 = also render qualitative panels per survey
export DATASETS DET_THRESH
export ACTIVE_CLASSES="${ACTIVE_CLASSES:-fault}"          # fault-scoped deployment
export N_TEST="${N_TEST:-100000}"                         # uncapped eval
export REAL_CAP="${REAL_CAP:-1000000}"

echo "############ DEPLOY · measure-ON RR-joint · ratio=$WEIGHTS · DILATE_R=$DILATE_R · base=reader.pt(frozen) ############"

# 1) TRAIN the deployed adapter — attributes UN-GATED (TRAIN_MEASURE=1), class on, from the frozen base
DILATE_R="$DILATE_R" WEIGHTS="$WEIGHTS" TOTAL_STEPS="$TOTAL_STEPS" JOINT_EPOCHS=1 \
  TRAIN_CLASS=1 TRAIN_MEASURE=1 JOINT_SAVE="$DEPLOY_CKPT" \
  "$PY" -m hybrid.eval.run_joint_rr 2>&1 | tee "$RUN_DIR/deploy_train.log"

# 2) BENCHMARK the deployed checkpoint (same metrics, at the deployed dilation)
DILATE_R="$DILATE_R" CKPT="$DEPLOY_CKPT" DATASETS="$DATASETS" DET_THRESH="$DET_THRESH" N_TEST="$N_TEST" \
  "$PY" -m hybrid.eval.benchmark 2>&1 | tee "$RUN_DIR/deploy_bench.log"

# 3) (optional) render qualitative inference panels per survey for the packaged figures
if [ "$RUN_INFER" = "1" ]; then
  for DS in ${DATASETS//,/ }; do
    echo "==== inference panels · $DS ===="
    DILATE_R="$DILATE_R" DATASET="$DS" READER="$DEPLOY_CKPT" CKPT="$NARRATOR" \
      ACTIVE_CLASSES=fault N=6 BOXES=1 OUT="hybrid/inference/deploy_$DS" \
      "$PY" -m hybrid.eval.inference 2>&1 | tee "$RUN_DIR/deploy_infer_$DS.log"
  done
fi

# 4) SUMMARY
echo "############ DEPLOY SUMMARY (DILATE_R=$DILATE_R · DET_THRESH=$DET_THRESH) ############"
grep -h '^\[METRICS\] ' "$RUN_DIR/deploy_bench.log" 2>/dev/null | sed 's/^\[METRICS\] //' | "$PY" - <<'PY'
import json, sys
def f(v): return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "  -  "
for line in sys.stdin:
    try: d = json.loads(line)
    except Exception: continue
    print(f"  {d.get('dataset',''):9} detF1 {f(d.get('detF1'))} · pooled_iou {f(d.get('pooled_iou'))} "
          f"· tol-F1 {f(d.get('tolf1'))} · dip {f(d.get('dip'))} (const {f(d.get('dip_const'))}) "
          f"· throw {f(d.get('throw'))} (const {f(d.get('throw_const'))})")
PY
echo "DEPLOY_DONE · checkpoint = $DEPLOY_CKPT"
echo "  → inference:  READER=$DEPLOY_CKPT NARRATOR=$NARRATOR python -m hybrid.infer   (or hybrid.eval.inference)"
echo "  → REMINDER: if DILATE_R=$DILATE_R > 0, re-state the mask tables at this dilation (not comparable to pure DR=0)."
