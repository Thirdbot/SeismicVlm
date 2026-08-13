#!/usr/bin/env bash
# ============================================================================================
# A/B ATTRIBUTE-GENERALIZATION EXPERIMENT — one script, stage 1 → conclusion.
#
# THE QUESTION: does real-field data need ground-truth ATTRIBUTES (dip/throw), or do the
# synthetic-trained attribute heads generalize when the real fine-tune supplies only IMAGE+MASK?
# If A ≈ B, real datasets need only image+mask — the attribute-annotation dependency is dropped.
#
#   A  =  NO real attribute training anywhere            (TRAIN_MEASURE=0)  → attrs from synthetic heads
#   B  =  ONLY Smeaheia's trustable dip/throw train      (TRAIN_MEASURE=1, data-gated to Smeaheia)
#
# For each of A and B: a per-survey {alone} baseline AND the weighted round-robin {joint}. Every
# checkpoint is benchmarked on ALL datasets (you trained on multiple → judge on multiple). Attribute
# claims are read on Smeaheia (only valid dip/throw GT); mask/detF1 across all confirm A stays
# in-distribution. Weights are saved separately + meaningfully under $OUT; logs under runs/.
#
# RUN:   scripts/ab_experiment.sh
# KNOBS (env): DATASETS · WEIGHTS · TOTAL_STEPS · ALONE_STEPS · SME_STEPS · READER_EPOCHS · DET_THRESH
#              THEBE_VERSION · OUT · SKIP_SYNTH (reuse reader.pt) · FAST (light signal settings)
# ============================================================================================
source "$(dirname "$0")/config.sh"

DATASETS="${DATASETS:-thebe,cracks,smeaheia}"          # which real surveys (dataset-select feature)
WEIGHTS="${WEIGHTS:-thebe:4,cracks:3,smeaheia:3}"      # rr-joint per-survey sampling weights (match DATASETS)
READER_EPOCHS="${READER_EPOCHS:-120}"                  # synthetic reader-from-scratch epochs
TOTAL_STEPS="${TOTAL_STEPS:-100000}"                   # rr-joint steps (full Thebe coverage ≈ 94640 @ 4:3:3)
ALONE_STEPS="${ALONE_STEPS:-40000}"                    # per-survey {alone} steps (small surveys cap themselves)
SME_STEPS="${SME_STEPS:-1000}"                         # Smeaheia {alone} steps (small survey)
DET_THRESH="${DET_THRESH:-0.9}"                        # detection operating point used for ALL benchmarks
THEBE_SOURCE="${THEBE_SOURCE:-patches}"                # 'patches' = Kaggle thebe-fault-patches-256 (reliable); 'volume' = Dataverse
THEBE_VERSION="${THEBE_VERSION:-}"                     # (volume only) set =1.0 to try the version-zip API; "" = raw/ files
OUT="${OUT:-$CKPT_DIR/ab_experiment}"                  # where this experiment's weights land
export ACTIVE_CLASSES="${ACTIVE_CLASSES:-fault}"       # real is fault-only
export N_TEST="${N_TEST:-100000}"                      # ALWAYS uncapped eval (no split contamination)
export GRAD_CKPT="${GRAD_CKPT:-0}"                     # big VRAM → faster

if [ "${FAST:-0}" = "1" ]; then                        # quick signal instead of the solid full run
  READER_EPOCHS=40; TOTAL_STEPS=10000; ALONE_STEPS=4000
  export THEBE_MAX_PATCHES="${THEBE_MAX_PATCHES:-8000}"   # cap the ~170k Kaggle patches for a bounded build
fi
export THEBE_MAX_PATCHES="${THEBE_MAX_PATCHES:-40000}"   # bound the patch build even for the full run (raise for all ~170k)
mkdir -p "$OUT"
export DATASETS WEIGHTS THEBE_VERSION THEBE_SOURCE DET_THRESH

# benchmark one checkpoint on ALL datasets → its own log (tagged, [METRICS] lines collected later)
bench () {   # $1 = ckpt path   $2 = tag
  echo "==== BENCH $2  (on $DATASETS, DET_THRESH=$DET_THRESH) ===="
  DET_THRESH="$DET_THRESH" ACTIVE_CLASSES=fault CKPT="$1" DATASETS="$DATASETS" \
    "$PY" -m hybrid.eval.benchmark 2>&1 | tee "$RUN_DIR/ab_bench_$2.log"
}

echo "############ 0 · PREPARE DATASETS ($DATASETS) ############"
for S in ${DATASETS//,/ }; do
  case "$S" in
    thebe)
      if [ "$THEBE_SOURCE" = "patches" ]; then                 # Kaggle thebe-fault-patches-256 (reliable; needs kaggle auth)
        "$PY" -m hybrid.data.thebe.build_from_patches 2>&1 | tee "$RUN_DIR/ab_build_thebe.log" || true
        if [ ! -f data/real_data/thebe/thebe_patches.csv ]; then
          echo "[ab] !! Thebe patches build failed (kagglehub / Kaggle auth — see the log) — SKIPPING Thebe."
          echo "[ab] !!   set up ~/.kaggle/kaggle.json, or THEBE_SOURCE=volume + place raw/ files, then re-run."
          DATASETS="$(echo ",$DATASETS," | sed 's/,thebe,/,/' | sed 's/^,//;s/,$//')"
        fi
      elif compgen -G "data/real_data/thebe/raw/*.npy" >/dev/null 2>&1 && compgen -G "data/real_data/thebe/raw/*.npz" >/dev/null 2>&1; then
        "$PY" -m hybrid.data.thebe.build_csv 2>&1 | tee "$RUN_DIR/ab_build_thebe.log"          # volume: local files (page download)
      elif [ -n "$THEBE_VERSION" ]; then
        THEBE_VERSION="$THEBE_VERSION" "$PY" -m hybrid.data.thebe.build_csv 2>&1 | tee "$RUN_DIR/ab_build_thebe.log"   # volume: version-zip api
      else
        echo "[ab] !! Thebe volume: no raw/ files and THEBE_VERSION unset — SKIPPING Thebe (or use THEBE_SOURCE=patches)."
        DATASETS="$(echo ",$DATASETS," | sed 's/,thebe,/,/' | sed 's/^,//;s/,$//')"
      fi
      ;;
    smeaheia) "$PY" -m hybrid.data.smeaheia.build_from_cube 2>&1 | tee "$RUN_DIR/ab_build_smeaheia.log" ;;
    cracks)   : ;;   # CRACKS auto-builds lazily on first use
  esac
done
export DATASETS                                         # (may have been trimmed above)

echo "############ 1 · SYNTHETIC READER BASE (READER_ONLY) ############"
if [ "${SKIP_SYNTH:-0}" != "1" ]; then
  READER_ONLY=1 ACTIVE_CLASSES=fault,closure,onlap READER_EPOCHS="$READER_EPOCHS" DILATE_R=0 \
    "$PY" -m hybrid.run_train 2>&1 | tee "$RUN_DIR/ab_synth_reader.log"
  cp "$CKPT_DIR/reader.pt" "$OUT/reader_synth.pt"
fi
bench "$CKPT_DIR/reader.pt" "synth"                     # synthetic held-out baseline (the ceiling)

echo "############ 2 · PLAN A — no real attributes (TRAIN_MEASURE=0) ############"
TRAIN_MEASURE=0 WEIGHTS="$WEIGHTS" TOTAL_STEPS="$TOTAL_STEPS" SAVE="$OUT/A_joint.pt" \
  scripts/joint.sh 2>&1 | tee "$RUN_DIR/ab_train_A_joint.log"
bench "$OUT/A_joint.pt" "A_joint"
for S in ${DATASETS//,/ }; do
  st="$ALONE_STEPS"; [ "$S" = "smeaheia" ] && st="$SME_STEPS"; [ "$S" = "cracks" ] && st="$SME_STEPS"
  TRAIN_MEASURE=0 SURVEY="$S" STEPS="$st" SAVE="$OUT/A_alone_$S.pt" \
    scripts/alone.sh 2>&1 | tee "$RUN_DIR/ab_train_A_alone_$S.log"
  bench "$OUT/A_alone_$S.pt" "A_alone_$S"
done

echo "############ 3 · PLAN B — Smeaheia's real attributes train (TRAIN_MEASURE=1) ############"
TRAIN_MEASURE=1 WEIGHTS="$WEIGHTS" TOTAL_STEPS="$TOTAL_STEPS" SAVE="$OUT/B_joint.pt" \
  scripts/joint.sh 2>&1 | tee "$RUN_DIR/ab_train_B_joint.log"
bench "$OUT/B_joint.pt" "B_joint"
TRAIN_MEASURE=1 SURVEY=smeaheia STEPS="$SME_STEPS" SAVE="$OUT/B_alone_smeaheia.pt" \
  scripts/alone.sh 2>&1 | tee "$RUN_DIR/ab_train_B_alone_smeaheia.log"
bench "$OUT/B_alone_smeaheia.pt" "B_alone_smeaheia"

echo "############ 4 · SUMMARY (decision table) ############"
grep -h '^\[METRICS\] ' "$RUN_DIR"/ab_bench_*.log 2>/dev/null | sed 's/^\[METRICS\] //' > "$OUT/metrics.jsonl"
"$PY" - "$OUT/metrics.jsonl" <<'PY'
import json, sys, collections
by = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    try: d = json.loads(line)
    except Exception: continue
    by[d["ckpt"]][d["dataset"]] = d
def f(d, k):
    v = d.get(k)
    return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "  -  "
print(f"\n{'checkpoint':24}{'dataset':10}{'pIoU':>7}{'detF1':>7}{'dip':>7}{'dipConst':>9}{'throw':>8}{'thrConst':>9}")
print("-" * 90)
for ck in sorted(by):
    for ds in ("thebe", "cracks", "smeaheia"):
        d = by[ck].get(ds)
        if not d: continue
        print(f"{ck:24}{ds:10}{f(d,'pooled_iou'):>7}{f(d,'detF1'):>7}{f(d,'dip'):>7}"
              f"{f(d,'dip_const'):>9}{f(d,'throw'):>8}{f(d,'throw_const'):>9}")
print("\nREAD: attributes on SMEAHEIA only (valid dip/throw GT). A≈B & A not worse on thebe/cracks "
      "mask/detF1 ⇒ real attributes OPTIONAL (drop the dependency). B better on smeaheia dip/throw, "
      "or A degrades others ⇒ real attributes still pay. alone-vs-joint per survey = the complementarity ratio.")
PY

echo "AB_EXPERIMENT_DONE · weights in $OUT · logs in $RUN_DIR/ab_*.log"
