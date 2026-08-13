#!/usr/bin/env bash
# ============================================================================================
# A/B ATTRIBUTE-GENERALIZATION EXPERIMENT — one script, synthetic reader → conclusion.
#
# STEPS (this is reader-based — the answer is decided by the vision reader, so geology/LM are SKIPPED
# via READER_ONLY; not run):
#   0. build datasets (DATASETS): Thebe=Kaggle patches, Smeaheia=cube, CRACKS=auto
#   1. synthetic base — READER_ONLY (reader.pt only, fast) OR SYNTH_FULL=1 (reader + geology + language
#      narrator + language eval: copy/CHAIR/BLEU — settles language ONCE in synthetic, then A/B = syn→real)
#      → ZERO-SHOT benchmark on all 3 real surveys (no fine-tune) = the transfer baseline
#   2. RATIO SELECTION: {alone} per survey + rr-joint at multiple RATIOS (all TRAIN_MEASURE=0)
#      → benchmark each on ALL datasets → AUTO-PICK the best ratio (mean SELECT_METRIC)
#   3. A vs B on the CHOSEN ratio (joint only): A = winning-ratio joint (reused, TRAIN_MEASURE=0);
#      B = same ratio, TRAIN_MEASURE=1 (real attributes, data-gated to Smeaheia) → benchmark both
#   4. summary decision table (checkpoint × dataset) + best-ratio + A/B verdict
# CONTROLS: class-to-train = ACTIVE_CLASSES (per-class gradient scope) · attributes = TRAIN_MEASURE ×
# data-gate · data = DATASETS/WEIGHTS · uncapped = REAL_CAP high + N_TEST high + SCENE_CAP unset.
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
# KNOBS (env): DATASETS · WEIGHTS · RATIOS · SELECT_METRIC · TOTAL_STEPS · ALONE_STEPS · SME_STEPS ·
#              READER_EPOCHS · DET_THRESH · REAL_CAP · ACTIVE_CLASSES · THEBE_SOURCE · OUT
#              SYNTH_FULL (reader+language) · SKIP_SYNTH (reuse reader.pt) · FAST (light signal)
# ============================================================================================
source "$(dirname "$0")/config.sh"

DATASETS="${DATASETS:-thebe,cracks,smeaheia}"          # which real surveys (dataset-select feature)
WEIGHTS="${WEIGHTS:-thebe:4,cracks:3,smeaheia:3}"      # rr-joint per-survey sampling weights (match DATASETS)
READER_EPOCHS="${READER_EPOCHS:-120}"                  # synthetic reader-from-scratch epochs
TOTAL_STEPS="${TOTAL_STEPS:-100000}"                   # rr-joint steps (full Thebe coverage ≈ 94640 @ 4:3:3)
ALONE_STEPS="${ALONE_STEPS:-40000}"                    # per-survey {alone} steps (small surveys cap themselves)
SME_STEPS="${SME_STEPS:-1000}"                         # Smeaheia {alone} steps (small survey)
DET_THRESH="${DET_THRESH:-0.9}"                        # detection operating point used for ALL benchmarks
RATIOS="${RATIOS:-thebe:4,cracks:3,smeaheia:3 thebe:8,cracks:1,smeaheia:1 thebe:1,cracks:1,smeaheia:1}"  # candidate rr-joint ratios
SELECT_METRIC="${SELECT_METRIC:-pooled_iou}"           # auto-pick the ratio by mean of this metric across surveys
THEBE_SOURCE="${THEBE_SOURCE:-patches}"                # 'patches' = Kaggle thebe-fault-patches-256 (reliable); 'volume' = Dataverse
THEBE_VERSION="${THEBE_VERSION:-}"                     # (volume only) set =1.0 to try the version-zip API; "" = raw/ files
OUT="${OUT:-$CKPT_DIR/ab_experiment}"                  # where this experiment's weights land
export ACTIVE_CLASSES="${ACTIVE_CLASSES:-fault}"       # CLASS filter: only these class heads train (fault) + detect masks to them
export N_TEST="${N_TEST:-100000}"                      # ALWAYS uncapped eval (no split contamination)
export REAL_CAP="${REAL_CAP:-1000000}"                # UNCAPPED load: use ALL built scenes (Thebe ~170k). Lower if RAM-bound.
export GRAD_CKPT="${GRAD_CKPT:-0}"                     # big VRAM → faster
SYNTH_FULL="${SYNTH_FULL:-0}"                          # 1 = FULL synthetic base (reader + geology + language narrator)
                                                      #     + language eval (copy/CHAIR/BLEU); 0 = READER_ONLY (fast, vision)
# ATTRIBUTE control = TRAIN_MEASURE (A=0 frozen / B=1 trains) × which survey has GT (data-gated at build).

if [ "${FAST:-0}" = "1" ]; then                        # quick signal instead of the solid full run
  READER_EPOCHS=40; TOTAL_STEPS=10000; ALONE_STEPS=4000
  export THEBE_MAX_PATCHES="${THEBE_MAX_PATCHES:-8000}"   # FAST only: cap the ~170k Kaggle patches
fi
# full run = UNCAPPED Thebe patches (THEBE_MAX_PATCHES unset → all ~170k). A prior build is reused (idempotent).
mkdir -p "$OUT"
export DATASETS WEIGHTS THEBE_VERSION THEBE_SOURCE DET_THRESH

# benchmark one checkpoint on ALL datasets → its own log (tagged, [METRICS] lines collected later)
bench () {   # $1 = ckpt path   $2 = tag   $3 = split (val|test, default test)
  local split="${3:-test}"
  local logf="$RUN_DIR/ab_bench_$2.log"; [ "$split" = "val" ] && logf="$RUN_DIR/ab_val_$2.log"
  echo "==== BENCH $2  (on $DATASETS, split=$split, DET_THRESH=$DET_THRESH) ===="
  EVAL_SPLIT="$split" DET_THRESH="$DET_THRESH" ACTIVE_CLASSES=fault CKPT="$1" DATASETS="$DATASETS" \
    "$PY" -m hybrid.eval.benchmark 2>&1 | tee "$logf"
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

echo "############ 1 · SYNTHETIC BASE ############"
if [ "${SKIP_SYNTH:-0}" != "1" ]; then
  if [ "$SYNTH_FULL" = "1" ]; then                     # FULL: reader + geology + language narrator, and eval language ONCE
    GEO_DIR="$("$PY" -c 'from hybrid.model.geology import adapter_dir; print(adapter_dir())')"
    if [ -d "$GEO_DIR" ]; then
      echo "[ab] geology adapter present ($GEO_DIR) — skip stage 1"
    else
      echo "[ab] stage 1 · geology CoT adapter"
      "$PY" -m hybrid.stages.stage1_geology 2>&1 | tee "$RUN_DIR/ab_stage1_geology.log"
    fi
    echo "[ab] full synthetic train (reader → grounding COPY → fuse fold ANSWER)"
    ACTIVE_CLASSES=fault,closure,onlap READER_EPOCHS="$READER_EPOCHS" DILATE_R=0 \
      "$PY" -m hybrid.run_train 2>&1 | tee "$RUN_DIR/ab_synth_full.log"
    echo "[ab] LANGUAGE eval on synthetic held-out (copy · CHAIR · BLEU/METEOR/CIDEr)"
    scripts/eval.sh 2>&1 | tee "$RUN_DIR/ab_language_eval.log" || true
  else                                                 # READER_ONLY: vision only (skips geology/grounding/fold) — the A/B answer
    READER_ONLY=1 ACTIVE_CLASSES=fault,closure,onlap READER_EPOCHS="$READER_EPOCHS" DILATE_R=0 \
      "$PY" -m hybrid.run_train 2>&1 | tee "$RUN_DIR/ab_synth_reader.log"
  fi
  cp "$CKPT_DIR/reader.pt" "$OUT/reader_synth.pt"
fi
bench "$CKPT_DIR/reader.pt" "zeroshot"                  # ZERO-SHOT: synthetic reader on all 3 real surveys, NO fine-tune

echo "############ 2 · RATIO SELECTION — {alone} vs rr-joint across ratios (TRAIN_MEASURE=0) ############"
for S in ${DATASETS//,/ }; do                          # {alone} baselines = the complementarity reference
  st="$ALONE_STEPS"; [ "$S" != "thebe" ] && st="$SME_STEPS"
  TRAIN_MEASURE=0 SURVEY="$S" STEPS="$st" SAVE="$OUT/alone_$S.pt" \
    scripts/alone.sh 2>&1 | tee "$RUN_DIR/ab_train_alone_$S.log"
  bench "$OUT/alone_$S.pt" "alone_$S"
done
: > "$OUT/ratio_map.txt"
i=0
for R in $RATIOS; do                                   # rr-joint at each candidate ratio (all TRAIN_MEASURE=0)
  i=$((i + 1)); tag="ratio${i}"
  echo "-- $tag  WEIGHTS=$R --"
  TRAIN_MEASURE=0 WEIGHTS="$R" TOTAL_STEPS="$TOTAL_STEPS" SAVE="$OUT/${tag}.pt" \
    scripts/joint.sh 2>&1 | tee "$RUN_DIR/ab_train_${tag}.log"
  bench "$OUT/${tag}.pt" "$tag" val                    # SELECT on VAL (no peeking at test)
  echo "$tag $R" >> "$OUT/ratio_map.txt"
done
read BEST_TAG BEST_R < <("$PY" - "$RUN_DIR" "$OUT/ratio_map.txt" "$SELECT_METRIC" <<'PY'
import json, os, sys
rundir, mapf, metric = sys.argv[1], sys.argv[2], sys.argv[3]
best = None
for line in open(mapf):
    tag, R = line.split()
    vals = [json.loads(l[10:]).get(metric) for l in open(os.path.join(rundir, f"ab_val_{tag}.log"))
            if l.startswith("[METRICS] ")]
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    score = sum(vals) / len(vals) if vals else -1.0
    print(f"#   {tag} {R}: mean {metric} = {score:.4f}", file=sys.stderr)
    if best is None or score > best[1]:
        best = (tag, score, R)
print(best[0], best[2])
PY
)
echo "[ab] BEST RATIO → $BEST_R  (from $BEST_TAG, by mean $SELECT_METRIC across surveys)"

echo "############ 3 · A vs B on the CHOSEN ratio ($BEST_R) — joint only ############"
cp "$OUT/${BEST_TAG}.pt" "$OUT/A_joint.pt"              # A = the winning-ratio joint (TRAIN_MEASURE=0), reused (no retrain)
bench "$OUT/A_joint.pt" "A_joint"
TRAIN_MEASURE=1 WEIGHTS="$BEST_R" TOTAL_STEPS="$TOTAL_STEPS" SAVE="$OUT/B_joint.pt" \
  scripts/joint.sh 2>&1 | tee "$RUN_DIR/ab_train_B_joint.log"   # B = same ratio + real attributes (data-gated to Smeaheia)
bench "$OUT/B_joint.pt" "B_joint"

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
print("\nREAD: zeroshot = synthetic reader, no fine-tune (baseline). ratio* = ratio selection (mean "
      f"{os.environ.get('SELECT_METRIC','pooled_iou')} picks the winner). A_joint/B_joint = A vs B on that ratio.")
print("VERDICT (on SMEAHEIA dip/throw — the only valid attribute GT): A_joint ≈ B_joint AND A_joint not "
      "worse than the ratios on thebe/cracks mask/detF1 ⇒ real attributes OPTIONAL (drop the dependency). "
      "B_joint clearly better on smeaheia dip/throw ⇒ real attributes still pay.")
PY

echo "[ab] BEST RATIO was $BEST_R ($BEST_TAG) · A_joint = that ratio @ TRAIN_MEASURE=0 · B_joint = @ TRAIN_MEASURE=1"
echo "AB_EXPERIMENT_DONE · weights in $OUT · logs in $RUN_DIR/ab_*.log"
