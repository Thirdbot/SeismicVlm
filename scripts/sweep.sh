#!/usr/bin/env bash
# SWEEP the joint over configurations (weightings and/or loss), train + benchmark each → one log per config
# for side-by-side comparison. Edit the CONFIGS array (name | weights | steps | extra-env). Sequential.
#
#   scripts/sweep.sh
#
# Each row's extra-env can override loss/toggles, e.g. "TVERSKY=0.3,0.7,1.0" for a loss sweep at fixed weights.
source "$(dirname "$0")/config.sh"
S="$(dirname "$0")"

# name | WEIGHTS | TOTAL_STEPS | extra env (space-separated KEY=VAL, or empty)
CONFIGS=(
  "w811|thebe:8,cracks:1,smeaheia:1|48000|"
  "w622|thebe:6,cracks:2,smeaheia:2|60000|"
  "w433|thebe:4,cracks:3,smeaheia:3|100000|"
)

for row in "${CONFIGS[@]}"; do
  IFS='|' read -r name weights steps extra <<< "$row"
  save="$CKPT_DIR/reader_joint_${name}.pt"
  echo "===== sweep: $name  weights=$weights steps=$steps  extra=[$extra] ====="
  env $extra WEIGHTS="$weights" TOTAL_STEPS="$steps" SAVE="$save" bash "$S/joint.sh"
  env CKPT="$save" DATASETS=thebe,cracks,smeaheia TAG="sweep_${name}" bash "$S/benchmark.sh"
done
echo "===== report (paper tables) ====="
bash "$S/report.sh" "$RUN_DIR"/bench_sweep_*.log
echo "SWEEP_DONE — report at $RUN_DIR/report.md"
