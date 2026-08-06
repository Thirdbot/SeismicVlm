#!/usr/bin/env bash
# FULL PAPER ABLATION — one command reproduces every number in the complementarity + config-tradeoff tables.
# Sequential on one GPU (~a day). Every training stage checkpoints every 10k steps (crash-survivable).
#
#   scripts/ablation.sh
#
# Produces, under $RUN_DIR: bench_* logs for — {alone} baselines (matched exposure), both joint configs
# (4:3:3 mask-best, 8:1:1 attribute-best), and the synthetic base. Assemble with scripts/report.sh.
source "$(dirname "$0")/config.sh"
S="$(dirname "$0")"

echo "===== 1/3  {alone} baselines (matched exposure, same loss/toggles as joint) ====="
SURVEY=thebe    STEPS=40000 SAVE=$CKPT_DIR/reader_alone_thebe_full.pt bash "$S/alone.sh"
SURVEY=cracks   STEPS=1000  SAVE=$CKPT_DIR/reader_alone_cracks.pt     bash "$S/alone.sh"
SURVEY=smeaheia STEPS=1000  SAVE=$CKPT_DIR/reader_alone_smeaheia.pt   bash "$S/alone.sh"

echo "===== 2/3  joint configs (full data) ====="
WEIGHTS=thebe:4,cracks:3,smeaheia:3 TOTAL_STEPS=100000 SAVE=$CKPT_DIR/reader_joint_full.pt     bash "$S/joint.sh"   # mask-best
WEIGHTS=thebe:8,cracks:1,smeaheia:1 TOTAL_STEPS=48000  SAVE=$CKPT_DIR/reader_joint_full_811.pt bash "$S/joint.sh"   # attribute-best

echo "===== 3/3  comprehensive uncapped benchmarks (per-dataset, never pooled) ====="
CKPT=$CKPT_DIR/reader.pt                  DATASETS=synthetic            bash "$S/benchmark.sh"
CKPT=$CKPT_DIR/reader_joint_full.pt       DATASETS=thebe,cracks,smeaheia bash "$S/benchmark.sh"
CKPT=$CKPT_DIR/reader_joint_full_811.pt   DATASETS=thebe,cracks,smeaheia bash "$S/benchmark.sh"
CKPT=$CKPT_DIR/reader_alone_thebe_full.pt DATASETS=thebe                bash "$S/benchmark.sh"
CKPT=$CKPT_DIR/reader_alone_cracks.pt     DATASETS=cracks               bash "$S/benchmark.sh"
CKPT=$CKPT_DIR/reader_alone_smeaheia.pt   DATASETS=smeaheia             bash "$S/benchmark.sh"
echo "ABLATION_DONE — assemble with: scripts/report.sh"
