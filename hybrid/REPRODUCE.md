# Reproduce — training, sweeps, ablation, benchmark, reports

Everything is env-configurable and reproducible (seeded round-robin, fixed loss/toggles). The `scripts/`
wrappers source `scripts/config.sh` (central config) and encode the **validated methodology**: weighted
round-robin complementary-joint finetune · Dice+Focal-Tversky(0.4/0.6)+pos-weight-clamp-15 mask loss · derive
heads off on real · **uncapped, per-dataset (never pooled)** evaluation. Nothing caps, pools, or falls back
silently (a missing SFM encoder now errors, not silently NCS).

## 0. Prerequisites
- SFM encoder at `hybrid/checkpoints/SFM-Base-512.pth` (**required** — no silent NCS fallback).
- Synthetic CSV (external simulator) at the path in `hybrid/data/synthetic/__init__.py`.
- venv: `./.venv/bin/python`.

## 1. Synthetic base (one time) — the frozen substrate everything builds on
```
python -m hybrid.stages.stage1_geology        # frozen geology LM adapter
python -m hybrid.run_train                     # reader.pt (vision) + narrator (grounding+fuse LoRA)
```
Stage independence: `run_train` checkpoints `reader.pt` after the reader and `stage2_grounding.pt` after
grounding, so the fuse fold (stage 3) is re-runnable alone via `SKIP_GROUNDING=1 python -m hybrid.run_train`.

## 2. Real-field joint (the deployed model) — trains one shared reader over all surveys
```
scripts/joint.sh                                              # default 4:3:3, full data (thebe 100k steps)
WEIGHTS=thebe:8,cracks:1,smeaheia:1 TOTAL_STEPS=48000 \
  SAVE=hybrid/checkpoints/reader_joint_full_811.pt scripts/joint.sh   # the 8:1:1 config
```
Full Thebe coverage needs `TOTAL_STEPS ≥ thebe_pool / thebe_fraction` (4:3:3 → ≥94 640; 8:1:1 → ≥48 000).
Checkpoints every `CKPT_EVERY` (10k) steps; `RESUME=1` warm-restarts from the last one.

## 3. `{alone}` baselines (ablation control) — each survey trained by itself, same loss/toggles
```
SURVEY=thebe STEPS=40000 SAVE=hybrid/checkpoints/reader_alone_thebe_full.pt scripts/alone.sh
SURVEY=cracks STEPS=1000 scripts/alone.sh ; SURVEY=smeaheia STEPS=1000 scripts/alone.sh
```

## 4. Everything at once — the full paper ablation (+ report)
```
scripts/ablation.sh        # {alone} + both joint configs + uncapped benchmarks + runs/report.md
scripts/sweep.sh           # sweep weightings/loss (edit CONFIGS[]) + report
```

## 5. Benchmark + report (standalone)
```
CKPT=hybrid/checkpoints/reader_joint_full.pt DATASETS=thebe,cracks,smeaheia scripts/benchmark.sh
scripts/report.sh          # assemble runs/bench_*.log → runs/report.md (VISION paper tables)
scripts/eval.sh            # INTERNAL/LANGUAGE report → runs/eval_report.txt (copy · CHAIR · BLEU/METEOR/CIDEr)
```
The language report (`eval.sh`) needs the synthetic data + narrator (real datasets are vision-only); the
vision report (`benchmark.sh`/`report.sh`) runs on any dataset.

## What the report contains (honest by construction)
- **Full metrics** per dataset · per checkpoint (never pooled): Dice **oracle/deploy**, pixel P/R/F1, tol-F1,
  **gated** detection F1, class, dip(const), throw(const), with `n`.
- **Complementarity**: `{alone}` → `{joint}` per survey (Smeaheia untrainable-alone → functional-in-joint).
- **Config tradeoff**: 4:3:3 (mask-best) vs 8:1:1 (attribute-best: dip **and** throw).
- Dice is the **deploy** number (`detect()`, not teacher-forced); **dip claimable only for Smeaheia**
  (Thebe/CRACKS dip is mask-derived → circular); `✓` = beats the constant predictor.

## Config knobs (`scripts/config.sh`, all env-overridable)
`TVERSKY` · `POS_WEIGHT_MAX` · `TRAIN_CLASS/MEASURE/DERIVED` · `N_TEST` · `CKPT_EVERY` · `WEIGHTS` · `TOTAL_STEPS`.
Single-seed (round-robin seed fixed at 0) — the one stated limitation.
