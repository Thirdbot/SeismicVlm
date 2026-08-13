# RUNBOOK — fast-but-solid signal run (~1.5 h, from scratch)

Tuned for a **24 GB VRAM / ~64 GB RAM / multi-core** box. This gets the plan's core answer —
*do the synthetic-trained attribute heads generalize to real features?* — in about **1.5 hours**, without
wasting time on parts that don't affect that answer.

**Why it's fast *and* solid:** the question is decided by the **vision reader** (`benchmark.py` scores the
reader's attributes; the LM narrator plays no part). So this run trains only the reader (`READER_ONLY=1`,
skipping geology/grounding/fold) and tests on **Smeaheia** directly (the only survey with trustworthy
dip/throw) — leaving the splits leak-free and the eval uncapped, which is where "solid" actually lives.

---

## 0 · Env + prerequisites (once)

```bash
export SFM_CKPT=hybrid/checkpoints/SFM-Base-512.pth   # place this file first (required encoder)
export HF_HUB_DISABLE_XET=1        # reliable HF downloads
export GRAD_CKPT=0                 # 24 GB: faster (moot under READER_ONLY, harmless)
export N_TEST=100000               # ALWAYS uncapped eval (no split contamination)
```

Smeaheia is **required** here (it's the eval target) and is a **manual** dataset — place its three inputs per
[`data/real_data/README.md`](data/real_data/README.md), then build it (CPU, ~5 min):

```bash
python -m hybrid.data.smeaheia.build_from_cube     # → real_field_cube.csv (throw needs the horizons/*.shp)
```

No Thebe/CRACKS download needed for the signal — they don't affect the Smeaheia attribute test.

## 1 · Synthetic reader only  (~25–35 min)

```bash
READER_ONLY=1 ACTIVE_CLASSES=fault,closure,onlap READER_EPOCHS=40 DILATE_R=0 \
  python -m hybrid.run_train
```
Stops after `reader.pt` — skips the LM stages (and geology, so no `stage1_geology` needed). Watch the
`[reader test(held-out)]` line: dip MAE should be well below the constant baseline. (40 epochs is the *signal*
setting; the solid run uses ~120 — see bottom.)

## 2 · Detection operating point  (~3 min, no retrain)

```bash
for T in 0.5 0.7 0.9; do echo "== $T =="; DET_THRESH=$T ACTIVE_CLASSES=fault,closure,onlap \
  CKPT=hybrid/checkpoints/reader.pt DATASETS=synthetic python -m hybrid.eval.benchmark; done
```
Pick the `T` with the best detF1 + pooled-IoU → use it as `<T>` below.

## 3 · The test — Plan A vs Plan B on Smeaheia  (~20–30 min)

```bash
# Plan A — attributes FROZEN from synthetic (does the head generalize to real features?)
ACTIVE_CLASSES=fault TRAIN_MEASURE=0 SURVEY=smeaheia STEPS=1000 \
  SAVE=hybrid/checkpoints/reader_sme_noattr.pt scripts/alone.sh
# Plan B — Smeaheia's real dip/throw train (the domain-gap control)
ACTIVE_CLASSES=fault TRAIN_MEASURE=1 SURVEY=smeaheia STEPS=1000 \
  SAVE=hybrid/checkpoints/reader_sme_attr.pt scripts/alone.sh
```

## 4 · Read the result — DIP first  (~5 min)

```bash
DET_THRESH=<T> ACTIVE_CLASSES=fault CKPT=hybrid/checkpoints/reader_sme_noattr.pt DATASETS=smeaheia python -m hybrid.eval.benchmark
DET_THRESH=<T> ACTIVE_CLASSES=fault CKPT=hybrid/checkpoints/reader_sme_attr.pt   DATASETS=smeaheia python -m hybrid.eval.benchmark
```

Look at the `ATTR … dip MAE (const …)` line:

- **Plan A dip beats its constant and is close to Plan B** → synthetic heads **generalize**; real attributes are optional. (Primary claim holds.)
- **Plan A dip ≈ constant, Plan B much better** → **domain gap**; real attributes needed.

Judge **dip** — it's 48 %-covered on synthetic and geometry-derived (trustworthy). **Throw** is only ~18 %
covered on synthetic, so a weak throw result is likely undertraining, not a real conclusion.

---

## Scaling this signal up to the full solid run

When the signal looks right, promote it (this is the ~1-day version):

- **Reader:** drop `READER_ONLY`, raise `READER_EPOCHS=120`, and run the full `run_train` (adds the LM
  narrator: geology → grounding → fuse fold). Prefetch the geology model/dataset first (`hf download …`).
- **Real:** replace Smeaheia-alone with the **weighted round-robin joint** over all three surveys:
  `WEIGHTS=thebe:4,cracks:3,smeaheia:3 TOTAL_STEPS=100000 scripts/joint.sh` — needs Thebe
  (`THEBE_VERSION=1.0`, ~30 GB, run in the background) and CRACKS (auto).
- Everything else (uncapped eval, leak-free splits, `DILATE_R=0`, `ACTIVE_CLASSES`) stays identical.
