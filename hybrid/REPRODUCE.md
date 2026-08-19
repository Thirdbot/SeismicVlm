# REPRODUCE — the reported tables from scratch

This is the full pipeline that produces the paper's numbers: the **synthetic base**, then the
**native-resolution real-field run** (`run_all.sh`), then the report/eval/inference passes. Every step
chains the committed primitives — no hidden training logic. For downloading pretrained weights and running
**inference only** (no retraining), see [`../SETUP.md`](../SETUP.md) §3d instead.

> **Hardware.** ~6 GB VRAM is enough for **inference** ([`../SETUP.md`](../SETUP.md) §3d/§4). The **full
> reproduce below is not** — the native-res decoder (`MASK_UPSAMPLE=4`) on the full Thebe volume needs a
> **large-VRAM GPU (24 GB class)**. It is **single-seed** and takes **~a day on one GPU**.
> `RESUME=1` warm-restarts an interrupted training run; checkpoints land every `CKPT_EVERY` (10 000) steps.

## 0. Prerequisites
Environment + assets per [`../SETUP.md`](../SETUP.md): the **SFM encoder** (§3a, required), and the datasets
(§3b — synthetic/CRACKS/Thebe auto-download, **Smeaheia is manual**). For the **full Thebe volume** (50,484
panels, the build the tables use) set **`N_CHUNKS=18`** — the default `N_CHUNKS=2` is a ~3 GB smoke subset.

```bash
export SFM_CKPT=hybrid/checkpoints/SFM-Base-512.pth   # frozen encoder — REQUIRED
export N_CHUNKS=18                                     # full Thebe volume (not the 2-chunk smoke set)
```

## 1. Synthetic base (reader + narrator + geology adapter)
The real-field run trains **on top of** the frozen synthetic base. Build it first:

```bash
python -m hybrid.stages.stage1_geology     # frozen geology LoRA (→ hybrid/checkpoints/stage1_<hash>/)
python -m hybrid.run_train                  # reader.pt + narrator (needs the synthetic data; auto-downloads)
```

(`scripts/ab_experiment.sh SYNTH_FULL=1` drives both for you if you prefer one command.) The mask decoder is
**native resolution** by default (`MASK_UPSAMPLE=4`); checkpoints built at `=4` will **not** load under `=3`.

## 2. The real-field pipeline — one sequence
`run_all.sh` runs the whole real-field methodology in order and writes every checkpoint + log under
`$CKPT_DIR/run_all/`. **Loss and dilation are untouched** (Focal-Tversky 0.4/0.6/1.0, pos-weight clamp 15,
`DILATE_R=0` pure masks); all metrics score PURE masks.

```bash
N_CHUNKS=18 scripts/run_all.sh
```

It performs, in order:
1. **Alone** per survey — coverage-matched steps (`ALONE_EPOCHS=3 ×` that survey's own train pool, floored at
   5 000), giving each survey's single-survey ceiling → `alone_<survey>.pt`.
2. **Threshold** — sweep `DET_THRESH ∈ {0.5,0.7,0.8,0.9,0.95}` per survey, pick each survey's F1-optimal, and
   the average (used for the joint eval).
3. **Cross-eval** — a 3×3 train-one → eval-all matrix (cIoU / tol-F1 / detF1): the "why joint" +
   segmentation-transfer diagnostic. Read-only over the `alone_*.pt`. (`CROSS_EVAL=0` to skip.)
4. **Ratio-selection** (joint) — train each candidate mix `RATIOS="1:1:1 8:1:1 4:3:3"`, benchmark all
   surveys, pick the **best mean detF1** → `ratio_<tag>.pt`.
5. **A/B** on the winning ratio — **A** = attributes off (`TRAIN_MEASURE=0`, reuses the ratio ckpt) vs **B** =
   real dip/throw on (`TRAIN_MEASURE=1`) → `A_joint.pt`, `B_joint.pt`.
6. **Final benchmark** — alone + A + B at the average threshold, PURE masks, common metrics
   (cIoU/gIoU/Pr@X · det AP · formal dip/throw MAE·RMSE·skill) → `final_*.log`.

## 3. Reports, language, qualitative
```bash
scripts/report.sh        # → runs/report.md          — vision paper tables (per-survey, never pooled)
scripts/eval.sh          # → runs/eval_report.txt    — language: copy · CHAIR · (BLEU/ROUGE-L/CIDEr = lower-bound comparability)
scripts/inference.sh     # → runs/inference_<survey>.txt + hybrid/inference/ overlays (BOXES=1 adds boxes)
```

## 4. Deployment decision (read the final table, don't assume)
`run_all.sh` does **not** pick the deployed model — you decide from `final_*` by comparing each survey's
**alone** ceiling against the **A/B joint**:
- **Thebe** — strongest alone; deploy `alone_thebe.pt` (its detection prefers the single-survey model).
- **CRACKS + Smeaheia** — data-limited; deploy the **joint `B_joint.pt`** (Smeaheia is untrainable alone —
  detF1 0 → joint rescues it). `A≈B` on a survey ⇒ real attribute supervision wasn't needed there.

This is the per-domain deployment the tables report — one flagship survey (Thebe, alone) plus a joint
extension for the two small surveys. Any single ratio/threshold is a training-time choice; deployment is
chosen **after**, from the measured table.
