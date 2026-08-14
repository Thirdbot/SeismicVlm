# Run scripts — reproducible, configurable seismic-VLM training & evaluation

Every script sources [`config.sh`](config.sh) (central config; every value env-overridable) and encodes the
**current validated methodology**: SFM-512 frozen encoder, native tiling, DETR reader, weighted-round-robin
real-field finetune, fixed mask loss (Focal-Tversky 0.4/0.6/1.0 + pos-weight clamp 15), and **uncapped
per-dataset** evaluation (never pooled, nothing falls back silently). Change behaviour only via env —
the defaults are the validated ones; don't change a default without re-validating.

Convention: no argparse. Everything is env knobs (`FOO=bar scripts/x.sh`). `$PY`, `$CKPT_DIR`, `$RUN_DIR`,
`$SFM_CKPT` come from `config.sh`.

## End-to-end experiments
| script | what | run |
|---|---|---|
| [`ab_experiment.sh`](ab_experiment.sh) | **The A/B attribute experiment, one command.** Builds datasets → synthetic base (reader, or `SYNTH_FULL=1` for reader+geology+language) → zero-shot real benchmark → ratio selection (train candidate `RATIOS`, pick winner by `SELECT_METRIC`) → **A** (no real attributes) vs **B** (real dip/throw) on the winning ratio → summary table. Weights land in `$CKPT_DIR/ab_experiment/`. | `SYNTH_FULL=1 scripts/ab_experiment.sh` |
| [`ablation.sh`](ablation.sh) | Full paper ablation: `{alone}` baselines + joint configs + all benchmarks. Sequential, ~a day, checkpointed every 10k. | `scripts/ablation.sh` |
| [`sweep.sh`](sweep.sh) | Sweep the joint over weightings / loss (edit the `CONFIGS` array), train + benchmark each. | `scripts/sweep.sh` |

Key `ab_experiment.sh` knobs: `DATASETS` · `RATIOS` · `SELECT_METRIC` (default `detF1`) · `TOTAL_STEPS` ·
`SYNTH_FULL` (reader+language vs reader-only) · `ACTIVE_CLASSES` (which class heads train, default `fault`) ·
`DET_THRESH` (default 0.9) · `THEBE_SOURCE` (default `patches`) · `FAST=1` (quick signal). Header has the full list.

## Training building blocks
| script | what | example |
|---|---|---|
| [`joint.sh`](joint.sh) | complementary-joint real-field finetune (one round-robin config) | `WEIGHTS=thebe:4,cracks:3,smeaheia:3 TOTAL_STEPS=100000 scripts/joint.sh` |
| [`alone.sh`](alone.sh) | single-survey `{alone}` baseline (the complementarity control) | `SURVEY=smeaheia STEPS=1000 scripts/alone.sh` |

(The synthetic base — geology adapter, reader, grounding, fuse-fold — trains via `python -m hybrid.run_train`
and `python -m hybrid.stages.stage1_geology`; `ab_experiment.sh SYNTH_FULL=1` drives both for you.)

## Evaluation
| script | what | example |
|---|---|---|
| [`benchmark.sh`](benchmark.sh) | **vision** per-dataset report for one checkpoint (mask pooled-IoU/Dice, detF1, dip/throw vs constant, n). Per-dataset, never pooled, uncapped. | `CKPT=…/reader_joint_full.pt DATASETS=thebe,cracks,smeaheia scripts/benchmark.sh` |
| [`eval.sh`](eval.sh) | full **internal** report → `runs/eval_report.txt`: schema · copy · reader mechanism · attrs · boxes · CHAIR. (Synthetic for the language parts; real sets are vision-only.) | `scripts/eval.sh` |
| [`eval_language.sh`](eval_language.sh) | **language only** (copy fidelity + CHAIR faithfulness), no vision/box metrics. The narrator trains independently, so this re-scores just the language half on saved weights. **Synthetic-only** (real has no narration GT). | `scripts/eval_language.sh` |
| [`inference.sh`](inference.sh) | narration-on-real qualitative report + malform tally; saves mask overlays + the ungated dataset GT per scene. | `DATASET=smeaheia READER=…/B_joint.pt scripts/inference.sh` |
| [`report.sh`](report.sh) | assemble benchmark logs into the paper tables (markdown) → `$RUN_DIR/report.md`. | `scripts/report.sh` |

Note: narration-overlap metrics (BLEU/METEOR/CIDEr) were **removed** — free-generated narration doesn't
match the templated dataset answer, so overlap under-measures a grounded caption. Faithfulness = CHAIR only,
and it backs the **full** injected marker set (count/bbox/center/dip/throw/area/derived), not just dip/throw/area.

## Run management
| script | what |
|---|---|
| [`finish.sh`](finish.sh) | eval-only finish: run the uncapped benchmarks + report on already-trained checkpoints (no training). |
| [`recover.sh`](recover.sh) | power-cut recovery: warm-resume a partial joint, then the fresh configs + benchmarks. |
| [`download/`](download/) | dataset fetch helpers. |

## Reproducibility
- Round-robin feeder is seeded (deterministic sequence for a given config); global `SEED=42`.
- Full-data coverage needs `TOTAL_STEPS ≥ thebe_pool / thebe_weight_fraction`.
- Training checkpoints every `CKPT_EVERY` (default 10 000) steps → crash-survivable.
- Logs land in `$RUN_DIR` (default `runs/`). Every checkpoint's benchmark is a separate process → a re-cut only loses the in-flight one.

## Choosing the round-robin ratio
No single per-survey weighting wins every axis, and the winner depends on the **build and the metric** you
optimize — so `ab_experiment.sh` selects it *empirically*: it trains the candidate `RATIOS` and picks the
one that maximizes `SELECT_METRIC` (default `detF1`, since attribute values are only read on *detected*
faults). On the Kaggle Thebe-patches build that selects **1:1:1** (balanced — best detection, donates
detection to the small surveys). The canonical Dataverse ablation (`ablation.sh`) instead reports the
explicit tradeoff **4:3:3 (mask-best)** vs **8:1:1 (attribute-best)**. **Compare ratios only within one
build** — patches and Dataverse numbers are not comparable.
