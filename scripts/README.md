# Run scripts — reproducible, configurable seismic-VLM training & evaluation

Every script sources [`config.sh`](config.sh) (central config; every value env-overridable) and encodes the
**current validated methodology**: SFM-512 frozen encoder, native tiling, DETR reader, native-resolution
mask decoder (`MASK_UPSAMPLE=4`), weighted-round-robin real-field finetune, fixed mask loss (Focal-Tversky
0.4/0.6/1.0 + pos-weight clamp 15), and **uncapped per-dataset** evaluation (never pooled, nothing falls back
silently). Change behaviour only via env — the defaults are the validated ones; don't change a default
without re-validating. (Checkpoints are `MASK_UPSAMPLE=4`; setting `=3` builds a 3-stage decoder that
cannot load them.)

Convention: no argparse. Everything is env knobs (`FOO=bar scripts/x.sh`). `$PY`, `$CKPT_DIR`, `$RUN_DIR`,
`$SFM_CKPT` come from `config.sh`.

## End-to-end experiments
| script | what | run |
|---|---|---|
| [`run_all.sh`](run_all.sh) | **The full real-field pipeline, one sequence** (ungated volume + all-pos/neg): alone per survey (coverage-matched `ALONE_EPOCHS`) → per-survey `DET_THRESH` sweep → cross-eval 3×3 → ratio-selection (joint) → A/B → final benchmark. Deploy (joint vs per-domain alone) decided AFTER, from the printed table. | `N_CHUNKS=18 scripts/run_all.sh` |
| [`ab_experiment.sh`](ab_experiment.sh) | **The A/B attribute experiment, one command.** Builds datasets → synthetic base (reader, or `SYNTH_FULL=1` for reader+geology+language) → zero-shot real benchmark → ratio selection (train candidate `RATIOS`, pick winner by `SELECT_METRIC`) → **A** (no real attributes) vs **B** (real dip/throw) on the winning ratio → summary table. Weights land in `$CKPT_DIR/ab_experiment/`. | `SYNTH_FULL=1 scripts/ab_experiment.sh` |
| [`deploy.sh`](deploy.sh) | **Consolidation** (not exploration): one measure-ON RR-joint run at the SETTLED config (best ratio + attrs un-gated + chosen `DILATE_R`) → `deploy.pt` + benchmark (+ optional panels). Grounding guard: pass the ratio selected on the SAME base+dilation you deploy. | `WEIGHTS=thebe:1,cracks:1,smeaheia:1 scripts/deploy.sh` |
| [`ablation.sh`](ablation.sh) | Full paper ablation: `{alone}` baselines + joint configs + all benchmarks. Sequential, ~a day, checkpointed every 10k. | `scripts/ablation.sh` |
| [`sweep.sh`](sweep.sh) | Sweep the joint over weightings / loss (edit the `CONFIGS` array), train + benchmark each. | `scripts/sweep.sh` |
| [`dilate_sweep.sh`](dilate_sweep.sh) | **Future-work diagnostic** — vary ONLY real-field `DILATE_R` (ratio fixed), benchmark self-vs-pure. detF1 is dilation-independent; only `pure_*` rows + tol-F1 compare segmentation honestly. The paper stays at `DILATE_R=0` (pure); this quantifies the tradeoff, it is NOT part of the reported pipeline. | `scripts/dilate_sweep.sh` |

Key `ab_experiment.sh` knobs: `DATASETS` · `RATIOS` · `SELECT_METRIC` (default `detF1`) · `TOTAL_STEPS` ·
`SYNTH_FULL` (reader+language vs reader-only) · `ACTIVE_CLASSES` (which class heads train, default `fault`) ·
`DET_THRESH` (default 0.9) · `THEBE_SOURCE` (default `volume` = Dataverse 3-D volume; `patches` = opt-in Kaggle set) · `N_CHUNKS` (volume chunks, default 2 — set `18` for the full 50,484-panel build) · `FAST=1` (quick signal). Header has the full list.

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
| [`benchmark.sh`](benchmark.sh) | **vision** per-dataset report (RES cIoU=pooled-IoU / gIoU / Pr@0.5/0.7/0.9 · tol-F1@2px · threshold-free detection AP · formal dip/throw MAE·RMSE·skill·n). Per-dataset, never pooled, uncapped. `DETECT_ONLY=1` skips the (threshold-independent) oracle mask metrics for a fast threshold sweep. | `CKPT=…/B_joint.pt DATASETS=thebe,cracks,smeaheia scripts/benchmark.sh` |
| [`threshold.sh`](threshold.sh) | `DET_THRESH` sweep on an existing checkpoint (no training) → per-survey F1-optimal + average. Scores PURE masks. `EVAL_SPLIT=val` selects on val. | `CKPT=…/B_joint.pt scripts/threshold.sh` |
| [`cross_eval.sh`](cross_eval.sh) | cross-survey 3×3 (train-one → eval-all) reusing the `run_all` `alone_*.pt` — cIoU/tolF1/detF1 matrix (the "why joint" + segmentation-transfer diagnostic). Read-only. | `scripts/cross_eval.sh` |
| [`eval.sh`](eval.sh) | full **internal** report → `runs/eval_report.txt`: schema · copy · reader mechanism · attrs · boxes · CHAIR. (Synthetic for the language parts; real sets are vision-only.) | `scripts/eval.sh` |
| [`eval_language.sh`](eval_language.sh) | **language faithfulness**: copy fidelity (GT-injected ceiling + reader-piped deployment) · answer-CHAIR · think-CHAIR · think→answer consistency · caption-overlap (BLEU/ROUGE-L/CIDEr-D, comparability only). Narrator trains independently → re-scores the language half on saved weights. Runs on synthetic *or* a real survey (`DATASET=smeaheia READER=…/B_joint.pt` = deployment copy). | `ACTIVE_CLASSES=fault scripts/eval_language.sh` |
| [`eval_swap.sh`](eval_swap.sh) | **causal-swap** seam faithfulness: intervene on each injected value, measure how often the narration FOLLOWS the swap (flip) vs states the un-swapped value (baseline). `USE_READER=1` swaps reader-measured facts (deployment). | `DATASET=smeaheia READER=…/B_joint.pt scripts/eval_swap.sh` |
| [`inference.sh`](inference.sh) | narration-on-real qualitative report + malform tally; saves mask overlays (+ boxes when `BOXES=1`) + the ungated dataset GT per scene. | `N=6 BOXES=1 DATASET=smeaheia READER=…/B_joint.pt scripts/inference.sh` |
| [`report.sh`](report.sh) | assemble benchmark logs into the paper tables (markdown) → `$RUN_DIR/report.md`. | `scripts/report.sh` |

Note on faithfulness vs overlap: the faithfulness metrics are **copy fidelity + CHAIR + causal swap** — CHAIR
backs the **full** injected marker set (count/bbox/center/dip/throw/area/derived). Caption-overlap
(BLEU/ROUGE-L/CIDEr-D in `hybrid/eval/caption_metrics.py`) is reported **for field comparability only, clearly
labelled a LOWER BOUND** — the narration is free-generated and doesn't template-match the dataset answer, so
overlap under-measures a correct grounded caption; it is NOT a faithfulness axis. Overlap applies to synthetic
only (real surveys have no reference captions → it skips gracefully).

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
optimize — so `run_all.sh` (and `ab_experiment.sh`) select it *empirically*: train the candidate `RATIOS`,
pick the one that maximizes mean `detF1` (attribute values are only read on *detected* faults). On the
native-res **Dataverse volume** run, **1:1:1 wins** (mean detF1 0.348 > 8:1:1 0.338 > 4:3:3 0.326) — balanced,
donates detection to the small surveys. Ratio is a JOINT-only choice; deployment is decided *after* from the
final table (per-domain: Thebe alone; CRACKS+Smeaheia joint). **Compare ratios only within one build** —
Kaggle-patches and Dataverse-volume numbers are not comparable.
