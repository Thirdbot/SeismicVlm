# Run scripts — reproducible, configurable seismic-VLM training & evaluation

Every script sources `config.sh` (central config; env-overridable) and encodes the **current validated
methodology**: weighted-round-robin complementary-joint finetune, fixed mask loss (Dice + Focal-Tversky
0.4/0.6 + pos-weight clamp 15), derive-heads off on real, **uncapped per-dataset** evaluation (never pooled).

Nothing is capped or pooled; nothing falls back silently. Change behavior only via env — defaults are the
validated ones.

## Building blocks
| script | what | example |
|---|---|---|
| `config.sh` | central config (paths, loss, toggles, eval policy) — sourced, not run | — |
| `joint.sh` | complementary-joint finetune (one config) | `WEIGHTS=thebe:4,cracks:3,smeaheia:3 TOTAL_STEPS=100000 scripts/joint.sh` |
| `alone.sh` | single-survey `{alone}` baseline (ablation control) | `SURVEY=smeaheia STEPS=1000 scripts/alone.sh` |
| `benchmark.sh` | comprehensive per-dataset report for one checkpoint | `CKPT=…/reader_joint_full.pt DATASETS=thebe,cracks,smeaheia scripts/benchmark.sh` |

## Compositions
| script | what |
|---|---|
| `ablation.sh` | full paper ablation: `{alone}` baselines + both joint configs + all benchmarks |
| `sweep.sh` | sweep the joint over weightings / loss (edit the `CONFIGS` array), train + benchmark each |

## Reproducibility
- Round-robin feeder is seeded (deterministic sequence for a given config).
- Full-data coverage requires `TOTAL_STEPS ≥ thebe_pool / thebe_weight_fraction` (4:3:3 → ≥ 94 640; 8:1:1 → ≥ 48 000).
- Training checkpoints every `CKPT_EVERY` (default 10 000) steps → crash-survivable long runs.
- Logs land in `$RUN_DIR` (default `runs/`).

## Config tradeoff (why two joint configs)
`4:3:3` is **mask-best**; `8:1:1` is **attribute-best** (dip *and* throw). No single weighting wins both — the
ablation reports both so the tradeoff is explicit.
