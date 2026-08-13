# Grounded Seismic VLM — user guide

A vision-language model for **2-D seismic interpretation**. The vision stack **measures** the geology
(fault count, dip, throw, location, per-instance masks); a frozen-base language model **copies** those
numbers across a deliberately non-differentiable *digit seam* and narrates them in words. Because no
gradient crosses the seam, the LM **cannot invent a number the vision stack did not measure**.

```
seismic image ──▶ SFM ViT (frozen) ──▶ DETR reader ──▶ measured facts ──▶ digit tokens ──▶ Qwen-1.5B LoRA ──▶ grounded answer
                  native tiling         dip/throw/mask   (dip 61°, [86,505])   "seam"          copies + reasons   + mask overlay
```

This guide covers the whole loop: **configure → add attributes → train → evaluate/benchmark → infer**.
Deeper docs: [`SETUP.md`](SETUP.md) (env + assets), [`hybrid/REPRODUCE.md`](hybrid/REPRODUCE.md) (full
pipeline), [`hybrid/ARCHITECTURE.md`](hybrid/ARCHITECTURE.md) (mechanism), [`scripts/README.md`](scripts/README.md)
(runners), [`hybrid/MODEL_RESEARCH.md`](hybrid/MODEL_RESEARCH.md) (results/provenance).

---

## 0. Install (once)

Full detail in [`SETUP.md`](SETUP.md). TL;DR (uv, Python 3.12, CUDA 12.8):
```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
export SFM_CKPT=hybrid/checkpoints/SFM-Base-512.pth     # the frozen encoder — REQUIRED (hard error if absent)
```
Datasets: **CRACKS** + **synthetic** (`thirdExec/synthetic-seismic-vlm`) auto-download from HuggingFace, **Thebe**
auto-downloads from Harvard Dataverse; **Smeaheia** you supply (see SETUP.md §3). The synthetic materializes to
`data/synthetic/…csv` on first use (`SYNTH_CSV` overrides it — e.g. point at a locally-generated newer version).
The scripts expect `./.venv/bin/python`.

---

## 1. Configure

Two surfaces, both **env-overridable** (the project uses env knobs, *not* argparse):

**A) `scripts/config.sh`** — central, sourced by every runner. Defaults encode the *validated* methodology;
override per-run with `KEY=value scripts/<runner>.sh`.

| knob | default | what |
|---|---|---|
| `SFM_CKPT` | `hybrid/checkpoints/SFM-Base-512.pth` | frozen encoder (required) |
| `TVERSKY` | `0.4,0.6,1.0` | mask loss: Dice + Focal-Tversky(α,β,γ) |
| `POS_WEIGHT_MAX` | `15` | BCE positive-weight clamp (over-prediction control) |
| `TRAIN_CLASS` / `TRAIN_MEASURE` / `TRAIN_DERIVED` | `1` / `1` / `0` | which reader heads train (derive off on real) |
| `N_TEST` | `100000` | held-out cap (≥ split ⇒ **uncapped**, per-dataset, never pooled) |

**B) Per-module / per-runner knobs** (read via `os.environ`):

| knob | where | what |
|---|---|---|
| `DILATE_R` | `hybrid/data/loader.py` | mask dilation; **0 = pure/undilated GT** (tolerance from the loss, not the target) |
| `SYNTH_CSV` | `hybrid/data/synthetic` | synthetic dataset CSV path |
| `SMEAHEIA_LINES` | `hybrid/data/smeaheia` | `0` = GN1101 3-D-cube GT (default), `1` = legacy 2-D lines |
| `WEIGHTS`, `TOTAL_STEPS` | `scripts/joint.sh` | joint survey weighting (e.g. `thebe:4,cracks:3,smeaheia:3`) + steps |
| `SURVEY`, `STEPS` | `scripts/alone.sh` | single-survey baseline |
| `DATASETS`, `CKPT` | `scripts/benchmark.sh` | which held-out splits, which checkpoint |
| `CKPT_EVERY`, `RESUME` | runners | checkpoint interval / warm-restart from last |
| `EVIDENCE_TOKENS`, `ANSWER_TOKENS` | `hybrid/stages/stage3_answer.py` | narrator generation budgets (320 / 512) |
| `DET_TAU` | `hybrid/eval/benchmark.py` | detection F1 gate (normalized-centroid τ) |

---

## 2. Add an attribute

Everything routes through **`hybrid/model/registry.py`** — the single source of truth. Three cases:

**Add a TIER-1 measurement** (like dip/throw/area — read by its own head):
1. Add one row to `MEASURE` — `"newattr": ("spatial"|"pooled", scale)`. `spatial` reads the mask's 2nd-moment
   footprint (orientation/extent → dip/area); `pooled` reads the pooled feature (magnitudes → throw).
2. Map its dataset key in `MEASURE_KEY` (`"newattr": "values.measure key"`).
3. List it under the classes that carry it in `CLASS_SCHEMA` (e.g. `"fault": ["dip","throw","newattr"]`).

The reader builds one head per measure automatically (`nn.ModuleDict`) — **no new head class**. Presence is
data-gated, so a class may declare a measure the dataset doesn't fill (simply not supervised).

**Add a TIER-2 derived attribute** (reasoned after the read, one shared query-conditioned head):
- Add **one row** to `DERIVED`: `(dataset_key, marker_word, kind, labels, scope, klass)`. `kind` ∈
  `scalar|cat|bool`; `scope` ∈ `SECTION|OBJECT`. Its list position is the head's query index. `scenes` routes
  its GT, `reader` supervises that index, the narrator states it via `marker_word`. **Never a new head.**

**Add an object class** (fault/closure/salt/onlap → a new one):
- Add one row to `CLASS_ID` (`"newclass": 5`) and widen `reader.class_head`. The name is also the word the
  narrator states.

> Tip: `registry.schema_from_csv(csv_path)` scans a dataset and prints the `{class: {measure, derive}}` map so
> you can see exactly which attributes your data carries before wiring them in.

---

## 3. Train

Full pipeline + timings in [`hybrid/REPRODUCE.md`](hybrid/REPRODUCE.md). From scratch:

```bash
# 1) synthetic base (the frozen substrate) — needs SYNTH_CSV
python -m hybrid.stages.stage1_geology          # frozen geology-CoT adapter
python -m hybrid.run_train                       # reader.pt (vision) + narrator (grounding + fuse LoRA)

# 2) real-field joint (the deployed model) — one shared reader over all surveys, weighted round-robin
WEIGHTS=thebe:4,cracks:3,smeaheia:3 TOTAL_STEPS=100000 \
  SAVE=hybrid/checkpoints/reader_joint_full.pt scripts/joint.sh

# 3) {alone} baselines (ablation control), or run everything at once:
scripts/ablation.sh                              # {alone} + both joint configs + all benchmarks + report
```
`RESUME=1` warm-restarts from the last checkpoint. Training checkpoints every `CKPT_EVERY` (10k) steps.
Cloud (big-GPU, ~1–3 h from scratch): see `~/Desktop/Modal_instance/trainer/`.

---

## 4. Evaluate & benchmark

**Uncapped, per-dataset, never pooled** (the datasets are incommensurable). See [`scripts/README.md`](scripts/README.md).

```bash
CKPT=hybrid/checkpoints/reader_joint_full.pt DATASETS=thebe,cracks,smeaheia scripts/benchmark.sh
scripts/report.sh          # assemble runs/bench_*.log → runs/report.md (VISION paper tables)
scripts/eval.sh            # LANGUAGE report → runs/eval_report.txt (copy · CHAIR · BLEU/METEOR/CIDEr)
scripts/inference.sh       # narration-on-real + malform tally → runs/inference_<survey>.txt
```
What the report contains: **pooled IoU** (the paper metric — deployed fault-union I/U accumulated over the
split, divided once), Dice **oracle**(teacher-forced) **and deploy**(`detect()`), pixel P/R/F1, tol-F1,
**gated** detection F1, class, dip/throw with their **constant-predictor baselines** and `n`. `✓` = beats the
constant. Metrics are pinned by unit tests (`python -m hybrid.tests.test_benchmark`).

---

## 5. Use it — VLM-style inference

Give it an **image** and (optionally) a **question**; get the measured geology + a grounded answer + a mask overlay:

```bash
IMAGE=path/to/section.png python -m hybrid.infer
IMAGE=path/to/section.png QUESTION="Where is the fault and what does its dip imply?" \
  READER=hybrid/checkpoints/reader_joint_full.pt python -m hybrid.infer
```
Output:
```
=== section.png (192x1251) · reader=reader_joint_full.pt ===
[measured] 1 fault(s):
   fault 0: dip 60.7° · center [90, 459] · throw 51 ms
[Q] Where is the fault and what does its dip imply?
[reasoning] The fault at [90,459] ... dips at about 60.7 degrees ...
[answer] The fault at [90,459] has throw of about 51 ms and dips at about 60.7 degrees.
[overlay] hybrid/inference/vlm_overlay.png
```
Knobs: `IMAGE` (required), `QUESTION`, `READER` (use `reader_real_<survey>.pt` or `reader_joint_full.pt` for
real data; `reader.pt` is the synthetic base), `NARRATOR`, `OUT` (overlay PNG; `""` to skip). Every number in the
answer is *measured by the vision stack and copied* — the LM cannot fabricate one.

For a **batch** run over a dataset's held-out split (chains + overlays + malform tally):
```bash
DATASET=thebe READER=hybrid/checkpoints/reader_real_thebe.pt N=3 python -m hybrid.eval.inference
```

---

## Repo map

| path | what |
|---|---|
| `hybrid/infer.py` | **VLM-style inference** — image + question → grounded answer + overlay |
| `hybrid/run_train.py` | from-scratch training entry (reader + narrator) |
| `hybrid/model/` | reader (DETR), captioner (LM+adapters), encoder (SFM), **registry** (attributes), geometry |
| `hybrid/stages/` | training curriculum: geology → reader → grounding → fuse-fold → real finetune |
| `hybrid/data/` | loader (native tiling), schema, dataset converters (thebe / cracks / smeaheia) |
| `hybrid/eval/` | benchmark, metrics, components, inference.py (batch held-out), runners |
| `scripts/` | `config.sh` + reproducible runners (train / joint / alone / benchmark / report / eval) |
| `hybrid/checkpoints/` | SFM encoder + trained readers/narrator (not in git — see SETUP.md) |
