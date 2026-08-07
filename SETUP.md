# SETUP — from a fresh clone to a run

The repo ships the **code, scripts, and docs**. The **environment, model weights, and datasets are not in git**
(too large / external — see `.gitignore`). This gets you from `git clone` to running. It uses **uv**.

## 1. Prerequisites
- A **GPU with ~6 GB** (built for an RTX 3060 5.67 GB — 4-bit LM + frozen encoder keep it small), **CUDA 12.8**.
- **Python 3.12** and **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## 2. Environment (uv)
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate            # the scripts expect ./.venv/bin/python
```
`requirements.txt` is the exact working env. `torch==2.10.0` is the **CUDA 12.8** build (the `nvidia-*-cu12`
wheels are pinned). For a different CUDA or CPU-only, install `torch` first from the matching index, then run the
`uv pip install -r requirements.txt`. (`import unsloth` must precede transformers/peft — the code already does this.)

## 3. Assets you must supply (not in git)

### 3a. SFM encoder — **REQUIRED** (absent → hard error, by design; no silent fallback)
Place the Seismic Foundation Model ViT-B/16 @512 checkpoint at:
```
hybrid/checkpoints/SFM-Base-512.pth        # or set SFM_CKPT=/path/to/it
```

### 3b. Datasets
| dataset | how to get it | where the code looks |
|---|---|---|
| **CRACKS** | **auto-downloads** from HuggingFace (`gOLIVES/CRACKS`) on first use (internet; optional `HF_TOKEN`) | `hybrid/data/cracks/build_csv.py` |
| **Synthetic** | synthoseis-generated multimodal CSV + images (external simulator) | `SYNTH_CSV=/path/to/….csv` (default `data/synthetic/…csv`) |
| **Thebe** | published fault-segmentation benchmark volume | `data/real_data/…` (`hybrid/data/thebe`) |
| **Smeaheia** | SEG-Y + fault sticks (`.zip`) + horizons (`.shp`) | `data/real_data/{segy,raw,horizons}` (`hybrid/data/smeaheia/segy.py`) |

Only **CRACKS** is turnkey; the others you place yourself. The **synthetic** data drives the *language* half —
without it the vision + real-field pipeline still runs, but the copy/CHAIR/BLEU language report cannot.

### 3c. Shortcut — pretrained weights
If someone shares the trained `reader.pt` + narrator (`stage3_answer.pt`) + `SFM-Base-512.pth`, drop them into
`hybrid/checkpoints/` and you can **benchmark / run inference without retraining**. Otherwise train from scratch (§5).

## 4. Smoke test (no heavy GPU)
```bash
export SFM_CKPT=hybrid/checkpoints/SFM-Base-512.pth
python -m hybrid.tests.test_round_robin        # scheduler unit test → ALL PASS
python -m hybrid.tests.test_benchmark          # metric unit tests   → ALL PASS
```

## 5. Reproduce the results
Full pipeline in **`hybrid/REPRODUCE.md`**. TL;DR:
```bash
python -m hybrid.stages.stage1_geology         # frozen geology adapter
python -m hybrid.run_train                      # reader.pt + narrator (needs synthetic data)
scripts/ablation.sh                             # real-field joint (4:3:3 + 8:1:1) + uncapped benchmark + report
scripts/report.sh                               # → runs/report.md (vision paper tables)
scripts/eval.sh                                 # → runs/eval_report.txt (language: copy · CHAIR · BLEU/METEOR/CIDEr)
scripts/inference.sh                            # → runs/inference_<survey>.txt (narration + malform tally)
```
All config is env-overridable in `scripts/config.sh` (loss, head toggles, weights, N_TEST). **Single-seed;**
a full ablation is ~a day on one GPU. `RESUME=1` warm-restarts an interrupted training run from its last checkpoint.
