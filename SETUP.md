# SETUP — from a fresh clone to a run

The repo ships the **code, scripts, and docs**. The **environment, model weights, and datasets are not in git**
(too large / external — see `.gitignore`). This gets you from `git clone` to running. It uses **uv**.

## 1. Prerequisites
- A **consumer GPU with ~6 GB VRAM** (4-bit LM + frozen encoder keep the footprint small), **CUDA 12.8**.
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

### 3b. Datasets — what auto-downloads, what you place, and how much

| dataset | how to get it | capping (default) |
|---|---|---|
| **Synthetic** | **auto-downloads** from HuggingFace [`thirdExec/synthetic-seismic-vlm`](https://huggingface.co/datasets/thirdExec/synthetic-seismic-vlm) → `data/synthetic/…csv`. Override with `SYNTH_CSV=/path` to use your own generator output. | full |
| **CRACKS** | **auto-downloads** from HuggingFace [`gOLIVES/CRACKS`](https://huggingface.co/datasets/gOLIVES/CRACKS) (needs internet; `HF_TOKEN` if rate-limited) | full download · `REAL_CAP` caps panels **built** (default 10 000) |
| **Thebe** | **auto-downloads** from Harvard Dataverse [DOI `10.7910/DVN/YBYGBK`](https://doi.org/10.7910/DVN/YBYGBK), streamed by chunk | ⚠ **`N_CHUNKS=2`** (~200 of 1803 crosslines, ~3 GB). Set **`N_CHUNKS=18`** for the full ~30 GB volume. `REAL_CAP` also caps panels built (100 000). |
| **Smeaheia** | **manual** — download from [co2datashare.org/dataset/smeaheia-dataset](https://co2datashare.org/dataset/smeaheia-dataset) and place under `data/real_data/` per [`data/real_data/README.md`](data/real_data/README.md) | all built |

Only **CRACKS/synthetic/Thebe auto-download**; **Smeaheia is manual**. The **synthetic** data drives the
*language* half — without it the vision + real-field pipeline still runs, but the copy/CHAIR/BLEU language
report cannot.

**Thebe — if you hit `np.load: No data left in file`:** an interrupted chunk download left a truncated cache.
The loader now validates + re-fetches automatically; if it persists, delete `data/real_data/thebe/raw/` and re-run.

**Smeaheia — throw needs HORIZONS.** SEG-Y + fault sticks give you masks + dip (e.g. `215 fault / 215 background`).
**Throw** is the horizon offset across a fault, so **without the horizon files it comes out `throw 0/N`** (the build
now warns). Place the horizons per `data/real_data/README.md` to get throw GT — masks and dip are unaffected either way.

### 3c. Geology adapter (stage 1) — needs a model + dataset download
`python -m hybrid.stages.stage1_geology` fine-tunes [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
on [`GeoGPT-Research-Project/GeoGPT-CoT-QA`](https://huggingface.co/datasets/GeoGPT-Research-Project/GeoGPT-CoT-QA)
(both from HuggingFace) to build the frozen geology adapter. **If the in-process download fails**, authenticate
first and/or pre-fetch:
```bash
huggingface-cli login            # or: export HF_TOKEN=hf_...
hf download GeoGPT-Research-Project/GeoGPT-CoT-QA --repo-type dataset
```
This stage is a reconstruction — if your GeoGPT-CoT-QA columns differ, adjust `format_example` in the stage.
A shared, pre-built geology adapter drops in without re-running it.

### 3d. Shortcut — pretrained weights
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
