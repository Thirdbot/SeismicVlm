# SETUP — from a fresh clone to a run

The repo ships the **code, scripts, and docs**. The **environment, model weights, and datasets are not in git**
(too large / external — see `.gitignore`). This gets you from `git clone` to running. It uses **uv**.

## 1. Prerequisites
- A **consumer GPU with ~6 GB VRAM** (4-bit LM + frozen encoder keep the footprint small) for **inference**,
  **CUDA 12.8**. Reproducing the tables from scratch (native-res train + full-volume benchmark) needs a
  **large-VRAM GPU (24 GB class)** — see [`hybrid/REPRODUCE.md`](hybrid/REPRODUCE.md).
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
The frozen encoder is the **Seismic Foundation Model** (ViT-B/16 @512) from Sheng et al. — a
**third-party** model we use **frozen and un-modified** (never fine-tuned), so it is **not redistributed
here**. Download the pretrained checkpoint from the authors' Model Zoo and place it locally:

- **Source:** [shenghanlin/SeismicFoundationModel](https://github.com/shenghanlin/SeismicFoundationModel#rocket-model-zoo-data-release) (Model Zoo & Data Release)
- **Paper / cite:** Sheng et al., *Seismic Foundation Model (SFM)* — [arXiv:2309.02791](https://arxiv.org/abs/2309.02791)

```
hybrid/checkpoints/SFM-Base-512.pth        # or set SFM_CKPT=/path/to/it
```

> **Full placement map (tree + per-dataset + sizes):** [`DATASETS.md`](DATASETS.md).

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
(both from HuggingFace) to build the frozen geology adapter.

**If the model download fails on a fresh machine** (`huggingface-hub 1.x` uses the **Xet** transfer backend by
default, which is a common cause of stalled/failed pulls): **first try disabling Xet**, then pre-fetch into the
cache and re-run the stage:
```bash
export HF_HUB_DISABLE_XET=1        # fall back off the Xet backend → classic HTTP/LFS download (the usual fix)
hf auth login                      # hf-hub ≥1.0 CLI — NOTE: the old `huggingface-cli` no longer exists; or: export HF_TOKEN=hf_...
hf download Qwen/Qwen2.5-1.5B-Instruct                                  # the MODEL (the large pull that tends to fail)
hf download GeoGPT-Research-Project/GeoGPT-CoT-QA --repo-type dataset   # the dataset (public — login only if rate-limited)
python -m hybrid.stages.stage1_geology                                 # now runs against the warm cache
# still stalling? HF_HUB_ENABLE_HF_TRANSFER=1 (faster classic transfer) is an alternative.
```
This stage is a reconstruction — if your GeoGPT-CoT-QA columns differ, adjust `format_example` in the stage.
A shared, pre-built geology adapter drops in without re-running it.

### 3d. Pretrained weights — download, place, run (no retraining)
The trained checkpoints are released at **[`thirdExec/seisground-weights`](https://huggingface.co/thirdExec/seisground-weights)**
(reader + narrator + deployed real-field adapter). They run **on top of** the frozen SFM encoder from §3a —
the SFM weight itself is a third-party model and is **not** in that repo (get it from the §3a link).

Download the whole set into the checkpoints folder (it mirrors this repo's `hybrid/checkpoints/` tree):
```bash
hf download thirdExec/seisground-weights --local-dir hybrid/checkpoints
```
Key files (the repo also carries the full stage + A/B-ablation set, so any table reproduces without retraining):

| File | Path | What it is |
|---|---|---|
| `reader.pt`           | `hybrid/checkpoints/reader.pt`                 | synthetic base reader (measures faults + masks) |
| `stage3_narrator.pt`  | `hybrid/checkpoints/stage3_narrator.pt`        | LM narrator (copies the measured facts) |
| `run_all/B_joint.pt`     | `hybrid/checkpoints/run_all/B_joint.pt`     | deployed **joint** adapter — CRACKS + Smeaheia (native-res) |
| `run_all/alone_thebe.pt` | `hybrid/checkpoints/run_all/alone_thebe.pt` | deployed **Thebe** adapter (Thebe deploys alone; native-res) |
| geology adapter       | `hybrid/checkpoints/stage1_30784a0a20/`        | frozen geology LoRA (stage 1) |

Deployment is **per-domain** (see [`hybrid/REPRODUCE.md`](hybrid/REPRODUCE.md) §4): Thebe → `run_all/alone_thebe.pt`;
CRACKS + Smeaheia → `run_all/B_joint.pt`. All deployment readers are native-resolution (`MASK_UPSAMPLE=4`, the
default) — the `run_all/` folder is the reproducible output set. (The `ab_experiment/` folder, if present, is the
earlier native/2 build — do not mix it with the `=4` default.)

Then drop `SFM-Base-512.pth` (§3a) into the same folder and run inference — nothing else needed:
```bash
# synthetic (in-distribution): base reader + narrator → overlays + narrated chains in hybrid/inference/
DATASET=synthetic python -m hybrid.eval.inference

# a real survey: use the deployed per-domain adapter as the reader (its real-adapter weights auto-load)
DATASET=thebe    READER=hybrid/checkpoints/run_all/alone_thebe.pt python -m hybrid.eval.inference
DATASET=smeaheia READER=hybrid/checkpoints/run_all/B_joint.pt     python -m hybrid.eval.inference

# a single image of your own (any seismic section)
IMAGE=path/to/section.png READER=hybrid/checkpoints/run_all/B_joint.pt python -m hybrid.infer
```
The narrator defaults to `stage3_narrator.pt` in `hybrid/checkpoints/`; override with `CKPT=<file>`
(`hybrid.eval.inference`) or `NARRATOR=<file>` (`hybrid.infer`). Otherwise train from scratch (§5).

## 4. Smoke test (no heavy GPU)
```bash
export SFM_CKPT=hybrid/checkpoints/SFM-Base-512.pth
python -m hybrid.tests.test_round_robin        # scheduler unit test → ALL PASS
python -m hybrid.tests.test_benchmark          # metric unit tests   → ALL PASS
```

## 5. Reproduce the results
Full pipeline (native-res base → `run_all.sh` → reports) in **[`hybrid/REPRODUCE.md`](hybrid/REPRODUCE.md)**. TL;DR:
```bash
python -m hybrid.stages.stage1_geology         # frozen geology adapter
python -m hybrid.run_train                      # reader.pt + narrator (needs synthetic data)
N_CHUNKS=18 scripts/run_all.sh                  # real-field pipeline (alone → threshold → cross-eval → ratio → A/B → final)
scripts/report.sh                               # → runs/report.md (vision paper tables)
scripts/eval.sh                                 # → runs/eval_report.txt (language: copy · CHAIR · BLEU/ROUGE-L/CIDEr)
scripts/inference.sh                            # → runs/inference_<survey>.txt (narration + malform tally)
```
All config is env-overridable in `scripts/config.sh` (loss, head toggles, weights, N_TEST). **Single-seed;**
the full run is **~a day on a large-VRAM GPU (24 GB class)** — the `~6 GB` in §1 covers **inference only**, not
the native-res train/benchmark. `RESUME=1` warm-restarts an interrupted training run from its last checkpoint.
