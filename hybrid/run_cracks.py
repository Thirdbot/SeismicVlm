"""CRACKS end-to-end runner — retrain the reader on synthetic (native-aspect tiling), eval, then
adapter-isolation finetune on CRACKS real fault masks. Thin orchestration over the EXISTING stage
functions (train_reader / finetune_real / evaluate) — no new training logic. Edit the UPPER_CASE
config below; no argparse.

PREREQS
  1. SFM checkpoint at hybrid/checkpoints/SFM-Base-512.pth.
  2. Download CRACKS (HuggingFace gOLIVES/CRACKS):
        huggingface-cli download gOLIVES/CRACKS --repo-type dataset
  3. Build the CRACKS CSV (once):
        python -m hybrid.data.cracks.build_csv

RUN  (DATASET picks the real set — cracks by default; both auto-download + build their CSV)
  SFM_CKPT=hybrid/checkpoints/SFM-Base-512.pth PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python -m hybrid.run_cracks                       # finetune on CRACKS
  DATASET=thebe N_CHUNKS=2 SFM_CKPT=… python -m hybrid.run_cracks   # finetune on Thebe (2 chunks ≈ 3 GB)
"""
import os
os.environ.setdefault("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")

import torch
from tqdm.auto import tqdm

import importlib

from hybrid.data import synthetic
from hybrid.model.reader import RegionReader, scene_to_gt, FAULT
from hybrid.model.geometry import field_dice
from hybrid.stages.stage2_reader import train_reader, reader_accuracy, _build_encoder
from hybrid.stages.finetune_vision import finetune_real
from hybrid.eval.real_transfer import evaluate, fmt

device = torch.device("cuda")

READER_EPOCHS = int(os.environ.get("READER_EPOCHS", 80))     # synthetic reader retrain (native-aspect front-end)
REAL_EPOCHS = int(os.environ.get("REAL_EPOCHS", 60))         # CRACKS adapter + mask-decoder finetune
CLDICE_W = float(os.environ.get("CLDICE_W", 0.0))            # thin-structure (centerline) loss weight (0 = off)
TRAINABLE_BLOCKS = int(os.environ.get("TRAINABLE_BLOCKS", 0))  # unfreeze last-N ViT blocks — REQUIRED when PATCH
                                                             # is resampled (8px patches are off-distribution
                                                             # for a 16px-pretrained encoder; it must re-adapt)
SCENE_CAP = int(os.environ["SCENE_CAP"]) if os.environ.get("SCENE_CAP") else 10_000
DATASET = os.environ.get("DATASET", "cracks")               # real-field set to finetune on: cracks | thebe


def _synth_mask_dice(reader, scenes):
    """Reader mask dice on synthetic held-out (fault objects, GT-forced tf_masks)."""
    ds = []
    for s in tqdm(scenes, desc="synth-dice", unit="sc", leave=False):
        gt = scene_to_gt(s)
        if not gt:
            continue
        ml = reader.tf_masks(reader.encode(s), gt)
        ds += [field_dice(ml[i], o["mask_full"].to(device)) for i, o in enumerate(gt) if o["cls"] == FAULT]
    return (sum(ds) / len(ds), len(ds)) if ds else (None, 0)


def _f(m):
    return f"{m[0]:.2f}(n{m[1]})" if m and m[0] is not None else "n0"


def main():
    # 1) RETRAIN the reader on synthetic — loader tiles the native image → SFM → scatter → DETR + mask
    res = synthetic.split(max_scenes=SCENE_CAP)
    tr, te = res[1], res[2]
    from hybrid.data.loader import PATCH, TILE, DILATE_R
    print(f"[synthetic] train {len(tr)} · test {len(te)} · reader epochs {READER_EPOCHS} · clDice {CLDICE_W}", flush=True)
    print(f"[resolution] PATCH {PATCH} · TILE {TILE or 'encoder-native'} · DILATE_R {DILATE_R} · "
          f"trainable_blocks {TRAINABLE_BLOCKS} · target/token {(3.16 + 2 * DILATE_R) / PATCH:.2f}", flush=True)
    reader = train_reader(tr, epochs=READER_EPOCHS, cldice_w=CLDICE_W,
                          trainable_blocks=TRAINABLE_BLOCKS)               # saves hybrid/checkpoints/reader.pt

    # 2) EVAL synthetic held-out (count · dip · class · mask dice)
    a = reader_accuracy(reader, te)
    md = _synth_mask_dice(reader, te)
    print(f"[synth held-out] count {_f(a['count'])} · dip {_f(a['dip'])}deg · "
          f"class {a['cls'][0]}/{a['cls'][1]} · mask dice {_f(md)}", flush=True)

    # 3) FINETUNE on CRACKS real fault masks (adapter isolation: real adapter + mask decoder train;
    #    synthetic base + LM frozen). Reports before (synthetic zero-shot) → after on the CRACKS held-out.
    realds = importlib.import_module(f"hybrid.data.{DATASET}")   # cracks | thebe — auto-downloads + builds its CSV
    _, ctr, cte = realds.scenes()
    print(f"[{DATASET}] train {len(ctr)} · test {len(cte)}", flush=True)
    print(f"[BEFORE] {fmt(evaluate(reader, cte))}", flush=True)            # synthetic reader zero-shot on CRACKS
    finetune_real(ctr, epochs=REAL_EPOCHS)                                 # saves hybrid/checkpoints/reader_real.pt
    after = RegionReader().to(device)
    after.add_real_adapter()
    after.load_state_dict(torch.load("hybrid/checkpoints/reader_real.pt", map_location=device))
    after.eval()
    after.set_encoder(_build_encoder())
    print(f"[AFTER ] {fmt(evaluate(after, cte))}", flush=True)
    print("CRACKS_DONE", flush=True)


if __name__ == "__main__":
    main()
