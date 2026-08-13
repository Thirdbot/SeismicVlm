"""Train + evaluate the promptable SAM mask decoder — FUTURE WORK (not the deployed head).

The deployed mask is the DETR reader's linear `<mask_q(h), pixfeat>`. This SAM/SAG promptable decoder is
an exploratory alternative kept as future work: under honest pooled-IoU it ties/loses to the linear head
(its Dice wins were over-paint), so it is NOT deployed. Loads the FROZEN joint reader + the real surveys,
trains the SAG head deploy-consistent on the reader's matched mask loss (hybrid/stages/mask_decoder.py),
and reports the full per-survey panel (IoU / Dice / Precision / Recall / tol-F1 / pixel-BCE). No argparse.

  DATASETS=thebe,cracks,smeaheia WEIGHTS=thebe:16,cracks:1,smeaheia:1 TOTAL_STEPS=42000 DILATE_R=2 \
    N_EVAL=0 SAG_SAVE=hybrid/checkpoints/sag_head.pt python -m hybrid.run_mask_decoder
"""
import os
os.environ.setdefault("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")   # reclaim reserved-unallocated (~6 GB consumer GPU)
import importlib

import torch

from hybrid.model.reader import RegionReader
from hybrid.stages.stage2_reader import _build_encoder
from hybrid.stages.mask_decoder import train_mask_decoder, eval_mask_decoder

device = torch.device("cuda")
DATASETS = os.environ.get("DATASETS", "thebe,cracks,smeaheia").split(",")
WEIGHTS = {kv.split(":")[0]: int(kv.split(":")[1])
           for kv in os.environ.get("WEIGHTS", "thebe:16,cracks:1,smeaheia:1").split(",") if kv}
READER = os.environ.get("READER", "hybrid/checkpoints/reader_joint_cube_w433.pt")
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 42000))
EPOCHS = int(os.environ.get("EPOCHS", 1))
N_EVAL = int(os.environ.get("N_EVAL", 0))                 # 0 = uncapped (full held-out test)
SAVE = os.environ.get("SAG_SAVE", "hybrid/checkpoints/sag_head.pt")


def load_reader():
    r = RegionReader().to(device)
    sd = torch.load(READER, map_location=device)
    if any(k.startswith("real_adapter") for k in sd):
        r.add_real_adapter()
    r.load_state_dict(sd); r.eval(); r.set_encoder(_build_encoder())
    for p in r.parameters():
        p.requires_grad_(False)                            # reader FROZEN — only the SAG head trains
    return r


def main():
    reader = load_reader()
    scenes_by_ds, tests = {}, {}
    for name in DATASETS:
        _, tr, te = importlib.import_module(f"hybrid.data.{name}").scenes()
        scenes_by_ds[name] = tr; tests[name] = te
    head = train_mask_decoder(reader, scenes_by_ds, WEIGHTS, total_steps=TOTAL_STEPS, epochs=EPOCHS, save=SAVE)
    eval_mask_decoder(reader, head, tests, n_eval=N_EVAL)
    print("MASK_DECODER_DONE", flush=True)


if __name__ == "__main__":
    main()
