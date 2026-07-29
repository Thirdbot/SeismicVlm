"""Qualitative inference viz — LM response + predicted mask overlay, synthetic or real.

For a few held-out scenes: run the end-to-end pipeline (reader measures facts + masks → LM copies into
grounded narration) and render the predicted instance masks over the seismic image. Prints the LM chain
and saves an overlay PNG per scene.

  DATASET=syn|real  · READER=…/reader.pt|reader_real.pt · CKPT=…narrator.pt · N=3 · OUT=<dir>
Run:  DATASET=syn python -m hybrid.eval.inference
      DATASET=real READER=hybrid/checkpoints/reader_real.pt python -m hybrid.eval.inference
"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from hybrid.model.captioner import Captioner, region_metadata
from hybrid.model.reader import RegionReader
from hybrid.stages.stage2_reader import reader_facts
from hybrid.stages.stage3_answer import generate_chain
from hybrid.checkpoints import load_narrator

device = torch.device("cuda")
DATASET = os.environ.get("DATASET", "synthetic")     # synthetic | smeaheia
N = int(os.environ.get("N", 3))
SCENES = int(os.environ.get("SCENES", 10_000))
OUT = os.environ.get("OUT", "/home/third/Desktop/Unsloth/hybrid/inference")
COLORS = [(255, 60, 60), (60, 160, 255), (60, 255, 120), (255, 200, 40), (200, 80, 255)]


def held_out():
    if DATASET == "smeaheia":
        from hybrid.data import smeaheia
        return smeaheia.scenes()[2]
    from hybrid.data import synthetic
    return synthetic.split(max_scenes=SCENES)[2]


def overlay(img_path, masks_hw, out_path):
    """Composite each predicted instance mask (thresholded, one colour each) over the seismic image."""
    base = Image.open(img_path).convert("RGB")
    W, H = base.size
    canvas = np.array(base).astype(np.float32)
    for i, m in enumerate(masks_hw):
        m = F.interpolate(m[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        mask = (m.sigmoid() > 0.5).cpu().numpy()
        col = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        canvas[mask] = 0.45 * canvas[mask] + 0.55 * col            # semi-transparent fill
    Image.fromarray(canvas.clip(0, 255).astype(np.uint8)).save(out_path)
    return out_path


def main():
    reader = RegionReader().to(device)
    rpt = os.environ.get("READER", "hybrid/checkpoints/reader.pt")
    sd = torch.load(rpt, map_location=device)
    if any(k.startswith("real_adapter") for k in sd):
        reader.add_real_adapter()
    reader.load_state_dict(sd); reader.eval()
    nar = Captioner(); nar.set_stage("s3")
    load_narrator(nar, os.environ.get("CKPT", "stage3_answer.pt")); nar.eval_mode()

    te = held_out()
    print(f"[infer] DATASET={DATASET} · reader={rpt} · held-out {len(te)} · showing {N}", flush=True)
    shown = 0
    for s in tqdm(te, desc="inference", unit="sc", leave=False):
        gf = region_metadata(s)
        if not (gf["faults"] or gf.get("closures")):
            continue
        facts = reader_facts(reader, s)                            # deployment: reader measures the facts
        objs, masks = reader.detect(s["smap"], want_masks=True)    # predicted instances + masks
        chain = generate_chain(nar, facts, reader, s).replace("\n", " ")
        base = os.path.splitext(os.path.basename(s["img"]))[0]
        png = os.path.join(OUT, f"infer_{DATASET}_{shown}_{base}.png")
        overlay(s["img"], masks, png)
        gt_dips = [round(float(x["dip"]), 1) for x in gf["faults"]]
        print(f"\n=== {DATASET} #{shown}  img={base} ===", flush=True)
        print(f"[GT]     {len(gf['faults'])} faults dips={gt_dips} · {len(gf.get('closures', []))} closures", flush=True)
        print(f"[reader] {len(objs)} objects, classes={[o['cls'] for o in objs]}, "
              f"dips={[round(o['dip'], 1) for o in objs if o['cls'] == 1]}", flush=True)
        print(f"[LM]     {chain}", flush=True)
        print(f"[overlay] {png}", flush=True)
        shown += 1
        if shown >= N:
            break
    print("INFER_DONE", flush=True)


if __name__ == "__main__":
    main()
