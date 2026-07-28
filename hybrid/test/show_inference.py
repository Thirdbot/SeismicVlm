"""Qualitative inference viz — LM response + predicted mask overlay, synthetic or real.

For a few held-out scenes: run the end-to-end pipeline (reader measures facts + masks → LM copies into
grounded narration) and render the predicted instance masks over the seismic image. Prints the LM chain
and saves an overlay PNG per scene.

  SOURCE=syn|real  · READER=…/reader.pt|reader_real.pt · CKPT=…narrator.pt · N=3 · OUT=<dir>
Run:  SOURCE=syn python -m hybrid.test.show_inference
      SOURCE=real READER=hybrid/checkpoints/reader_real.pt python -m hybrid.test.show_inference
"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

import hybrid.model.scenes as sc
sc.MAX_SCENES = int(os.environ.get("SCENES", 10_000))

from hybrid.model.narrator import Narrator, objects_of, scene_facts
from hybrid.model.reader import InstanceReader
from hybrid.train.stage_reader_mask import reader_facts
from hybrid.train.stage_fold import fold_chain
from hybrid.checkpoints import load_narrator

device = torch.device("cuda")
SOURCE = os.environ.get("SOURCE", "syn")
N = int(os.environ.get("N", 3))
OUT = os.environ.get("OUT", "/home/third/Desktop/Unsloth/hybrid/inference")
COLORS = [(255, 60, 60), (60, 160, 255), (60, 255, 120), (255, 200, 40), (200, 80, 255)]


def held_out():
    if SOURCE == "real":
        from hybrid.data.real import real_scenes
        _, _, te = real_scenes()
        return te
    rng = random.Random(42)
    scenes = [s for s in sc.build_scenes() if objects_of(s["objs"])]
    idx = list(range(len(scenes))); rng.shuffle(idx)
    return [scenes[i] for i in idx[int(len(scenes) * 0.75):]]


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
    reader = InstanceReader().to(device)
    rpt = os.environ.get("READER", "hybrid/checkpoints/reader.pt")
    sd = torch.load(rpt, map_location=device)
    if any(k.startswith("real_adapter") for k in sd):
        reader.add_real_adapter()
    reader.load_state_dict(sd); reader.eval()
    nar = Narrator(); nar.set_stage("s3")
    load_narrator(nar, os.environ.get("CKPT", "stage_fold_narrator.pt")); nar.eval_mode()

    te = held_out()
    print(f"[infer] SOURCE={SOURCE} · reader={rpt} · held-out {len(te)} · showing {N}", flush=True)
    shown = 0
    for s in tqdm(te, desc="inference", unit="sc", leave=False):
        gf = scene_facts(s)
        if not (gf["faults"] or gf.get("closures")):
            continue
        facts = reader_facts(reader, s)                            # deployment: reader measures the facts
        objs, masks = reader.detect(s["smap"], want_masks=True)    # predicted instances + masks
        chain = fold_chain(nar, facts, reader, s).replace("\n", " ")
        base = os.path.splitext(os.path.basename(s["img"]))[0]
        png = os.path.join(OUT, f"infer_{SOURCE}_{shown}_{base}.png")
        overlay(s["img"], masks, png)
        gt_dips = [round(float(x["dip"]), 1) for x in gf["faults"]]
        print(f"\n=== {SOURCE} #{shown}  img={base} ===", flush=True)
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
