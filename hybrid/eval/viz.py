"""Shared inference visualization — per-instance mask overlay COLORED BY CLASS, with a legend.

`reader.detect(..., want_masks=True)` returns (objs, masks) appended in one loop, so masks[i] is the mask
for objs[i]; we color it by objs[i]['cls'] (fault/closure/salt/onlap) instead of an arbitrary per-instance
index. Used by hybrid/infer.py (VLM) and hybrid/eval/inference.py (batch held-out)."""
import os

import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw

from hybrid.model.registry import ID_CLASS

# class id -> RGB. fault=red · closure=blue · salt=green · onlap=yellow; ∅/unknown -> grey.
CLASS_COLOR = {1: (255, 60, 60), 2: (60, 160, 255), 3: (60, 235, 120), 4: (255, 200, 40)}
_UNK = (170, 170, 170)


def overlay_classes(img_path, masks, classes, out_path, alpha=0.55):
    """Composite each predicted instance mask over the seismic image, colored by its CLASS, plus a legend
    of the classes present (top-left swatch + word). `classes[i]` is objs[i]['cls'] for mask i."""
    base = Image.open(img_path).convert("RGB")
    W, H = base.size
    canvas = np.array(base).astype(np.float32)
    present = []
    for m, c in zip(masks, classes):
        col = np.array(CLASS_COLOR.get(int(c), _UNK), np.float32)
        m = F.interpolate(m[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        mk = (m.sigmoid() > 0.5).cpu().numpy()
        canvas[mk] = (1 - alpha) * canvas[mk] + alpha * col
        if int(c) not in present:
            present.append(int(c))
    im = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)                                   # legend
    y = 4
    for c in present:
        d.rectangle([4, y, 16, y + 12], fill=CLASS_COLOR.get(c, _UNK))
        d.text((20, y), ID_CLASS.get(c, f"cls{c}"), fill=(255, 255, 255))
        y += 16
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    im.save(out_path)
    return out_path
