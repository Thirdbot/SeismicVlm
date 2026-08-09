"""Shared inference visualization — per-instance overlay COLORED BY CLASS: translucent mask, plus an
OPTIONAL bounding box + class label (toggle), with a legend.

Both the prediction and the ground truth render through ONE shared drawer (`_draw_instances`) so the two
panels are visually identical in style — same colors, same box/label treatment — and directly comparable.
`boxes=False` (env BOXES=0 in the runners) gives the clean mask-only figure preferred for paper main figures;
`boxes=True` adds the localization boxes + labels for a detailed/appendix view:
  · overlay_classes(img, objs, masks, out)  — PREDICTION (reader.detect objs carry cls + 0-100 bbox; masks are logits)
  · overlay_gt(scene, out)                   — GROUND TRUTH (scene objs carry cls + 0-1 bbox + mask_path)
Used by hybrid/infer.py (VLM) and hybrid/eval/inference.py (batch held-out)."""
import os

import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from hybrid.model.registry import ID_CLASS

# class id -> RGB. fault=red · closure=blue · salt=green · onlap=yellow; ∅/unknown -> grey.
CLASS_COLOR = {1: (255, 60, 60), 2: (60, 160, 255), 3: (60, 235, 120), 4: (255, 200, 40)}
_UNK = (170, 170, 170)
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(sz):
    try:
        return ImageFont.truetype(_FONT_PATH, sz)
    except Exception:
        return ImageFont.load_default()


def _draw_instances(base, instances, out_path, alpha=0.45, boxes=True):
    """Shared renderer. Each instance is {cls, mask(bool HxW ndarray | None), box(pixel [x1,y1,x2,y2] | None)}:
    composite the translucent class-colored mask, and (when `boxes`) draw the class-colored bounding box +
    class label; then a legend of the classes present (top-left). `boxes=False` = the clean mask-only figure
    (crisper for paper main figures, esp. dense scenes); `boxes=True` adds the localization boxes + labels.
    Prediction and GT both go through here so their panels are style-identical."""
    W, H = base.size
    canvas = np.array(base.convert("RGB")).astype(np.float32)
    present = []
    for it in instances:                                     # masks first (under the boxes)
        c = int(it["cls"]); mk = it.get("mask")
        if mk is not None and mk.any():
            col = np.array(CLASS_COLOR.get(c, _UNK), np.float32)
            canvas[mk] = (1 - alpha) * canvas[mk] + alpha * col
        if c not in present:
            present.append(c)
    im = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    flab, fleg = _font(12), _font(13)
    if boxes:
        for it in instances:                                 # then boxes + labels on top
            box = it.get("box")
            if box is None:
                continue
            c = int(it["cls"]); col = CLASS_COLOR.get(c, _UNK)
            x1, x2 = sorted((box[0], box[2])); y1, y2 = sorted((box[1], box[3]))
            x1, x2 = max(0, min(W - 1, x1)), max(0, min(W - 1, x2))
            y1, y2 = max(0, min(H - 1, y1)), max(0, min(H - 1, y2))
            d.rectangle([x1, y1, x2, y2], outline=col, width=2)
            lbl = ID_CLASS.get(c, f"cls{c}")
            bb = flab.getbbox(lbl); tw, th = bb[2] - bb[0], bb[3] - bb[1]
            ly = y1 - th - 3 if y1 - th - 3 >= 0 else y1 + 1  # label above the box, or just inside if at the top
            d.rectangle([x1, ly, x1 + tw + 4, ly + th + 3], fill=col)
            d.text((x1 + 2, ly + 1), lbl, fill=(0, 0, 0), font=flab)
    y = 4                                                    # legend
    for c in present:
        d.rectangle([4, y, 18, y + 13], fill=CLASS_COLOR.get(c, _UNK))
        d.text((22, y), ID_CLASS.get(c, f"cls{c}"), fill=(255, 255, 255), font=fleg)
        y += 18
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    im.save(out_path)
    return out_path


def overlay_classes(img_path, objs, masks, out_path, alpha=0.45, boxes=True):
    """PREDICTION overlay — class-colored mask (+ bounding box + class label when `boxes`) per detected
    instance. `objs[i]` (from reader.detect) carries cls + a 0-100 bbox; `masks[i]` is the matching mask
    logits (H',W'). `boxes=False` = clean mask-only."""
    base = Image.open(img_path).convert("RGB")
    W, H = base.size
    inst = []
    for o, m in zip(objs, masks):
        mm = F.interpolate(m[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        mk = (mm.sigmoid() > 0.5).cpu().numpy()
        b = o.get("bbox")                                    # 0-100 (detect scale) -> pixels
        box = [b[0] / 100 * W, b[1] / 100 * H, b[2] / 100 * W, b[3] / 100 * H] if b else None
        inst.append({"cls": o["cls"], "mask": mk, "box": box})
    return _draw_instances(base, inst, out_path, alpha, boxes)


def overlay_gt(scene, out_path, alpha=0.45, boxes=True):
    """GROUND-TRUTH overlay — same style as the prediction: each labelled object's binary GT mask (from its
    mask_path), plus its bounding box + class label when `boxes`. `scene["objs"][i]` carries cls + a 0-1
    bbox + mask_path. `boxes=False` = clean mask-only."""
    base = Image.open(scene["img"]).convert("RGB")
    W, H = base.size
    inst = []
    for o in scene.get("objs", []):
        mk = None
        mp = o.get("mask_path")
        if mp and os.path.exists(mp):
            a = np.array(Image.open(mp).convert("L").resize((W, H), Image.NEAREST))
            mk = a > 40
            if not mk.any():
                mk = None
        b = o.get("bbox")                                    # 0-1 (loader scale) -> pixels
        box = [b[0] * W, b[1] * H, b[2] * W, b[3] * H] if b else None
        if mk is None and box is None:
            continue
        inst.append({"cls": o["cls"], "mask": mk, "box": box})
    return _draw_instances(base, inst, out_path, alpha, boxes)


def panel(img_path, gt_path, pred_path, header_lines, qas, out_path, title="", wrap=118):
    """Presentation composite (one PNG for slides): the three panels [Seismic | Ground truth | Prediction]
    on top, then the grounded text underneath — `header_lines` (measured facts + evidence) and each item
    of `qas` ({tag, q, think, answer}). Every number in the text was measured by the vision stack and
    copied across the digit seam, so it is verifiable against the two mask panels."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def wr(label, text):
        segs = textwrap.wrap(str(text), wrap) or [""]
        pad = " " * len(label)
        return [label + segs[0]] + [pad + s for s in segs[1:]]

    body = []
    for lab, txt in header_lines:
        body += wr(f"{lab:<10}", txt)
    body.append("")
    for qa in qas:
        body += wr(f"[{qa['tag']}] Q: ", qa["q"])
        body += wr("  think   ", qa.get("think") or "—")
        body += wr("  answer  ", qa.get("answer") or "—")
        body.append("")
    text = "\n".join(body).rstrip()
    n_lines = text.count("\n") + 1

    img_h, line_h = 3.5, 0.185
    fig_h = img_h + 0.55 + n_lines * line_h
    fig = plt.figure(figsize=(13.0, fig_h))
    gs = fig.add_gridspec(2, 3, height_ratios=[img_h, n_lines * line_h + 0.4], hspace=0.06, wspace=0.04)
    for j, (p, ttl) in enumerate([(img_path, "Seismic"), (gt_path, "Ground truth"), (pred_path, "Prediction")]):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(ttl, fontsize=12, fontweight="bold", color="#c9d1d9")
        ax.axis("off")
    tx = fig.add_subplot(gs[1, :]); tx.axis("off")
    tx.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=9.6,
            color="#e6e9ee", linespacing=1.28, transform=tx.transAxes)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", color="#e6e9ee", y=0.995)
    fig.patch.set_facecolor("#0d1117")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return out_path
