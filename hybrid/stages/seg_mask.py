"""Stage — LM-side <SEG> → PER-INSTANCE MASK (referring segmentation).

The reader produces masks from its DETR query h_i; this path produces them from the LM hidden at each
<SEG> token, over the reader's pixel features. One <SEG> per <region>, one mask per object.

⚠️ WAS UNION / SEMANTIC, NOW PER-INSTANCE (2026-08-02). The original took ONE hidden from a literal
`<evidence> </evidence> <SEG>` (an EMPTY evidence, so nothing referred to anything) and trained it
against `amax(0)` of every object mask — semantic segmentation of "all objects", one blob per scene.
Its dice was therefore NOT comparable to the reader's per-instance `mask_dice` (union dice is far
easier: it only needs the right region, not separated instances — it is the same area-weighted global
dice the architecture audit identified as a broken objective). The old 0.24-vs-0.06 "the referring path
generalises 4x better" comparison was an artefact of that mismatch and is VOID.

THE ORDERING CONTRACT (the whole correctness risk): the injected facts, the <region> blocks, and the GT
masks must agree on object order. `_ordered()` builds all three from ONE pass over `scene["objs"]` in
the canonical injection order (faults → closures → salts → onlaps), applying the same degenerate-mask
filter as `scene_to_gt`, so alignment holds BY CONSTRUCTION rather than by assumption. If it were wrong
every mask would train against the wrong object and the result would look like a capacity failure.

Isolation: reader (pixel substrate) and LM (LoRA) FROZEN; only SegMaskHead trains (plus feat_proj /
feat_gate when use_feature — but the feature is inert-to-harmful, see reference_masked_value_result).

Run (after the main train):  python -m hybrid.stages.seg_mask
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from hybrid.model.captioner import (Captioner, region_metadata, MAX_OBJ, INSTRUCTION_ROLE,
                                    object_derived_facts, derived_facts)
from hybrid.model.reader import RegionReader, scene_to_gt, _dice_loss, FAULT
from hybrid.model.geometry import field_dice
from hybrid.stages.stage3_answer import aligned_feats
from hybrid.checkpoints import load_narrator

device = torch.device("cuda")

# "<SEG>" tokenises to ['<','SEG','>'] — id 78759 is the stable 'SEG' piece in both the spaced and
# unspaced forms, so every <SEG> in a sequence is found by searching for it. The hidden is read at the
# CLOSING '>' (anchor+1), i.e. the last token of the marker.
SEG_ANCHOR = 78759


class SegMaskHead(nn.Module):
    """LM <SEG> hidden (lm_dim) → query (d) → mask logits over the reader pixel features (d,H',W')."""
    def __init__(self, lm_dim, d=256):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(lm_dim, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, seg_hidden, pixfeat):
        q = self.q(seg_hidden)                              # (K,d) or (d,)
        if q.dim() == 1:
            return torch.einsum("d,dhw->hw", q, pixfeat)    # (H',W')
        return torch.einsum("kd,dhw->khw", q, pixfeat)      # (K,H',W') — one mask per <SEG>


def _ordered(scene):
    """(facts, masks) sharing ONE canonical object order — see THE ORDERING CONTRACT above.

    Facts are rebuilt here (rather than taken from `region_metadata`) so they contain EXACTLY the
    objects whose masks survive the degenerate filter; otherwise a dropped mask would silently shift
    every later region onto the wrong object."""
    from hybrid.data.loader import load_mask_hw, dilate, encoder_tiling
    _, P = encoder_tiling()
    fH, fW = scene["grid"]; mh, mw = fH * P, fW * P
    H, W = scene["hw"]
    buckets = {"faults": [], "closures": [], "salts": [], "onlaps": []}
    masks = {k: [] for k in buckets}
    for o in scene["objs"]:
        cls = int(o["cls"])
        mp = o.get("mask_path")
        if not mp:
            continue
        m = dilate(load_mask_hw(Image.open(mp), (mh, mw)))
        if cls == FAULT:                                    # same degenerate filter as scene_to_gt
            fr = float((m > 0.5).float().mean())
            if fr > 0.4 or fr < 5e-4:
                continue
        x1, y1, x2, y2 = o["bbox"]; cx, cy = o["center"]
        bbox = [int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)]
        center = [int(cx * W), int(cy * H)]
        # ⚠️ POPULATION: a region/mask exists for EVERY object that has a usable mask. Gating this on
        # `mmask` (i.e. "has a measured dip/area") — as `region_metadata` does, because it builds FACTS
        # — silently dropped ~79 of 100 held-out faults, since dip labels are sparse, and scored the
        # referring path on only the best-labelled instances. The contract that must hold is
        # REGION i ↔ MASK i; the injected facts are a SEPARATE list and need not be 1:1 with regions.
        if cls == FAULT:
            masks["faults"].append(m)                       # a region for EVERY fault …
            if float(o["mmask"][0]) > 0:                    # … but a FACT only when measured, because
                f = {"dip": float(o["meas"][0]), "bbox": bbox, "center": center}   # region_markers reads
                if float(o["mmask"][1]) > 0:                # f['dip'] unconditionally
                    f["throw"] = float(o["meas"][1])
                buckets["faults"].append(f)
        elif cls in (2, 3, 4):
            key = "closures" if cls == 2 else ("salts" if cls == 3 else "onlaps")
            masks[key].append(m)
            if float(o["mmask"][2]) > 0:
                obj = {"area_pct": float(o["meas"][2]), "bbox": bbox, "center": center}
                if cls == 2:
                    obj["derive"] = object_derived_facts(o.get("derive"))
                buckets[key].append(obj)
    facts = {**buckets, "derived": derived_facts(scene.get("derived"))}
    ordered = masks["faults"] + masks["closures"] + masks["salts"] + masks["onlaps"]
    n_fault = len(masks["faults"])
    return facts, ordered, n_fault


def _seg_hiddens(nar, facts, n_regions, feats, use_feature):
    """LM hiddens at EVERY <SEG> (grad-enabled), one per <region>, in region order → (K, lm_dim).

    Teacher-forced: the evidence skeleton is built from the known object count, so region i is object i
    by construction. (At inference the model's OWN generated evidence supplies the anchors instead.)"""
    nar.use_feature = use_feature
    nar.set_stage("s3")
    prompt = nar.build_prefix(nar._ft(facts, feats if use_feature else None), INSTRUCTION_ROLE, question="")
    ev = "<evidence> " + " ".join("<region> <SEG> </region>" for _ in range(n_regions)) + " </evidence>"
    ids = nar.tok(ev, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    seq = torch.cat([prompt, nar.emb(ids)], 0)
    out = nar.dec(inputs_embeds=seq.unsqueeze(0), output_hidden_states=True)
    h = out.hidden_states[-1][0]                            # (T, lm_dim)
    anchors = (ids == SEG_ANCHOR).nonzero(as_tuple=True)[0] + 1      # hidden at the closing '>'
    return h[prompt.shape[0] + anchors]                     # (K, lm_dim)


@torch.no_grad()
def _pixfeat(reader, scene):
    """Reader pixel features (d,H',W') — the FROZEN substrate the mask query attends over."""
    memory, _coord, _iso, (fH, fW) = reader._grid(reader.encode(scene))
    return reader._mask_features(memory, fH, fW)[0]


def _mask_loss(ml, masks):
    """PER-INSTANCE BCE(adaptive pos-weight) + dice — the SAME `_dice_loss` the reader uses, so the two
    mask paths are optimised and scored by identical functions."""
    g = torch.stack([m.to(device) for m in masks])          # (K,H,W)
    p = F.interpolate(ml[None], size=g.shape[-2:], mode="bilinear", align_corners=False)[0]
    rate = g.mean().clamp(1e-4, 0.5)
    pw = ((1 - rate) / rate).clamp(1.0, float(os.environ.get("POS_WEIGHT_MAX", 50.0)))   # honor the reader's env knob
    return F.binary_cross_entropy_with_logits(p, g, pos_weight=pw) + _dice_loss(p.sigmoid(), g)


def train_seg_mask(nar, reader, scenes, epochs=15, lr=1e-3, use_feature=True,
                   save="hybrid/checkpoints/seg_mask_head.pt"):
    """Train SegMaskHead (LM + reader FROZEN) on the PER-INSTANCE mask loss, one <SEG> per object."""
    head = SegMaskHead(nar.emb.embedding_dim).to(device)
    for p in nar.dec.parameters():
        p.requires_grad_(False)                            # isolate: only head (+feature) learn
    nar.use_feature = use_feature
    params = list(head.parameters())
    if use_feature:                                        # feature is inert-to-harmful; skip its params when OFF
        params += list(nar.feat_proj.parameters()) + [nar.feat_gate]
    opt = torch.optim.AdamW(params, lr=lr)
    data = []
    for s in tqdm(scenes, desc="segmask-prep", unit="sc", leave=False):
        facts, masks, n_fault = _ordered(s)
        if not masks:
            continue
        feats = aligned_feats(reader, s, facts) if use_feature else None
        data.append((s, facts, feats, [m.cpu() for m in masks]))   # masks lazy on CPU (pix recomputed per step)
    print(f"[segmask] {len(data)} scenes · per-instance · feature {'ON' if use_feature else 'OFF'} "
          f"· LM+reader frozen", flush=True)
    if not data:
        print("[segmask] no data — abort", flush=True); return head
    nar.train_mode()
    ebar = tqdm(range(epochs), desc="segmask", unit="ep")
    for ep in ebar:
        tot = 0.0
        for s, facts, feats, masks in tqdm(data, desc=f"ep {ep}", unit="sc", leave=False):
            opt.zero_grad()
            pix = _pixfeat(reader, s)                       # big; recomputed here and freed each step
            sh = _seg_hiddens(nar, facts, len(masks), feats, use_feature)
            loss = _mask_loss(head(sh, pix), masks)
            loss.backward(); opt.step(); tot += loss.item(); del pix
        ebar.set_postfix(loss=f"{tot/len(data):.3f}", gate=f"{float(nar.feat_gate):.4f}")
        tqdm.write(f"[segmask] ep {ep}/{epochs} loss {tot/len(data):.3f} · gate {float(nar.feat_gate):.4f}")
    nar.eval_mode()
    torch.save({"head": head.state_dict(), "feat_proj": nar.feat_proj.state_dict(),
                "feat_gate": nar.feat_gate.detach()}, save)
    print(f"[segmask] saved {save}", flush=True)
    return head


@torch.no_grad()
def eval_seg_dice(nar, reader, head, scenes, use_feature):
    """Mean PER-FAULT mask dice over the SAME population as `stage2_reader.mask_dice(oracle=True)` —
    every GT fault in `scenes`, teacher-forced regions. Directly comparable to the reader's mask head."""
    dices = []
    for s in tqdm(scenes, desc="segmask-eval", unit="sc", leave=False):
        facts, masks, n_fault = _ordered(s)
        if not n_fault:                                     # denominator = GT FAULTS (matches mask_dice)
            continue
        pix = _pixfeat(reader, s)
        feats = aligned_feats(reader, s, facts) if use_feature else None
        sh = _seg_hiddens(nar, facts, len(masks), feats, use_feature)
        ml = head(sh, pix)                                  # (K,H',W') in the same order as `masks`
        for i in range(n_fault):                            # faults are FIRST in the canonical order
            g = masks[i].to(device)
            p = F.interpolate(ml[i][None, None], size=g.shape, mode="bilinear", align_corners=False)[0, 0]
            dices.append(field_dice(p, g))
        del pix
    return (sum(dices) / len(dices), len(dices)) if dices else (None, 0)


def main():
    from hybrid.data import synthetic
    _, tr, te = synthetic.split(test_frac=0.25, max_scenes=int(os.environ.get("SCENES", 10_000)))

    reader = RegionReader().to(device)
    reader.load_state_dict(torch.load("hybrid/checkpoints/reader.pt", map_location=device)); reader.eval()
    from hybrid.stages.stage2_reader import _build_encoder, mask_dice
    reader.set_encoder(_build_encoder())               # encoder in-model (pixels -> grid)
    nar = Captioner(); nar.set_stage("s3")
    load_narrator(nar, os.environ.get("CKPT", "stage3_answer.pt")); nar.eval_mode()

    use_feature = os.environ.get("USE_FEATURE", "0") == "1"   # feature is inert-to-harmful → OFF by default
    head = train_seg_mask(nar, reader, tr, epochs=int(os.environ.get("SEG_EPOCHS", 15)),
                          use_feature=use_feature)
    seg = eval_seg_dice(nar, reader, head, te, use_feature=use_feature)
    rdr = mask_dice(reader, te, oracle=True)                # SAME metric, SAME population — the real A/B
    def fmt(x): return f"{x[0]:.3f}(n{x[1]})" if x and x[0] is not None else "n0"
    print(f"[SEG per-instance] referring {fmt(seg)} · reader head {fmt(rdr)} "
          f"· feature {'ON' if use_feature else 'OFF'} · gate {float(nar.feat_gate):.4f}", flush=True)
    print("SEG_MASK_DONE", flush=True)


if __name__ == "__main__":
    main()
