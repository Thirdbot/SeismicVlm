"""Stage — LM-side <SEG> → MASK, to test whether the <feature>_i soft token helps MASKING.

Today masks come only from the reader's h_i head; <SEG> is a purely textual marker. Here we add a
SegMaskHead that maps the LM hidden state AT the <SEG> token to a mask over the reader's pixel
features. Because the <feature>_i soft tokens sit in the prompt, the <SEG> hidden carries feature
info — so training this head with a Dice reward gives the feature gate a DENSE localization gradient
(the "SEG path" for feature activation). The experiment is the A/B: dice with feature ON vs OFF.

Isolation: the reader (pixel-feature substrate) and the LM (LoRA) are FROZEN; only SegMaskHead +
feat_proj + feat_gate train, so any ON>OFF dice gap is attributable to the feature, not to the LM
re-fitting. Target = the union of the scene's object masks (one <SEG> → one mask).

Run (after the main train):  python -m hybrid.train.stage_seg_mask
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from hybrid.model.narrator import Narrator, scene_facts, MAX_OBJ, INSTRUCTION_ROLE
from hybrid.model.reader import InstanceReader, scene_to_gt
from hybrid.model.geometry import field_dice
from hybrid.train.stage_fold import aligned_feats
from hybrid.checkpoints import load_narrator

device = torch.device("cuda")


class SegMaskHead(nn.Module):
    """LM <SEG> hidden (lm_dim) → query (d) → mask logits over the reader pixel features (d,H',W')."""
    def __init__(self, lm_dim, d=256):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(lm_dim, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, seg_hidden, pixfeat):
        q = self.q(seg_hidden)                              # (d,)
        return torch.einsum("d,dhw->hw", q, pixfeat)        # (H',W') mask logits


def _seg_hidden(nar, facts, feats, use_feature):
    """LM hidden AT the final <SEG> token (grad-enabled). Feature tokens (when use_feature) sit in the
    injected facts, so this hidden carries the feature — the seam through which the mask reward reaches
    feat_proj/feat_gate (the LM stays frozen; gradient still passes through it to the input embeds)."""
    nar.use_feature = use_feature
    nar.set_stage("s3")
    prompt = nar.build_prompt(nar._ft(facts, feats if use_feature else None), INSTRUCTION_ROLE, question="")
    seq = torch.cat([prompt, nar._emb_text("<evidence> </evidence> <SEG>")], 0)
    out = nar.dec(inputs_embeds=seq.unsqueeze(0), output_hidden_states=True)
    return out.hidden_states[-1][0, -1]                     # (lm_dim,)


@torch.no_grad()
def _pixfeat_and_union(reader, scene):
    """Reader pixel features (d,H',W') [FROZEN substrate] + the union GT mask (H,W) of all objects."""
    memory, _coord, (fH, fW) = reader._grid(scene["smap"])
    pix = reader._mask_features(memory, fH, fW)[0]          # (d,H',W')
    masks = [o["mask"] for o in scene["objs"] if "mask" in o]
    if not masks:
        return pix, None
    union = torch.stack([m.to(device) for m in masks]).amax(0).clamp(0, 1)   # (H,W)
    return pix, union


def _mask_loss(ml, union):
    """BCE(pos-weighted) + soft-Dice, pred (H',W') interpolated to the GT (H,W)."""
    Ht, Wt = union.shape
    ml = F.interpolate(ml[None, None], size=(Ht, Wt), mode="bilinear", align_corners=False)[0, 0]
    pw = torch.tensor([40.0], device=device)
    p = ml.sigmoid()
    return (F.binary_cross_entropy_with_logits(ml, union, pos_weight=pw)
            + 1 - (2 * (p * union).sum() + 1) / (p.sum() + union.sum() + 1))


def train_seg_mask(nar, reader, scenes, epochs=15, lr=1e-3, use_feature=True,
                   save="hybrid/checkpoints/seg_mask_head.pt"):
    """Train SegMaskHead + feat_proj + feat_gate (LM + reader FROZEN) on the union-mask Dice reward."""
    head = SegMaskHead(nar.emb.embedding_dim).to(device)
    for p in nar.dec.parameters():
        p.requires_grad_(False)                            # isolate: only feature + head learn
    nar.use_feature = use_feature
    params = list(head.parameters()) + list(nar.feat_proj.parameters()) + [nar.feat_gate]
    opt = torch.optim.AdamW(params, lr=lr)
    data = []
    for s in tqdm(scenes, desc="segmask-prep", unit="sc", leave=False):
        f = scene_facts(s)
        if not (f["faults"] or f.get("closures")):         # need injected object facts to carry a feature
            continue
        pix, union = _pixfeat_and_union(reader, s)
        if union is None:
            continue
        feats = aligned_feats(reader, s, f) if use_feature else None
        data.append((f, feats, pix, union))
    print(f"[segmask] {len(data)} scenes · feature {'ON' if use_feature else 'OFF'} · LM+reader frozen", flush=True)
    if not data:
        print("[segmask] no data — abort", flush=True); return head
    nar.train_mode()
    ebar = tqdm(range(epochs), desc="segmask", unit="ep")
    for ep in ebar:
        tot = 0.0
        for f, feats, pix, union in tqdm(data, desc=f"ep {ep}", unit="sc", leave=False):
            opt.zero_grad()
            sh = _seg_hidden(nar, f, feats, use_feature)
            loss = _mask_loss(head(sh, pix), union)
            loss.backward(); opt.step(); tot += loss.item()
        ebar.set_postfix(loss=f"{tot/len(data):.3f}", gate=f"{float(nar.feat_gate):.4f}")
        tqdm.write(f"[segmask] ep {ep}/{epochs} loss {tot/len(data):.3f} · gate {float(nar.feat_gate):.4f}")
    nar.eval_mode()
    torch.save({"head": head.state_dict(), "feat_proj": nar.feat_proj.state_dict(),
                "feat_gate": nar.feat_gate.detach()}, save)
    print(f"[segmask] saved {save}", flush=True)
    return head


@torch.no_grad()
def eval_seg_dice(nar, reader, head, scenes, use_feature):
    """Mean union-mask dice on held-out, feature ON or OFF — the masking A/B metric."""
    dices = []
    for s in tqdm(scenes, desc="segmask-eval", unit="sc", leave=False):
        f = scene_facts(s)
        if not (f["faults"] or f.get("closures")):
            continue
        pix, union = _pixfeat_and_union(reader, s)
        if union is None:
            continue
        feats = aligned_feats(reader, s, f) if use_feature else None
        sh = _seg_hidden(nar, f, feats, use_feature)
        Ht, Wt = union.shape
        ml = F.interpolate(head(sh, pix)[None, None], size=(Ht, Wt), mode="bilinear", align_corners=False)[0, 0]
        dices.append(field_dice(ml, union))
    return (sum(dices) / len(dices), len(dices)) if dices else (None, 0)


def main():
    import random
    import hybrid.data.loader as sc
    sc.MAX_SCENES = int(os.environ.get("SCENES", 10_000))
    from hybrid.data.loader import build_scenes
    from hybrid.model.narrator import objects_of
    rng = random.Random(42)
    scenes = [s for s in build_scenes() if objects_of(s["objs"])]
    idx = list(range(len(scenes))); rng.shuffle(idx)
    cut = int(len(idx) * 0.75)
    tr = [scenes[i] for i in idx[:cut]]; te = [scenes[i] for i in idx[cut:]]

    reader = InstanceReader().to(device)
    reader.load_state_dict(torch.load("hybrid/checkpoints/reader.pt", map_location=device)); reader.eval()
    nar = Narrator(); nar.set_stage("s3")
    load_narrator(nar, os.environ.get("CKPT", "stage_fold_narrator.pt")); nar.eval_mode()

    head = train_seg_mask(nar, reader, tr, use_feature=True)
    d_on = eval_seg_dice(nar, reader, head, te, use_feature=True)
    d_off = eval_seg_dice(nar, reader, head, te, use_feature=False)
    def fmt(x): return f"{x[0]:.3f}(n{x[1]})" if x[0] is not None else "n0"
    print(f"[SEG-mask A/B] feature ON dice {fmt(d_on)} · feature OFF dice {fmt(d_off)} "
          f"· gate {float(nar.feat_gate):.4f}", flush=True)
    print("SEG_MASK_DONE", flush=True)


if __name__ == "__main__":
    main()
