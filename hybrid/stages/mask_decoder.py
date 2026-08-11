"""Promptable SAM mask decoder — the DEPLOYED mask head (supersedes the reader's linear ⟨mask_q,pixfeat⟩).

Frozen SFM + frozen reader (detection) → the detection centroid prompts a frozen SAM ViT-B decoder + LoRA
adapters (`hybrid/model/sag_seg.SAGSegHead`) → per-fault mask. Trained DEPLOY-CONSISTENT (prompt with the
reader's detections, Hungarian-matched to GT) on the reader's EXACT mask loss
(`BCE(pw≤15)+Dice+FocalTversky(0.4,0.6)`, `reader.py:435`), so the SAM-vs-linear comparison is decoder-only.
On the fair matched loss it beats the linear decoder 1.5–3.3× deploy-Dice / 1.6–2.8× tol-F1 on all real
surveys (Thebe/CRACKS/Smeaheia) → it is the deployed mask decoder.

API:
    head = train_mask_decoder(reader, scenes_by_ds, weights, total_steps=..., save="sag_head.pt")
    eval_mask_decoder(reader, head, tests_by_ds)          # per-survey full panel

Promoted from experiments/sag_ab.py (validated 2026-08-11). No argparse — module UPPER_CASE config + env knobs.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from hybrid.model.reader import scene_to_gt, FAULT, NO_OBJ, tversky_loss, _dice_loss
from hybrid.model.sag_seg import SAGSegHead
from hybrid.model.geometry import field_dice
from hybrid.stages.stage2_reader import match_pred_gt
from hybrid.eval.benchmark import tol_f1
from hybrid.data.round_robin import weighted_round_robin

device = torch.device("cuda")
TVERSKY = tuple(float(x) for x in os.environ.get("TVERSKY", "0.4,0.6,1.0").split(","))   # matched to reader.tversky
POS_WEIGHT_MAX = float(os.environ.get("POS_WEIGHT_MAX", 15.0))
DET_THRESH = float(os.environ.get("DET_THRESH", 0.9))
LORA_R = int(os.environ.get("LORA_R", 4))


def mask_loss(logit, g):
    """The reader's EXACT per-instance mask loss (reader.py:435): BCE(pw≤15) + Dice + Focal-Tversky(0.4,0.6),
    raw target — so the SAM head is trained under the same objective as the linear decoder it replaces."""
    p = logit.sigmoid()
    r = g.mean().clamp(1e-4, 0.5); pw = ((1 - r) / r).clamp(1, POS_WEIGHT_MAX)
    return (F.binary_cross_entropy_with_logits(logit, g, pos_weight=pw)
            + _dice_loss(p[None], g[None])
            + tversky_loss(p[None], g[None], *TVERSKY))


def _grid(reader, smap):
    memory, _coord, _iso, (fH, fW) = reader._grid(smap)
    return memory, (fH, fW)


def _up(m, shape):
    return F.interpolate(m[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]


def train_mask_decoder(reader, scenes_by_ds, weights, total_steps=42000, epochs=1, lr=1e-4,
                       save="hybrid/checkpoints/sag_head.pt"):
    """Train the SAGSegHead over the FROZEN reader, deploy-consistent (detection-prompted, matched to GT),
    weighted round-robin across surveys. Returns the trained head; saves to `save`."""
    for p in reader.parameters():
        p.requires_grad_(False)
    reader.eval()
    seq = weighted_round_robin(scenes_by_ds, {n: weights.get(n, 1) for n in scenes_by_ds}, total_steps=total_steps)
    head = SAGSegHead(feat_dim=reader.d, lora_r=LORA_R).to(device)
    opt = torch.optim.AdamW(head.trainable_parameters(), lr=lr)
    print(f"[mask-dec] round-robin {len(seq)}×{epochs}ep · weights {weights} · trainable {head.n_trainable()/1e6:.3f}M "
          f"· loss BCE+Dice+FocalTversky{TVERSKY} (matched to reader) · DILATE_R={os.environ.get('DILATE_R','0')}", flush=True)
    head.train()
    for ep in range(epochs):
        tot = n = 0
        for step, (name, s) in enumerate(seq):
            gt = [o for o in scene_to_gt(s) if o["cls"] == FAULT and o.get("mask_full") is not None]
            if not gt:
                continue
            with torch.no_grad():
                smap = reader.encode(s); memory, fhw = _grid(reader, smap)
                pred = reader.detect(smap)                                   # deploy-consistent: prompt with detections
            pairs = match_pred_gt(pred, gt) if pred else []
            if not pairs:
                continue
            loss = 0.0; k = len(pairs)
            for pr, o in pairs:
                g = (o["mask_full"].to(device) > 0.5).float()
                m = _up(head(memory, fhw, pr["ctr"].to(device)), g.shape)
                loss = loss + mask_loss(m, g)
            opt.zero_grad(); (loss / k).backward(); opt.step()
            tot += float(loss.detach()) / k; n += 1
            if step % 500 == 0:
                print(f"[mask-dec] ep{ep} {step}/{len(seq)} loss {tot/max(n,1):.3f} ({name})", flush=True)
        print(f"[mask-dec] ep{ep} mean-loss {tot/max(n,1):.3f} ({n} matched-scenes)", flush=True)
    if save:
        torch.save(head.state_dict(), save); print(f"[mask-dec] saved {save}", flush=True)
    return head


def load_mask_decoder(reader, ckpt="hybrid/checkpoints/sag_head.pt"):
    head = SAGSegHead(feat_dim=reader.d, lora_r=LORA_R).to(device)
    head.load_state_dict(torch.load(ckpt, map_location=device)); head.eval()
    return head


def _panel(logit, g):
    """Full pixel panel for ONE mask vs GT — IoU · Dice · Precision · Recall · tol-F1 · pixel-BCE."""
    p = logit.sigmoid(); pb = (p > 0.5).float(); gb = (g > 0.5).float()
    inter = float((pb * gb).sum()); ps = float(pb.sum()); gs = float(gb.sum())
    return dict(IoU=inter / (ps + gs - inter + 1e-6), Dice=2 * inter / (ps + gs + 1e-6),
                P=inter / (ps + 1e-6), R=inter / (gs + 1e-6), tolF1=tol_f1(pb, gb)[0],
                BCE=float(F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), gb)))


OVER = ("IoU", "Dice", "P", "R", "tolF1")           # overlap metrics: a MISS = 0
KEYS = OVER + ("BCE",)                               # BCE matched-only (a miss has no mask)


@torch.no_grad()
def eval_mask_decoder(reader, head, tests_by_ds, n_eval=0):
    """Per-survey full panel, linear reader mask vs SAG head, DEPLOY basis (detect-matched, misses=0)."""
    reader.eval(); head.eval()
    print(f"\n[mask-dec eval — linear vs SAG · deploy (misses=0) · DILATE_R={os.environ.get('DILATE_R','0')}]", flush=True)
    out = {}
    for name, te in tests_by_ds.items():
        te = te[:n_eval] if n_eval else te
        C = {k: [] for k in KEYS}; S = {k: [] for k in KEYS}
        for s in te:
            gt = [o for o in scene_to_gt(s) if o["cls"] == FAULT and o.get("mask_full") is not None]
            if not gt:
                continue
            smap = reader.encode(s); memory, fhw = _grid(reader, smap)
            pred, cur_masks = reader.detect(smap, want_masks=True)
            matched = {id(g): pr for pr, g in (match_pred_gt(pred, gt) if pred else [])}
            idx = {id(p): i for i, p in enumerate(pred)}
            for o in gt:
                g = o["mask_full"].to(device); pr = matched.get(id(o))
                if pr is None:
                    for A in (C, S):
                        for kk in OVER:
                            A[kk].append(0.0)
                    continue
                cm = _up(cur_masks[idx[id(pr)]], g.shape)
                nm = _up(head(memory, fhw, pr["ctr"].to(device)), g.shape)
                for A, m in ((C, _panel(cm, g)), (S, _panel(nm, g))):
                    for kk in KEYS:
                        A[kk].append(m[kk])
        def row(A):
            return " · ".join(f"{k} {np.mean(A[k]):.3f}" for k in KEYS)
        print(f"  {name:9s} n={len(C['Dice']):4d} (BCE matched n={len(C['BCE'])})", flush=True)
        print(f"    linear : {row(C)}", flush=True)
        print(f"    SAG    : {row(S)}", flush=True)
        out[name] = {"linear": {k: float(np.mean(C[k])) for k in KEYS}, "sag": {k: float(np.mean(S[k])) for k in KEYS}}
    return out
