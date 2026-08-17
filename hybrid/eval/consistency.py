"""Consistency-gate GO / NO-GO — does label-free self-consistency predict real correctness?

The self-training idea (adapt the reader to real seismic on UNLABELED data) only works if a prediction's
GEOMETRIC SELF-CONSISTENCY tells you whether it's right, WITHOUT ground truth. This script measures
exactly that, on the surveys where GT exists, so you learn if the loop is real before building it.

Per PREDICTED fault instance it computes two things:
  GATE SCORES (label-free, from the prediction alone):
    · dip-consistency   : |dip_head − line_dip(pred_mask)|     (head vs mask geometry agree?)
    · center-consistency: ‖ctr_head − centroid(pred_mask)‖     (two spatial paths agree?)
    · elongation        : mask 2nd-moment eigen-ratio          (a fault is a LINE, not a blob)
    · connectivity      : #connected components (1 = clean)    (fragmented = unreliable)
    · COMPOSITE         : geometric mean of the four (AND — all must hold)
  TRUE CORRECTNESS (needs GT — this is the validation, not used at train time):
    · mask IoU vs GT · is-TP (centroid within DET_TAU) · dip error vs GT dip (where GT dip exists)

Then it reports whether the gates PREDICT correctness:
  · Spearman(gate, IoU)                — does agreement track segmentation quality?
  · Spearman(dip-gate, −dip_err)       — (Smeaheia) does dip-consistency track dip accuracy?
  · precision@gate / coverage sweep    — of gate-PASSING preds, what fraction are actually correct?
  · VERDICT: is there an operating point with precision ≥ 0.75 AND coverage ≥ 0.3?  → build vs kill.

Run (set ACTIVE_CLASSES=fault to match the deployed fault-scoped reader):
  ACTIVE_CLASSES=fault CKPT=hybrid/checkpoints/ab_experiment/B_joint.pt \
    DATASETS=thebe,smeaheia N_TEST=2000 python -m hybrid.eval.consistency
Reads: Thebe (mask gates, big n) · Smeaheia (dip gate + independent dip GT). Trains nothing, edits nothing.
"""
import os
os.environ.setdefault("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")
import importlib
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label

from hybrid.model.reader import RegionReader, scene_to_gt, FAULT
from hybrid.model.geometry import line_dip
from hybrid.stages.stage2_reader import _build_encoder

device = torch.device("cuda")
CKPT = os.environ.get("CKPT", "hybrid/checkpoints/reader.pt")
DATASETS = os.environ.get("DATASETS", "thebe,smeaheia").split(",")
N_TEST = int(os.environ.get("N_TEST", 2000))                 # scenes/dataset (correlation study — a few k is plenty)
IOU_THR = float(os.environ.get("IOU_THR", 0.3))             # "correct" = mask IoU above this
DET_TAU = float(os.environ.get("DET_TAU", 0.1))            # a pred is a TP within this normalised centroid dist


def held_out(name):
    if name == "synthetic":
        from hybrid.data import synthetic
        return synthetic.split(max_scenes=10 ** 9)[2]
    return importlib.import_module(f"hybrid.data.{name}").scenes()[2]


def spearman(a, b):
    """Rank correlation, NaN-safe, no scipy.stats dependency."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


@torch.no_grad()
def gate_scores(pr, m):
    """Label-free consistency scores from ONE prediction. Returns raw values + a 0..1 composite goodness,
    or None if the mask is empty (nothing to score)."""
    msq = F.interpolate(m.sigmoid()[None, None], size=(128, 128), mode="bilinear", align_corners=False)[0, 0]
    mb = (msq > 0.5).cpu().numpy()
    if mb.sum() < 4:
        return None
    ys, xs = np.where(mb)
    pts = np.stack([xs, ys], 1).astype(float)                 # isotropic 128² → comparable to the iso dip head
    # dip-consistency: head vs line_dip of its own mask
    ld = line_dip(pts)
    dip_gap = abs(float(pr["dip"]) - float(ld)) if ld is not None else 90.0
    # center-consistency: head centroid vs mask centroid (both normalised [row,col])
    mc = np.array([ys.mean() / 128.0, xs.mean() / 128.0])
    ctr = pr["ctr"].detach().cpu().numpy().reshape(-1)[:2]     # [row,col] normalised (out["mu"])
    ctr_gap = float(np.linalg.norm(mc - ctr))
    # elongation: covariance eigen-ratio (line ≫ blob)
    c = pts - pts.mean(0); cov = (c.T @ c) / max(1, len(c))
    ev = np.linalg.eigvalsh(cov)                               # ascending [λ_lo, λ_hi]
    elong = float((ev[1] / ev[0]) ** 0.5) if ev[0] > 1e-9 else 99.0
    # connectivity
    ncomp = int(cc_label(mb)[1])
    # → 0..1 goodness per gate, AND-combined (geometric mean)
    g_dip = 1.0 - min(1.0, dip_gap / 30.0)
    g_ctr = 1.0 - min(1.0, ctr_gap / 0.15)
    g_el = min(1.0, max(0.0, (elong - 1.0) / 3.0))            # elong 1(round)→0 · ≥4(line)→1
    g_cn = 1.0 if ncomp == 1 else 1.0 / ncomp
    comp = float((max(g_dip, 1e-4) * max(g_ctr, 1e-4) * max(g_el, 1e-4) * max(g_cn, 1e-4)) ** 0.25)
    return dict(dip_gap=dip_gap, ctr_gap=ctr_gap, elong=elong, ncomp=ncomp, composite=comp)


@torch.no_grad()
def correctness(pr, m, gt_faults):
    """True correctness of one predicted fault vs GT: nearest-centroid match → IoU, TP, dip error."""
    ctr = pr["ctr"].detach().cpu().numpy().reshape(-1)[:2]
    best, bd = None, 1e9
    for go in gt_faults:
        gc = go["ctr"].detach().cpu().numpy().reshape(-1)[:2]
        d = float(np.linalg.norm(ctr - gc))
        if d < bd:
            bd, best = d, go
    if best is None or bd > DET_TAU:
        return 0.0, False, None                                # FP → IoU 0
    gm = (best["mask_full"].to(device) > 0.5)
    mi = F.interpolate(m.sigmoid()[None, None], size=gm.shape, mode="bilinear", align_corners=False)[0, 0] > 0.5
    inter = float((mi & gm).sum()); union = float((mi | gm).sum())
    iou = inter / (union + 1e-6)
    dip_err = abs(float(pr["dip"]) - float(best["dip"])) if best.get("dip") is not None else None
    return iou, True, dip_err


@torch.no_grad()
def run(reader, name):
    pool = list(held_out(name))
    random.Random(0).shuffle(pool)
    G = {k: [] for k in ("dip_gap", "ctr_gap", "elong", "ncomp", "composite")}
    IOU, TP, DIPE = [], [], []
    for s in pool[:N_TEST]:
        gt = scene_to_gt(s)
        gt_faults = [o for o in gt if o["cls"] == FAULT]
        if not gt_faults:
            continue
        smap = reader.encode(s)
        pred, masks = reader.detect(smap, want_masks=True)
        for pr, m in zip(pred, masks):
            if pr["cls"] != FAULT:
                continue
            gs = gate_scores(pr, m)
            if gs is None:
                continue
            iou, tp, de = correctness(pr, m, gt_faults)
            for k in G:
                G[k].append(gs[k])
            IOU.append(iou); TP.append(tp); DIPE.append(de if de is not None else float("nan"))

    n = len(IOU)
    if n == 0:
        print(f"[gates {name}] no fault predictions.", flush=True)
        return
    IOU = np.array(IOU); comp = np.array(G["composite"]); correct = IOU > IOU_THR
    # gates should track correctness: HIGHER goodness → HIGHER IoU. (dip_gap/ctr_gap are LOWER=better → negate)
    corr = {"dip": spearman([-x for x in G["dip_gap"]], IOU),
            "center": spearman([-x for x in G["ctr_gap"]], IOU),
            "elong": spearman(G["elong"], IOU),
            "connect": spearman([-x for x in G["ncomp"]], IOU),
            "COMPOSITE": spearman(comp, IOU)}
    print(f"\n[gates {name}] n_pred={n} · TP={int(sum(TP))} · FP={n-int(sum(TP))} · correct(IoU>{IOU_THR})={int(correct.sum())}", flush=True)
    print("  Spearman(gate → IoU):  " + " · ".join(f"{k} {v:+.2f}" for k, v in corr.items()), flush=True)
    dipe = np.array(DIPE)
    if np.isfinite(dipe).sum() >= 3:                            # only where independent dip GT exists (Smeaheia)
        sc = spearman([-x for x in G["dip_gap"]], -dipe)        # dip-consistency vs dip ACCURACY (−error)
        print(f"  Spearman(dip-gate → dip accuracy) = {sc:+.2f}   (does dip-consistency predict true dip error?)", flush=True)
    # precision / coverage sweep on the composite
    print(f"  PR sweep (precision = P(correct | pass) · coverage = frac kept · recall = frac of correct kept):", flush=True)
    best = (0.0, 0.0, 0.0)                                      # (precision, coverage, thr) with coverage≥0.3
    verdict = False
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        keep = comp >= thr
        cov = float(keep.mean())
        prec = float(correct[keep].mean()) if keep.any() else float("nan")
        rec = float(correct[keep].sum() / max(1, correct.sum()))
        print(f"     thr {thr:.1f}:  precision {prec:.2f} · coverage {cov:.2f} · recall {rec:.2f}", flush=True)
        if cov >= 0.3 and prec == prec and prec > best[0]:
            best = (prec, cov, thr)
        if cov >= 0.3 and prec >= 0.75:
            verdict = True
    print(f"  best precision @ coverage≥0.3 = {best[0]:.2f} (thr {best[2]:.1f}, coverage {best[1]:.2f})", flush=True)
    print(f"  VERDICT [{name}]: precision≥0.75 AND coverage≥0.3 reachable?  {'YES → loop is viable' if verdict else 'NO → gates do not separate correct from wrong here'}", flush=True)
    print("[METRICS] " + json.dumps({"dataset": name, "n": n, "tp": int(sum(TP)),
          "corr_composite_iou": corr["COMPOSITE"], "best_precision_at_cov30": best[0],
          "viable": verdict}), flush=True)


def main():
    reader = RegionReader().to(device)
    sd = torch.load(CKPT, map_location=device)
    if any(k.startswith("real_adapter") for k in sd):
        reader.add_real_adapter()
    reader.load_state_dict(sd); reader.eval()
    reader.set_encoder(_build_encoder())
    print(f"[consistency] reader={CKPT} · datasets={DATASETS} · N_TEST={N_TEST} · ACTIVE_CLASSES={os.environ.get('ACTIVE_CLASSES','(all)')}", flush=True)
    for name in DATASETS:
        run(reader, name)
    print("CONSISTENCY_DONE", flush=True)


if __name__ == "__main__":
    main()
