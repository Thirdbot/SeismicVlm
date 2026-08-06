"""COMPLETE per-dataset vision benchmark with ACADEMIC metrics (self-baseline framing).

For each dataset's HELD-OUT split, on one checkpoint, reports (never pooled across datasets):
  MASK (per-instance, oracle tf_masks): IoU · soft Dice · thresholded Dice · pixel precision/recall/F1
  MASK (deployment detect(), misses=0): soft Dice
  DETECTION: precision / recall / F1 (Hungarian-matched) · count MAE
  ATTRIBUTES: class acc · dip MAE (+ CONSTANT-predictor baseline) · throw MAE (+ constant baseline)
All numbers carry n. Constant-predictor = always-median; the reviewer baseline for narrow-distribution
attributes. Imports frozen hybrid.* — changes nothing in main.

  CKPT=hybrid/checkpoints/reader_joint_rr.pt  N_TEST=300  python -m hybrid.eval.benchmark
"""
import os
os.environ.setdefault("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")
import importlib
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

from hybrid.model.reader import RegionReader, scene_to_gt, FAULT
from hybrid.model.geometry import field_dice
from hybrid.eval.metrics import mask_iou
from hybrid.stages.stage2_reader import _build_encoder, match_pred_gt

device = torch.device("cuda")
CKPT = os.environ.get("CKPT", "hybrid/checkpoints/reader_joint_rr.pt")   # current joint (old reader_joint.pt retired)
N_TEST = int(os.environ.get("N_TEST", 300))
DATASETS = os.environ.get("DATASETS", "synthetic,thebe,cracks,smeaheia").split(",")
DET_TAU = float(os.environ.get("DET_TAU", 0.1))    # detection: a pred counts as TP only within this normalized
                                                   # centroid distance of a GT (else count-only F1 → 1.0 for any G boxes)


def held_out(name):
    if name == "synthetic":
        from hybrid.data import synthetic
        return synthetic.split(max_scenes=10**9)[2]
    return importlib.import_module(f"hybrid.data.{name}").scenes()[2]


def px_prf(p, g):                                  # pixel precision/recall/f1 at 0.5
    pb, gb = (p > 0.5), (g > 0.5)
    tp = float((pb & gb).sum()); fp = float((pb & ~gb).sum()); fn = float((~pb & gb).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def tol_f1(pb, g, tau=2):
    """Per-instance TOLERANCE-BAND F1 + coverage-recall (support metric). A pred pixel is a hit within
    tau px of a GT pixel (precision), a GT pixel covered within tau of a pred pixel (recall). Forgives
    sub-pixel offset but still penalizes over-prediction (precision) — separates localization from the
    exact-width demand that strict Dice bundles. NEVER inflates the GT (preserves the thin-line label)."""
    from scipy.ndimage import distance_transform_edt
    pn = (pb > 0.5).cpu().numpy(); gn = (g > 0.5).cpu().numpy()
    if not gn.any() or not pn.any():
        return 0.0, 0.0
    egt = distance_transform_edt(~gn); epr = distance_transform_edt(~pn)
    P = float((egt[pn] <= tau).mean()); R = float((epr[gn] <= tau).mean())
    return (2 * P * R / (P + R) if P + R else 0.0), R


def bench(reader, name):
    pool = list(held_out(name))                    # copy so we never mutate the cached split
    random.Random(0).shuffle(pool)                 # fixed-seed shuffle → an N_TEST cap is a RANDOM sample, not a
    scenes = pool[:N_TEST]                          # contiguous crossline slice (which biased the capped grid)
    iou, sdice, tdice, pP, pR, pF = [], [], [], [], [], []
    tolf, tolr = [], []                            # tolerance-band F1 + coverage-recall (localization support)
    ddice = []                                     # deployment
    dtp = dfp = dfn = 0
    cls_hit = cls_tot = 0
    dip_e, throw_e, ctr_e = [], [], []             # ctr_e = localization: the [x,y] the LM copies
    dip_gt, throw_gt = [], []
    for s in scenes:
        gt = scene_to_gt(s)
        faults = [o for o in gt if o["cls"] == FAULT]
        if not faults:
            continue
        smap = reader.encode(s)
        ml = reader.tf_masks(smap, gt)             # oracle per-instance
        for i, o in enumerate(gt):
            if o["cls"] != FAULT:
                continue
            g = (o["mask_full"].to(device) > 0.5).float()
            p = F.interpolate(ml[i][None, None], size=g.shape, mode="bilinear", align_corners=False)[0, 0].sigmoid()
            inter = float((p * g).sum())           # soft-dice uses soft p (no tested leaf; kept inline)
            sdice.append(2 * inter / (float(p.sum()) + float(g.sum()) + 1e-6))
            pb = (p > 0.5).float()
            iou.append(_iou(pb, g)); tdice.append(_tdice(pb, g))       # the TESTED leaf fns (test_benchmark pins these,
            Pp, Rr, Fp = px_prf(pb, g); pP.append(Pp); pR.append(Rr); pF.append(Fp)   # so bench math can't drift untested)
            tf, tr = tol_f1(pb, g); tolf.append(tf); tolr.append(tr)
        # deployment + detection
        pred, masks = reader.detect(smap, want_masks=True)
        pairs = match_pred_gt(pred, faults) if pred else []
        n_tp = 0
        for pr, go in pairs:
            pc, gc = pr["ctr"], go["ctr"]              # normalized centroid L2
            dist = float(((pc[0] - gc[0]) ** 2 + (pc[1] - gc[1]) ** 2) ** 0.5)
            if dist > DET_TAU:                         # a FAR match is NOT a detection — else DETECT-F1 → 1.0 for
                continue                               # any model firing |GT| boxes anywhere (count-only agreement)
            n_tp += 1
            cls_tot += 1; cls_hit += int(pr["cls"] == go["cls"])
            ctr_e.append(100.0 * dist)                 # localization of TRUE detections, ×100 (% of extent)
            if go.get("dip") is not None:              # constant baseline on the SAME (matched) population as the
                dip_e.append(abs(pr["dip"] - go["dip"])); dip_gt.append(go["dip"])   # model error — apples-to-apples
            if go.get("throw") is not None:
                throw_e.append(abs(pr.get("throw", 0) - go["throw"])); throw_gt.append(go["throw"])
        dtp += n_tp; dfp += len(pred) - n_tp; dfn += len(faults) - n_tp
        # deployment dice: matched pred masks vs gt (misses=0)
        pg = {id(g): p for p, g in zip(*_pair(pred, masks, faults))} if pred else {}
        for o in faults:
            m = pg.get(id(o)); gmf = o["mask_full"].to(device)
            if m is None:
                ddice.append(0.0)
            else:
                mi = F.interpolate(m[None, None], size=gmf.shape, mode="bilinear", align_corners=False)[0, 0]
                ddice.append(field_dice(mi, gmf))
    return dict(name=name, n_inst=len(sdice),
                iou=_m(iou), sdice=_m(sdice), tdice=_m(tdice), ppr=(_m(pP), _m(pR), _m(pF)),
                tolf=_m(tolf), tolr=_m(tolr),
                ddice=_m(ddice), det=_prf(dtp, dfp, dfn),
                cls=(cls_hit, cls_tot), ctr=_m(ctr_e), dip=_m(dip_e), dip_const=_const(dip_gt),
                throw=_m(throw_e), throw_const=_const(throw_gt))


def _iou_ok(l, g): return l.shape == g.shape
def _iou(p, g):
    pb, gb = (p > 0.5), (g > 0.5); inter = float((pb & gb).sum()); uni = float((pb | gb).sum())
    return inter / uni if uni else 0.0
def _tdice(p, g):
    pb, gb = (p > 0.5), (g > 0.5); inter = float((pb & gb).sum())
    return 2 * inter / (float(pb.sum()) + float(gb.sum())) if (pb.sum() + gb.sum()) else 0.0
def _m(x): return float(np.mean(x)) if x else float("nan")
def _const(vals):
    if not vals: return float("nan")
    a = np.array(vals); return float(np.mean(np.abs(a - np.median(a))))
def _prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return (P, R, 2 * P * R / (P + R) if P + R else 0.0)
def _pair(pred, masks, gt):
    if not pred: return [], []
    pairs = match_pred_gt(pred, gt)
    idx = {id(p): i for i, p in enumerate(pred)}
    pm, go = [], []
    for pr, g in pairs:
        pm.append(masks[idx[id(pr)]]); go.append(g)
    return pm, go


def main():
    r = RegionReader().to(device)
    sd = torch.load(CKPT, map_location=device)
    if any(k.startswith("real_adapter") for k in sd):
        r.add_real_adapter()
    r.load_state_dict(sd); r.eval(); r.set_encoder(_build_encoder())
    print(f"[benchmark] {CKPT} · per-dataset (never pooled) · uncapped when N_TEST≥split else random N={N_TEST}\n", flush=True)
    failed = []
    for name in DATASETS:
        try:
            d = bench(r, name)
        except Exception as e:
            print(f"  {name}: FAILED {e}", flush=True); failed.append(name); continue
        P, R, Fp = d["ppr"]; dP, dR, dF = d["det"]
        print(f"[{d['name']}] n_inst={d['n_inst']}", flush=True)
        print(f"  MASK oracle : IoU {d['iou']:.3f} · Dice(soft) {d['sdice']:.3f} · Dice(0.5) {d['tdice']:.3f} "
              f"· pixP {P:.3f}/R {R:.3f}/F1 {Fp:.3f}", flush=True)
        print(f"  MASK tol    : tol-F1@2px {d['tolf']:.3f} · coverage-R {d['tolr']:.3f} "
              f"(localization; forgives offset, keeps precision)", flush=True)
        print(f"  MASK deploy : Dice {d['ddice']:.3f}", flush=True)
        print(f"  DETECT      : P {dP:.3f}/R {dR:.3f}/F1 {dF:.3f}", flush=True)
        cls = f"{d['cls'][0]}/{d['cls'][1]}" if d['cls'][1] else "n/a"
        print(f"  ATTR        : class {cls} · location(centroid) MAE {d['ctr']:.2f}%extent · "
              f"dip MAE {d['dip']:.2f} (const {d['dip_const']:.2f}) "
              f"· throw MAE {d['throw']:.2f} (const {d['throw_const']:.2f})\n", flush=True)
        # machine-readable line for scripts/report.sh (never pooled — one per dataset+checkpoint)
        print("[METRICS] " + json.dumps({
            "ckpt": os.path.basename(CKPT), "dataset": d["name"], "n": d["n_inst"],
            "dice_oracle": d["tdice"], "dice_deploy": d["ddice"], "iou": d["iou"],
            "pixP": P, "pixR": R, "pixF1": Fp, "tolf1": d["tolf"],
            "detP": dP, "detR": dR, "detF1": dF, "cls_hit": d["cls"][0], "cls_tot": d["cls"][1],
            "ctr_mae": d["ctr"], "dip": d["dip"], "dip_const": d["dip_const"],
            "throw": d["throw"], "throw_const": d["throw_const"],
            "dip_claimable": d["name"] == "smeaheia"}), flush=True)
    if failed:                                          # a dropped dataset must NOT masquerade as a complete report
        print(f"BENCHMARK_INCOMPLETE — FAILED: {','.join(failed)}", flush=True)
        raise SystemExit(1)
    print("BENCHMARK_DONE", flush=True)


if __name__ == "__main__":
    main()
