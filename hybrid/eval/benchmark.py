"""COMPLETE per-dataset vision benchmark with ACADEMIC metrics (self-baseline framing).

For each dataset's HELD-OUT split, on one checkpoint, reports (never pooled across datasets):
  MASK (per-instance, oracle tf_masks): IoU · soft Dice · thresholded Dice · pixel precision/recall/F1
  MASK (deployment detect(), misses=0): soft Dice
  DETECTION: precision / recall / F1 (Hungarian-matched) · count MAE
  ATTRIBUTES: class acc · dip MAE (+ CONSTANT-predictor baseline) · throw MAE (+ constant baseline)
All numbers carry n. Constant-predictor = always-median; the reviewer baseline for narrow-distribution
attributes. Imports frozen hybrid.* — changes nothing in main.

  CKPT=hybrid/checkpoints/reader.pt  N_TEST=300  python -m hybrid.eval.benchmark
  # point CKPT at any reader: reader.pt (synthetic base), reader_real_<survey>.pt, or a joint reader.
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
from hybrid.stages.stage2_reader import _build_encoder, match_pred_gt

device = torch.device("cuda")
CKPT = os.environ.get("CKPT", "hybrid/checkpoints/reader.pt")   # synthetic base; override for real/joint readers
N_TEST = int(os.environ.get("N_TEST", 300))
DATASETS = os.environ.get("DATASETS", "synthetic,thebe,cracks,smeaheia").split(",")
DETECT_ONLY = os.environ.get("DETECT_ONLY") == "1"  # skip the threshold-INDEPENDENT oracle mask metrics (tol_f1's
                                                    # per-instance CPU distance-transforms) — for the DET_THRESH sweep
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


_DISK = {}


def _disk(tau, device):
    """Cached binary Euclidean-disk kernel of radius tau (dy²+dx² ≤ tau²) — the exact 'within tau px' SE."""
    key = (round(float(tau), 4), str(device))
    k = _DISK.get(key)
    if k is None:
        r = int(tau) + (1 if int(tau) < tau else 0)               # ceil(tau); box that contains the disk
        ys, xs = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij")
        k = ((ys.float() ** 2 + xs.float() ** 2) <= tau * tau).float().to(device)[None, None]
        _DISK[key] = k
    return k


def tol_f1(pb, g, tau=2):
    """Per-instance TOLERANCE-BAND F1 + coverage-recall (support metric). A pred pixel is a hit within
    tau px of a GT pixel (precision), a GT pixel covered within tau of a pred pixel (recall). Forgives
    sub-pixel offset but still penalizes over-prediction (precision) — separates localization from the
    exact-width demand that strict Dice bundles. NEVER inflates the GT (preserves the thin-line label).

    GPU-native: 'within tau px of a pixel' == dilation by a Euclidean disk of radius tau, i.e. one conv
    (dilated = conv2d(mask, disk) > 0). Numerically identical to the old scipy distance_transform_edt<=tau
    (see test), but stays on-device and vectorized — no per-instance .cpu()/EDT (the native-res bottleneck)."""
    gb = (g > 0.5).float(); pf = (pb > 0.5).float()
    gs, ps = float(gb.sum()), float(pf.sum())
    if gs == 0 or ps == 0:
        return 0.0, 0.0
    k = _disk(tau, pb.device); r = k.shape[-1] // 2
    g_dil = F.conv2d(gb[None, None], k, padding=r)[0, 0] > 0     # pixels within tau of a GT pixel
    p_dil = F.conv2d(pf[None, None], k, padding=r)[0, 0] > 0     # pixels within tau of a pred pixel
    P = float((pf.bool() & g_dil).sum()) / ps                    # pred pixels within tau of GT
    R = float((gb.bool() & p_dil).sum()) / gs                    # GT pixels within tau of pred
    return (2 * P * R / (P + R) if P + R else 0.0), R


@torch.no_grad()                                   # eval only: no autograd graph. Without this, encode/tf_masks/detect
def bench(reader, name):                           # retain graphs across the whole held-out pool → RAM creep → freeze
    pool = list(held_out(name))                    # on the uncapped (12628-scene) Thebe pass. Changes NO number.
    random.Random(0).shuffle(pool)                 # fixed-seed shuffle → an N_TEST cap is a RANDOM sample, not a
    scenes = pool[:N_TEST]                          # contiguous crossline slice (which biased the capped grid)
    iou, sdice, tdice, pP, pR, pF = [], [], [], [], [], []
    tolf, tolr = [], []                            # tolerance-band F1 + coverage-recall (localization support)
    ddice = []                                     # deployment
    pool_i = pool_u = 0.0                           # pooled IoU (paper metric): section-union I/U accumulated, divided ONCE
    dtp = dfp = dfn = 0
    cls_hit = cls_tot = 0
    dip_e, throw_e, ctr_e = [], [], []             # ctr_e = localization: the [x,y] the LM copies
    dip_gt, throw_gt = [], []
    ap_dets, ap_gts = [], []                        # threshold-free detection AP: (ctr, conf, img) vs (ctr, img)
    for s in scenes:
        gt = scene_to_gt(s)
        faults = [o for o in gt if o["cls"] == FAULT]
        if not faults:
            continue
        smap = reader.encode(s)
        # ORACLE per-instance mask metrics (IoU/Dice/pixPRF/tol-F1) — GT-matched, so THRESHOLD-INDEPENDENT.
        # tol_f1 runs 2 scipy distance-transforms per instance on CPU (the bottleneck at native res). In a
        # DET_THRESH sweep these don't change, so DETECT_ONLY=1 skips them — sweep only needs detF1 (below).
        if not DETECT_ONLY:
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
        # POOLED IoU (paper metric) — union of DEPLOYED predicted fault masks vs union of GT faults over
        # THIS section; intersection & union accumulate across the whole held-out and are divided ONCE at
        # the end (NOT a per-instance average). This is the honest deployed-segmentation number.
        H, W = faults[0]["mask_full"].shape[-2:]
        gu = torch.zeros(H, W, dtype=torch.bool, device=device)
        for o in faults:
            gu |= (o["mask_full"].to(device) > 0.5)
        pu = torch.zeros(H, W, dtype=torch.bool, device=device)
        for pr, m in zip(pred, masks):
            if pr["cls"] != FAULT:
                continue                               # only fault-class predictions enter the fault union
            mi = F.interpolate(m[None, None].float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            pu |= (mi > 0.5)
        pool_i += float((pu & gu).sum()); pool_u += float((pu | gu).sum())
        pairs = match_pred_gt(pred, faults) if pred else []
        n_tp = 0
        for pr, go in pairs:
            pc, gc = pr["ctr"], go["ctr"]              # normalized centroid L2
            dist = float(((pc[0] - gc[0]) ** 2 + (pc[1] - gc[1]) ** 2) ** 0.5)
            if dist > DET_TAU:                         # a FAR match is NOT a detection — else DETECT-F1 → 1.0 for
                continue                               # any model firing |GT| boxes anywhere (count-only agreement)
            cls_tot += 1; cls_hit += int(pr["cls"] == go["cls"])   # class accuracy over near-matches (any class)
            n_tp += 1
            ctr_e.append(100.0 * dist)                 # localization of TRUE detections, ×100 (% of extent)
            if go.get("dip") is not None:              # constant baseline on the SAME (matched) population as the
                dip_e.append(abs(pr["dip"] - go["dip"])); dip_gt.append(go["dip"])   # model error — apples-to-apples
            if go.get("throw") is not None:
                throw_e.append(abs(pr.get("throw", 0) - go["throw"])); throw_gt.append(go["throw"])
        dtp += n_tp; dfp += len(pred) - n_tp; dfn += len(faults) - n_tp
        for pr in reader.detect(smap, thresh=0.05):        # AP: low-threshold candidates + objectness, threshold-free
            if pr["cls"] == FAULT:
                ap_dets.append((pr["ctr"].detach().cpu().numpy(), pr.get("conf", 1.0), s["img"]))
        for o in faults:
            ap_gts.append((o["ctr"].detach().cpu().numpy(), s["img"]))
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
                iou=_m(iou), pooled_iou=(pool_i / pool_u if pool_u else float("nan")),
                pr50=_pr(iou, 0.5), pr70=_pr(iou, 0.7), pr90=_pr(iou, 0.9),   # RES: Pr@t = frac instances IoU>t
                sdice=_m(sdice), tdice=_m(tdice), ppr=(_m(pP), _m(pR), _m(pF)),
                tolf=_m(tolf), tolr=_m(tolr),
                ddice=_m(ddice), det=_prf(dtp, dfp, dfn), det_ap=_det_ap(ap_dets, ap_gts, DET_TAU),
                cls=(cls_hit, cls_tot), ctr=_m(ctr_e), ctr_n=len(ctr_e),
                dip=_m(dip_e), dip_rmse=_rmse(dip_e), dip_n=len(dip_e), dip_const=_const(dip_gt),
                throw=_m(throw_e), throw_rmse=_rmse(throw_e), throw_n=len(throw_e), throw_const=_const(throw_gt))


def _iou(p, g):
    pb, gb = (p > 0.5), (g > 0.5); inter = float((pb & gb).sum()); uni = float((pb | gb).sum())
    return inter / uni if uni else 0.0
def _tdice(p, g):
    pb, gb = (p > 0.5), (g > 0.5); inter = float((pb & gb).sum())
    return 2 * inter / (float(pb.sum()) + float(gb.sum())) if (pb.sum() + gb.sum()) else 0.0
def _m(x): return float(np.mean(x)) if x else float("nan")
def _pr(x, t): return float(np.mean([v > t for v in x])) if x else float("nan")   # Pr@t (referring-seg): frac of instances with IoU>t
def _rmse(x): return float(np.sqrt(np.mean(np.square(x)))) if x else float("nan")
def _skill(mae, const):   # skill vs the constant-median predictor: 1 - MAE/MAE_const (>0 beats it, <0 worse)
    return float(1.0 - mae / const) if (const and const == const and const > 0) else float("nan")
def _det_ap(dets, gts, tau):
    """Threshold-free detection AP: rank candidates by objectness, greedy centroid-match within `tau` (the
    SAME criterion as det F1), 101-point-interpolated precision-recall. Answers 'det F1 is one threshold'.
    dets: [(ctr np[2], conf, img)]; gts: [(ctr np[2], img)]. Returns AP in [0,1]."""
    from collections import defaultdict
    if not gts or not dets:
        return float("nan")
    gt_by = defaultdict(list)
    for c, img in gts:
        gt_by[img].append(c)
    n_gt = len(gts)
    used, tp, fp = defaultdict(set), [], []
    for ctr, _c, img in sorted(dets, key=lambda x: -x[1]):
        best, bd = -1, tau
        for j, gc in enumerate(gt_by[img]):
            if j in used[img]:
                continue
            dist = float(np.linalg.norm(ctr - gc))
            if dist <= bd:
                bd, best = dist, j
        if best >= 0:
            used[img].add(best); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec, prec = tp / max(n_gt, 1), tp / np.maximum(tp + fp, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        m = prec[rec >= t]
        ap += (float(m.max()) if m.size else 0.0) / 101.0
    return float(ap)
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
        print(f"  MASK pooled : IoU {d['pooled_iou']:.3f}  (paper metric — deployed fault-union I/U, accumulated)", flush=True)
        print(f"  MASK oracle : IoU {d['iou']:.3f} · Dice(soft) {d['sdice']:.3f} · Dice(0.5) {d['tdice']:.3f} "
              f"· pixP {P:.3f}/R {R:.3f}/F1 {Fp:.3f}", flush=True)
        print(f"  MASK tol    : tol-F1@2px {d['tolf']:.3f} · coverage-R {d['tolr']:.3f} "
              f"(localization; forgives offset, keeps precision)", flush=True)
        print(f"  MASK (RES)  : cIoU {d['pooled_iou']:.3f} · gIoU {d['iou']:.3f} · "
              f"Pr@0.5/0.7/0.9 {d['pr50']:.2f}/{d['pr70']:.2f}/{d['pr90']:.2f}  "
              f"(referring-seg standard — cIoU=cumulative, gIoU=mean per-instance)", flush=True)
        print(f"  MASK deploy : Dice {d['ddice']:.3f}", flush=True)
        print(f"  DETECT      : P {dP:.3f}/R {dR:.3f}/F1 {dF:.3f} @ thr {os.environ.get('DET_THRESH','0.9')} · "
              f"AP {d['det_ap']:.3f} (threshold-free, centroid-matched τ={DET_TAU})", flush=True)
        cls_acc = d['cls'][0] / d['cls'][1] if d['cls'][1] else float("nan")
        sk_dip, sk_throw = _skill(d['dip'], d['dip_const']), _skill(d['throw'], d['throw_const'])
        print(f"  ATTR class     : accuracy {cls_acc:.3f}  (n={d['cls'][1]})", flush=True)
        print(f"  ATTR centroid  : MAE {d['ctr']:.2f} %extent  (n={d['ctr_n']})", flush=True)
        print(f"  ATTR dip       : MAE {d['dip']:.2f}° · RMSE {d['dip_rmse']:.2f}° · baseline(const) {d['dip_const']:.2f}° "
              f"· skill {sk_dip:+.2f} · n={d['dip_n']}", flush=True)
        print(f"  ATTR throw     : MAE {d['throw']:.2f} ms · RMSE {d['throw_rmse']:.2f} ms · baseline(const) {d['throw_const']:.2f} ms "
              f"· skill {sk_throw:+.2f} · n={d['throw_n']}\n", flush=True)
        # machine-readable line for scripts/report.sh (never pooled — one per dataset+checkpoint)
        print("[METRICS] " + json.dumps({
            "ckpt": os.path.basename(CKPT), "dataset": d["name"], "n": d["n_inst"],
            "dice_oracle": d["tdice"], "dice_deploy": d["ddice"], "iou": d["iou"], "pooled_iou": d["pooled_iou"],
            "ciou": d["pooled_iou"], "giou": d["iou"], "pr50": d["pr50"], "pr70": d["pr70"], "pr90": d["pr90"],
            "pixP": P, "pixR": R, "pixF1": Fp, "tolf1": d["tolf"],
            "detP": dP, "detR": dR, "detF1": dF, "det_ap": d["det_ap"], "cls_hit": d["cls"][0], "cls_tot": d["cls"][1],
            "ctr_mae": d["ctr"], "ctr_n": d["ctr_n"],
            "dip": d["dip"], "dip_rmse": d["dip_rmse"], "dip_n": d["dip_n"], "dip_const": d["dip_const"],
            "dip_skill": _skill(d["dip"], d["dip_const"]),
            "throw": d["throw"], "throw_rmse": d["throw_rmse"], "throw_n": d["throw_n"], "throw_const": d["throw_const"],
            "throw_skill": _skill(d["throw"], d["throw_const"]),
            "dip_claimable": d["name"] == "smeaheia"}), flush=True)
    if failed:                                          # a dropped dataset must NOT masquerade as a complete report
        print(f"BENCHMARK_INCOMPLETE — FAILED: {','.join(failed)}", flush=True)
        raise SystemExit(1)
    print("BENCHMARK_DONE", flush=True)


if __name__ == "__main__":
    main()
