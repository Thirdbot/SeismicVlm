"""Geometry helpers for masks and line-fault dip — no model, no training.

field_dice : thresholded Dice between a predicted mask logit and a GT mask (eval metric).
line_dip   : single-line RANSAC on an instance's pixels → its apparent dip in degrees (the SAME
             angle read the reader learns to reproduce; used to derive GT dip from projected fault
             sticks on real data). (Salvaged from the retired dense-segmenter module.)
"""
import math

import numpy as np
import torch


def field_dice(logit, gt, thresh=0.5):
    """Thresholded Dice between a mask logit and a {0,1} GT mask (both same H×W)."""
    p = (torch.sigmoid(logit) > thresh).float()
    return float(2 * (p * gt).sum() / (p.sum() + gt.sum()).clamp_min(1e-6))


def line_dip(pts, inlier_dist=3.0, iters=300):
    """RANSAC apparent dip (deg) of one line-instance's pixel coordinates. Reads ANGLE only (not count).
    pts: (N,2) array. Returns None if too few points."""
    if len(pts) < 5:
        return None
    rng = np.random.default_rng(0)
    best_in, best_d, best_a = -1, None, 0
    ii, jj = rng.integers(0, len(pts), iters), rng.integers(0, len(pts), iters)
    for a, b in zip(ii.tolist(), jj.tolist()):
        if a == b:
            continue
        d = pts[b] - pts[a]
        nr = math.hypot(float(d[0]), float(d[1]))
        if nr < 1e-6:
            continue
        d = d / nr
        n = np.array([-d[1], d[0]])
        k = int((np.abs((pts - pts[a]) @ n) < inlier_dist).sum())
        if k > best_in:
            best_in, best_d, best_a = k, d, a
    n = np.array([-best_d[1], best_d[0]])
    inl = pts[np.abs((pts - pts[best_a]) @ n) < inlier_dist]
    if len(inl) >= 2:
        c = inl - inl.mean(0)
        _, v = np.linalg.eigh(c.T @ c)
        best_d = v[:, -1]
    ang = abs(math.degrees(math.atan2(float(best_d[1]), float(best_d[0]))))
    return ang if ang <= 90 else 180 - ang


# Backward-compatible alias (old call sites used `_line_dip`).
_line_dip = line_dip
