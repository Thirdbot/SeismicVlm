"""Academic evaluation metrics — canonical definitions shared by the benchmarks.

Detection : precision / recall / F1 + count MAE (objects per scene).
Measurement: per-attribute MAE (dip°, throw ms, area %) by sorted within-class matching.
Segmentation: mask Dice and IoU (thresholded).
Copy / grounding are measured in components.py (they need the LM in the loop).

All functions are pure (lists / tensors in, numbers out) so results are reproducible and citable.
"""
import torch

from hybrid.model.geometry import field_dice   # thresholded Dice (re-exported as the canonical mask metric)


def mask_iou(logit, gt, thresh=0.5):
    """Thresholded IoU between a mask logit and a {0,1} GT mask (same H×W)."""
    p = (torch.sigmoid(logit) > thresh).float()
    inter = (p * gt).sum()
    union = p.sum() + gt.sum() - inter
    return float(inter / union.clamp_min(1e-6))


def mask_dice(logit, gt, thresh=0.5):
    """Thresholded Dice (canonical) — alias of geometry.field_dice."""
    return field_dice(logit, gt, thresh)


def mae(errors):
    """Mean absolute error over a list of |pred-gt| values → (mae, n) or (None, 0)."""
    return (sum(errors) / len(errors), len(errors)) if errors else (None, 0)


def count_mae(pred_counts, gt_counts):
    """Mean |#pred - #gt| objects per scene."""
    return mae([abs(p - g) for p, g in zip(pred_counts, gt_counts)])


def detection_prf(tp, fp, fn):
    """Precision / recall / F1 from tp/fp/fn tallies."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn)


def match_counts(pred_objs, gt_objs, key="cls"):
    """Greedy per-scene detection tally by class presence order → (tp, fp, fn). A prediction counts as
    TP if a same-class GT remains unmatched (position-free within class — count/class quality, not IoU)."""
    from collections import Counter
    gc, pc = Counter(o[key] for o in gt_objs), Counter(o[key] for o in pred_objs)
    tp = sum(min(gc[c], pc[c]) for c in gc)
    fp = sum(max(pc[c] - gc[c], 0) for c in pc)
    fn = sum(max(gc[c] - pc[c], 0) for c in gc)
    return tp, fp, fn


def attr_mae(pred_objs, gt_objs, attr, cls):
    """Per-attribute MAE for one class by SORTED within-class matching (order-free). Returns list of
    |errors| to accumulate across scenes."""
    g = sorted(o[attr] for o in gt_objs if o["cls"] == cls and o.get(attr) is not None)
    p = sorted(o[attr] for o in pred_objs if o["cls"] == cls and o.get(attr) is not None)
    return [abs(p[i] - g[i]) for i in range(min(len(g), len(p)))]


# ---- Faithfulness (hallucination) — CHAIR, the standard grounded-captioning metric --------------
# This task is "region-conditioned text generation" (a.k.a. grounded captioning): the reader supplies
# per-object measurements, the LM narrates them. The risk the literature names is optimization
# short-circuiting — the LM copying the injected clue and inventing the prose. CHAIR_I measures exactly
# that: the fraction of NUMBERS the narration states that are NOT backed by a measured fact.
import re as _re

_NUM = _re.compile(r"\d+\.?\d*")


def narrated_numbers(text):
    """Every numeric token the narration states (as rounded strings, matching the injected forms)."""
    return _NUM.findall(text)


def fact_numbers(facts):
    """The measured/derived numbers that ARE backed (the injected values), as strings."""
    vals = [f"{round(float(x['dip']), 1):g}" for x in facts.get("faults", [])]
    vals += [f"{round(float(x['throw'])):g}" for x in facts.get("faults", []) if x.get("throw") is not None]
    vals += [f"{round(float(c['area_pct'])):g}" for c in facts.get("closures", []) if c.get("area_pct") is not None]
    return set(vals)


def chair(narration, facts):
    """CHAIR_I on numeric claims: (# narrated numbers NOT backed by a fact) / (# narrated numbers).
    0 = perfectly faithful (every stated number is a measured fact); higher = more hallucination.
    Our non-differentiable copy seam is designed to drive this toward 0 structurally."""
    stated = narrated_numbers(narration)
    if not stated:
        return 0.0, 0
    backed = fact_numbers(facts)
    hallucinated = sum(1 for n in stated if n not in backed)
    return hallucinated / len(stated), len(stated)
