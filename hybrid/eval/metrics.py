"""Academic evaluation metrics — canonical definitions shared by the benchmarks.

Detection : precision / recall / F1 + count MAE (objects per scene).
Measurement: per-attribute MAE (dip°, throw ms, area %) by sorted within-class matching.
Segmentation: mask Dice and IoU (thresholded).
Copy / grounding are measured in components.py (they need the LM in the loop).

All functions are pure (lists / tensors in, numbers out) so results are reproducible and citable.
"""
import re

import torch

from hybrid.model.geometry import field_dice   # thresholded Dice (re-exported as the canonical mask metric)


# ---- Narration malformation tally (the real-field degenerate-language check) -------------------------
_MALFORM_TAGS = ["evidence", "region", "think", "answer"]
_CONFAB = re.compile(r"\b(onlap|salt|closure|graben|horst|hydrocarbon)\b", re.I)


def malform(chain):
    """Malformation tally on a generated chain — the real-field OOD-degeneration check. A frozen LM fed OOD
    reader values / a stray derived tier can produce: unclosed tags · class_N key-leak · non-fault confab
    words (on fault-only data) · truncation (no </answer>). Lower = cleaner; the derive-off + train-measure
    fixes drive these toward 0. Baseline before fix: unclosed 2.0 · keyleak 3.0 · confab 2.4."""
    unclosed = sum(max(0, len(re.findall(f"<{t}>", chain)) - len(re.findall(f"</{t}>", chain))) for t in _MALFORM_TAGS)
    return dict(unclosed=unclosed, keyleak=len(re.findall(r"class_\d", chain)),
                confab=len(_CONFAB.findall(chain)),
                truncated=0 if re.search(r"</answer>", chain) else 1, length=len(chain))


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


# ---- Spatial precision (boxes): IoU · GIoU · mAP@0.5 ---------------------------------------------
# Boxes are [x1, y1, x2, y2] in ANY single consistent coordinate (caller normalizes pred & GT alike).

def _inter_union(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter, union


def box_iou(a, b):
    inter, union = _inter_union(a, b)
    return inter / union if union > 0 else 0.0


def giou(a, b):
    """Generalized IoU ∈ [-1, 1] — IoU minus the fraction of the enclosing box that is neither box."""
    inter, union = _inter_union(a, b)
    iou = inter / union if union > 0 else 0.0
    cx1, cy1, cx2, cy2 = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
    area_c = (cx2 - cx1) * (cy2 - cy1)
    return iou - (area_c - union) / area_c if area_c > 0 else iou


def _ap_from_pr(rec, prec):
    """VOC all-points AP: integrate the precision envelope over recall."""
    mrec, mpre = [0.0] + list(rec) + [1.0], [0.0] + list(prec) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1))


def map50(preds, gts, iou_thresh=0.5):
    """mAP@0.5 (VOC style). preds: list of (box, score, cls, image_id); gts: (box, cls, image_id).
    Greedy IoU matching per class. NOTE: our reader emits no per-object confidence → pass score=1.0
    and the PR curve collapses to one operating point (AP ≈ precision at that recall) — report as such.
    Returns (mAP, {cls: AP})."""
    import itertools
    from collections import defaultdict
    classes = {c for _, _, c, _ in preds} | {c for _, c, _ in gts}
    aps = {}
    for cls in classes:
        cp = sorted([(b, s, i) for b, s, c, i in preds if c == cls], key=lambda x: -x[1])
        cg = defaultdict(list)
        for b, c, i in gts:
            if c == cls:
                cg[i].append(b)
        n_gt = sum(len(v) for v in cg.values())
        matched, tp, fp = defaultdict(set), [], []
        for b, _s, i in cp:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(cg.get(i, [])):
                if j in matched[i]:
                    continue
                iou = box_iou(b, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            hit = best_iou >= iou_thresh and best_j >= 0
            if hit:
                matched[i].add(best_j)
            tp.append(int(hit)); fp.append(int(not hit))
        ctp, cfp = list(itertools.accumulate(tp)), list(itertools.accumulate(fp))
        rec = [t / n_gt if n_gt else 0.0 for t in ctp]
        prec = [t / (t + f) if (t + f) else 0.0 for t, f in zip(ctp, cfp)]
        aps[cls] = _ap_from_pr(rec, prec) if n_gt else 0.0
    return (sum(aps.values()) / len(aps) if aps else 0.0), aps


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
    """The measured numbers that ARE backed — the EXACT set the soft-prompt markers inject (count,
    per-object bbox/center, dip/throw/area, derived counts). Delegates to the marker serializer so it
    can never drift from what the LM is actually given. (Previously backed only dip/throw/area, which
    miscounted every stated count/centroid/box coordinate as a hallucination.)"""
    from hybrid.model.captioner import marker_numbers
    return marker_numbers(facts)


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

# NOTE: caption-overlap metrics (BLEU-1..4 / ROUGE-L / CIDEr-D / METEOR) live in
# hybrid/eval/caption_metrics.py — the single, unit-tested, field-standard implementation. An earlier
# unused BLEU-4/METEOR/CIDEr-D block lived here (no callers, no ROUGE-L, exact-match "METEOR" with no
# WordNet); it was removed to avoid two divergent overlap implementations.
