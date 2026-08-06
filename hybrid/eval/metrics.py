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


# ---- Narration overlap metrics (grounded-captioning standard: BLEU-4 · METEOR · CIDEr-D) ----------
# Self-contained (no nltk / pycocoevalcap). hyp = generated narration string; ref = the dataset's
# answer string (or a list of reference strings). Corpus-level BLEU/CIDEr, sentence METEOR.
import math as _math
from collections import Counter as _Counter


def _ngrams(tokens, n):
    return _Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu4(hyps, refs):
    """Corpus BLEU-4: clipped n-gram precision (n=1..4) × brevity penalty, +1 smoothing on empty orders."""
    pn, pd, hyp_len, ref_len = [0] * 4, [0] * 4, 0, 0
    for h, r in zip(hyps, refs):
        ht = h.split(); rts = [(r if isinstance(r, list) else [r])][0]; rts = [x.split() for x in rts]
        hyp_len += len(ht)
        ref_len += min((len(rt) for rt in rts), key=lambda l: (abs(l - len(ht)), l))
        for n in range(1, 5):
            hc = _ngrams(ht, n)
            mx = _Counter()
            for rt in rts:
                for g, c in _ngrams(rt, n).items():
                    mx[g] = max(mx[g], c)
            pn[n - 1] += sum(min(c, mx[g]) for g, c in hc.items())
            pd[n - 1] += max(sum(hc.values()), 1)
    precs = []
    for i in range(4):
        num, den = (pn[i], pd[i]) if pn[i] else (1, pd[i] + 1)     # smoothing
        precs.append(num / den)
    bp = 1.0 if hyp_len > ref_len else _math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * _math.exp(sum(0.25 * _math.log(p) for p in precs))


def meteor(hyp, ref, alpha=0.9, gamma=0.5, beta=3.0):
    """Sentence METEOR (EXACT-match unigram variant — no WordNet/stemming). F-mean(P,R) recall-weighted,
    minus a fragmentation penalty on the number of matched chunks."""
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    rc = _Counter(r); m = sum(min(c, rc[g]) for g, c in _Counter(h).items())
    if m == 0:
        return 0.0
    P, R = m / len(h), m / len(r)
    fmean = P * R / (alpha * P + (1 - alpha) * R)
    # chunks = maximal contiguous runs of MATCHED h-tokens. Cap chunks at m (a run can't have more
    # chunks than matched tokens) so frag ∈ [0,1] and the penalty stays in [0,1] — fixes negative
    # METEOR when a hypothesis word repeats more often than in the reference.
    rset = set(r); chunks, prev = 0, False
    for w in h:
        here = w in rset
        chunks += int(here and not prev); prev = here
    frag = min(1.0, chunks / m)
    return fmean * (1 - gamma * frag ** beta)


def cider_d(hyps, refs, nmax=4, sigma=6.0):
    """Corpus CIDEr-D: TF-IDF n-gram (1..nmax) cosine between hyp and refs, with a Gaussian length
    penalty. IDF from the reference corpus (document = one reference)."""
    ref_lists = [(r if isinstance(r, list) else [r]) for r in refs]
    # document frequency of each n-gram across all references
    df = _Counter(); ndoc = 0
    for rl in ref_lists:
        for ref in rl:
            ndoc += 1
            seen = set()
            for n in range(1, nmax + 1):
                seen |= set(_ngrams(ref.split(), n).keys())
            for g in seen:
                df[g] += 1
    logN = _math.log(max(ndoc, 1))

    def vec(tokens):
        v, norm = {}, [0.0] * nmax
        for n in range(1, nmax + 1):
            for g, c in _ngrams(tokens, n).items():
                w = c * max(logN - _math.log(df.get(g, 1)), 0.0)
                v[g] = w; norm[n - 1] += w * w
        return v, [_math.sqrt(x) for x in norm]

    total = 0.0
    for h, rl in zip(hyps, ref_lists):
        hv, hn = vec(h.split()); scores = []
        for ref in rl:
            rv, rn = vec(ref.split())
            per = 0.0
            for n in range(1, nmax + 1):
                dot = sum(hv.get(g, 0.0) * rv.get(g, 0.0) for g in rv if len(g) == n)
                denom = (hn[n - 1] * rn[n - 1]) or 1.0
                sim = dot / denom
                delta = len(h.split()) - len(ref.split())
                per += sim * _math.exp(-(delta ** 2) / (2 * sigma ** 2))
            scores.append(per / nmax)
        total += (sum(scores) / len(scores)) if scores else 0.0
    return 10.0 * total / max(len(hyps), 1)          # ×10 per the CIDEr convention
