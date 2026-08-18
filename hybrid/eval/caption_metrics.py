"""Standard image-captioning OVERLAP metrics — BLEU-1..4, ROUGE-L, CIDEr-D (+ optional METEOR).

Dependency-free re-implementations of the COCO-caption toolkit formulas (Papineni-2002 BLEU with
brevity penalty; Lin-2004 ROUGE-L LCS-F, beta=1.2; Vedantam-2015 CIDEr-D with corpus IDF, n-gram
clipping and the sigma=6 gaussian length penalty, x10). These are reported for FIELD COMPARABILITY
only — see hybrid/eval/components.overlap_table for the honesty caveat (free-generated narration vs a
templated reference makes these a LOWER BOUND, not the faithfulness axis the paper claims).

METEOR needs WordNet synonymy; it is computed only if `nltk` + the wordnet corpus are installed, else
reported as n/a (install: pip install nltk && python -m nltk.downloader wordnet omw-1.4).

Everything here is a pure function of (hypotheses, references) token strings — unit-testable, no model.
"""
import math
import re
from collections import Counter, defaultdict

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")   # words + decimals (12.5 stays one token, ° dropped)


def tokenize(s):
    """Lowercase, drop tags/punctuation, keep words and decimal numbers as single tokens."""
    s = re.sub(r"</?[a-zA-Z][^>]*>", " ", s or "")     # strip skeleton/stray tags
    return _WORD.findall(s.lower())


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)) if len(toks) >= n else Counter()


# ---------------------------------------------------------------- BLEU (corpus, up to 4-gram)
def bleu(hyps, refs_list, max_n=4):
    """Corpus BLEU-1..max_n (cumulative, geometric mean + brevity penalty). refs_list[i] = list of
    reference strings for hyps[i]. Returns {"BLEU-1":.., .., "BLEU-4":..}."""
    clip = [0] * max_n
    total = [0] * max_n
    c_len = r_len = 0
    for hyp, refs in zip(hyps, refs_list):
        h = tokenize(hyp)
        rs = [tokenize(r) for r in refs] or [[]]
        c_len += len(h)
        closest = min(rs, key=lambda r: (abs(len(r) - len(h)), len(r)))  # closest-length ref (BLEU spec)
        r_len += len(closest)
        for n in range(1, max_n + 1):
            hn = _ngrams(h, n)
            if not hn:
                continue
            # clip each hyp n-gram count by the max count in any single reference
            maxref = Counter()
            for r in rs:
                rn = _ngrams(r, n)
                for g, cnt in rn.items():
                    if cnt > maxref[g]:
                        maxref[g] = cnt
            clip[n - 1] += sum(min(cnt, maxref[g]) for g, cnt in hn.items())
            total[n - 1] += sum(hn.values())
    # geometric-mean precisions with brevity penalty
    out, log_sum = {}, 0.0
    bp = 1.0 if c_len > r_len else math.exp(1 - r_len / c_len) if c_len > 0 else 0.0
    for n in range(1, max_n + 1):
        p = clip[n - 1] / total[n - 1] if total[n - 1] else 0.0
        log_sum += math.log(p) if p > 0 else -1e9
        out[f"BLEU-{n}"] = bp * math.exp(log_sum / n)
    return out


# ---------------------------------------------------------------- ROUGE-L (LCS F, beta=1.2)
def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(hyps, refs_list, beta=1.2):
    """Mean sentence-level ROUGE-L F-score (LCS; recall-weighted beta=1.2, COCO default), max over refs."""
    scores = []
    for hyp, refs in zip(hyps, refs_list):
        h = tokenize(hyp)
        best = 0.0
        for r in (refs or [""]):
            rt = tokenize(r)
            l = _lcs(h, rt)
            if l == 0 or not h or not rt:
                continue
            p, rec = l / len(h), l / len(rt)
            best = max(best, (1 + beta ** 2) * p * rec / (rec + beta ** 2 * p))
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------- CIDEr-D (n=4, sigma=6, x10)
def _counts2vec(toks, df, ref_len, n=4):
    vec = [defaultdict(float) for _ in range(n)]
    norm = [0.0] * n
    length = len(toks)
    for k in range(1, n + 1):
        for g, tf in _ngrams(toks, k).items():
            idf = ref_len - math.log(max(1.0, df[k - 1].get(g, 0.0)))
            vec[k - 1][g] = tf * idf
            norm[k - 1] += vec[k - 1][g] ** 2
    norm = [math.sqrt(x) for x in norm]
    return vec, norm, length


def cider_d(hyps, refs_list, n=4, sigma=6.0):
    """CIDEr-D (Vedantam-2015 / COCO): corpus IDF over the reference documents, per-n cosine of clipped
    TF-IDF vectors with the sigma gaussian length penalty, averaged over n and refs, x10."""
    hyp_toks = [tokenize(h) for h in hyps]
    ref_toks = [[tokenize(r) for r in (refs or [""])] for refs in refs_list]
    # document frequency: count each n-gram once per image if present in ANY of its references
    df = [defaultdict(float) for _ in range(n)]
    for refs in ref_toks:
        seen = [set() for _ in range(n)]
        for r in refs:
            for k in range(1, n + 1):
                for g in _ngrams(r, k):
                    seen[k - 1].add(g)
        for k in range(n):
            for g in seen[k]:
                df[k][g] += 1.0
    ref_len = math.log(max(1, len(ref_toks)))
    scores = []
    for h, refs in zip(hyp_toks, ref_toks):
        vh, nh, lh = _counts2vec(h, df, ref_len, n)
        acc = [0.0] * n
        for r in refs:
            vr, nr, lr = _counts2vec(r, df, ref_len, n)
            delta = lh - lr
            pen = math.exp(-(delta ** 2) / (2 * sigma ** 2))
            for k in range(n):
                s = sum(min(vh[k][g], vr[k].get(g, 0.0)) * vr[k].get(g, 0.0) for g in vh[k])
                if nh[k] and nr[k]:
                    s /= (nh[k] * nr[k])
                acc[k] += s * pen
        score = sum(a / max(1, len(refs)) for a in acc) / n * 10.0
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------- METEOR (optional, nltk+wordnet)
def meteor(hyps, refs_list):
    """Mean METEOR via nltk (needs the wordnet corpus). Returns None if nltk/wordnet are unavailable so
    the caller can report it as n/a rather than shipping a non-standard approximation."""
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.corpus import wordnet
        wordnet.synsets("test")                                   # force the corpus to load / fail fast
    except Exception:
        return None
    vals = [meteor_score([tokenize(r) for r in (refs or [""])], tokenize(h)) for h, refs in zip(hyps, refs_list)]
    return sum(vals) / len(vals) if vals else 0.0


def all_overlap(hyps, refs_list):
    """The full standard block as one dict (BLEU-1..4, ROUGE-L, CIDEr-D, METEOR-or-None)."""
    out = bleu(hyps, refs_list)
    out["ROUGE-L"] = rouge_l(hyps, refs_list)
    out["CIDEr-D"] = cider_d(hyps, refs_list)
    out["METEOR"] = meteor(hyps, refs_list)
    return out
