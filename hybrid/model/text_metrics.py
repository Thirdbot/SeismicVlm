"""Text metrics on the model's generated chain — MEASURE the output, never manufacture it.

These are the regexes + checks the fold eval and fold-prep use to judge a generated
`<evidence>…<think>…<answer>` chain: extract sections, strip base-model artifacts that make a
FAITHFUL chain fail (formatting tags / bracket placeholders), and test whether the answer is clean,
well-tagged, and grounded in the measured facts. (Salvaged from the retired stage4_star / STaR path.)
"""
import re

# ---- section extractors ----
_EV = re.compile(r"<evidence>(.*?)</evidence>", re.S)
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_ANS = re.compile(r"<answer>(.*?)</answer>", re.S)

# ---- artifact stripping: keep skeleton tags (evidence/think/answer/seg), drop stray tags + [placeholders] ----
_STRAY_TAG = re.compile(r"</?(?!(?:evidence|think|answer|seg)\b)[a-zA-Z][^>]*>", re.I)
_PLACEHOLDER = re.compile(r"\[[^\]\n]*\]")
_TAG = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)")
_OK_TAGS = {"evidence", "think", "answer", "seg"}


def _clean(ch):
    """Strip non-skeleton tags (keep their inner text) and bracket placeholders from a chain."""
    return _PLACEHOLDER.sub("", _STRAY_TAG.sub("", ch))


def degenerate(t):
    """True if the text contains non-ASCII garbage (a common 1.5B degeneracy mode)."""
    return any(ord(ch) > 127 for ch in t)


def stray_tags(t):
    """True if the chain has any tag outside the expected skeleton (evidence/think/answer/seg)."""
    return any(m.lower() not in _OK_TAGS for m in _TAG.findall(t))


def grounded(t, f):
    """True if the text cites ANY real measured value — fault dip/throw OR closure area (multi-object)."""
    vals = [f"{round(float(x['dip']), 1):g}" for x in f.get("faults", [])]
    vals += [f"{round(float(x['throw'])):g}" for x in f.get("faults", []) if x.get("throw") is not None]
    vals += [f"{round(float(c['area_pct'])):g}" for c in f.get("closures", []) if c.get("area_pct") is not None]
    return any(v in t for v in vals)
