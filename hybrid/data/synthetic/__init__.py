"""Synthetic dataset (synthoseis-generated). The CSV is produced by the external simulator; this
module just points the UNIFIED loader (data.loader) at it. It shares the COMMON vision contract with
every real-field dataset — image · mask · regions — and additionally carries the language columns
(instruction/question/answer/evidence) that only synthetic has (they supervise the LM half and are the
BLEU/CIDEr reference). Real datasets omit those → vision-only. The loader keys on the common subset.
"""
import os
import random

import hybrid.data.loader as loader
from hybrid.model.captioner import objects_of

# ---- CONFIG ----
CSV = os.environ.get("SYNTH_CSV", "data/synthetic/multimodal_multi_image_dataset.csv")   # portable default; SYNTH_CSV env overrides
SEED = 42


def scenes(max_scenes=float("inf")):
    """All synthetic scenes (encoded) via the unified loader."""
    return loader.build_scenes(csv=CSV, max_scenes=max_scenes)


def split(test_frac=0.25, seed=SEED, max_scenes=float("inf")):
    """Deterministic image-level train/test split (any-object selection). Returns (all, train, test)."""
    rng = random.Random(seed)
    scs = [s for s in scenes(max_scenes) if objects_of(s["objs"])]
    idx = list(range(len(scs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    return scs, [scs[i] for i in idx[:cut]], [scs[i] for i in idx[cut:]]
