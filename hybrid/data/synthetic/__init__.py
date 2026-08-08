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


def _ensure_csv():
    """Auto-build the synthetic CSV from HuggingFace (thirdExec/synthetic-seismic-vlm) if the DEFAULT path is
    missing — same on-demand pattern as cracks/thebe. A custom SYNTH_CSV that's missing is a hard error (the
    user pointed at their own file, e.g. the generator's output; don't silently overwrite it with the HF one)."""
    if os.path.exists(CSV):
        return
    from hybrid.data.synthetic.build_csv import CSV_OUT, build_synthetic_csv
    if os.path.abspath(CSV) == os.path.abspath(str(CSV_OUT)):
        build_synthetic_csv()                                   # download + materialize the HF synthetic
    else:
        raise FileNotFoundError(f"SYNTH_CSV={CSV} not found. Provide it, or unset SYNTH_CSV to auto-download "
                                f"the HF synthetic (thirdExec/synthetic-seismic-vlm).")


def scenes(max_scenes=float("inf")):
    """All synthetic scenes (encoded) via the unified loader (auto-downloads the HF synthetic if missing)."""
    _ensure_csv()
    return loader.build_scenes(csv=CSV, max_scenes=max_scenes)


def split(test_frac=0.25, seed=SEED, max_scenes=float("inf")):
    """Deterministic image-level train/test split (any-object selection). Returns (all, train, test)."""
    rng = random.Random(seed)
    scs = [s for s in scenes(max_scenes) if objects_of(s["objs"])]
    idx = list(range(len(scs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    return scs, [scs[i] for i in idx[:cut]], [scs[i] for i in idx[cut:]]
