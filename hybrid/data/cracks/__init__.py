"""CRACKS real-field fault dataset (Netherlands F3, Georgia Tech OLIVES, arXiv:2408.11185).

`build_csv.py` converts the HF dataset (`gOLIVES/CRACKS`) into the UNIFIED CSV (fault-instance regions
+ per-instance mask PNGs, NO attributes); `scenes()` loads it via the shared loader, so stages/eval
consume CRACKS exactly like synthetic + Smeaheia. Mask-only → the reader's attribute losses auto-skip
(present-gated); this is the DATA-VOLUME lever for the still-data-bound mask head.

Same pattern as hybrid/data/smeaheia: a folder with its own build_csv that emits the shared schema.
"""
import os
import re
import random

from hybrid.data.cracks.build_csv import build_cracks_csv, CSV_OUT


def scenes(test_frac=0.25, seed=42):
    """CRACKS scenes (build the CSV if missing, then load + split). Returns (all, train, test).
    CONTIGUOUS split by section index — CRACKS sections are adjacent F3 inlines, so a random split
    leaks the same fault into train+test; holding out the LAST test_frac of section indices keeps a
    fault's neighbours on one side (same lesson as the Smeaheia line-level split)."""
    if not os.path.exists(CSV_OUT):
        build_cracks_csv()
    import hybrid.data.loader as sc
    sc.CSV = str(CSV_OUT)
    sc.MAX_SCENES = int(os.environ.get("REAL_CAP", 10_000))     # cap real sections (REAL_CAP) for laptop-feasible runs
    scenes = sc.build_scenes()
    # CRACKS "dip" is DEGENERATE (annotator-stroke orientation, pinned near-vertical), not a true fault dip.
    # Gate it HERE at the data level (null the dip mmask slot) so the shared dip head can never be trained or
    # scored on it — regardless of which runner loads CRACKS. scene_to_gt attaches dip only when mmask[0]>0, so
    # this makes CRACKS genuinely mask-only (matching this module's docstring). Slot 0 = dip (MEASURE_SLOTS).
    for s in scenes:
        for o in s.get("objs", []):
            if int(o["cls"]) == 1 and len(o.get("mmask", ())) > 0:
                o["mmask"][0] = 0.0

    def idx_of(s):                                             # cracks_0007.png -> 7
        m = re.search(r"cracks_(\d+)", os.path.basename(s["img"]))
        return int(m.group(1)) if m else 0
    idxs = sorted({idx_of(s) for s in scenes})
    cut = idxs[int(len(idxs) * (1 - test_frac))] if idxs else 0
    tr = [s for s in scenes if idx_of(s) < cut]
    te = [s for s in scenes if idx_of(s) >= cut]
    random.Random(seed).shuffle(tr)
    npos = sum(1 for s in scenes if s["objs"])
    print(f"[cracks] scenes {len(scenes)} (fault {npos} / bg {len(scenes) - npos}) · "
          f"train {len(tr)} / test {len(te)} · CONTIGUOUS split @ section {cut}", flush=True)
    return scenes, tr, te
