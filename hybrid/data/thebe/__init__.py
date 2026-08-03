"""Thebe real-field fault dataset (NW-Australia, An et al. 2021, doi:10.7910/DVN/YBYGBK).

`build_csv.py` auto-downloads from Harvard Dataverse and converts the 3D volumes into the UNIFIED CSV
(panel images + per-instance fault masks + apparent dip, NO throw); `scenes()` loads it via the shared
loader, so stages/eval consume Thebe exactly like synthetic / CRACKS / Smeaheia. Same pattern as
hybrid/data/cracks — a folder with its own build_csv that emits the shared schema.
"""
import os
import re
import random

from hybrid.data.thebe.build_csv import build_thebe_csv, CSV_OUT


def scenes(test_frac=0.25, seed=42):
    """Thebe scenes (build the CSV if missing, then load + split). Returns (all, train, test). CONTIGUOUS
    split by global crossline index — a crossline is tiled into many panels, so a random split would leak
    the same fault across train/test; holding out the LAST test_frac of crosslines keeps them apart
    (same lesson as the Smeaheia line-level / CRACKS contiguous split)."""
    if not os.path.exists(CSV_OUT):
        build_thebe_csv()
    import hybrid.data.loader as sc
    sc.CSV = str(CSV_OUT)
    sc.MAX_SCENES = int(os.environ.get("REAL_CAP", 100_000))    # cap real panels (REAL_CAP) for laptop-feasible runs
    scenes = sc.build_scenes()

    def cx_of(s):                                             # thebe_00042_1024_512.png -> 42 (crossline)
        m = re.search(r"thebe_(\d+)", os.path.basename(s["img"]))
        return int(m.group(1)) if m else 0
    cxs = sorted({cx_of(s) for s in scenes})
    cut = cxs[int(len(cxs) * (1 - test_frac))] if cxs else 0
    tr = [s for s in scenes if cx_of(s) < cut]
    te = [s for s in scenes if cx_of(s) >= cut]
    random.Random(seed).shuffle(tr)
    npos = sum(1 for s in scenes if s["objs"])
    print(f"[thebe] scenes {len(scenes)} (fault {npos} / bg {len(scenes) - npos}) · "
          f"train {len(tr)} / test {len(te)} · CONTIGUOUS split @ crossline {cut}", flush=True)
    return scenes, tr, te
