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
    """Thebe scenes (build the CSV if missing, then load + split). Returns (all, train, test).

    Two sources:
      · DEFAULT (THEBE_SOURCE=patches) → the Kaggle thebe-fault-patches-256 set (image+mask patches, reliable
        download); split HONORS the author's train/val/test folders (leak-safe) — see build_from_patches.py.
      · THEBE_SOURCE=volume → Harvard Dataverse 3-D volumes; CONTIGUOUS split by global crossline index (kept
        for the full-res crosslines, but the access API 202-stages from cold storage — slow/unreliable)."""
    import hybrid.data.loader as sc
    if os.environ.get("THEBE_SOURCE", "patches") != "volume":
        from hybrid.data.thebe.build_from_patches import build as build_patches, CSV_OUT as PCSV
        if not os.path.exists(PCSV):
            build_patches()
        sc.CSV = str(PCSV); sc.MAX_SCENES = int(os.environ.get("REAL_CAP", 100_000))
        scs = sc.build_scenes(csv=str(PCSV))
        def _bn(s): return os.path.basename(s["img"])
        tr = [s for s in scs if "_train_" in _bn(s)]                                  # author split (leak-safe)
        va = [s for s in scs if "_val_" in _bn(s)]
        te = [s for s in scs if "_test_" in _bn(s)]
        if not va:                                                                   # some builds lack val → carve from test
            va, te = te[:len(te) // 2], te[len(te) // 2:]
        held = va if os.environ.get("EVAL_SPLIT") == "val" else te                   # EVAL_SPLIT: val=selection, test=report
        npos = sum(1 for s in scs if s["objs"])
        print(f"[thebe] PATCHES · scenes {len(scs)} (fault {npos}) · train {len(tr)} · val {len(va)} · "
              f"test {len(te)} · eval={os.environ.get('EVAL_SPLIT', 'test')} (author split, leak-safe)", flush=True)
        return scs, tr, held
    if not os.path.exists(CSV_OUT):
        build_thebe_csv()
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
