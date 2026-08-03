"""CRACKS (real F3 fault masks) → unified CSV (synthetic schema). Same shape as Smeaheia's build_csv.

CRACKS (Georgia Tech OLIVES, arXiv:2408.11185) = 400 Netherlands-F3 sections (701×255 px), each with an
EXPERT fault delineation whose colors encode confidence: BLUE = fault (certain), GREEN = fault
(uncertain), ORANGE = no-fault (certain), else background. Per section we emit the shared schema:

  • fault = cool colors (blue|green), split into CONNECTED-COMPONENT instances (one region per fault line);
  • DIP (apparent) MEASURED from each instance's mask geometry with `line_dip` (RANSAC — same as Smeaheia);
    THROW omitted (needs horizons CRACKS lacks) → the reader's throw loss is present-gated and auto-skips;
  • CONFIDENCE (certain/uncertain) rides in values.derive — the reader's derived head is registry-driven,
    so it's ignored (just stored) until you add one registry row.

One row per section; the loader resizes each to the encoder size the normal way (same as all data). No
tiling, no stitching — the standard pipeline. This is the DATA-VOLUME lever for the still-data-bound mask.

DOWNLOAD (user): HF `gOLIVES/CRACKS`. `load_dataset` uses the HF cache — pre-download
(`huggingface-cli download gOLIVES/CRACKS --repo-type dataset`) or just run this (it fetches on demand).

Run:  python -m hybrid.data.cracks.build_csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label as cc_label

from hybrid.model.geometry import line_dip

REPO = "gOLIVES/CRACKS"                        # HF dataset id
REAL_ROOT = Path("data/real_data/cracks")      # lives with the other real-field data
CSV_OUT = REAL_ROOT / "cracks.csv"
IMG_DIR = REAL_ROOT / "images"                 # per-section seismic PNGs
MASK_DIR = REAL_ROOT / "masks"                 # per-instance fault mask PNGs
LABEL = "expert"                               # gold annotation column (swap for a practitioner/novice)
INCLUDE_UNCERTAIN = True                       # fault = blue(certain) + green(uncertain); False = blue only
MIN_AREA = 12                                  # drop connected components smaller than this (annotation specks)


def _confidence_masks(label_img):
    """CRACKS color-coded label → (blue, green) boolean masks. BLUE = fault (certain), GREEN = fault
    (uncertain); ORANGE (no-fault, R dominant) + neutral background are excluded. The +10 margin
    ignores near-grey seismic bleed-through."""
    a = np.asarray(label_img.convert("RGB")).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (b > r + 10), ((g > r + 10) & (g > b))             # blue = certain, green = uncertain


def build_cracks_csv():
    """One CSV row per section: seismic PNG + per-fault-instance mask PNGs (with apparent dip + confidence).
    Sections with no fault pixels become NEGATIVE rows (train the reader to suppress false faults on real)."""
    from datasets import load_dataset, concatenate_datasets
    REAL_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True); MASK_DIR.mkdir(exist_ok=True)
    dd = load_dataset(REPO)                                     # HF cache; downloads on demand
    ds = concatenate_datasets([dd[s] for s in dd.keys()])      # merge splits — re-split by section index in the loader
    rows, npos, nneg, ninst = [], 0, 0, 0
    for i, ex in enumerate(ds):
        if ex.get("image") is None or ex.get(LABEL) is None:   # a few CRACKS rows lack the expert label — skip
            continue
        img = ex["image"].convert("RGB"); W, H = img.size
        blue, green = _confidence_masks(ex[LABEL])
        fault = (blue | green) if INCLUDE_UNCERTAIN else blue
        comps, ncomp = cc_label(fault)                         # connected fault components → instances
        img_png = IMG_DIR / f"cracks_{i:04d}.png"; img.save(img_png)
        regions, mask_paths = [], []
        for c in range(1, ncomp + 1):
            m = comps == c
            if int(m.sum()) < MIN_AREA:
                continue
            ys, xs = np.where(m)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            conf = "certain" if int(blue[m].sum()) >= int(m.sum()) * 0.5 else "uncertain"   # dominant confidence
            meas = {}
            dip = line_dip(np.stack([xs, ys], 1).astype(float))    # apparent dip from the trace (RANSAC); throw omitted
            if dip is not None:
                meas["dip_deg"] = round(float(dip), 2)
            mp = MASK_DIR / f"cracks_{i:04d}_{len(mask_paths)}.png"
            Image.fromarray((m * 255).astype(np.uint8)).save(mp)
            regions.append({"object_type": "fault", "class_id": 1,
                            "bbox": [x1, y1, x2, y2], "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                            "values": {"measure": meas, "derive": {"confidence": conf}},
                            "mask_idx": len(mask_paths)})
            mask_paths.append(str(mp))
        if not regions:                                        # NEGATIVE section (no fault)
            regions = [{"object_type": "background", "bbox": [0, 0, W, H]}]; nneg += 1
        else:
            npos += 1; ninst += len(regions)
        rows.append({"sample_id": f"cracks_{i:04d}", "images": json.dumps([str(img_png)]),
                     "masks": json.dumps(mask_paths), "regions": json.dumps(regions)})
    rows.sort(key=lambda r: 0 if '"object_type": "fault"' in r["regions"] else 1)   # positives first (capped-encode safe)
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"[cracks] wrote {CSV_OUT} · {len(rows)} sections (fault {npos} / background {nneg}) · "
          f"{ninst} fault instances", flush=True)
    return str(CSV_OUT)


if __name__ == "__main__":
    build_cracks_csv()
