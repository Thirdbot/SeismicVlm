"""UNGATED full-coverage real-field → CSV (synthetic schema), so one loader/trainer handles both.

Every 2D line is used (not just the ~63 that cross a mapped fault): each line is tiled edge-to-edge
into fixed-width panels (bounded size = the laptop-RAM lever, NOT a label gate). A panel gets a fault
region wherever a 3D fault stick projects into it (mask + apparent dip + throw), and is a NEGATIVE
(no object) otherwise. Negatives are the point: they teach the reader "no fault here" and fight the
false-fault detection seen on synthetic held-out.

CPU-only (render + stick projection + rasterize + dip/throw) — NO NCS encode here; encoding happens
later in build_scenes when the CSV is loaded. Streams one line at a time (no smap held), so it never
touches the 15 GB RAM wall.

Output: data/real_data/real_field.csv (+ per-panel image/mask PNGs) with the synthetic columns
(images/masks/regions/values{measure,derive}). object_type "fault" for positives, "background" for
negatives. Load via hybrid/model/scenes.build_scenes (negative-aware) → reader train/test.

Run:  python -m hybrid.data.real_csv          # extracts all SEG-Y, then builds the CSV
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from hybrid.data.real import (SEGY_DIR, REAL_ROOT, load_fault_sticks, load_horizons,
                              read_segy, _to_image, project_faults, _rasterize, _throw, MIN_FAULT_PTS)
from hybrid.model.geometry import _line_dip

LINES_ZIP = REAL_ROOT / "raw" / "seismic_2d_lines.zip"
CSV_OUT = REAL_ROOT / "real_field.csv"
IMG_DIR = REAL_ROOT / "csv_panels"                 # per-panel image PNGs
MASK_DIR = REAL_ROOT / "csv_masks"                 # per-panel fault mask PNGs
W_TILE = 512                                       # panel width in traces (bounds the encoded smap → RAM)
MIN_TILE_W = 128                                   # drop a trailing sliver narrower than this


def extract_all_lines():
    """Extract EVERY SEG-Y sub-line from the 2D-lines bag into SEGY_DIR/<survey>/<subline> (skip xml
    and already-extracted). This is what unlocks all ~462 lines (only ~63 were extracted before)."""
    if not LINES_ZIP.exists():
        print(f"[real-csv] {LINES_ZIP} missing — cannot extract lines", flush=True); return 0
    SEGY_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(LINES_ZIP) as z:
        for info in z.infolist():
            name = info.filename
            if "/data/segy/" not in name or name.endswith("/"):
                continue
            if name.endswith(".xml") or "crsmeta" in name:
                continue
            rel = name.split("/data/segy/", 1)[1]              # <survey>/<subline>
            dst = SEGY_DIR / rel
            if dst.exists() and dst.stat().st_size == info.file_size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as out:
                out.write(src.read())
            n += 1
    print(f"[real-csv] extracted {n} new SEG-Y lines (total on disk: "
          f"{sum(1 for p in SEGY_DIR.rglob('*') if p.is_file() and p.suffix.lower() not in ('.xml',))})", flush=True)
    return n


def _panel_regions(crossings, c0, c1, H, cx, cy, horizons, stem, pi):
    """Fault regions for the panel [c0:c1) — stick points falling in the panel → mask/dip/throw.
    Returns (regions, mask_paths, evidence). Empty regions ⇒ caller writes a NEGATIVE row."""
    regions, mask_paths, ev = [], [], []
    pw = c1 - c0
    for name, poly in crossings:
        sel = (poly[:, 0] >= c0) & (poly[:, 0] < c1)
        if int(sel.sum()) < MIN_FAULT_PTS:
            continue
        p = poly[sel].astype(float).copy(); p[:, 0] -= c0            # shift to panel columns
        dip = _line_dip(p)
        if dip is None:
            continue
        thr = _throw(poly, cx, cy, horizons)                        # full-line throw (needs original cols)
        mask = _rasterize(p, (H, pw)).numpy().astype(np.uint8) * 255  # fat polyline (built-in width)
        mp = MASK_DIR / f"{stem}__p{pi}__{len(mask_paths)}.png"
        Image.fromarray(mask).save(mp)
        x1, y1, x2, y2 = int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())
        meas = {"dip_deg": round(float(dip), 2)}
        if thr is not None:
            meas["throw"] = round(float(thr), 2)
        regions.append({"object_type": "fault", "class_id": 1,
                        "bbox": [x1, y1, x2, y2], "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                        "values": {"measure": meas, "derive": {}}, "mask_idx": len(mask_paths)})
        mask_paths.append(str(mp))
    return regions, mask_paths, " "


def build_real_csv(w_tile=W_TILE, limit=None):
    """Tile every extracted line into W_TILE-trace panels; write one CSV row per panel (positive fault
    regions where sticks project, negative otherwise). Streamed, CPU-only."""
    extract_all_lines()
    faults, horizons = load_fault_sticks(), load_horizons()
    IMG_DIR.mkdir(parents=True, exist_ok=True); MASK_DIR.mkdir(parents=True, exist_ok=True)
    lines = [p for p in sorted(SEGY_DIR.rglob("*"))
             if p.is_file() and p.suffix.lower() not in (".xml", ".txt", ".pdf")]
    if limit:
        lines = lines[:limit]
    print(f"[real-csv] {len(lines)} lines · {len(faults)} faults · {len(horizons)} horizons · panel {w_tile}w", flush=True)
    rows, npos, nneg = [], 0, 0
    for li, segy in enumerate(lines):
        try:
            data, cx, cy, samples = read_segy(segy)
            img = _to_image(data); H, W = img.shape
            crossings = project_faults(faults, cx, cy, samples, (H, W))
        except Exception as e:
            print(f"[real-csv] skip {segy.name}: {e}", flush=True); continue
        stem = f"{segy.parent.name}__{segy.name}"
        for pi, c0 in enumerate(range(0, W, w_tile)):
            c1 = min(W, c0 + w_tile)
            if c1 - c0 < MIN_TILE_W:
                break
            regions, mask_paths, ev = _panel_regions(crossings, c0, c1, H, cx, cy, horizons, stem, pi)
            img_png = IMG_DIR / f"{stem}__p{pi}.png"
            Image.fromarray(img[:, c0:c1]).save(img_png)
            if not regions:                                          # NEGATIVE panel (no fault)
                regions = [{"object_type": "background", "bbox": [0, 0, c1 - c0, H]}]
                nneg += 1
            else:
                npos += 1
            rows.append({"sample_id": f"{stem}__p{pi}", "images": json.dumps([str(img_png)]),
                         "masks": json.dumps(mask_paths), "regions": json.dumps(regions)})
        if li % 25 == 0:
            print(f"[real-csv] line {li}/{len(lines)} · rows {len(rows)} (pos {npos} / neg {nneg})", flush=True)
    # positives FIRST → a capped encode (build_scenes MAX_SCENES) still gets every fault panel.
    rows.sort(key=lambda r: 0 if '"object_type": "fault"' in r["regions"] else 1)
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"[real-csv] wrote {CSV_OUT} · {len(rows)} panels (positive {npos} / negative {nneg})", flush=True)
    return str(CSV_OUT)


def real_csv_scenes(test_frac=0.25, neg_per_pos=3, seed=42):
    """Load the ungated real CSV → scenes (positives + negatives), stratified train/test split. Encodes
    all fault panels + neg_per_pos× negatives (bounded for the 15 GB RAM). Returns (all, train, test).
    Same scene format as synthetic → the SAME reader/tester consume it."""
    import os
    import random
    import hybrid.model.scenes as sc
    from hybrid.data.dataset import load_local_csv
    os.environ["OFFLOAD_SMAP"] = "1"                     # big real panels → CPU smaps, paged to GPU per-use
    sc.CSV = str(CSV_OUT)
    rows = load_local_csv(csv_path=str(CSV_OUT))
    npos = sum(1 for r in rows
               if any(reg.get("object_type") == "fault" for reg in (r.get("regions") or [])))
    sc.MAX_SCENES = npos + neg_per_pos * npos + 10       # positives-first CSV → all pos + capped neg
    scenes = sc.build_scenes()
    pos = [s for s in scenes if s["objs"]]
    neg = [s for s in scenes if not s["objs"]]
    rng = random.Random(seed); rng.shuffle(pos); rng.shuffle(neg)

    def split(lst):
        c = int(len(lst) * (1 - test_frac)); return lst[:c], lst[c:]
    ptr, pte = split(pos); ntr, nte = split(neg)
    tr, te = ptr + ntr, pte + nte
    rng.shuffle(tr); rng.shuffle(te)
    print(f"[real-csv] scenes {len(scenes)} (pos {len(pos)} / neg {len(neg)}) · train {len(tr)} · test {len(te)}", flush=True)
    return pos + neg, tr, te


if __name__ == "__main__":
    build_real_csv()
