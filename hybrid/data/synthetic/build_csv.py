"""Synthetic seismic-VLM dataset from HuggingFace -> unified CSV (vision + language supervision).

Source: HF `thirdExec/synthetic-seismic-vlm` (public, parquet, ~3530 rows). Each row carries the full schema
as EMBEDDED features + text: images(1) · masks(0-8) · regions(JSON) · instruction · question · answer ·
evidence. This downloads it, MATERIALIZES the images/masks to PNG files, and writes the CSV the unified loader
+ the narrator consume (at the SYNTH_CSV default path). Same download->convert pattern as cracks/build_csv.py.

The evidence/answer already carry the <evidence>/<region>/<SEG>/<answer> skeleton the fold expects, so no tag
surgery — the regions pass through as-is (extra fields image_idx/region_idx/view/object_id are ignored by the
loader; missing values.measure is simply not supervised).

DOWNLOAD: HF `thirdExec/synthetic-seismic-vlm` (public — no token needed; set HF_TOKEN for higher rate limits).
Run:  python -m hybrid.data.synthetic.build_csv          # full build
      LIMIT=20 python -m hybrid.data.synthetic.build_csv  # a small sample (smoke)
"""
import json
import os
from pathlib import Path

import pandas as pd

HF_REPO = os.environ.get("SYNTH_HF_REPO", "thirdExec/synthetic-seismic-vlm")
ROOT = Path("data/synthetic")
CSV_OUT = ROOT / "multimodal_multi_image_dataset.csv"          # the SYNTH_CSV default (synthetic/__init__.py)
IMG_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"


def build_synthetic_csv(limit=None):
    """Download the HF dataset and write the unified CSV (materialized PNGs + regions + language columns)."""
    from datasets import load_dataset
    ROOT.mkdir(parents=True, exist_ok=True); IMG_DIR.mkdir(exist_ok=True); MASK_DIR.mkdir(exist_ok=True)
    ds = load_dataset(HF_REPO, split="train")
    if limit:
        ds = ds.select(range(min(int(limit), len(ds))))
    print(f"[synthetic] HF {HF_REPO} · {len(ds)} rows -> {CSV_OUT}", flush=True)
    rows = []
    for i, ex in enumerate(ds):
        imgs = ex.get("images") or []
        if not imgs:
            continue
        img_paths = []
        for j, im in enumerate(imgs):
            p = IMG_DIR / f"synth_{i:05d}_{j}.png"; im.convert("RGB").save(p); img_paths.append(str(p))
        mask_paths = []
        for j, mk in enumerate(ex.get("masks") or []):
            p = MASK_DIR / f"synth_{i:05d}_{j}.png"; mk.convert("L").save(p); mask_paths.append(str(p))
        rows.append({"sample_id": f"synth_{i:05d}",
                     "images": json.dumps(img_paths),
                     "masks": json.dumps(mask_paths),
                     "regions": ex.get("regions") or "[]",
                     "instruction": ex.get("instruction") or "",
                     "question": ex.get("question") or "",
                     "answer": ex.get("answer") or "",
                     "evidence": ex.get("evidence") or ""})
        if i % 500 == 0:
            print(f"[synthetic] {i}/{len(ds)} materialized", flush=True)
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"[synthetic] wrote {CSV_OUT} · {len(rows)} rows (images {IMG_DIR}/ · masks {MASK_DIR}/)", flush=True)
    return str(CSV_OUT)


if __name__ == "__main__":
    build_synthetic_csv(limit=os.environ.get("LIMIT"))
