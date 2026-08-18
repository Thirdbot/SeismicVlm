"""Thebe from the Kaggle **thebe-fault-patches-256** dataset (mycarta) → unified CSV.

Chosen over the Dataverse 3-D volumes because those 202-stage from cold storage (slow/unreliable). The Kaggle
set is 256×256 **seismic + fault** patch pairs — exactly Thebe's role in this project (fault
detection/segmentation; its dip is mask-derived, NOT ground truth, so we emit image+mask only). Files:
`<split>_part<NN>_seismic.npz` + `<split>_part<NN>_fault.npz` (+ `_meta.csv`), each an (N,256,256) stack, with
the author's **train/val/test** split in the filename — which we honor (leak-safe by construction).

AUTO-DOWNLOAD needs Kaggle auth: `pip install kagglehub` + `~/.kaggle/kaggle.json` (or KAGGLE_USERNAME/
KAGGLE_KEY). Or download the dataset yourself and drop the `*_seismic.npz`/`*_fault.npz` into
`data/real_data/thebe/patches/`.

  THEBE_SOURCE=patches python -m hybrid.data.thebe.build_from_patches
  THEBE_ATTRS=1 python -m hybrid.data.thebe.build_from_patches   # + apparent dip (transfer probe; mask-derived, NOT GT)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label as cc_label

from hybrid.model.geometry import line_dip   # apparent dip from mask geometry (opt-in, THEBE_ATTRS)

KAGGLE_DS = os.environ.get("THEBE_KAGGLE", "mycarta/thebe-fault-patches-256")
ROOT = Path("data/real_data/thebe")
PATCH_DIR = ROOT / "patches"          # drop *_seismic.npz/*_fault.npz here to skip the Kaggle download
IMG_DIR = ROOT / "images_patches"
MASK_DIR = ROOT / "masks_patches"
CSV_OUT = ROOT / "thebe_patches.csv"
MIN_AREA = int(os.environ.get("THEBE_MIN_AREA", 12))
NEG_PER_POS = float(os.environ.get("THEBE_NEG_PER_POS", 1))   # background patches kept per fault patch
MAX_PATCHES = int(os.environ.get("THEBE_MAX_PATCHES", 0))     # total cap (0 = all); ~170k available — set for fast/bounded builds
# DEFAULT ON: emit apparent dip (line_dip on the fault mask) as a region attribute, so the full ungated data
# is stored and the A/B evaluation can score Thebe dip (the RESULT decides validity, not a gate). NOTE it is
# mask-DERIVED, not independent GT — circular as an accuracy target, so report it as learnability, not accuracy
# (only Smeaheia's 3-D dip is independent). Throw stays omitted (no horizons). OPT-OUT THEBE_ATTRS=0 for a
# mask-only build. Same operator as the volume build (build_csv.py), which also keeps dip.
THEBE_ATTRS = os.environ.get("THEBE_ATTRS", "1").lower() not in ("0", "false", "no")


def _source_dir():
    """A dir holding the `*_seismic.npz` — local `patches/` if present, else a kagglehub download."""
    if any(PATCH_DIR.rglob("*_seismic.npz")):
        return PATCH_DIR
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("[thebe-patches] need `pip install kagglehub` + Kaggle auth (~/.kaggle/kaggle.json "
                         "or KAGGLE_USERNAME/KAGGLE_KEY), OR drop the dataset's *_seismic.npz/*_fault.npz "
                         f"into {PATCH_DIR}/ and re-run.")
    print(f"[thebe-patches] downloading Kaggle {KAGGLE_DS} (needs Kaggle auth) …", flush=True)
    return Path(kagglehub.dataset_download(KAGGLE_DS))


def _arr(p):
    with np.load(p) as z:
        return z[z.files[0]]                            # first array in the npz → (N, 256, 256)


def build():
    if CSV_OUT.exists() and os.environ.get("THEBE_REBUILD") != "1":     # idempotent: never clobber a prior (uncapped) build
        print(f"[thebe-patches] {CSV_OUT} exists — using it (THEBE_REBUILD=1 to rebuild).", flush=True)
        return str(CSV_OUT)
    src = _source_dir()
    seis = sorted(Path(src).rglob("*_seismic.npz"))
    if not seis:
        raise SystemExit(f"[thebe-patches] no *_seismic.npz found under {src}")
    IMG_DIR.mkdir(parents=True, exist_ok=True); MASK_DIR.mkdir(parents=True, exist_ok=True)
    per_file = (MAX_PATCHES // max(len(seis), 1)) if MAX_PATCHES else 0   # spread the cap over ALL files/splits
    rows, ninst, npos, nneg = [], 0, 0, 0
    for sf in seis:
        ff = sf.with_name(sf.name.replace("_seismic.npz", "_fault.npz"))
        if not ff.exists():
            print(f"[thebe-patches] no fault pair for {sf.name} — skip", flush=True); continue
        stem = sf.name.replace("_seismic.npz", "")      # e.g. 'train_part02' — begins with the split token
        S, F = _arr(sf), _arr(ff)
        N = min(len(S), len(F))
        print(f"[thebe-patches] {sf.name}: {N} patches · seismic{S.shape} fault{F.shape}", flush=True)
        kept = 0
        for i in range(N):
            if per_file and kept >= per_file:           # per-file budget → every split (train/val/test) is represented
                break
            fault = np.asarray(F[i]) > 0
            has = bool(fault.any())
            if not has and nneg >= NEG_PER_POS * max(npos, 1):
                continue                                # POSITIVES-FIRST: keep every fault patch; bound the negatives
            a = np.asarray(S[i], dtype=np.float32)
            lo, hi = np.percentile(a, 2), np.percentile(a, 98)   # 2–98% clip → 8-bit grey (faults stay visible)
            im = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
            sid = f"thebe_{stem}_{i:05d}"               # sid carries the split token (train/val/test)
            ip = IMG_DIR / f"{sid}.png"
            Image.fromarray(im.astype(np.uint8)).convert("RGB").save(ip)
            regs, mpaths = [], []
            if has:
                comps, nc = cc_label(fault)             # connected components → fault instances
                for c in range(1, nc + 1):
                    m = comps == c
                    if int(m.sum()) < MIN_AREA:
                        continue
                    ys, xs = np.where(m)
                    bb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    mp = MASK_DIR / f"{sid}_{len(mpaths)}.png"
                    Image.fromarray((m * 255).astype(np.uint8)).save(mp)
                    reg = {"object_type": "fault", "bbox": bb,
                           "center": [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2],
                           "mask_idx": len(mpaths)}   # default: NO attributes (mask-derived dip isn't GT)
                    if THEBE_ATTRS:                   # opt-in transfer probe: apparent dip from the mask (circular; eval-only)
                        dip = line_dip(np.stack([xs, ys], 1).astype(float))
                        if dip is not None:
                            reg["values"] = {"measure": {"dip_deg": round(float(dip), 2)}, "derive": {}}
                    regs.append(reg)
                    mpaths.append(str(mp))
            npos += int(has); nneg += int(not has); kept += 1
            rows.append({"sample_id": sid, "images": json.dumps([str(ip)]),
                         "masks": json.dumps(mpaths), "regions": json.dumps(regs),
                         "instruction": "", "question": "", "answer": "", "evidence": ""})
            ninst += len(mpaths)
    import random
    random.Random(42).shuffle(rows)                    # de-interleave splits — files are sorted test→train→val,
                                                       # so a REAL_CAP on CSV order would otherwise starve 'train'
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"[thebe-patches] wrote {CSV_OUT} · {len(rows)} patches ({npos} fault / {nneg} bg) · "
          f"{ninst} fault instances", flush=True)
    return str(CSV_OUT)


if __name__ == "__main__":
    build()
