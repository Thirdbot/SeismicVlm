"""Thebe (real NW-Australia fault masks) → unified CSV. AUTO-DOWNLOADS from Harvard Dataverse, streams
crossline slices, tiles into square panels, emits the shared schema — same contract as CRACKS/Smeaheia.

Thebe (An et al. 2021, doi:10.7910/DVN/YBYGBK, CC-BY-4.0) = the largest public real fault-segmentation
set: 1803 crossline sections, pixel-level expert labels, stored as ~100-crossline chunks. Verified
format: fault volume `(100, 3174, 1537)` bool (key arr_0), seismic `(100, 3174, 1537)` float32 — a
crossline `arr[c]` is (inlines 3174, samples 1537), TRANSPOSED here to (depth 1537, inlines 3174) so
faults sit near-vertical (apparent dip meaningful). Per panel: fault = label → connected-component
instances (mask + apparent dip via line_dip; NO throw → present-gated skip, like CRACKS).

Downloads by Dataverse file id (access API): fault .npy (~488 MB) + seismic .npz (~1 GB) per chunk;
streamed one chunk at a time so limited RAM never holds more than one sub-volume. N_CHUNKS caps how
many ~100-crossline chunks to pull+convert (default 2 ≈ 3 GB, ~200 crosslines → thousands of panels;
18 = the full ~30 GB volume).

Run:  python -m hybrid.data.thebe.build_csv        (N_CHUNKS=4 python -m … for more)
"""
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label as cc_label

from hybrid.model.geometry import line_dip

REAL_ROOT = Path("data/real_data/thebe")
CSV_OUT = REAL_ROOT / "thebe.csv"
IMG_DIR = REAL_ROOT / "images"
MASK_DIR = REAL_ROOT / "masks"
RAW_DIR = REAL_ROOT / "raw"
DVN = "https://dataverse.harvard.edu/api/access/datafile/"
DOI = os.environ.get("THEBE_DOI", "doi:10.7910/DVN/YBYGBK")           # the Thebe dataset persistent id (An et al. 2021)
DVN_LATEST = "https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest"
PANEL = 512                     # square panel → one encoder tile each (fast); tiles the big 3174×1537 section
MIN_AREA = 12
N_CHUNKS = int(os.environ.get("N_CHUNKS", 2))     # ~100-crossline chunks to pull+convert (18 = full ≈ 30 GB)

# FALLBACK ONLY — file ids of ONE pinned Dataverse version. The build resolves the LATEST version's ids
# dynamically (`_resolve_chunks`), because Dataverse file ids change across versions; these are used only if
# that lookup fails. (name, fault .npy id, seismic .npz id) — train chunks are densest, then val/test.
CHUNKS = [
    ("train1", 4607333, 4862642), ("train2", 4607334, 4862655), ("train3", 4607335, 4862656),
    ("train4", 4607336, 4862781), ("train5", 4607332, 4862788), ("train6", 4607315, 4862793),
    ("train7", 4607320, 4862823), ("train8", 4607317, 4863049), ("train9", 4607316, 4863068),
    ("val1", 4607324, 4863099), ("val2", 4607323, 4863098),
    ("test1", 4607325, 4863110), ("test2", 4607329, 4863111), ("test3", 4607330, 4863109),
    ("test4", 4607327, 4863126), ("test5", 4607328, 4863125), ("test6", 4607331, 4863123),
    ("test7", 4607326, 4863124),
]


def _chunk_key(fn):
    """Normalize a Dataverse filename → a chunk key, so a fault .npy pairs with its seismic .npz regardless
    of exact naming (drop extension + the fault/seismic/label words + separators). e.g. both
    'Thebe_faults_train_1.npy' and 'Thebe_seismic_train_1.npz' → 'train1'."""
    s = os.path.splitext(fn.lower())[0]
    for w in ("faults", "fault", "seismic", "seis", "labels", "label", "amplitude", "amp", "data", "thebe"):
        s = s.replace(w, "")
    return re.sub(r"[^a-z0-9]", "", s)


def _chunk_order(key):
    m = re.match(r"(train|val|test)?\D*(\d+)?", key)
    split = {"train": 0, "val": 1, "test": 2}.get(m.group(1) if m else None, 3)
    num = int(m.group(2)) if m and m.group(2) else 0
    return (split, num)


def _resolve_chunks():
    """AUTO — resolve the LATEST published dataset version's files by DOI → [(key, fault_id, seis_id)],
    pairing each fault .npy with its seismic .npz by normalized filename. File ids change across Dataverse
    versions, so this (not the pinned CHUNKS table) is the source of truth and always tracks the latest."""
    req = urllib.request.Request(f"{DVN_LATEST}?persistentId={DOI}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        files = json.load(r)["data"]["files"]
    faults, seis = {}, {}
    for f in files:
        df = f.get("dataFile", {})
        fn = df.get("filename") or f.get("label") or ""
        fid = df.get("id")
        if fid is None:
            continue
        low = fn.lower()
        if low.endswith(".npy"):
            faults[_chunk_key(fn)] = fid
        elif low.endswith(".npz"):
            seis[_chunk_key(fn)] = fid
    paired = [(k, faults[k], seis[k]) for k in sorted(set(faults) & set(seis), key=_chunk_order)]
    if len(paired) < max(len(faults), len(seis)):                # some files didn't pair → surface the keys to tune _chunk_key
        print(f"[thebe] note: latest version has {len(faults)} .npy + {len(seis)} .npz → {len(paired)} paired · "
              f"unpaired .npy={sorted(set(faults) - set(seis))[:5]} .npz={sorted(set(seis) - set(faults))[:5]}", flush=True)
    return paired


def _valid(path):
    """True iff a cached .npy/.npz actually LOADS. Catches a truncated/interrupted download left >0
    bytes — the classic `np.load` 'No data left in file' that otherwise re-fires on every retry because
    the old size>0 cache gate accepts the partial file. .npz forces the zip central directory (EOF);
    .npy checks the file is at least as long as its header declares."""
    try:
        obj = np.load(path, mmap_mode="r", allow_pickle=False)
        if hasattr(obj, "files"):                         # NpzFile → read central dir (raises if truncated)
            _ = obj.files
        elif path.stat().st_size < obj.nbytes:            # .npy header promises more data than the file holds
            return False
        return True
    except Exception:
        return False


def _download(file_id, dst, _retry=True):
    """Stream a Dataverse datafile (by id) to disk ATOMICALLY, and only trust a cache that actually
    loads. Downloads to a `.part` file and renames into place only on success, so an interrupted stream
    never becomes a poisoned cache; a pre-existing corrupt/truncated cache (from an older run) is caught
    by `_valid` and re-fetched. This fixes the 'No data left in file' np.load error on retry."""
    if dst.exists() and dst.stat().st_size > 0 and _valid(dst):
        return dst
    dst.unlink(missing_ok=True)                           # 0-byte OR corrupt/truncated → re-fetch, don't reuse
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[thebe] downloading {dst.name} (id {file_id}) …", flush=True)
    # Dataverse 303-redirects to a presigned S3 URL whose bucket policy REJECTS the default
    # "Python-urllib/x.y" User-Agent with 403 — send a normal UA so the redirect completes.
    req = urllib.request.Request(f"{DVN}{file_id}", headers={"User-Agent": "Mozilla/5.0"})
    tmp = dst.with_suffix(dst.suffix + ".part")
    written = 0
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        clen = r.headers.get("Content-Length")
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b); written += len(b)
    expected = int(clen) if clen and clen.isdigit() else None
    if (expected is not None and written != expected) or not _valid(tmp):   # truncated / unreadable stream
        print(f"[thebe] WARNING {dst.name}: got {written} bytes"
              f"{f', expected {expected}' if expected else ''}, cache invalid — "
              f"{'re-fetching' if _retry else 'giving up'}", flush=True)
        tmp.unlink(missing_ok=True)
        if _retry:
            return _download(file_id, dst, _retry=False)
        raise IOError(f"[thebe] {dst.name} (id {file_id}) failed to download intact — check the network "
                      f"and free disk, then re-run (the partial cache has been removed).")
    tmp.rename(dst)                                       # atomic: only a fully-verified file lands at dst
    return dst


def _load(path):
    """fault .npy → memmap (488 MB, never fully resident); seismic .npz → first array (must load)."""
    if path.suffix == ".npz":
        z = np.load(path)
        return z[z.files[0]]
    return np.load(path, mmap_mode="r")


def _seismic_png(slice2d):
    """Amplitude slice → 8-bit grey PNG (2–98% clip so faults/reflectors stay visible)."""
    a = slice2d.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


def _panel_starts(W, H, p=PANEL):
    def starts(n):
        xs = list(range(0, max(n - p, 0) + 1, p)) or [0]
        if xs[-1] != max(n - p, 0):
            xs.append(max(n - p, 0))
        return xs
    return [(x, y) for y in starts(H) for x in starts(W)]


def build_thebe_csv():
    REAL_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True); MASK_DIR.mkdir(exist_ok=True)
    rows, npos, nneg, ninst, gc = [], 0, 0, 0, 0     # gc = global crossline index (drives the loader split)
    try:
        chunks = _resolve_chunks()                                     # LATEST version — file ids resolved by DOI
        print(f"[thebe] resolved {len(chunks)} fault↔seismic chunk pairs from the LATEST dataset version "
              f"(DOI {DOI}); using first {min(N_CHUNKS, len(chunks))}", flush=True)
    except Exception as e:
        chunks = []
        print(f"[thebe] WARNING: latest-version lookup failed ({type(e).__name__}: {e}) — "
              f"falling back to the pinned file ids", flush=True)
    if not chunks:
        chunks = CHUNKS                                                # network/API failure → pinned-version fallback
    for name, fid, sid in chunks[:N_CHUNKS]:
        fault = _load(_download(fid, RAW_DIR / f"fault{name}.npy"))     # (C, 3174, 1537) bool
        seis = _load(_download(sid, RAW_DIR / f"seis{name}.npz"))       # (C, 3174, 1537) float32
        C = min(fault.shape[0], seis.shape[0])
        print(f"[thebe] chunk {name}: {C} crosslines · volume slice {fault.shape[1:]}", flush=True)
        for c in range(C):
            fs = np.asarray(fault[c]).T > 0                             # → (depth 1537, inlines 3174), faults near-vertical
            img = _seismic_png(np.asarray(seis[c]).T)
            H, W = fs.shape
            for (x0, y0) in _panel_starts(W, H):
                crop = img.crop((x0, y0, x0 + PANEL, y0 + PANEL))
                comps, ncomp = cc_label(fs[y0:y0 + PANEL, x0:x0 + PANEL])
                sid_str = f"thebe_{gc:05d}_{x0}_{y0}"
                img_png = IMG_DIR / f"{sid_str}.png"; crop.save(img_png)
                regions, mask_paths = [], []
                for k in range(1, ncomp + 1):
                    m = comps == k
                    if int(m.sum()) < MIN_AREA:
                        continue
                    ys, xs = np.where(m)
                    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    meas = {}
                    dip = line_dip(np.stack([xs, ys], 1).astype(float))    # apparent dip; throw omitted (no horizons)
                    if dip is not None:
                        meas["dip_deg"] = round(float(dip), 2)
                    mp = MASK_DIR / f"{sid_str}_{len(mask_paths)}.png"
                    Image.fromarray((m * 255).astype(np.uint8)).save(mp)
                    regions.append({"object_type": "fault", "class_id": 1, "bbox": [x1, y1, x2, y2],
                                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                                    "values": {"measure": meas, "derive": {}}, "mask_idx": len(mask_paths)})
                    mask_paths.append(str(mp))
                if not regions:
                    regions = [{"object_type": "background", "bbox": [0, 0, crop.size[0], crop.size[1]]}]; nneg += 1
                else:
                    npos += 1; ninst += len(regions)
                rows.append({"sample_id": sid_str, "images": json.dumps([str(img_png)]),
                             "masks": json.dumps(mask_paths), "regions": json.dumps(regions)})
            gc += 1
        del fault, seis
        print(f"[thebe] {name} done · rows {len(rows)} (fault {npos} / bg {nneg})", flush=True)
    rows.sort(key=lambda r: 0 if '"object_type": "fault"' in r["regions"] else 1)   # positives first
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"[thebe] wrote {CSV_OUT} · {len(rows)} panels (fault {npos} / background {nneg}) · "
          f"{ninst} fault instances", flush=True)
    return str(CSV_OUT)


if __name__ == "__main__":
    build_thebe_csv()
