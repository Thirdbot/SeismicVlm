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
import time
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
DVN_DATASET = "https://dataverse.harvard.edu/api/access/dataset/:persistentId"   # whole-VERSION zip endpoint
DOI = os.environ.get("THEBE_DOI", "doi:10.7910/DVN/YBYGBK")     # Thebe dataset persistent id (An et al. 2021)
THEBE_VERSION = os.environ.get("THEBE_VERSION", "")             # e.g. "1.0" → pull that VERSION as one zip and
                                                               # extract it (version-pinned, reliable vs the flaky
                                                               # per-file access). "" (default) = per-file api.
PANEL = 512                     # square panel → one encoder tile each (fast); tiles the big 3174×1537 section
MIN_AREA = 12
N_CHUNKS = int(os.environ.get("N_CHUNKS", 2))     # ~100-crossline chunks to pull+convert (18 = full ≈ 30 GB)

# (name, fault .npy id, seismic .npz id) — Dataverse file ids (verified downloadable). Train chunks are
# densest, then val/test. The download itself is validated + atomic (see `_download`), which is the real fix
# for the earlier `np.load: No data left in file` (a truncated cache), NOT the file ids.
CHUNKS = [
    ("train1", 4607333, 4862642), ("train2", 4607334, 4862655), ("train3", 4607335, 4862656),
    ("train4", 4607336, 4862781), ("train5", 4607332, 4862788), ("train6", 4607315, 4862793),
    ("train7", 4607320, 4862823), ("train8", 4607317, 4863049), ("train9", 4607316, 4863068),
    ("val1", 4607324, 4863099), ("val2", 4607323, 4863098),
    ("test1", 4607325, 4863110), ("test2", 4607329, 4863111), ("test3", 4607330, 4863109),
    ("test4", 4607327, 4863126), ("test5", 4607328, 4863125), ("test6", 4607331, 4863123),
    ("test7", 4607326, 4863124),
]


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


STAGE_TRIES = int(os.environ.get("THEBE_STAGE_TRIES", 20))   # polls while Dataverse stages a file from cold storage
STAGE_WAIT = int(os.environ.get("THEBE_STAGE_WAIT", 15))     # base seconds between polls (grows, capped)


def _download(file_id, dst):
    """Stream a Dataverse datafile (by id) to disk ATOMICALLY. Handles the two REAL failure modes seen on
    Harvard Dataverse:
      1. HTTP 202 + empty body — the large file is being STAGED from archival/cold storage; the first
         request triggers staging and every poll returns 202 until it is ready. We poll with backoff.
      2. a truncated/interrupted stream — the `.part` file fails `_valid` and is re-fetched.
    Downloads to a `.part` file and renames into place only once fully verified, so a 202-empty or partial
    response never becomes a poisoned cache (the old 'No data left in file' np.load error)."""
    if dst.exists() and dst.stat().st_size > 0 and _valid(dst):
        return dst
    dst.unlink(missing_ok=True)                           # 0-byte OR corrupt/truncated → re-fetch, don't reuse
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[thebe] downloading {dst.name} (id {file_id}) …", flush=True)
    # Dataverse 303-redirects to a presigned S3 URL whose bucket policy REJECTS the default
    # "Python-urllib/x.y" User-Agent with 403 — send a normal UA so the redirect completes.
    req = urllib.request.Request(f"{DVN}{file_id}", headers={"User-Agent": "Mozilla/5.0"})
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, STAGE_TRIES + 1):
        written = 0
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            status = getattr(r, "status", None) or r.getcode()
            clen = r.headers.get("Content-Length")
            retry_after = r.headers.get("Retry-After")
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b); written += len(b)
        if status == 202 or written == 0:                # cold-storage staging — not ready yet, poll again
            tmp.unlink(missing_ok=True)
            wait = int(retry_after) if retry_after and retry_after.isdigit() else min(STAGE_WAIT * attempt, 120)
            print(f"[thebe] {dst.name}: Dataverse is staging this file from archival storage (HTTP 202) — "
                  f"waiting {wait}s, poll {attempt}/{STAGE_TRIES} …", flush=True)
            time.sleep(wait)
            continue
        expected = int(clen) if clen and clen.isdigit() else None
        if (expected is not None and written != expected) or not _valid(tmp):   # truncated / unreadable stream
            tmp.unlink(missing_ok=True)
            print(f"[thebe] {dst.name}: got {written} bytes"
                  f"{f', expected {expected}' if expected else ''}, cache invalid — retry {attempt}/{STAGE_TRIES}", flush=True)
            time.sleep(min(STAGE_WAIT, 15))
            continue
        tmp.rename(dst)                                   # atomic: only a fully-verified file lands at dst
        return dst
    raise IOError(f"[thebe] {dst.name} (id {file_id}) still not served after {STAGE_TRIES} polls — Dataverse "
                  f"is staging it from archival storage (this can take a while for a cold file). Re-run later, "
                  f"or raise THEBE_STAGE_TRIES / THEBE_STAGE_WAIT; it caches once staged.")


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


def _valid_zip(path):
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None                   # None = every CRC ok (not truncated)
    except Exception:
        return False


def _fetch_version_zip():
    """THEBE_VERSION path: download the whole dataset VERSION as one zip via the Dataverse dataset-access API
    (`/api/access/dataset/:persistentId/versions/<v>`) and extract its fault `.npy` + seismic `.npz` into
    RAW_DIR — version-pinned and one request for the full volume, which is reliable when the per-file access is
    202-staging/flaky. Streams to a `.part` file, polls the same 202 cold-storage staging as `_download`,
    validates the zip, then extracts. No-op once the numpy files are present. NOTE: this pulls the FULL version
    (~30 GB) regardless of N_CHUNKS (a zip can't be partially fetched); N_CHUNKS still caps what gets BUILT."""
    import zipfile
    if any(RAW_DIR.glob("*.npy")) and any(RAW_DIR.glob("*.npz")):
        return                                           # already downloaded + extracted
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zpath = RAW_DIR / f"thebe_v{THEBE_VERSION}.zip"
    url = f"{DVN_DATASET}/versions/{THEBE_VERSION}?persistentId={DOI}"
    if not (zpath.exists() and _valid_zip(zpath)):
        zpath.unlink(missing_ok=True)
        print(f"[thebe] downloading dataset VERSION {THEBE_VERSION} as a zip (full volume, ~30 GB) …", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        tmp = zpath.with_suffix(".part")
        for attempt in range(1, STAGE_TRIES + 1):
            written = 0
            with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
                status = getattr(r, "status", None) or r.getcode()
                retry_after = r.headers.get("Retry-After")
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); written += len(b)
            if status == 202 or written == 0:            # cold-storage staging — poll again
                tmp.unlink(missing_ok=True)
                wait = int(retry_after) if retry_after and retry_after.isdigit() else min(STAGE_WAIT * attempt, 120)
                print(f"[thebe] version zip staging (HTTP 202) — waiting {wait}s, poll {attempt}/{STAGE_TRIES} …", flush=True)
                time.sleep(wait); continue
            if not _valid_zip(tmp):                      # truncated/invalid → retry
                tmp.unlink(missing_ok=True)
                print(f"[thebe] version zip invalid ({written} bytes) — retry {attempt}/{STAGE_TRIES}", flush=True)
                time.sleep(min(STAGE_WAIT, 15)); continue
            tmp.rename(zpath); break
        else:
            raise IOError(f"[thebe] dataset version {THEBE_VERSION} zip not served after {STAGE_TRIES} polls "
                          f"(Dataverse staging). Re-run later, or raise THEBE_STAGE_TRIES / THEBE_STAGE_WAIT.")
    print(f"[thebe] extracting .npy/.npz from {zpath.name} …", flush=True)
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():                           # flatten — write each numpy file into RAW_DIR by basename
            if m.lower().endswith((".npy", ".npz")):
                (RAW_DIR / Path(m).name).write_bytes(z.read(m))
    print(f"[thebe] extracted {sum(1 for _ in RAW_DIR.glob('*.npy'))} .npy + "
          f"{sum(1 for _ in RAW_DIR.glob('*.npz'))} .npz into {RAW_DIR}", flush=True)


def _toks(p):
    return set(re.findall(r"[a-z]+|\d+", p.stem.lower())) - {
        "fault", "faults", "seismic", "seis", "label", "labels", "data", "amp", "amplitude", "thebe"}


def _local_chunks():
    """MANUAL/OFFLINE source: files the user downloaded from the dataset PAGE and dropped in RAW_DIR — the
    reliable path when the access API is 202-staging/unavailable. Discovers fault `.npy` + seismic `.npz`
    and pairs each fault with the seismic that shares the most filename tokens (falls back to sorted order
    when names carry no shared token but counts match). Returns [(name, fault_path, seis_path)] or []. Logs
    every pair so the pairing is verifiable before a 30 GB build."""
    npy = sorted(p for p in RAW_DIR.glob("*.npy") if not p.name.endswith(".part"))
    npz = sorted(p for p in RAW_DIR.glob("*.npz") if not p.name.endswith(".part"))
    if not npy or not npz:
        return []
    used, pairs = set(), []
    for f in npy:
        cand = sorted(((len(_toks(f) & _toks(s)), s) for s in npz if s not in used), reverse=True)
        if cand and cand[0][0] > 0:
            s = cand[0][1]; used.add(s); pairs.append((f.stem, f, s))
    if not pairs and len(npy) == len(npz):                    # no shared tokens → assume parallel sorted order
        pairs = [(f.stem, f, s) for f, s in zip(npy, npz)]
    for _n, f, s in pairs:
        print(f"[thebe] local pair: {f.name}  ↔  {s.name}", flush=True)
    return pairs


def build_thebe_csv():
    REAL_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True); MASK_DIR.mkdir(exist_ok=True)
    rows, npos, nneg, ninst, gc = [], 0, 0, 0, 0     # gc = global crossline index (drives the loader split)
    if THEBE_VERSION:                                # THEBE_VERSION=<v> → auto-pull that version zip + extract
        _fetch_version_zip()
    local = _local_chunks()
    if local:                                        # user-placed files OR the extracted version zip — no per-file api/202
        print(f"[thebe] BUILDING from {len(local)} locally-placed chunk pair(s) in {RAW_DIR} — no API download", flush=True)
        source = [(name, ("f", f), ("f", s)) for name, f, s in local]
    else:
        source = [(name, ("a", fid), ("a", sid)) for name, fid, sid in CHUNKS]
    for name, (fk, fv), (sk, sv) in source[:N_CHUNKS]:
        fault = _load(fv if fk == "f" else _download(fv, RAW_DIR / f"fault{name}.npy"))     # (C, 3174, 1537) bool
        seis = _load(sv if sk == "f" else _download(sv, RAW_DIR / f"seis{name}.npz"))       # (C, 3174, 1537) float32
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
