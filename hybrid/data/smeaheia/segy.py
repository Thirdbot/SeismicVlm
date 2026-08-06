"""Real-field scene builder — Smeaheia 2D lines -> the SAME scene structure the
synthetic pipeline uses, so the vision code (the instance reader) runs unchanged.
(Imports `_line_dip` from segmenter only as a RANSAC angle helper for GT sticks.)

Source: Smeaheia Dataset (c) Equinor & Gassnova, CO2DataShare — modified CC BY 4.0,
ATTRIBUTION REQUIRED (see data/real_data/README.md).

Per 2D SEG-Y sub-line:
  SEG-Y            -> amplitude image (samples x traces) + per-trace (X,Y) + TWT axis
  3D fault sticks  -> PROJECT onto the line (world-XY match) -> per-fault polyline
                      (trace col, TWT row) -> rasterized fault mask + bbox
  apparent dip     -> read off the mask with the SAME reader the model uses (RANSAC)
  throw            -> horizon offset across the fault (see `_throw`, first-pass)
count / closure / class are NOT on 2D lines -> ignored (real = fault channel only).
true dip needs 3D + view -> NOT an ML target; kept only as an eval-extra (o["true_dip"]).

Scene contract (identical to hybrid/model/scenes.build_scenes):
  scene = dict(smap, hw, objs, img, fault_field, closure_field)
  obj   = dict(cls=1, bbox=[x1,y1,x2,y2](norm), mask=(H,W), meas=[dip,throw,0],
               mmask=[1, throw?1:0, 0], true_dip=None)
"""
import math
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from hybrid.model.encoder import NcsEncoder, stitch
from hybrid.data.loader import dilate
from hybrid.model.geometry import _line_dip

try:
    import segyio
except Exception:
    segyio = None

device = torch.device("cuda")

REAL_ROOT = Path("data/real_data")
SEGY_DIR = REAL_ROOT / "segy"                 # extracted SEG-Y: <survey>/<subline> (no ext)
FAULT_ZIP = REAL_ROOT / "raw/fault_sticks.zip"
HORIZON_DIR = REAL_ROOT / "horizons"          # POINTZ horizon shapefiles (.shp only needed)
RENDER_DIR = REAL_ROOT / "render"             # cached PNGs for encoder + overlays
MATCH_THRESH_M = 150.0                         # fault point within this of a trace -> on the line
MIN_FAULT_PTS = 6                              # min projected points to accept a crossing
HZ_MATCH_M = 90.0                              # horizon point within this of a trace to sample it
THROW_WIN = 45                                 # traces each side of the fault for the throw trend fit
DILATE_R = 3
SEED = 42


# --------------------------- fault sticks (3D world) ---------------------------

def load_fault_sticks():
    """Both fault-stick files -> {namespaced_fault: np.array([[X,Y,Z], ...])}.
    Namespaced by source file so identically-named faults don't merge."""
    faults = {}
    with zipfile.ZipFile(FAULT_ZIP) as z:
        for n in z.namelist():
            base = Path(n).name
            if base not in ("fault_sticks_2010", "fault_Sticks_GN1101_2012"):
                continue
            src = "2010" if "2010" in base else "GN1101"
            for ln in z.read(n).decode("latin-1").splitlines():
                p = ln.split()
                if len(p) < 7:
                    continue
                try:
                    x, y, zz = float(p[3]), float(p[4]), float(p[5])
                except ValueError:
                    continue
                name = f"{src}:{' '.join(p[6:-1])}"
                faults.setdefault(name, []).append((x, y, zz))
    return {k: np.array(v) for k, v in faults.items()}


# --------------------------- SEG-Y -> image + geometry ---------------------------

def _coord_scale(hdr):
    sc = hdr[segyio.TraceField.SourceGroupScalar] or 1
    return (1.0 / abs(sc)) if sc < 0 else (float(sc) if sc > 0 else 1.0)


def read_segy(path):
    """SEG-Y sub-line -> (amplitude (samples,traces), trace X (traces,), trace Y,
    TWT samples (ms)). CDP X/Y from headers (fallback Source X/Y), scaled."""
    with segyio.open(str(path), ignore_geometry=True) as f:
        n = f.tracecount
        data = np.stack([f.trace[i] for i in range(n)], axis=1).astype(np.float32)
        samples = np.array(f.samples, dtype=np.float32)     # TWT ms
        s = _coord_scale(f.header[0])
        cx = np.array([f.header[i][segyio.TraceField.CDP_X] for i in range(n)], float)
        cy = np.array([f.header[i][segyio.TraceField.CDP_Y] for i in range(n)], float)
        if not cx.any():                                    # fall back to source coords
            cx = np.array([f.header[i][segyio.TraceField.SourceX] for i in range(n)], float)
            cy = np.array([f.header[i][segyio.TraceField.SourceY] for i in range(n)], float)
    return data, cx * s, cy * s, samples


def _to_image(data):
    """Amplitude (samples,traces) -> uint8 grayscale, per-trace balanced (AGC-ish)."""
    a = data.copy()
    a -= a.mean(axis=0, keepdims=True)
    scale = np.percentile(np.abs(a), 99) + 1e-6
    a = np.clip(a / scale, -1, 1)
    return ((a + 1) * 127.5).astype(np.uint8)               # (H=samples, W=traces)


# --------------------------- project faults onto the line ---------------------------

def _rasterize(poly, hw):
    H, W = hw
    im = Image.new("L", (W, H), 0)
    # width=1 (THIN): store the stick polyline at 1px, matching the skeleton convention — the loader's
    # single dilate(r=DILATE_R) then brings it to the standard ~7px. Was 2*DILATE_R+1 (=7px) which,
    # DOUBLE-dilated by the loader, gave the ~20px blobby Smeaheia masks (gt_audit 2026-08-05).
    ImageDraw.Draw(im).line([(int(c), int(r)) for c, r in poly], fill=1, width=1)
    return torch.from_numpy(np.array(im, dtype=np.float32))


def project_faults(faults, cx, cy, samples, hw):
    """For each fault, keep its 3D points within MATCH_THRESH_M of a trace, project to
    (trace col, TWT row), and return [(fault_name, polyline (col,row))] crossings."""
    H, W = hw
    tree = cKDTree(np.c_[cx, cy])
    t0, dt = float(samples[0]), float(samples[1] - samples[0])
    out = []
    for name, P in faults.items():
        d, idx = tree.query(P[:, :2])
        keep = d < MATCH_THRESH_M
        if int(keep.sum()) < MIN_FAULT_PTS:
            continue
        cols = idx[keep].astype(float)
        rows = (P[keep, 2] - t0) / dt                       # TWT(ms) -> sample row
        inb = (rows >= 0) & (rows < H)
        if int(inb.sum()) < MIN_FAULT_PTS:
            continue
        poly = np.c_[cols[inb], rows[inb]]
        poly = poly[np.argsort(poly[:, 1])]                 # order top->bottom for a clean stick
        out.append((name, poly))
    return out


def load_horizons():
    """Key horizons (POINTZ shapefiles, .shp only) -> {name: (Z_abs, cKDTree(XY))}.
    Z is TWT ms (confirmed empirically: Draupne aligns with fault TWT), down-positive."""
    import shapefile
    hz = {}
    for shp in sorted(HORIZON_DIR.glob("*.shp")):
        r = shapefile.Reader(shp=str(shp))
        XY, Z = [], []
        for s in r.iterShapes():
            if s.points and getattr(s, "z", None):
                XY.append(s.points[0]); Z.append(s.z[0])
        if XY:
            hz[shp.stem] = (np.abs(np.array(Z)), cKDTree(np.array(XY)))
    return hz


def _throw(poly, cx, cy, horizons):
    """Throw = TWT offset of a horizon across the fault via a TWO-SIDED trend fit:
    fit the horizon along the line on each side of the fault, extrapolate both to the
    fault column, difference = throw (removes the horizon's regional dip). Max over
    horizons present on both sides; None if none straddles the fault. TWT ms."""
    if not horizons:
        return None
    fc = int(np.median(poly[:, 0]))

    def side(rng, tree, Z):
        ts, zs = [], []
        for t in rng:
            if 0 <= t < len(cx):
                d, i = tree.query([cx[t], cy[t]])
                if d < HZ_MATCH_M:
                    ts.append(t); zs.append(Z[i])
        return np.array(ts, float), np.array(zs, float)

    best = None
    for Z, tree in horizons.values():
        tL, zL = side(range(fc - THROW_WIN, fc - 4), tree, Z)
        tR, zR = side(range(fc + 4, fc + THROW_WIN), tree, Z)
        if len(zL) >= 6 and len(zR) >= 6:
            jump = abs(np.polyval(np.polyfit(tL, zL, 1), fc)
                       - np.polyval(np.polyfit(tR, zR, 1), fc))
            best = jump if best is None else max(best, jump)
    return best


