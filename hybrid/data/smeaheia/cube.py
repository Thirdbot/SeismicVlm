"""Smeaheia GT from the GN1101 3-D cube — slice the LABELLED volume (Thebe-style) instead of projecting
fault sticks onto 2-D lines. Fixes the 2-D-line-projection misplacement and yields dense, correct sections.

Pipeline: cube geometry (this file) -> fault trace on a slice (Z-as-TWT) -> horizon throw
-> build_from_cube.py writes the image/mask/regions CSV consumed by the unified loader.

The cube is a Petrel SEG-Y export: 1251 samples @ 4 ms TWT, ~581 inlines x ~2457 crosslines, IEEE float.
Fault-stick Z (fault_Sticks_GN1101_2012) is in TWT ms, NOT depth — VERIFIED: at Z-as-TWT the horizons
(Seabed/Statfjord) sit on their reflectors AND are offset across the fault. NO velocity conversion is applied;
`row = (Z - t0) / dt` maps a stick straight to a sample row. (The old 2-D-line projection was the misplacement;
do NOT reintroduce a depth->time conversion here — it re-displaces every fault on the project's #1-concern axis.)
"""
import numpy as np
import segyio
from pathlib import Path
from scipy.spatial import cKDTree

from hybrid.data.smeaheia.segy import _coord_scale

CUBE = Path("data/real_data/smeaheia_3d/Seismic_3D_Surveys/data/GN1101_Scaled(Realized)")
GEOM_CACHE = CUBE.parent / "gn1101_geom.npz"


def load_geometry(rebuild=False):
    """Per-trace (inline, crossline, X, Y) + TWT samples, cached. Vectorised header read (fast).
    Returns dict{il, xl, x, y, samples, ilines, xlines, tree(KDTree over XY), idx_of(il,xl)->trace}."""
    if GEOM_CACHE.exists() and not rebuild:
        z = np.load(GEOM_CACHE, allow_pickle=True)
        il, xl, x, y, samples = z["il"], z["xl"], z["x"], z["y"], z["samples"]
    else:
        with segyio.open(str(CUBE), ignore_geometry=True) as s:
            sc = _coord_scale(s.header[0])
            il = s.attributes(segyio.TraceField.INLINE_3D)[:].astype(np.int32)
            xl = s.attributes(segyio.TraceField.CROSSLINE_3D)[:].astype(np.int32)
            x = s.attributes(segyio.TraceField.CDP_X)[:].astype(np.float64) * sc
            y = s.attributes(segyio.TraceField.CDP_Y)[:].astype(np.float64) * sc
            if not x.any():
                x = s.attributes(segyio.TraceField.SourceX)[:].astype(np.float64) * sc
                y = s.attributes(segyio.TraceField.SourceY)[:].astype(np.float64) * sc
            samples = np.array(s.samples, np.float64)
        np.savez(GEOM_CACHE, il=il, xl=xl, x=x, y=y, samples=samples)
    ilines = np.unique(il); xlines = np.unique(xl)
    tree = cKDTree(np.c_[x, y])                                  # world XY -> nearest trace
    return dict(il=il, xl=xl, x=x, y=y, samples=samples, ilines=ilines, xlines=xlines, tree=tree,
                dt=float(samples[1] - samples[0]), t0=float(samples[0]), ns=len(samples))


def xy_to_ilxl(G, X, Y):
    """World (X,Y) arrays -> (inline, crossline) of the nearest trace + the match distance (m)."""
    d, i = G["tree"].query(np.c_[np.atleast_1d(X), np.atleast_1d(Y)])
    return G["il"][i], G["xl"][i], d


if __name__ == "__main__":
    import time
    t = time.time()
    G = load_geometry()
    n = len(G["il"])
    # is it a full regular grid?
    full = n == len(G["ilines"]) * len(G["xlines"])
    print(f"[cube] {n} traces · {len(G['ilines'])} inlines [{G['ilines'][0]},{G['ilines'][-1]}] · "
          f"{len(G['xlines'])} xlines [{G['xlines'][0]},{G['xlines'][-1]}] · full-grid={full}", flush=True)
    print(f"[cube] TWT {G['samples'][0]:.0f}..{G['samples'][-1]:.0f}ms dt={G['dt']:.1f} ns={G['ns']} · "
          f"X[{G['x'].min():.0f},{G['x'].max():.0f}] Y[{G['y'].min():.0f},{G['y'].max():.0f}] · geom in {time.time()-t:.1f}s", flush=True)
    # trace spacing (inline & crossline) in metres
    xl0 = int(G["xlines"][len(G["xlines"]) // 2]); sel = np.where(G["xl"] == xl0)[0]
    if len(sel) > 2:
        oo = sel[np.argsort(G["il"][sel])]
        dsp = np.hypot(np.diff(G["x"][oo]), np.diff(G["y"][oo]))
        print(f"[cube] inline spacing on xl {xl0}: median {np.median(dsp):.1f} m ({len(sel)} traces)", flush=True)
    print("CUBE_GEOM_OK", flush=True)
