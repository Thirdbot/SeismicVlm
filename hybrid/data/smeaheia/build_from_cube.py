"""Build Smeaheia GT by SLICING the GN1101 cube perpendicular to each fault's LOCAL strike, extending the
trace along its dip over the fault's interpreted depth extent. Correct placement + dense sections, replacing
the 150 m / depth-as-time 2-D projection. Output = the shared `sample_id,images,masks,regions` CSV.

    python -m hybrid.data.smeaheia.build_from_cube            # full build -> real_field_cube.csv
    python -m hybrid.data.smeaheia.build_from_cube sheet      # DEBUG contact sheet only
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import segyio
from pathlib import Path
from PIL import Image, ImageDraw

from hybrid.data.smeaheia.cube import load_geometry, depth_to_twt, CUBE
from hybrid.data.smeaheia.segy import load_fault_sticks, _to_image, _line_dip

MIN_PTS = 4            # min sticks on a slice to accept it
MAX_PERP = 45          # max horizontal span (traces) — wider = fault runs ALONG the slice (skip)
STEEP = 1.3            # require row_span >= STEEP*perp_span (locally perpendicular, not a smear)
HALF = 96              # half-width of the output panel (traces)
COH_TOP, COH_BOT = 40, 780     # coherent TWT window (rows) to extend the trace within
PAN = Path("data/real_data/cube_panels"); MSK = Path("data/real_data/cube_masks")
CSV_OUT = "data/real_data/real_field_cube.csv"


def fault_slice_coords(G, P):
    """Fault sticks -> (il, xl, row). Z is TWT ms (verified: at Z-as-TWT the horizons sit on their
    reflectors and are offset across the fault) — NOT depth, so no velocity conversion. The old
    misplacement was the 2-D-line projection geometry, which the cube fixes."""
    from hybrid.data.smeaheia.cube import xy_to_ilxl
    il, xl, _ = xy_to_ilxl(G, P[:, 0], P[:, 1])
    row = (P[:, 2] - G["t0"]) / G["dt"]
    return il.astype(int), xl.astype(int), row


def horizons_ilxl(G):
    """All horizons -> {name: (il, xl, twt)} in cube index space (Z already TWT ms)."""
    from hybrid.data.smeaheia.segy import load_horizons
    from hybrid.data.smeaheia.cube import xy_to_ilxl
    out = {}
    for name, (Z, tree) in load_horizons().items():
        XY = tree.data
        hil, hxl, _ = xy_to_ilxl(G, XY[:, 0], XY[:, 1])
        out[name] = (hil.astype(int), hxl.astype(int), Z.astype(float))
    return out


def throw_on_slice(HZ, axis, sid, fault_perp, win=90):
    """Throw (TWT ms) = horizon offset across the fault: two-sided trend fit of each horizon on the slice,
    extrapolated to the fault, differenced (removes regional dip). Max over horizons that straddle. A wide
    window catches horizon coverage even when the horizon is truncated near the fault; a horizon that is
    genuinely one-sided (footwall only) contributes nothing."""
    best = None
    for hil, hxl, htwt in HZ.values():
        key = hxl if axis == "xl" else hil
        perp = hil if axis == "xl" else hxl
        m = np.abs(key - sid) <= 3
        pp, tt = perp[m], htwt[m]
        if len(pp) < 8:
            continue
        L = (pp < fault_perp - 3) & (pp > fault_perp - win)
        R = (pp > fault_perp + 3) & (pp < fault_perp + win)
        if int(L.sum()) >= 4 and int(R.sum()) >= 4:
            jump = abs(np.polyval(np.polyfit(pp[L], tt[L], 1), fault_perp) -
                       np.polyval(np.polyfit(pp[R], tt[R], 1), fault_perp))
            if 0 < jump < 800:                                # sanity bound (ms); drop wild extrapolations
                best = jump if best is None else max(best, jump)
    return best


def clean_slices(il, xl, row, axis):
    """axis='xl' -> slice by crossline (section horizontal axis = inline). Yield (slice_id, perp_ids, rows)
    for slices the fault crosses CLEANLY (locally perpendicular = narrow + steep)."""
    key, perp = (xl, il) if axis == "xl" else (il, xl)
    for s in np.unique(key):
        m = np.abs(key - s) <= 1
        if int(m.sum()) < MIN_PTS:
            continue
        cp, rr = perp[m], row[m]
        ps, rs = cp.max() - cp.min(), rr.max() - rr.min()
        if ps > MAX_PERP or rs < STEEP * max(ps, 1):
            continue
        yield int(s), cp, rr


def _section(s, G, axis, sid):
    """Read a cube slice (only its traces) -> (grayscale img (samples,perp), perp trace ids)."""
    key = G["xl"] if axis == "xl" else G["il"]
    perp = G["il"] if axis == "xl" else G["xl"]
    sel = np.where(key == sid)[0]
    sel = sel[np.argsort(perp[sel])]
    sec = np.stack([s.trace[int(t)] for t in sel], axis=1).astype(np.float32)
    return _to_image(sec), perp[sel]


def render_trace(img, ax_ids, cp, rr, row_lo, row_hi, half=HALF):
    """Fit the fault line from the sticks and draw it extended over [row_lo,row_hi] (the fault's interpreted
    depth), clamped to the coherent window + panel. Returns (panel_img, mask_bool, local_trace)."""
    H, W = img.shape
    a2c = {int(v): i for i, v in enumerate(ax_ids)}
    cols = np.array([a2c.get(int(p), -1.0) for p in cp])
    ok = cols >= 0
    if int(ok.sum()) < MIN_PTS:
        return None
    cols, rows_ = cols[ok], rr[ok]
    o = np.argsort(rows_); rs, cs = rows_[o], cols[o]
    a_, b_ = np.polyfit(rs, cs, 1)                            # col = a_*row + b_
    r0 = int(np.clip(min(rs.min(), row_lo), COH_TOP, H - 1))
    r1 = int(np.clip(max(rs.max(), row_hi), COH_TOP, min(H - 1, COH_BOT)))
    if r1 - r0 < 8:
        return None
    rows_line = np.arange(r0, r1 + 1)
    cols_line = a_ * rows_line + b_
    c = int(np.clip(cols_line.mean(), half, W - half)) if W > 2 * half else W // 2
    a, b = max(0, c - half), min(W, c + half)
    panel = img[:, a:b]
    pm = Image.new("L", (panel.shape[1], H), 0)
    pts = [(int(x - a), int(y)) for x, y in zip(cols_line, rows_line) if 0 <= x - a < panel.shape[1] and 0 <= y < H]
    if len(pts) < 2:
        return None
    ImageDraw.Draw(pm).line(pts, fill=1, width=1)
    return panel, np.array(pm, bool), np.c_[cols_line - a, rows_line]


def iter_sections(s, G, faults):
    """Yield (name, axis, sid, panel, mask, dip) for every clean, extended fault section."""
    for name, P in faults.items():
        il, xl, row = fault_slice_coords(G, P)
        row_lo, row_hi = float(row.min()), float(row.max())   # the fault's interpreted depth extent
        prefer = "xl" if (xl.max() - xl.min()) >= (il.max() - il.min()) else "il"
        for axis in (prefer, "il" if prefer == "xl" else "xl"):
            for sid, cp, rr in clean_slices(il, xl, row, axis):
                out = render_trace(*_section(s, G, axis, sid), cp, rr, row_lo, row_hi)
                if out is None:
                    continue
                panel, mask, tr = out
                dip = _line_dip(np.c_[tr[:, 0], tr[:, 1]])
                yield name, axis, sid, panel, mask, dip


def build(neg_per_pos=1, step=6):
    """Full build -> real_field_cube.csv. `step` subsamples slices per fault (density knob)."""
    PAN.mkdir(parents=True, exist_ok=True); MSK.mkdir(parents=True, exist_ok=True)
    G = load_geometry()
    gn = {k: v for k, v in load_fault_sticks().items() if k.startswith("GN1101")}
    HZ = horizons_ilxl(G)                                    # for throw = horizon offset across the fault
    rows, npos = [], 0
    fault_throws = {}                                         # fault name -> [measured throws] for propagation
    used = set()                                              # (axis, sid, col-bucket) to de-dup near-identical panels
    fault_key = set()                                         # (axis, sid) that HOLD a fault -> keep negatives away
    with segyio.open(str(CUBE), ignore_geometry=True) as s:
        for fi, (name, P) in enumerate(gn.items()):
            il, xl, row = fault_slice_coords(G, P)
            row_lo, row_hi = float(row.min()), float(row.max())
            prefer = "xl" if (xl.max() - xl.min()) >= (il.max() - il.min()) else "il"
            for axis in (prefer, "il" if prefer == "xl" else "xl"):
                sids = [sid for sid, _, _ in clean_slices(il, xl, row, axis)]
                for sid in sids[::step]:                      # subsample for density control
                    m = np.abs((xl if axis == "xl" else il) - sid) <= 1
                    cp = (il if axis == "xl" else xl)[m]; rr = row[m]
                    if int(m.sum()) < MIN_PTS:
                        continue
                    out = render_trace(*_section(s, G, axis, sid), cp, rr, row_lo, row_hi)
                    fault_key.add((axis, sid))
                    if out is None:
                        continue
                    panel, mask, tr = out
                    ys, xs = np.where(mask)
                    if len(xs) < 6:
                        continue
                    dip = _line_dip(np.c_[tr[:, 0], tr[:, 1]])
                    fn = f"{name.split(':')[-1][:22]}__{axis}{sid}"
                    if fn in used:
                        continue
                    used.add(fn)
                    Image.fromarray(panel).save(PAN / f"{fn}.png")
                    Image.fromarray((mask * 255).astype(np.uint8)).save(MSK / f"{fn}__0.png")
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    meas = {"dip_deg": round(float(dip), 2)} if dip is not None else {}
                    thr = throw_on_slice(HZ, axis, sid, int(np.median(cp)))
                    reg = {"object_type": "fault", "class_id": 1, "bbox": bbox,
                           "center": [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
                           "values": {"measure": meas, "derive": {}}, "mask_idx": 0}
                    if thr is not None:                                 # per-slice measured throw
                        meas["throw"] = round(float(thr), 2); reg["throw_src"] = "measured"
                        fault_throws.setdefault(name, []).append(float(thr))
                    rows.append({"sample_id": fn, "images": json.dumps([str(PAN / f"{fn}.png")]),
                                 "masks": json.dumps([str(MSK / f"{fn}__0.png")]), "_reg": [reg], "_fault": name})
                    npos += 1
            if fi % 10 == 0:
                print(f"[cube-build] fault {fi}/{len(gn)} · panels {npos}", flush=True)

        # NEGATIVES: random crossline slices with no fault, cropped -> background panels
        rng = np.random.RandomState(42)
        xls = G["xlines"]; nneg = 0; target = int(npos * neg_per_pos)
        tries = 0
        while nneg < target and tries < target * 8:
            tries += 1
            sid = int(rng.choice(xls))
            if ("xl", sid) in fault_key or ("xl", sid + 1) in fault_key or ("xl", sid - 1) in fault_key:
                continue
            img, ax = _section(s, G, "xl", sid); H, W = img.shape
            if W < 2 * HALF:
                continue
            c = int(rng.randint(HALF, W - HALF)); panel = img[:, c - HALF:c + HALF]
            fn = f"neg__xl{sid}__{c}"
            Image.fromarray(panel).save(PAN / f"{fn}.png")
            reg = [{"object_type": "background", "bbox": [0, 0, panel.shape[1], H]}]
            rows.append({"sample_id": fn, "images": json.dumps([str(PAN / f"{fn}.png")]),
                         "masks": json.dumps([]), "_reg": reg, "_fault": None})
            nneg += 1
    # PROPAGATE throw fault-wise: throw is ~a fault property (its displacement), so a section that lacked a
    # two-sided horizon fit inherits the fault's MEDIAN measured throw — marked "propagated" (provenance kept
    # OUTSIDE values.measure so the loader still reads only the number). Boosts coverage past the sparse
    # per-slice horizon fits WITHOUT inventing throw on faults that were never measured at all.
    fmed = {k: round(float(np.median(v)), 2) for k, v in fault_throws.items() if v}
    nmeas = sum(len(v) for v in fault_throws.values())
    nprop = 0
    for r in rows:
        f = r.get("_fault")
        if f in fmed:
            m = r["_reg"][0]["values"]["measure"]
            if "throw" not in m:
                m["throw"] = fmed[f]; r["_reg"][0]["throw_src"] = "propagated"; nprop += 1
    for r in rows:                                              # serialize the deferred region objects
        r["regions"] = json.dumps(r.pop("_reg")); r.pop("_fault", None)
    pd.DataFrame(rows).sort_values("regions", key=lambda c: c.str.contains("background").astype(int)) \
        .to_csv(CSV_OUT, index=False)
    print(f"CUBE_BUILD_DONE {CSV_OUT} · {len(rows)} panels ({npos} fault / {nneg} background) · "
          f"throw {nmeas + nprop}/{npos} ({nmeas} measured + {nprop} propagated across {len(fmed)} faults)",
          flush=True)
    return CSV_OUT


def debug_sheet(N=9, per_row=3, TW=240, TH=300):
    G = load_geometry()
    gn = {k: v for k, v in load_fault_sticks().items() if k.startswith("GN1101")}
    tiles, seen = [], 0
    with segyio.open(str(CUBE), ignore_geometry=True) as s:
        for name, axis, sid, panel, mask, dip in iter_sections(s, G, gn):
            rgb = np.stack([panel] * 3, -1).astype(np.float32); rgb[mask] = [240, 70, 60]
            mr = np.where(mask.any(1))[0]
            if not len(mr):
                continue
            r0, r1 = max(0, int(mr.min()) - 70), min(panel.shape[0], int(mr.max()) + 70)
            im = Image.fromarray(rgb[r0:r1].clip(0, 255).astype(np.uint8)).resize((TW, TH))
            ImageDraw.Draw(im).text((4, 4), f"{axis}{sid} {dip:.0f}" if dip else f"{axis}{sid} --", fill=(255, 235, 0))
            tiles.append(im); seen += 1
            if seen >= N:
                break
    rows_ = (len(tiles) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * (TW + 2), rows_ * (TH + 2)), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % per_row) * (TW + 2), (i // per_row) * (TH + 2)))
    out = "/tmp/claude-1000/-home-third-Desktop-Unsloth/62fb9454-1e43-46b7-8387-1207d2e1a188/scratchpad/cube_sheet.png"
    sheet.save(out); print(f"CONTACT_SHEET {out} · {seen}", flush=True)


if __name__ == "__main__":
    debug_sheet() if (len(sys.argv) > 1 and sys.argv[1] == "sheet") else build()
