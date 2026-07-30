"""Scene building — dataset -> per-image training scenes for the vision stage.

Reads the CSV, groups rows by image (a region's dip may live in ANY of that
image's rows -> aggregate the evidence), encodes each image to a stitched NCS
feature map, and builds per-object GT (class, dip/throw/pct, dilated mask) plus
per-object centre. One scene per unique image (image-level split, no leakage).
Feeds the INSTANCE READER (`hybrid.model.reader`), which replaced both the DETR
detector and the dense segmenter. (The dense fault/closure fields are legacy from
the dense-seg era and are no longer read by the reader.)
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from hybrid.model.encoder import NcsEncoder, stitch
from hybrid.data.schema import load_local_csv
from hybrid.model.registry import (FAULT_MODES, CLASS_ID, ID_CLASS, SECTION_DERIVED, OBJECT_DERIVED,
                                    MEASURE_SLOTS, MEASURE_SCALE, MEASURE_KEY, CLASS_SCHEMA)

# ---- CONFIG ---- (the unified scene loader — synthetic & real both land here via their CSV)
CSV = None            # active dataset CSV; set by data.synthetic / data.smeaheia before build_scenes()
MAX_SCENES = float("inf")   # cap on scenes encoded (bounds GPU/RAM); inf = all
DILATE_R = 3          # fatten the thin fault line to ~1 feature-cell wide
device = torch.device("cuda")
# TIER-1 meas encoding is REGISTRY-DRIVEN (registry.MEASURE/MEASURE_KEY/CLASS_SCHEMA): the meas/mmask
# vector has one slot per MEASURE (order = MEASURE_SLOTS), and ATTRS = (dataset key, slot, {class ids
# that carry it}) is generated from the class→measure mapping. Add a measure/class = a registry edit,
# NOT a change here. bbox comes from reg["bbox"] (outside values).
MEAS_SCALE = torch.tensor(MEASURE_SCALE)              # per-slot scale from the registry
ATTRS = [(MEASURE_KEY[name], slot,
          {cid for cid, cname in ID_CLASS.items() if name in CLASS_SCHEMA.get(cname, [])})
         for slot, name in enumerate(MEASURE_SLOTS)]


def load_mask_hw(pil, hw):
    H, W = hw
    a = np.array(pil.convert("L").resize((W, H)), dtype=np.float32)
    return torch.from_numpy((a > 40).astype("float32")).to(device)


def dilate(m, r=DILATE_R):
    """Symmetric dilation (keeps the principal axis, so dip is unchanged)."""
    if r <= 0:
        return m
    return F.max_pool2d(m[None, None], 2 * r + 1, stride=1, padding=r)[0, 0]


def build_scenes(csv=None, max_scenes=None, encoder_ckpt=None):
    """Unified loader: a dataset CSV -> per-image scenes with encoded NCS feature map + registry GT.
    Keys only on the COMMON vision contract every dataset shares — image · mask · regions; the LM
    columns (instruction/question/answer/evidence) are synthetic-only and read as "" when absent, so
    synthetic and real both go through here unbranched.
    csv/max_scenes default to the module globals so `loader.CSV = …; build_scenes()` still works."""
    csv = csv or CSV
    max_scenes = MAX_SCENES if max_scenes is None else max_scenes
    rows = load_local_csv(csv_path=csv)
    enc = NcsEncoder().to(device).eval()
    _ckpt = encoder_ckpt or os.environ.get("ENCODER_CKPT")     # tuned encoder (joint reader+encoder train) → re-encode with it
    if _ckpt and os.path.exists(_ckpt):
        enc.load_state_dict(torch.load(_ckpt, map_location=device)); print(f"[loader] using TUNED encoder {_ckpt}", flush=True)
    # Each image recurs across many rows with different Q&A; a region's dip may
    # live in ANY of them -> group by image and aggregate the evidence, one
    # scene per image (a true image-level unit, no train/test leakage).
    by_img = {}
    for r in rows:
        ips = r.get("image_paths") or []
        if ips and Path(ips[0]).exists():
            by_img.setdefault(ips[0], []).append(r)
    scenes = []
    cap = min(len(by_img), max_scenes)
    for img, rr in tqdm(by_img.items(), total=cap, desc="encode scenes", unit="img"):
        if not any(r.get("regions") for r in rr):
            continue
        W, H = Image.open(img).size
        hw = (H, W)
        # UNION all objects across this image's rows (a fault may appear in only some rows),
        # dedup by (class, bbox); resolve each object's mask from the row it came from.
        uniq = {}
        for r in rr:
            rmps = r.get("mask_paths") or []
            for reg in (r.get("regions") or []):
                cid = CLASS_ID.get(reg.get("object_type"))     # class from object_type (consistent name)
                if cid is None:
                    continue
                key = (cid, tuple(reg.get("bbox") or []))
                if key in uniq:
                    continue
                mi = reg.get("mask_idx", 0)
                if not (isinstance(mi, int) and 0 <= mi < len(rmps) and Path(rmps[mi]).exists()):
                    continue
                uniq[key] = (reg, rmps[mi])
        objs = []
        for (cid, _bt), (reg, mp) in uniq.items():
            x1, y1, x2, y2 = reg["bbox"]
            ctr = reg.get("center") or [(x1 + x2) / 2, (y1 + y2) / 2]   # pixel center (fallback = bbox mid)
            vals = reg.get("values") or {}
            mvals = vals.get("measure") or {}                     # TIER-1 bucket (dataset routes it)
            meas = [0.0, 0.0, 0.0]
            mm = [0.0, 0.0, 0.0]                                   # supervise ONLY what is present
            for key, slot, klass in ATTRS:                        # name-filter → only what the reader trains
                if cid in klass and mvals.get(key) is not None:
                    meas[slot] = float(mvals[key]); mm[slot] = 1.0
            # OBJECT-scoped derived GT (RAW dataset values.derive, only its class's keys) — the reader's
            # object-derived head supervises h_i against these; narrator states them as marker words.
            dvals = vals.get("derive") or {}
            okeys = [k for _i, k, _m, _kd, _l, kl in OBJECT_DERIVED if CLASS_ID[kl] == cid]
            oder = {k: dvals[k] for k in okeys if dvals.get(k) is not None} or None
            objs.append(dict(cls=cid, bbox=[x1 / W, y1 / H, x2 / W, y2 / H],
                             center=[float(ctr[0]) / W, float(ctr[1]) / H],   # normalized; un-normalized at injection
                             mask=dilate(load_mask_hw(Image.open(mp), hw)),
                             meas=torch.tensor(meas, device=device),
                             mmask=torch.tensor(mm, device=device), derive=oder))
            # no object cap: the dense segmenter has no N-query limit, and count
            # comes from connected components over the whole field.
        # encode only images that carry a detectable object (saves NCS compute); multi-object → keep
        # closure/salt/onlap-only too. NEGATIVE-aware: a panel tagged object_type "background" (real
        # ungated CSV) has no valid object but IS kept — it trains the reader to predict "no fault
        # here" (count=0), fighting false-fault detection. Synthetic never emits "background".
        is_neg = any(reg.get("object_type") == "background"
                     for r in rr for reg in (r.get("regions") or []))
        if not objs and not is_neg:
            continue
        # TIER-2 SECTION-scoped DERIVED GT (scene-level: the section's pattern describes the whole
        # image, repeated on its regions) — registry-driven, take the first present value across rows.
        # mode → label index; bool → True/False; scalar → float. Add an attribute in registry.DERIVED.
        der = {marker: None for _i, _k, marker, _kd, _l in SECTION_DERIVED}
        for r in rr:
            for reg in (r.get("regions") or []):
                if CLASS_ID.get(reg.get("object_type")) is None:
                    continue
                dv = (reg.get("values") or {}).get("derive") or {}   # TIER-2 bucket (dataset routes it)
                for _i, key, marker, kind, labels in SECTION_DERIVED:
                    if der[marker] is not None or dv.get(key) is None:
                        continue
                    if kind == "cat":
                        if dv[key] in labels:
                            der[marker] = labels.index(dv[key])
                    elif kind == "bool":
                        der[marker] = bool(dv[key])
                    else:
                        der[marker] = float(dv[key])
        smap, _ = stitch(enc, img)
        if os.environ.get("OFFLOAD_SMAP"):     # big real panels: keep smaps in CPU RAM, page to GPU per-use
            smap = smap.cpu()                  # (reader._grid moves back per call); avoids holding all on 5.67GB GPU
        ff = torch.zeros(hw, device=device)   # dense targets: union of masks by class
        cf = torch.zeros(hw, device=device)
        for o in objs:
            if int(o["cls"]) == 1:
                ff = torch.maximum(ff, o["mask"])
            elif int(o["cls"]) == 2:
                cf = torch.maximum(cf, o["mask"])
        if os.environ.get("OFFLOAD_SMAP"):     # legacy dense fields are full-hw & big at real panel size
            ff, cf = ff.cpu(), cf.cpu()
            for o in objs:                     # per-object masks too (paged to GPU in scene_to_gt/forward)
                o["mask"] = o["mask"].cpu(); o["meas"] = o["meas"].cpu(); o["mmask"] = o["mmask"].cpu()
            torch.cuda.empty_cache()
        scenes.append(dict(smap=smap, hw=hw, objs=objs, img=img, derived=der,
                           fault_field=ff, closure_field=cf, is_neg=is_neg))
        if len(scenes) >= max_scenes:
            break
    del enc                                    # free the NCS encoder (~300MB) before LM training
    torch.cuda.empty_cache()
    return scenes
