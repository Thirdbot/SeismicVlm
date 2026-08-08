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
from torchvision.transforms.functional import pil_to_tensor
from tqdm.auto import tqdm
from hybrid.data.schema import load_local_csv
from hybrid.model.registry import (FAULT_MODES, CLASS_ID, ID_CLASS, SECTION_DERIVED, OBJECT_DERIVED,
                                    MEASURE_SLOTS, MEASURE_SCALE, MEASURE_KEY, CLASS_SCHEMA)

# ---- CONFIG ---- (the unified scene loader — synthetic & real both land here via their CSV)
CSV = None            # active dataset CSV; set by data.synthetic / data.smeaheia before build_scenes()
MAX_SCENES = float("inf")   # cap on scenes encoded (bounds GPU/RAM); inf = all
# ---- RESOLUTION SET — these three MOVE TOGETHER (all of them are functions of PATCH; each was
# hardcoded at the value it happened to have when PATCH was 16, which is how they silently desynced).
#   PATCH   = native px per token  (ACUITY). 16 = SFM's own; 8 = patch_embed resampled -> 4x the cells.
#   DILATE_R= the LOSS's capture radius, not a data choice. The invariant that matters is target width
#             in TOKEN units: measured native fault stroke is 3.16px, so 3.16+2r over PATCH. r=3 @P=16
#             gives 0.57 (the only ratio that has ever produced a working mask); r=0 @P=8 gives 0.40 —
#             THINNER than proven, so it is only safe paired with clDice (CLDICE_W>0), which scores the
#             centreline instead of the overlap and supplies tolerance without fattening the target.
#   TILE    = px of context per encoder pass. 0 = the encoder's own pretraining resolution (512 for SFM).
#             Keep it >= the image height where possible: a seam ACROSS a near-vertical fault severs it,
#             while a seam ALONG one is harmless.
PATCH = int(os.environ.get("PATCH", 16))
TILE = int(os.environ.get("TILE", 0))
DILATE_R = int(os.environ.get("DILATE_R", 0))   # 0 = PURE/undilated GT (2026-08-08): the mask is the TRUE fault.
                                                # Was 3, which fattened the native fault 1.7-8x — contaminating the GT
                                                # and breaking prior-comparability. Tolerance now comes from the LOSS
                                                # (Focal-Tversky/clDice/pos-weight), exactly as the note above intends
                                                # ("supplies tolerance without fattening the target"). Set DILATE_R=3 to revert.
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
    a = np.array(pil.convert("L").resize((W, H), Image.NEAREST), dtype=np.float32)  # NEAREST: keep the TRUE fault
    return torch.from_numpy((a > 40).astype("float32"))          # width (default resample blurred thin masks fatter)


def dilate(m, r=DILATE_R):
    """Symmetric dilation (keeps the principal axis, so dip is unchanged)."""
    if r <= 0:
        return m
    return F.max_pool2d(m[None, None], 2 * r + 1, stride=1, padding=r)[0, 0]


def _tile_image(img_path, tile_size, patch):
    """Image → a batch of NATIVE-SCALE tiles + their grid offsets + the stitched grid (fH,fW).

    THREE rules, all of them things that silently corrupted earlier runs:
      · NEVER pad — a mostly-black tile wrecks SFM's per-image normalisation and contaminates attention
        on the real content patches.
      · NEVER rescale — the image is used at its own resolution (only snapped to a whole number of
        patches, ≤8 px). Squashing every image into a square 512 applied a per-dataset geometric warp
        (synthetic 100x507 stretched 5.1x → a 60deg fault presents as 84deg), so the dip head learned a
        warp that does not hold on real data. Upsampling to "fill" adds compute, not information.
      · UNIFORM tiles — the tile is min(tile_size, side) per axis, edge-snapped, so every crop is exactly
        the same shape with no padding and they batch cleanly. A small image is simply ONE native tile
        (SFM interpolates its position embedding to that grid), a large one is cut into tile_size crops
        for memory. 50% overlap, averaged on the stitch."""
    im = Image.open(img_path).convert("RGB")
    W, H = _snap(*im.size, patch)
    if (W, H) != im.size:
        im = im.resize((W, H), Image.BILINEAR)
    tw, th = min(tile_size, W), min(tile_size, H)

    def starts(n, t):
        if n <= t:
            return [0]
        xs = list(range(0, n - t + 1, max(patch, t // 2)))
        if xs[-1] != n - t:
            xs.append(n - t)
        return xs
    tiles, offs = [], []
    for y in starts(H, th):
        for x in starts(W, tw):
            tiles.append(pil_to_tensor(im.crop((x, y, x + tw, y + th))).float() / 255.0)
            offs.append((y // patch, x // patch))
    return torch.stack(tiles), offs, (H // patch, W // patch)


def encoder_tiling():
    """(tile_size, patch) for the ACTIVE encoder — SFM 512/16 when its checkpoint is present, else NCS
    224/16. ONE resolver shared by the loader and reader.encode so the tiling can never disagree with
    the encoder that consumes it (they used to branch on the env var independently)."""
    from hybrid.stages.stage2_reader import SFM_DEFAULT
    sfm = os.environ.get("SFM_CKPT")
    if sfm and not os.path.exists(sfm):
        raise FileNotFoundError(f"SFM_CKPT={sfm} is set but the file does not exist.")
    sfm = sfm or (SFM_DEFAULT if os.path.exists(SFM_DEFAULT) else None)
    if sfm:
        return (TILE or 512, PATCH)
    if os.environ.get("ALLOW_NCS") == "1":
        return (TILE or 224, PATCH)
    raise FileNotFoundError(                                    # NO silent NCS tiling (mirrors _build_encoder):
        f"SFM checkpoint absent (SFM_CKPT unset and {SFM_DEFAULT} absent) — refusing to tile at NCS-224 for a "
        f"reader trained at 512. Set SFM_CKPT, or ALLOW_NCS=1 to force NCS.")


def _snap(W, H, patch):
    """Round each side to a whole number of patches (≤ patch/2 px of change, so scale/aspect are
    effectively untouched). The encoder needs both sides divisible by the patch size."""
    return max(patch, round(W / patch) * patch), max(patch, round(H / patch) * patch)


def _tile_grid(W, H, tile_size, patch):
    """The stitch grid (fH,fW) tiling WILL produce for a W×H image — from size ALONE (no pixel load),
    mirroring `_tile_image`. Lets lazy scenes store the grid without holding tiles."""
    W, H = _snap(W, H, patch)
    return (H // patch, W // patch)


def build_scenes(csv=None, max_scenes=None, encoder_ckpt=None):
    """Unified loader: a dataset CSV -> per-image scenes with encoded NCS feature map + registry GT.
    Keys only on the COMMON vision contract every dataset shares — image · mask · regions; the LM
    columns (instruction/question/answer/evidence) are synthetic-only and read as "" when absent, so
    synthetic and real both go through here unbranched.
    csv/max_scenes default to the module globals so `loader.CSV = …; build_scenes()` still works."""
    csv = csv or CSV
    max_scenes = MAX_SCENES if max_scenes is None else max_scenes
    rows = load_local_csv(csv_path=csv)
    tile_size, patch = encoder_tiling()          # encoder TILE geometry (shared resolver → loader/reader agree)
    # Each image recurs across many rows with different Q&A; a region's dip may
    # live in ANY of them -> group by image and aggregate the evidence, one
    # scene per image (a true image-level unit, no train/test leakage).
    by_img = {}
    n_missing_img = 0
    for r in rows:
        ips = r.get("image_paths") or []
        if not ips:
            continue                                  # LM-only row (no image column) — legitimately not a scene
        if not Path(ips[0]).exists():
            n_missing_img += 1; continue              # a labeled row whose image file is GONE — counted, not silent
        by_img.setdefault(ips[0], []).append(r)
    scenes = []
    n_missing_mask = 0
    cap = min(len(by_img), max_scenes)
    for img, rr in tqdm(by_img.items(), total=cap, desc="encode scenes", unit="img"):
        if not any(r.get("regions") for r in rr):
            continue
        W, H = Image.open(img).size
        hw = (H, W)
        grid = _tile_grid(W, H, tile_size, patch)                     # tiling grid from size only; pixels + masks load LAZILY per scene (RAM-safe)
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
                    n_missing_mask += 1; continue      # labeled region whose mask PNG is missing/invalid — counted
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
                             mask_path=mp, meas=meas, mmask=mm, derive=oder))   # LAZY: store path + value lists; mask loads on demand in scene_to_gt
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
        scenes.append(dict(grid=grid, hw=(H, W), objs=objs, img=img, derived=der, is_neg=is_neg))
        if len(scenes) >= max_scenes:
            break
    if n_missing_img or n_missing_mask:                # surface silent data loss instead of a quietly-smaller set
        print(f"[loader] WARNING: dropped {n_missing_img} row(s) with a missing image file + "
              f"{n_missing_mask} region(s) with a missing/invalid mask PNG — dataset may be incomplete "
              f"(check paths). This is reported, not silent.", flush=True)
    return scenes
