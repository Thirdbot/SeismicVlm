"""Instance reader — DETR set-prediction detector over a FROZEN vision encoder.

Reads a scene's feature grid and emits, per object: class · class-driven measured
attributes · occupancy footprint · a per-instance mask · the hidden state h_i that
prompts the LM side. Measure-only by contract: the LM never regresses a number, it
COPIES what this module measured across the non-differentiable digit seam.

ENCODER (set by the caller via `set_encoder`, built by stage2_reader._build_encoder):
SFM-Base-512 when its checkpoint is present, else NCS-v1-2d-base. Frozen either way —
`trainable_blocks>0` is an experiment, not the default. Never construct one here; the
shared resolver is what keeps the encoder and the loader's tiling from disagreeing.

DETR SET PREDICTION — and the bug it fixed. Earlier this was an autoregressive reader:
teacher-forced to exactly K objects during training, but at inference a separate count
head chose N. When N disagreed, a decoder never trained to stop had to invent objects
("4 closures" where GT = 1 fault) — a train/inference MISMATCH, not a capacity problem.
Now a FIXED set of learnable queries (`self.query`, N_QUERIES) cross-attends the grid in
ONE parallel pass; Hungarian matching (cost = −class_prob + centroid-L1) assigns
queries↔GT and every unmatched query is supervised as ∅ (NO_OBJ), down-weighted by the
DETR eos_coef. Identical path in training and inference — no teacher forcing, no count
head, no stop head. N_QUERIES MUST exceed the densest scene: below that the model
cannot represent the scene and GT is silently DROPPED from the loss.

THE MEASUREMENT RULE (see reference_dip_from_mask_geometry): angle/shape attributes read
the SPATIAL footprint (dip from its 2nd-moment stats — orientation IS the dip, r=1.000),
magnitudes read POOLED features (throw). A dip head on a pooled scalar collapses to a
constant ~73° prior and ignores the image — never do that. Which class carries which
measure comes from the registry (`measures_for_id`), so adding an attribute is a
registry row, not a new head here.

MASKS: a pixel-decoder trunk (`self.pixdec`, global self-attention over the STITCHED map)
reassembles cross-tile context that per-tile encoding loses, so a fault split across tiles
is whole BEFORE the reader reads it; `_mask_features` upsamples that trunk to per-pixel
features and each query paints its own instance mask over them. Scored PER INSTANCE — an
area-weighted union dice is a different, far easier metric and is not comparable.

`self.mask_attn` (Mask2Former masked attention — each query attends only to its previous
layer's occupancy) is implemented in `_decode` but OFF by default: a matched A/B showed no
gain, and the mask is limited by features and data, not by the attention pattern.

`add_real_adapter` gives real-field transfer a zero-init residual adapter on the grid
features while the whole synthetic reader stays frozen (adapter isolation).
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from hybrid.model.registry import (NUM_DERIVED, MAX_CAT, N_CLASS, SCALAR_SCALE,
                                    SECTION_DERIVED, OBJECT_DERIVED, CLASS_ID,
                                    MEASURE, MEASURE_SLOTS, measures_for_id)
from hybrid.model.heads import DerivedHead

device = torch.device("cuda")
NO_OBJ, FAULT, CLOSURE, SALT, ONLAP = 0, 1, 2, 3, 4       # class ids (∅ / fault / closure / salt / onlap)
N_QUERIES = int(os.environ.get("N_QUERIES", 48))          # DETR object slots — must exceed the densest scene
                                                          # (CRACKS max 42). ∅ imbalance is handled by the
                                                          # eos_coef class weight, as in DETR.


def _soft_erode(x): return -F.max_pool2d(-x, 3, 1, 1)
def _soft_open(x):  return F.max_pool2d(_soft_erode(x), 3, 1, 1)


def _dice_loss(p, g):
    """PER-INSTANCE soft-dice, averaged over instances. p,g are (K, ...) — dice is computed for EACH
    instance then meaned. A GLOBAL `.sum()` over all K (the previous form) is AREA-WEIGHTED: a big
    object dominates and a thin fault contributes almost nothing, so completely missing a thin fault
    cost the loss ~0.04 while the per-instance eval metric penalises ~0.50 (a 12x train/eval mismatch
    that taught the model to ignore exactly the thin faults we care about)."""
    p = p.flatten(1); g = g.flatten(1)
    inter = (p * g).sum(1)
    return (1 - (2 * inter + 1) / (p.sum(1) + g.sum(1) + 1)).mean()


def tversky_loss(p, g, alpha=0.3, beta=0.7, gamma=1.0, smooth=1.0):
    """PER-INSTANCE Focal-Tversky loss. TI = TP/(TP + α·FN + β·FP); loss = (1-TI)^γ, meaned over instances.
    β>α penalizes FALSE POSITIVES more than false negatives → discourages over-prediction (fat masks, the
    measured pixP≈0.43 defect) while allowing slight under-cover. γ>1 focuses on hard instances. α=β=0.5,
    γ=1 ≡ Dice (same +1 smoothing as _dice_loss). NEVER touches the GT — it reshapes the prediction."""
    p = p.flatten(1); g = g.flatten(1)
    tp = (p * g).sum(1); fp = (p * (1 - g)).sum(1); fn = ((1 - p) * g).sum(1)
    ti = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return ((1 - ti) ** gamma).mean()


def soft_skel(x, iters=5):
    """Differentiable morphological skeleton (Shit et al., clDice)."""
    x1 = _soft_open(x); skel = F.relu(x - x1)
    for _ in range(iters):
        x = _soft_erode(x); x1 = _soft_open(x); delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice(p, g, iters=5, eps=1e-5):
    """clDice — topology/centerline-aware dice; p,g are (K,H,W) in [0,1]. Rewards an unbroken skeleton,
    which the volumetric dice ignores → the natural loss for THIN faults. Returns 1 - clDice (a loss)."""
    p4, g4 = p.unsqueeze(1), g.unsqueeze(1)                     # (K,1,H,W)
    sp, sg = soft_skel(p4, iters), soft_skel(g4, iters)
    tprec = (sp * g4).sum((1, 2, 3)) / (sp.sum((1, 2, 3)) + eps)   # skeleton precision
    tsens = (sg * p4).sum((1, 2, 3)) / (sg.sum((1, 2, 3)) + eps)   # skeleton sensitivity
    return (1 - 2 * tprec * tsens / (tprec + tsens + eps)).mean()


def scene_to_gt(scene):
    """Scene objs -> reader GT list {cls, ctr, mask_fW, dip?, throw?, area?, derive?}, ordered by
    x-centre. Multi-class: fault(1) dip/throw · closure(2)/salt(3)/onlap(4) area · closures carry the
    RAW per-object derive dict (fluid/intersects_*) for the object-scoped derived head."""
    from hybrid.data.loader import load_mask_hw, dilate, encoder_tiling
    from PIL import Image
    _, P = encoder_tiling()                                 # GT resolution is grid*PATCH (was hardcoded *16,
    fH, fW = scene["grid"]; mh, mw = fH * P, fW * P         # which silently pinned the GT to a 16px encoder)
    objs = []
    for o in scene["objs"]:
        c = int(o["cls"])
        if c not in CLASS_ID.values():
            continue
        m = dilate(load_mask_hw(Image.open(o["mask_path"]), (mh, mw)))   # LAZY: load mask on demand at the tiled resolution
        if c == 1:                                          # faults are THIN — drop only DEGENERATE GT
            m01 = (m > 0.5).float()
            if float(m01.mean()) > 0.4 or float(m01.sum()) < 16:   # whole-image (>40%) OR empty/noise (<16px). ABSOLUTE
                continue                                           # px floor: the old 5e-4 FRACTION scaled with panel size
                                                                   # and dropped legitimate thin faults on big panels
                                                                   # (Smeaheia 50%); min real fault is ~50px (gt audit).
        x1, y1, x2, y2 = o["bbox"]
        ctr = torch.tensor([(y1 + y2) / 2, (x1 + x2) / 2], device=device, dtype=torch.float32)
        mfw = F.adaptive_avg_pool2d(m[None, None].float(), (fH, fW))[0, 0].clamp(0, 1)
        g = dict(cls=c, ctr=ctr, mask_fW=mfw, mask_full=m, bbox=[x1, y1, x2, y2])
        allowed = measures_for_id(c)                   # registry: which measures THIS class carries
        for slot, name in enumerate(MEASURE_SLOTS):    # present-gated per class + per data (mmask)
            g[name] = float(o["meas"][slot]) if (name in allowed and float(o["mmask"][slot]) > 0) else None
        g["derive"] = o.get("derive")                  # RAW dataset derive (closures only); loss present-gated
        objs.append(((x1 + x2) / 2, g))
    objs.sort(key=lambda t: t[0])
    return [g for _, g in objs]


class RegionReader(nn.Module):
    def __init__(self, vdim=768, d=256, layers=3, heads=8, max_steps=24, pixel_decoder=True):
        super().__init__()
        self.d, self.max_steps = d, max_steps
        self.real_adapter, self.use_real = None, False     # real-field adapter isolation (off by default)
        self.enc = None                                    # vision encoder (set by train/eval): pixels -> feature grid
        self.proj = nn.Linear(vdim, d)
        # Pixel decoder: global self-attention over the STITCHED map — reassembles
        # cross-tile context that per-tile NCS encoding loses, so a tall fault split
        # across tiles becomes whole-object BEFORE the reader reads it.
        self.pixdec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True), 2) if pixel_decoder else None
        self.pos = nn.Parameter(torch.zeros(1, d, 32, 32))     # learned 2-D pos GRID, interpolated to each scene's (fH,fW)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True)
        self.dec = nn.TransformerDecoder(dec, layers)
        # DETR set prediction: a FIXED set of learnable object queries (≥ max objects/scene). The decoder
        # cross-attends the grid in ONE parallel pass; each query → class (incl. ∅=NO_OBJ) + attrs + mask.
        # Hungarian matching assigns queries↔GT, unmatched → ∅. Same path train & inference (no teacher
        # forcing, no count head) → fixes the AR/count-head over-detection.
        self.N_QUERIES = N_QUERIES   # MUST be ≥ max objects/scene, else the model literally cannot
                                     # represent the scene and GT is silently DROPPED from the loss.
                                     # 12 was tuned on synthetic (≤10 objects) but CRACKS averages 19.3
                                     # faults/section (max 42) — 80% of sections exceeded 12 queries.
        self.query = nn.Parameter(torch.randn(1, self.N_QUERIES, d) * 0.02)
        self.mask_attn = False       # Mask2Former masked attention (query attends only to its occupancy region);
                                     # OFF by default — 24-scene proof gave mask dice 0.069 (likely early-occupancy
                                     # instability); turn ON only if a full-data train beats the mask baseline.
        self.cldice_w = 0.0          # clDice (thin-structure/centerline) weight added to the mask loss; 0 = off
        _tv = os.environ.get("TVERSKY", "0.4,0.6,1.0")   # SWEPT WINNER is now the DEFAULT (α,β,γ): additive Focal-Tversky
        self.tversky = tuple(float(x) for x in _tv.split(",")) if _tv else None   # (β>α penalizes over-prediction). TVERSKY="" disables.
        self.pos_weight_max = float(os.environ.get("POS_WEIGHT_MAX", 15.0))       # swept default 15 (was 50); env still overrides for sweeps
                                     # the over-prediction engine — sweep DOWN to thin masks. Default 15 (swept down from 50).
        self.class_head = nn.Linear(d, N_CLASS)                # ∅ / fault / closure / salt / onlap
        self.foot_q = nn.Linear(d, d)                          # pooling footprint (softmax → dip/pool)
        self.occ_q = nn.Linear(d, d)                           # occupancy footprint (sigmoid → mask/area)
        # TIER-1 MEASURE heads — built FROM the registry (one per MEASURE, shared across classes that
        # declare it). "spatial" heads read the 8-dim footprint stats (dip); "pooled" read the d-dim
        # pooled feature (throw/area). Add a measure = a registry row; NO code change here.
        self.measure_heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(8 if kind == "spatial" else d, 64), nn.GELU(), nn.Linear(64, 1))
            for name, (kind, _scale) in MEASURE.items()})
        # Mask head (Mask2Former-style): instance query · UPSAMPLED pixel-decoder features →
        # per-instance mask. Shares the trunk with the reader → mask loss co-trains the pixel
        # decoder (couples facts + masks in Stage 2; no LM dependency). The query is the content prompt.
        self.mask_q = nn.Linear(d, d)
        mlrs = [nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU()]
        for _ in range(3):
            mlrs += [nn.ConvTranspose2d(d, d, 4, stride=2, padding=1), nn.GroupNorm(8, d), nn.GELU()]
        self.mask_up = nn.Sequential(*mlrs, nn.Conv2d(d, d, 1))     # 8× upsample → mask features
        # TIER-2 DERIVED — ONE query-conditioned head for ALL derived attrs, SECTION- or OBJECT-scoped
        # (registry.DERIVED). Context = the section pool (section scope) or the per-object h_i (object
        # scope); the query index selects the attribute. Add a derived attribute = one registry row —
        # NEVER a new head. derived_section_ctx compresses the 2d section pool to the head's d-dim ctx.
        self.derived = DerivedHead(d, NUM_DERIVED, MAX_CAT)
        self.derived_section_ctx = nn.Linear(2 * d, d)

    def _section_ctx(self, memory, h):
        """Section-scope context = global memory ⊕ object-set summary (AFTER the tier-1 reads),
        compressed to d. Fed to the ONE derived head for the section-scoped queries."""
        g = memory.mean(1)[0]                                     # (d) scene embedding
        po = h[0].mean(0)                                         # ALL query states (`:-1` was an AR
                                                                  # "stop state" leftover that arbitrarily
                                                                  # excluded the last query under DETR)
        return F.gelu(self.derived_section_ctx(torch.cat([g, po])))   # (d)

    def _dloss(self, out, kind, val, labels, marker):
        """Loss for one derived attribute. cat -> CE (val = label index or word); bool -> BCE(logit,
        0/1); scalar -> smooth-L1 on value/scale (well-conditioned regression)."""
        if kind == "cat":
            idx = labels.index(val) if isinstance(val, str) else int(val)
            return F.cross_entropy(out.unsqueeze(0), torch.tensor([idx], device=device))
        if kind == "bool":
            return F.binary_cross_entropy_with_logits(out, torch.tensor(float(bool(val)), device=device))
        return F.smooth_l1_loss(out, torch.tensor(float(val) / SCALAR_SCALE.get(marker, 1.0), device=device))

    def set_encoder(self, enc):
        object.__setattr__(self, "enc", enc)               # plain attr, NOT a submodule → out of state_dict/parameters

    def encode(self, scene):
        """Loader-tiled scene -> stitched feature grid (vdim,fH,fW). The loader already cut the NATIVE
        image into tile_size² tiles (scene['tiles'], a BATCH) + each tile's grid offset (scene['tile_offs'])
        + the target grid (scene['grid']); here we run the encoder ONCE on the tile batch and SCATTER each
        tile's patch grid onto the (fH,fW) map (50%-overlap → averaged). No resize → native aspect, no
        distortion. Encoder frozen -> no_grad, unfrozen -> grad (one flag: enc.trainable_blocks)."""
        from hybrid.data.loader import _tile_image, encoder_tiling
        ts, pt = encoder_tiling()                              # SAME resolver the loader used for scene['grid']
        tiles, offs, (fH, fW) = _tile_image(scene["img"], ts, pt)   # LAZY: tile on demand — nothing held in RAM between scenes
        tiles = tiles.to(device)                               # (n_tiles, 3, T, T) — the encoder's clean input
        # enable_grad only when the encoder is unfrozen AND we're already in a grad context — an
        # unconditional enable_grad() OVERRODE the caller's @torch.no_grad in every eval path, building a
        # full encoder autograd graph per scene (a real OOM risk on the 5.67 GB GPU).
        want_grad = getattr(self.enc, "trainable_blocks", 0) > 0 and torch.is_grad_enabled()
        ctx = torch.enable_grad() if want_grad else torch.no_grad()
        with ctx:
            sp = self.enc(tiles, return_spatial=True)[1]        # (n_tiles, vdim, g, g)
        vdim, g = sp.shape[1], sp.shape[2]
        accum = torch.zeros(vdim, fH, fW, device=device)
        cnt = torch.zeros(1, fH, fW, device=device)
        for i, (fy, fx) in enumerate(offs):                    # place each tile's patches at its grid offset
            gy, gx = min(g, fH - fy), min(g, fW - fx)
            if gy <= 0 or gx <= 0:
                continue
            accum[:, fy:fy + gy, fx:fx + gx] += sp[i, :, :gy, :gx]
            cnt[:, fy:fy + gy, fx:fx + gx] += 1
        return accum / cnt.clamp_min(1.0)                      # (vdim, fH, fW) stitched map → _grid/pos/pixdec/DETR

    def _grid(self, smap):
        """smap (vdim,fH,fW) -> memory tokens (1,fHW,d), and the (row,col) coords."""
        smap = smap.to(device)                                 # no-op if already on GPU; moves CPU-offloaded real smaps
        fH, fW = smap.shape[1], smap.shape[2]
        m = self.proj(smap.flatten(1).t()).unsqueeze(0)        # (1, fHW, d)
        # 2-D positional encoding: interpolate the learned pos GRID to this scene's (fH,fW). Respects the
        # grid's true 2-D (row,col) structure at ANY resolution (NCS 31x6, SFM 162x32, real panels). The
        # old flat (1,4096,d) buffer 1-D-interpolated past 4096 tokens scrambled the grid → diffuse masks.
        pos = F.interpolate(self.pos, size=(fH, fW), mode="bicubic", align_corners=False)  # (1,d,fH,fW)
        m = m + pos.flatten(2).transpose(1, 2)                 # (1, fHW, d)
        if self.pixdec is not None:
            m = self.pixdec(m)                                 # cross-tile global attention
        if self.use_real and self.real_adapter is not None:    # real-field residual delta (base FROZEN)
            m = m + self.real_adapter(m)
        # TWO coordinate grids, because they answer different questions:
        #  · coord — each axis normalised to [0,1] INDEPENDENTLY, matching how GT bbox/centre are
        #    normalised (x/W, y/H). Used for the centroid mu so it is comparable to gt ctr.
        #  · iso   — a SHARED scale for both axes (cell units / max(fH,fW)), so ANGLES are preserved.
        #    Using `coord` for the dip covariance warped every non-square panel: a true 45deg fault
        #    read as 70deg on a 32x87 CRACKS grid and 18deg on a 96x32 Smeaheia panel. Dip is only
        #    meaningful in an isotropic space. (Cells are square in pixels — one patch — so cell
        #    units ARE physical units.) Centres use (i+0.5)/n, not i/(n-1), to sit mid-cell.
        ri = (torch.arange(fH, device=device).float() + 0.5)
        ci = (torch.arange(fW, device=device).float() + 0.5)
        rr, cc = torch.meshgrid(ri / fH, ci / fW, indexing="ij")
        coord = torch.stack([rr.flatten(), cc.flatten()], -1)  # (fHW, 2) GT-normalised
        s = float(max(fH, fW))
        ir, ic = torch.meshgrid(ri / s, ci / s, indexing="ij")
        iso = torch.stack([ir.flatten(), ic.flatten()], -1)    # (fHW, 2) angle-preserving
        return m, coord, iso, (fH, fW)

    def add_real_adapter(self, r=32, train_mask=True,
                         train_class=False, train_measure=False, train_derived=False):
        """ADAPTER ISOLATION for real-field finetune. FREEZE the synthetic reader (trunk + detection +
        attribute heads → every synthetic class/attribute preserved) and add a zero-init residual REAL
        adapter on the grid features (starts as identity). With train_mask (default), ALSO unfreeze the
        MASK DECODER (mask_q query + mask_up pixel-decoder upsampler): syn→real, the mask decoder itself
        carries the appearance gap (real fault masks look different), so it must adapt to real.

        TOGGLEABLE VALUE HEADS (per-domain decoders → no forgetting cost, since each checkpoint only ever
        serves its own survey). Unfreeze exactly the heads for which THIS domain has GT, so the reader is
        TRAINED to emit domain-correct values instead of the frozen SYNTHETIC prior (which is what drives
        the LM's OOD malformation — dip over-read, class misfire, confabulated relations). Do NOT unfreeze
        foot_q/occ_q here: they shape the mask geometry, so training them destabilises the mask; measure
        heads recalibrate dip from the FROZEN geometry stats.
          · train_class   — class word (single-class real → collapses to the correct 'fault')
          · train_measure — dip/throw map (recalibrates the synthetic dip over-read to the domain)
          · train_derived — relational tier + section ctx (needs REAL relational GT; on surveys without
                            relational labels this must come from GEOMETRY, else it has nothing to learn)
        Defaults keep the historical behavior (mask only). Returns the trainable params."""
        for p in self.parameters():
            p.requires_grad_(False)                            # freeze ALL synthetic params first
        self.real_adapter = nn.Sequential(
            nn.Linear(self.d, r), nn.GELU(), nn.Linear(r, self.d)).to(device)
        nn.init.zeros_(self.real_adapter[-1].weight)           # start = identity (residual delta = 0)
        nn.init.zeros_(self.real_adapter[-1].bias)
        self.use_real = True
        params = list(self.real_adapter.parameters())
        groups = []
        if train_mask:                                         # the syn→real mask-decoder gap
            groups += [self.mask_q, self.mask_up]
        if train_class:                                        # domain-correct class word
            groups += [self.class_head]
        if train_measure:                                      # domain-correct dip/throw
            groups += [self.measure_heads]
        if train_derived:                                      # domain-correct relational tier
            groups += [self.derived, self.derived_section_ctx]
        for mod in groups:
            for p in mod.parameters():
                p.requires_grad_(True); params.append(p)
        return params

    def _readout(self, h, memory, coord, iso):
        """h (B,T,d) decoder states -> per-step heads. Geometry (centroid/orientation/extent) is read
        from the SUPERVISED occupancy map; the pooled feature carries magnitudes (throw)."""
        logits = self.foot_q(h) @ memory.transpose(1, 2)        # (B,T,fHW) pooling scores
        w = logits.softmax(-1)                                  # pooling weights (→ feature pooling)
        occ_logits = self.occ_q(h) @ memory.transpose(1, 2)     # (B,T,fHW) occupancy (own head)
        foot = occ_logits.sigmoid()                             # per-cell occupancy (mask/area/geometry)
        pooled = w @ memory                                     # (B,T,d) — magnitudes (throw)
        # GEOMETRY comes from `foot`, the occupancy map the mask/foot loss actually SUPERVISES.
        # It used to come from `w` (the softmax pooling map), which no shape loss ever touches — so the
        # second moments the dip head reads were shaped by nothing but the dip loss itself, contradicting
        # the project's own rule that dip IS the mask's orientation.
        wn = foot / foot.sum(-1, keepdim=True).clamp_min(1e-6)  # occupancy as a distribution
        mu = wn @ coord                                        # (B,T,2) centroid in GT-normalised coords
        mi = wn @ iso                                          # centroid in isotropic coords
        d0 = iso.unsqueeze(0).unsqueeze(0) - mi.unsqueeze(2)   # (B,T,fHW,2) angle-preserving offsets
        cov = torch.einsum("btni,btnj,btn->btij", d0, d0, wn)  # (B,T,2,2)
        # occupied FRACTION of the grid — this IS the area, and it was previously unavailable to the
        # area head (which read `pooled`, a convex combination that is scale-invariant by construction
        # and therefore cannot represent extent at all).
        occ_frac = foot.mean(-1)
        stats = torch.stack([mu[..., 0], mu[..., 1], cov[..., 0, 0], cov[..., 1, 1],
                             cov[..., 0, 1], (mu[..., 0] - .5), (mu[..., 1] - .5), occ_frac], -1)
        # REGISTRY-DRIVEN measures: each head reads stats (spatial) or pooled (magnitude) per MEASURE.
        meas = {name: self.measure_heads[name](stats if kind == "spatial" else pooled).squeeze(-1)
                for name, (kind, _s) in MEASURE.items()}
        return dict(cls=self.class_head(h),
                    meas=meas, foot=foot, foot_logits=occ_logits, mu=mu)

    def _mask_features(self, memory, fH, fW):
        """Pixel-decoder trunk features (1,fHW,d) → upsampled per-pixel mask features (1,d,H',W')."""
        return self.mask_up(memory.transpose(1, 2).reshape(1, self.d, fH, fW))

    def _decode(self, memory):
        """Decoder pass over the query set. With mask_attn (Mask2Former MASKED ATTENTION), each layer
        restricts a query's cross-attention to the grid cells its OCCUPANCY predicted in the previous
        layer — sharper, better-oriented masks at fixed grid resolution + steadier query specialization.
        Empty-mask queries fall back to full attention (avoids a NaN softmax)."""
        h = self.query
        attn = None
        for layer in self.dec.layers:
            h = layer(h, memory, memory_mask=attn)
            if not self.mask_attn:
                continue
            occ = (self.occ_q(h) @ memory.transpose(1, 2)).sigmoid()[0]     # (N, fHW) per-query occupancy
            m = occ < 0.5                                                   # True = don't attend
            attn = (m & ~m.all(-1, keepdim=True)).detach()                 # un-mask fully-empty queries
        return h

    def forward(self, smap, gt, derived=None):
        """Teacher-forced loss on one scene. gt = list of dicts {cls, dip, throw?, area?,
        ctr(2), mask_fW(fH,fW), mask_full(H,W)}. derived = scene-level {intersect?, mode?} tier-2
        GT (registry). Returns (loss, parts)."""
        memory, coord, iso, (fH, fW) = self._grid(smap)
        K = len(gt)
        N = self.N_QUERIES
        h = self._decode(memory)                       # (1,N,d) — fixed queries, PARALLEL, no teacher forcing
        out = self._readout(h, memory, coord, iso)                  # per-query: cls (∅+4) · meas · foot · mu
        # HUNGARIAN MATCHING — N predictions ↔ K GT (class-prob + centroid-L1 cost). Unmatched → ∅.
        gt_cls = torch.tensor([o["cls"] for o in gt], device=device) if K else torch.zeros(0, dtype=torch.long, device=device)
        row = col = None
        if K:
            with torch.no_grad():
                prob = out["cls"][0].softmax(-1)               # (N, N_CLASS)
                gt_ctr = torch.stack([o["ctr"] for o in gt])   # (K, 2)
                cost = (-prob[:, gt_cls] + torch.cdist(out["mu"][0], gt_ctr, p=1)).cpu().numpy()
                ri, ci = linear_sum_assignment(cost)
            row = torch.as_tensor(ri, device=device); col = torch.as_tensor(ci, device=device)
        # CLASS loss over ALL queries: default ∅ (NO_OBJ), matched → GT class; ∅ down-weighted (DETR eos_coef).
        tgt_cls = torch.zeros(N, dtype=torch.long, device=device)
        if K:
            tgt_cls[row] = gt_cls[col]
        cw = torch.ones(N_CLASS, device=device); cw[NO_OBJ] = 0.1
        L = F.cross_entropy(out["cls"][0], tgt_cls, weight=cw)
        parts = {"cls": L.item()}
        if K:
            gd = [gt[c] for c in col.tolist()]                 # GT objects in matched (query) order
            # CENTROID loss (DETR L1) — pull each matched query's mu to its GT centre. mu was previously
            # UNSUPERVISED (it only entered the DETACHED matching cost) → degenerate: every detection's
            # centre collapsed to a panel corner, making reader_facts' "center" garbage and centre-based
            # matching read 0 TP. This term makes mu a real localization signal.
            ctr_gt = torch.stack([o["ctr"] for o in gd]).to(device)          # (K,2) GT (row,col) normalized
            cl = F.l1_loss(out["mu"][0, row], ctr_gt)
            L = L + 5.0 * cl; parts["ctr"] = cl.item()
            gm = torch.stack([o["mask_fW"] for o in gd]).to(device).flatten(1).clamp(0, 1)  # (K,fHW)
            occ = out["foot"][0, row]; pw = torch.tensor([8.0], device=device)   # was 20 → tighter occupancy
                                                                                # (footprint was 40× over-covering)
            fp = (F.binary_cross_entropy_with_logits(out["foot_logits"][0, row], gm, pos_weight=pw)
                  + _dice_loss(occ, gm))                                        # pos-BCE + PER-INSTANCE dice
            L = L + fp; parts["foot"] = fp.item()
            # REGISTRY class-driven measures (matched queries only): supervise a MEASURE iff the matched
            # object's CLASS declares it AND the GT value is present. One loop covers dip/throw/area.
            for name, (_kind, scale) in MEASURE.items():
                sel = torch.tensor([name in measures_for_id(o["cls"]) and o.get(name) is not None
                                    for o in gd], device=device)
                if sel.any():
                    gv = torch.tensor([o.get(name) or 0.0 for o in gd], device=device)
                    ml_ = F.smooth_l1_loss(out["meas"][name][0, row][sel], gv[sel] / scale)
                    L = L + ml_; parts[name] = ml_.item()
            mfull = [o["mask_full"] for o in gd if o.get("mask_full") is not None]   # filter, don't just test
            if mfull:
                mfeat = self._mask_features(memory, fH, fW)            # (1,d,H',W')
                ml = torch.einsum("kd,dhw->khw", self.mask_q(h[0, row]), mfeat[0])  # (K,H',W')
                Ht, Wt = mfull[0].shape
                ml = F.interpolate(ml.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False)[0]
                gm2 = torch.stack([mm.to(device) for mm in mfull]).float().clamp(0, 1)
                # pos_weight from the ACTUAL positive rate of this scene's masks (clamped), instead of a
                # hand-tuned constant: a thin fault is ~1-2% of the panel (≈70:1), so a fixed 15 left
                # positives at ~17% of the BCE — and any retune silently invalidated dice comparisons
                # taken at a fixed 0.5 threshold.
                pos_rate = gm2.mean().clamp(1e-4, 0.5)
                # pos_weight upper clamp is the OVER-PREDICTION engine (up-weights positives → fat masks);
                # POS_WEIGHT_MAX env-knob so it can be swept DOWN. Tversky is now ADDITIVE to Dice (Dice for
                # stable overlap + Focal-Tversky for the FP penalty), not a replacement.
                p2 = ml.sigmoid(); pw2 = ((1 - pos_rate) / pos_rate).clamp(1.0, self.pos_weight_max)
                mk = (F.binary_cross_entropy_with_logits(ml, gm2, pos_weight=pw2)
                      + _dice_loss(p2, gm2)
                      + (tversky_loss(p2, gm2, *self.tversky) if self.tversky else 0.0))
                if self.cldice_w:                                 # thin-structure (centerline) term
                    mk = mk + self.cldice_w * cldice(p2, gm2)
                L = L + mk; parts["mask"] = mk.item()
        # TIER-2 DERIVED losses — SECTION-scoped (section pool) + OBJECT-scoped (matched query h_i).
        if derived:
            sctx = self._section_ctx(memory, h)
            for idx, key, marker, kind, labels in SECTION_DERIVED:
                if derived.get(marker) is None:
                    continue
                o_ = self.derived(sctx.unsqueeze(0), torch.tensor([idx], device=device), kind)[0]
                dl = self._dloss(o_, kind, derived[marker], labels, marker)
                L = L + dl; parts[marker] = dl.item()
        if K:
            for qi, o in zip(row.tolist(), gd):
                raw = o.get("derive")
                if not raw:
                    continue
                for idx, key, marker, kind, labels, klass in OBJECT_DERIVED:
                    if o["cls"] != CLASS_ID[klass] or raw.get(key) is None:
                        continue
                    o_ = self.derived(h[0, qi].unsqueeze(0), torch.tensor([idx], device=device), kind)[0]
                    dl = self._dloss(o_, kind, raw[key], labels, marker)
                    L = L + dl; parts[f"{marker}_{qi}"] = dl.item()
        return L, parts

    @torch.no_grad()
    def tf_masks(self, smap, gt):
        """Per-GT-object mask logits (interp to GT mask size) — for mask eval. DETR: one query pass,
        Hungarian-match queries↔GT, return the matched queries' masks in GT order."""
        memory, coord, iso, (fH, fW) = self._grid(smap)
        h = self._decode(memory)
        out = self._readout(h, memory, coord, iso)
        prob = out["cls"][0].softmax(-1)
        gt_cls = torch.tensor([o["cls"] for o in gt], device=device)
        gt_ctr = torch.stack([o["ctr"] for o in gt])
        cost = (-prob[:, gt_cls] + torch.cdist(out["mu"][0], gt_ctr, p=1)).cpu().numpy()
        ri, ci = linear_sum_assignment(cost)
        # Scatter matched queries back to GT-INDEX order. `ci` is a SUBSET of GT indices when K > N_QUERIES
        # (CRACKS averages 19 faults/section), so indexing by matched RANK silently mis-paired masks to GT.
        # Unmatched GT gets a large negative logit (an empty mask) — scored as a miss, which is honest.
        Ht, Wt = gt[0]["mask_full"].shape
        mfeat = self._mask_features(memory, fH, fW)[0]
        mq = torch.einsum("kd,dhw->khw", self.mask_q(h[0, torch.as_tensor(ri, device=device)]), mfeat)
        full = torch.full((len(gt),) + tuple(mq.shape[1:]), -20.0, device=device)
        full[torch.as_tensor(ci, device=device)] = mq
        return F.interpolate(full.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False)[0]

    @torch.no_grad()
    def _decode_section(self, memory, h):
        """Decode all SECTION-scoped derived queries → {marker: value}: cat → label index, bool →
        True/False, scalar → int (×scale). Keys/forms match registry.derived_facts' expectations."""
        sctx = self._section_ctx(memory, h)
        out = {}
        for idx, key, marker, kind, labels in SECTION_DERIVED:
            o = self.derived(sctx.unsqueeze(0), torch.tensor([idx], device=device), kind)[0]
            if kind == "cat":
                out[marker] = int(o[:len(labels)].argmax())    # shared head is MAX_CAT-wide; clip to this attr
            elif kind == "bool":
                out[marker] = bool(torch.sigmoid(o) > 0.5)
            else:
                out[marker] = max(0, int(round(float(o) * SCALAR_SCALE.get(marker, 1.0))))
        return out

    @torch.no_grad()
    def _decode_object(self, h_i):
        """Decode all OBJECT-scoped derived queries from one object's h_i → {marker: word/yes-no} —
        the SAME fact-word form registry.object_derived_facts produces from GT (so both inject alike)."""
        out = {}
        for idx, key, marker, kind, labels, klass in OBJECT_DERIVED:
            o = self.derived(h_i.unsqueeze(0), torch.tensor([idx], device=device), kind)[0]
            if kind == "cat":
                out[marker] = labels[int(o[:len(labels)].argmax())]   # clip shared MAX_CAT head to this attr
            elif kind == "bool":
                out[marker] = "yes" if float(torch.sigmoid(o)) > 0.5 else "no"
            else:
                out[marker] = max(0, int(round(float(o) * SCALAR_SCALE.get(marker, 1.0))))
        return out

    @torch.no_grad()
    def read_derived(self, smap):
        """Scene-level tier-2 SECTION read for inference → {intersect,mode,nclosure,nonlap,salt}.
        DETR: one parallel query pass, then decode the section-scoped derived from the query states."""
        memory, coord, iso, (fH, fW) = self._grid(smap)
        h = self._decode(memory)
        return self._decode_section(memory, h)

    @torch.no_grad()
    def object_states(self, smap, gt):
        """Per-object hidden state h_i (teacher-forced over gt, DETACHED) — the shared object
        representation that also drives the mask head. Source for the narrator's <feature>_i
        (reader stays frozen/GT-trained; the LM only READS h_i, never reshapes it). Returns
        (K, d) aligned 1:1 with gt order; also returns each object's normalized centroid for
        matching to the fact objects."""
        memory, coord, iso, (fH, fW) = self._grid(smap)
        K = len(gt)
        if K == 0:
            return torch.zeros(0, self.d, device=device), []
        # Use the SAME trained decode path as forward()/detect(), then Hungarian-match queries↔GT.
        # This previously ran an autoregressive path (`_seq_embed` → bos/obj_cls/obj_ctr under a causal
        # mask) whose embeddings receive NO gradient anywhere — only `_decode`'s fixed `self.query` is
        # ever trained. So the h_i handed to the LM's <feature> token was the output of untrained
        # embeddings under a mask the decoder never saw in training: out-of-distribution noise. That is
        # a mechanical explanation for "<feature> is inert", independent of the LM seam.
        h = self._decode(memory)
        out = self._readout(h, memory, coord, iso)
        prob = out["cls"][0].softmax(-1)
        gt_cls = torch.tensor([o["cls"] for o in gt], device=device)
        gt_ctr = torch.stack([o["ctr"] for o in gt])
        ri, ci = linear_sum_assignment(
            (-prob[:, gt_cls] + torch.cdist(out["mu"][0], gt_ctr, p=1)).cpu().numpy())
        hs = torch.zeros(K, self.d, device=device)
        hs[torch.as_tensor(ci, device=device)] = h[0, torch.as_tensor(ri, device=device)]
        centroids = [(float(o["ctr"][0]), float(o["ctr"][1])) for o in gt]   # (row, col) normalized
        return hs.detach(), centroids

    @torch.no_grad()
    def detect(self, smap, thresh=0.9, want_masks=False):
        """DETR decode → list of {cls, dip, throw, area, ctr, bbox, derive}. One PARALLEL query pass;
        keep queries whose argmax class ≠ ∅ (NO_OBJ) with confidence > thresh (no count head, no AR
        loop). With want_masks, also returns per-kept-query mask logits (H',W'). thresh DEFAULT 0.9 =
        the swept operating point (det/scene≈GT/scene, best count-MAE + best IoU-F1); 0.5 over-fired 2.6×."""
        memory, coord, iso, (fH, fW) = self._grid(smap)
        h = self._decode(memory)                        # (1,N,d)
        out = self._readout(h, memory, coord, iso)
        mfeat = self._mask_features(memory, fH, fW) if want_masks else None
        prob = out["cls"][0].softmax(-1)                        # (N,N_CLASS)
        cls = prob[:, 1:].argmax(-1) + 1                        # best NON-∅ class
        conf = 1.0 - prob[:, NO_OBJ]                            # objectness = P(not ∅)
        # DETR-correct: threshold OBJECTNESS, not max-over-all-classes. `eos_coef=0.1` deliberately
        # under-trains ∅, so max-prob was systematically deflated and 0.9 was silently compensating for
        # it — a query at p(fault)=0.85 / p(∅)=0.10 was discarded despite ∅ being clearly rejected.
        # This form is stable under changes to the class weight.
        keep = (conf > thresh).nonzero(as_tuple=True)[0]
        objs, masks = [], []
        for qi in keep.tolist():
            c = int(cls[qi])
            occ = out["foot"][0, qi] > 0.5                            # footprint occupancy -> bbox
            if occ.any():
                cc = coord[occ]
                bb = [cc[:, 1].min(), cc[:, 0].min(), cc[:, 1].max(), cc[:, 0].max()]  # x1,y1,x2,y2 (0-1)
            else:
                mu = out["mu"][0, qi]; bb = [mu[1], mu[0], mu[1], mu[0]]
            od = self._decode_object(h[0, qi]) if c == CLOSURE else None   # object-scoped derived words
            mv = {n: float(out["meas"][n][0, qi] * MEASURE[n][1]) for n in MEASURE}   # registry-scaled measures
            objs.append(dict(cls=c, dip=mv.get("dip", 0.0), throw=mv.get("throw", 0.0),
                             area=mv.get("area", 0.0), meas=mv,
                             ctr=out["mu"][0, qi].detach(), derive=od,
                             bbox=[int(float(v) * 100) for v in bb]))     # 0-100, same scale as GT
            if want_masks:
                masks.append(torch.einsum("kd,dhw->khw", self.mask_q(h[0, qi:qi + 1]), mfeat[0])[0])
        return (objs, masks) if want_masks else objs
