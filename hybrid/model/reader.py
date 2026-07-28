"""Instance reader — autoregressive, query-free detector over frozen NCS features.

Replaces the dense-seg + RANSAC front-end (measure-only; masks live on the <SEG>
decoder). Emits objects as a SEQUENCE (emit-until-stop → cap-free): per step a
class + a soft footprint over the feature grid + class-driven LEARNABLE attribute
heads. THE RULE (reference_dip_from_mask_geometry): angle/shape attrs read the
SPATIAL footprint (dip from its 2nd-moment stats), magnitudes read POOLED features
(throw). A dip head on a pooled scalar collapses to 73° — never do that.

Whole-object attention: the decoder attends over ALL grid tokens of a window.
Teacher-forced during training (GT objects ordered by x-centre → no Hungarian).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid.model.registry import (NUM_DERIVED, MAX_CAT, N_CLASS, SCALAR_SCALE, FLUID_LABELS,
                                    SECTION_DERIVED, OBJECT_DERIVED, CLASS_ID, AREA_CLASSES,
                                    MEASURE, MEASURE_SLOTS, MEASURE_SCALE, measures_for_id)
from hybrid.model.derived_head import DerivedHead

device = torch.device("cuda")
NO_OBJ, FAULT, CLOSURE, SALT, ONLAP = 0, 1, 2, 3, 4       # class ids (∅ / fault / closure / salt / onlap)


def scene_to_gt(scene):
    """Scene objs -> reader GT list {cls, ctr, mask_fW, dip?, throw?, area?, derive?}, ordered by
    x-centre. Multi-class: fault(1) dip/throw · closure(2)/salt(3)/onlap(4) area · closures carry the
    RAW per-object derive dict (fluid/intersects_*) for the object-scoped derived head."""
    smap = scene["smap"]; fH, fW = smap.shape[1], smap.shape[2]
    objs = []
    for o in scene["objs"]:
        c = int(o["cls"])
        if c not in CLASS_ID.values():
            continue
        x1, y1, x2, y2 = o["bbox"]
        ctr = torch.tensor([(y1 + y2) / 2, (x1 + x2) / 2], device=device, dtype=torch.float32)
        mfw = F.adaptive_avg_pool2d(o["mask"][None, None].float(), (fH, fW))[0, 0].clamp(0, 1)
        g = dict(cls=c, ctr=ctr, mask_fW=mfw, mask_full=o["mask"])
        allowed = measures_for_id(c)                   # registry: which measures THIS class carries
        for slot, name in enumerate(MEASURE_SLOTS):    # present-gated per class + per data (mmask)
            g[name] = float(o["meas"][slot]) if (name in allowed and float(o["mmask"][slot]) > 0) else None
        g["derive"] = o.get("derive")                  # RAW dataset derive (closures only); loss present-gated
        objs.append(((x1 + x2) / 2, g))
    objs.sort(key=lambda t: t[0])
    return [g for _, g in objs]


class InstanceReader(nn.Module):
    def __init__(self, vdim=768, d=256, layers=3, heads=8, max_steps=24, pixel_decoder=True):
        super().__init__()
        self.d, self.max_steps = d, max_steps
        self.real_adapter, self.use_real = None, False     # real-field adapter isolation (off by default)
        self.proj = nn.Linear(vdim, d)
        # Pixel decoder: global self-attention over the STITCHED map — reassembles
        # cross-tile context that per-tile NCS encoding loses, so a tall fault split
        # across tiles becomes whole-object BEFORE the reader reads it.
        self.pixdec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True), 2) if pixel_decoder else None
        self.pos = nn.Parameter(torch.zeros(1, 4096, d))       # grid pos-enc (flattened)
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        self.obj_cls = nn.Embedding(N_CLASS, d)                 # embed a previous object's class (∅/4 classes)
        self.obj_ctr = nn.Linear(2, d)                         # + its footprint centroid
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True)
        self.dec = nn.TransformerDecoder(dec, layers)
        self.stop_head = nn.Linear(d, 1)
        self.count_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))  # global mem -> object count
        self.class_head = nn.Linear(d, N_CLASS)                # ∅ / fault / closure / salt / onlap
        self.foot_q = nn.Linear(d, d)                          # pooling footprint (softmax → dip/pool)
        self.occ_q = nn.Linear(d, d)                           # occupancy footprint (sigmoid → mask/area)
        # TIER-1 MEASURE heads — built FROM the registry (one per MEASURE, shared across classes that
        # declare it). "spatial" heads read the 7-dim footprint stats (dip); "pooled" read the d-dim
        # pooled feature (throw/area). Add a measure = a registry row; NO code change here.
        self.measure_heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(7 if kind == "spatial" else d, 64), nn.GELU(), nn.Linear(64, 1))
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
        po = h[0, :-1].mean(0) if h.shape[1] > 1 else torch.zeros(self.d, device=device)  # object states
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

    def _grid(self, smap):
        """smap (vdim,fH,fW) -> memory tokens (1,fHW,d), and the (row,col) coords."""
        smap = smap.to(device)                                 # no-op if already on GPU; moves CPU-offloaded real smaps
        fH, fW = smap.shape[1], smap.shape[2]
        m = self.proj(smap.flatten(1).t())                     # (fHW, d)
        n = m.shape[0]
        pos = self.pos                                         # learned grid pos-enc (1, 4096, d)
        if n > pos.shape[1]:                                   # big real panels exceed the buffer → interpolate up
            pos = F.interpolate(pos.transpose(1, 2), size=n, mode="linear", align_corners=False).transpose(1, 2)
        m = (m + pos[:, :n].squeeze(0)).unsqueeze(0)           # (1, fHW, d)
        if self.pixdec is not None:
            m = self.pixdec(m)                                 # cross-tile global attention
        if self.use_real and self.real_adapter is not None:    # real-field residual delta (base FROZEN)
            m = m + self.real_adapter(m)
        rr, cc = torch.meshgrid(torch.linspace(0, 1, fH, device=device),
                                torch.linspace(0, 1, fW, device=device), indexing="ij")
        coord = torch.stack([rr.flatten(), cc.flatten()], -1)  # (fHW, 2)
        return m, coord, (fH, fW)

    def add_real_adapter(self, r=32, train_mask=True):
        """ADAPTER ISOLATION for real-field finetune. FREEZE the synthetic reader (trunk + detection +
        attribute heads → every synthetic class/attribute preserved) and add a zero-init residual REAL
        adapter on the grid features (starts as identity). With train_mask (default), ALSO unfreeze the
        MASK DECODER (mask_q query + mask_up pixel-decoder upsampler): syn→real, the mask decoder itself
        carries the appearance gap (real fault masks look different), so it must adapt to real — the
        feature adapter alone can't close it. Detection/class/attr heads stay frozen (isolation still
        prevents forgetting on the sparse real windows). Returns the trainable params."""
        for p in self.parameters():
            p.requires_grad_(False)                            # freeze ALL synthetic params first
        self.real_adapter = nn.Sequential(
            nn.Linear(self.d, r), nn.GELU(), nn.Linear(r, self.d)).to(device)
        nn.init.zeros_(self.real_adapter[-1].weight)           # start = identity (residual delta = 0)
        nn.init.zeros_(self.real_adapter[-1].bias)
        self.use_real = True
        params = list(self.real_adapter.parameters())
        if train_mask:                                         # the syn→real mask-decoder gap
            for mod in (self.mask_q, self.mask_up):
                for p in mod.parameters():
                    p.requires_grad_(True); params.append(p)
        return params

    def _readout(self, h, memory, coord):
        """h (B,T,d) decoder states -> per-step heads. Footprint over the grid gives
        the spatial stats for dip; pooled feature gives throw/area."""
        logits = self.foot_q(h) @ memory.transpose(1, 2)        # (B,T,fHW) pooling scores
        w = logits.softmax(-1)                                  # pooling weights (→ dip/pool)
        occ_logits = self.occ_q(h) @ memory.transpose(1, 2)     # (B,T,fHW) occupancy (own head)
        foot = occ_logits.sigmoid()                             # per-cell occupancy (mask/area)
        pooled = w @ memory                                     # (B,T,d)
        # spatial 2nd-moment stats of the footprint -> dip (angle-preserving input)
        mu = w @ coord                                         # (B,T,2) centroid
        d0 = coord.unsqueeze(0).unsqueeze(0) - mu.unsqueeze(2)  # (B,T,fHW,2)
        cov = torch.einsum("btni,btnj,btn->btij", d0, d0, w)   # (B,T,2,2)
        stats = torch.stack([mu[..., 0], mu[..., 1], cov[..., 0, 0], cov[..., 1, 1],
                             cov[..., 0, 1], (mu[..., 0] - .5), (mu[..., 1] - .5)], -1)
        # REGISTRY-DRIVEN measures: each head reads stats (spatial) or pooled (magnitude) per MEASURE.
        meas = {name: self.measure_heads[name](stats if kind == "spatial" else pooled).squeeze(-1)
                for name, (kind, _s) in MEASURE.items()}
        return dict(stop=self.stop_head(h).squeeze(-1), cls=self.class_head(h),
                    meas=meas, foot=foot, foot_logits=occ_logits, mu=mu)

    def _mask_features(self, memory, fH, fW):
        """Pixel-decoder trunk features (1,fHW,d) → upsampled per-pixel mask features (1,d,H',W')."""
        return self.mask_up(memory.transpose(1, 2).reshape(1, self.d, fH, fW))

    def _seq_embed(self, classes, centroids):
        """Embed a prefix of GT/emitted objects for teacher forcing / AR decode."""
        classes = classes.to(torch.long)
        e = self.obj_cls(classes) + self.obj_ctr(centroids)    # (B,K,d)
        return torch.cat([self.bos.expand(e.shape[0], -1, -1), e], 1)  # prepend BOS

    def forward(self, smap, gt, derived=None):
        """Teacher-forced loss on one scene. gt = list of dicts {cls, dip, throw?, area?,
        ctr(2), mask_fW(fH,fW), mask_full(H,W)}. derived = scene-level {intersect?, mode?} tier-2
        GT (registry). Returns (loss, parts)."""
        memory, coord, (fH, fW) = self._grid(smap)
        K = len(gt)
        cls = torch.tensor([o["cls"] for o in gt], device=device).unsqueeze(0)
        ctr = torch.stack([o["ctr"] for o in gt]).unsqueeze(0) if K else torch.zeros(1, 0, 2, device=device)
        tgt = self._seq_embed(cls, ctr)                        # (1,K+1,d)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        h = self.dec(tgt, memory, tgt_mask=mask)               # (1,K+1,d); step t predicts object t
        out = self._readout(h, memory, coord)

        # COUNT HEAD (direct): predict N from the global memory and emit exactly N — robust for
        # high counts, replaces the fragile per-step emit-until-stop that over-emitted on multi-object.
        cpred = self.count_head(memory.mean(1))[0, 0]
        L = F.smooth_l1_loss(cpred, torch.tensor(float(K), device=device))
        parts = {"count": L.item()}
        if K:
            cl = F.cross_entropy(out["cls"][0, :K], cls[0])
            L = L + cl; parts["cls"] = cl.item()
            gm = torch.stack([o["mask_fW"] for o in gt]).to(device).flatten(1).clamp(0, 1)  # (K,fHW)
            pw = torch.tensor([20.0], device=device)
            occ = out["foot"][0, :K]
            fp = (F.binary_cross_entropy_with_logits(out["foot_logits"][0, :K], gm, pos_weight=pw)
                  + 1 - (2 * (occ * gm).sum() + 1) / (occ.sum() + gm.sum() + 1))    # pos-BCE + dice
            L = L + fp; parts["foot"] = fp.item()
            # REGISTRY-DRIVEN measure losses: for each MEASURE, supervise only the objects whose CLASS
            # declares it (CLASS_SCHEMA via measures_for_id) AND whose GT value is present. One loop
            # covers dip/throw/area and any future measure — no per-attribute code.
            for name, (_kind, scale) in MEASURE.items():
                sel = torch.tensor([name in measures_for_id(o["cls"]) and o.get(name) is not None
                                    for o in gt], device=device)
                if sel.any():
                    gv = torch.tensor([o.get(name) or 0.0 for o in gt], device=device)
                    ml_ = F.smooth_l1_loss(out["meas"][name][0, :K][sel], gv[sel] / scale)
                    L = L + ml_; parts[name] = ml_.item()
            mfull = [o.get("mask_full") for o in gt]
            if any(mm is not None for mm in mfull):
                mfeat = self._mask_features(memory, fH, fW)            # (1,d,H',W')
                ml = torch.einsum("kd,dhw->khw", self.mask_q(h[0, :K]), mfeat[0])  # (K,H',W')
                Ht, Wt = mfull[0].shape
                ml = F.interpolate(ml.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False)[0]
                gm2 = torch.stack([mm.to(device) for mm in mfull]).float().clamp(0, 1)
                p2 = ml.sigmoid(); pw2 = torch.tensor([40.0], device=device)
                mk = (F.binary_cross_entropy_with_logits(ml, gm2, pos_weight=pw2)
                      + 1 - (2 * (p2 * gm2).sum() + 1) / (p2.sum() + gm2.sum() + 1))
                L = L + mk; parts["mask"] = mk.item()
        # TIER-2 DERIVED losses — supervised by their OWN GT, never by the LM (same digit-seam
        # contract as dip). ONE head, two scopes. Present-only (a missing attribute is skipped).
        if derived:                              # SECTION-scoped: context = section pool
            sctx = self._section_ctx(memory, h)
            for idx, key, marker, kind, labels in SECTION_DERIVED:
                if derived.get(marker) is None:
                    continue
                out = self.derived(sctx.unsqueeze(0), torch.tensor([idx], device=device), kind)[0]
                dl = self._dloss(out, kind, derived[marker], labels, marker)
                L = L + dl; parts[marker] = dl.item()
        if K:                                    # OBJECT-scoped: context = per-object h_i = h[0, i]
            for i, o in enumerate(gt):
                raw = o.get("derive")
                if not raw:
                    continue
                for idx, key, marker, kind, labels, klass in OBJECT_DERIVED:
                    if o["cls"] != CLASS_ID[klass] or raw.get(key) is None:
                        continue
                    out = self.derived(h[0, i].unsqueeze(0), torch.tensor([idx], device=device), kind)[0]
                    dl = self._dloss(out, kind, raw[key], labels, marker)
                    L = L + dl; parts[f"{marker}_{i}"] = dl.item()
        return L, parts

    @torch.no_grad()
    def tf_masks(self, smap, gt):
        """Teacher-forced per-object mask logits (interp to GT mask size) — for mask eval."""
        memory, coord, (fH, fW) = self._grid(smap)
        K = len(gt)
        cls = torch.tensor([o["cls"] for o in gt], device=device).unsqueeze(0)
        cls = cls.long()
        ctr = torch.stack([o["ctr"] for o in gt]).unsqueeze(0)
        tgt = self._seq_embed(cls, ctr)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        h = self.dec(tgt, memory, tgt_mask=mask)
        ml = torch.einsum("kd,dhw->khw", self.mask_q(h[0, :K]), self._mask_features(memory, fH, fW)[0])
        Ht, Wt = gt[0]["mask_full"].shape
        return F.interpolate(ml.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False)[0]

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
        Decodes the object sequence (count head → N) to summarize the tier-1 reads, then the queries."""
        memory, coord, (fH, fW) = self._grid(smap)
        N = int(round(float(self.count_head(memory.mean(1))[0, 0].clamp(0, self.max_steps))))
        cls_hist, ctr_hist = [], []
        for _ in range(N):
            cls_t = torch.tensor(cls_hist, device=device).unsqueeze(0) if cls_hist else torch.zeros(1, 0, dtype=torch.long, device=device)
            ctr_t = torch.stack(ctr_hist).unsqueeze(0) if ctr_hist else torch.zeros(1, 0, 2, device=device)
            tgt = self._seq_embed(cls_t, ctr_t)
            m = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
            h = self.dec(tgt, memory, tgt_mask=m)
            out = self._readout(h[:, -1:], memory, coord)
            c = int(out["cls"][0, 0, 1:].argmax()) + 1
            cls_hist.append(c); ctr_hist.append(out["mu"][0, 0].detach())
        cls_t = torch.tensor(cls_hist, device=device).unsqueeze(0) if cls_hist else torch.zeros(1, 0, dtype=torch.long, device=device)
        ctr_t = torch.stack(ctr_hist).unsqueeze(0) if ctr_hist else torch.zeros(1, 0, 2, device=device)
        tgt = self._seq_embed(cls_t, ctr_t)
        m = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        hful = self.dec(tgt, memory, tgt_mask=m)                  # (1, N+1, d) full object states
        return self._decode_section(memory, hful)

    @torch.no_grad()
    def object_states(self, smap, gt):
        """Per-object hidden state h_i (teacher-forced over gt, DETACHED) — the shared object
        representation that also drives the mask head. Source for the narrator's <feature>_i
        (reader stays frozen/GT-trained; the LM only READS h_i, never reshapes it). Returns
        (K, d) aligned 1:1 with gt order; also returns each object's normalized centroid for
        matching to the fact objects."""
        memory, coord, (fH, fW) = self._grid(smap)
        K = len(gt)
        if K == 0:
            return torch.zeros(0, self.d, device=device), []
        cls = torch.tensor([o["cls"] for o in gt], device=device).unsqueeze(0)
        cls = cls.long()
        ctr = torch.stack([o["ctr"] for o in gt]).unsqueeze(0)

        tgt = self._seq_embed(cls, ctr)
        m = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        h = self.dec(tgt, memory, tgt_mask=m)                     # (1, K+1, d)
        centroids = [(float(o["ctr"][0]), float(o["ctr"][1])) for o in gt]   # (row, col) normalized
        return h[0, :K].detach(), centroids

    @torch.no_grad()
    def detect(self, smap, thresh=0.5, want_masks=False):
        """Greedy autoregressive decode → list of {cls, dip, throw, area, ctr}. With
        want_masks, also returns per-instance mask logits (H',W') from the query."""
        memory, coord, (fH, fW) = self._grid(smap)
        mfeat = self._mask_features(memory, fH, fW) if want_masks else None
        N = int(round(float(self.count_head(memory.mean(1))[0, 0].clamp(0, self.max_steps))))
        objs, masks, cls_hist, ctr_hist = [], [], [], []
        for _ in range(N):                                       # emit exactly N (count head)
            cls_t = torch.tensor(cls_hist, device=device).unsqueeze(0) if cls_hist else torch.zeros(1, 0, dtype=torch.long, device=device)
            ctr_t = torch.stack(ctr_hist).unsqueeze(0) if ctr_hist else torch.zeros(1, 0, 2, device=device)
            tgt = self._seq_embed(cls_t, ctr_t)
            m = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
            h = self.dec(tgt, memory, tgt_mask=m)
            out = self._readout(h[:, -1:], memory, coord)
            c = int(out["cls"][0, 0, 1:].argmax()) + 1           # fault/closure (count head already set N)
            occ = out["foot"][0, 0] > 0.5                             # footprint occupancy -> bbox
            if occ.any():
                cc = coord[occ]
                bb = [cc[:, 1].min(), cc[:, 0].min(), cc[:, 1].max(), cc[:, 0].max()]  # x1,y1,x2,y2 (0-1)
            else:
                mu = out["mu"][0, 0]; bb = [mu[1], mu[0], mu[1], mu[0]]
            od = self._decode_object(h[0, -1]) if c == CLOSURE else None   # object-scoped derived words
            mv = {n: float(out["meas"][n][0, 0] * MEASURE[n][1]) for n in MEASURE}   # registry-scaled measures
            objs.append(dict(cls=c, dip=mv.get("dip", 0.0), throw=mv.get("throw", 0.0),
                             area=mv.get("area", 0.0), meas=mv,
                             ctr=out["mu"][0, 0].detach(), derive=od,
                             bbox=[int(float(v) * 100) for v in bb]))     # 0-100, same scale as GT
            if want_masks:
                masks.append(torch.einsum("bd,dhw->bhw", self.mask_q(h[:, -1]), mfeat[0])[0])
            cls_hist.append(c); ctr_hist.append(out["mu"][0, 0].detach())
        return (objs, masks) if want_masks else objs
