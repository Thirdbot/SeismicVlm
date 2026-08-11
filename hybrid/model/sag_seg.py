"""SAG-style promptable mask head — the SAM-prior rebuild of PromptableSegHead.

Segment-Any-Geobodies' recipe verbatim: *"freeze the core pre-trained SAM weights, introduce small trainable
adapter modules that converge quickly on seismic patterns."* We keep YOUR frozen SFM as the image encoder
(seismic features already in-domain) and bolt SAM's FROZEN promptable decoder + prompt encoder — the
natural-image segmentation prior learned from ~1B masks — onto its pixel features, driven by the reader's
detection prompt. The only weights that train (on OUR surveys, nothing external):

  • proj      : SFM pixel features (256-ch) -> SAM's 256-ch image-embedding grid  (a small conv adapter)
  • LoRA(r)   : low-rank updates on every nn.Linear of the mask decoder            (the seismic adapter)

Frozen: SAM prompt encoder + mask-decoder base weights (the prior is never overwritten → no forgetting,
few params, fast convergence — exactly SAG's argument). Different decoder FAMILY than the reader's linear
`<mask_q(h), pixfeat>`, so it is NOT bounded by that path's oracle-query ~0.76 ceiling.

    head = SAGSegHead(feat_dim=256)                       # 256 = reader d (grid-token width)
    memory, coord, _iso, (fH, fW) = reader._grid(smap)    # (1,N,256), N=fH*fW — the FROZEN reader's grid
    m    = head(memory, (fH, fW), center)                 # center = detection [row,col] in [0,1]
    #  -> (256,256) mask logits at SAM's decoder resolution; interp to GT for the loss / score.

We take the compact fH×fW grid (not the reader's 8×-upsampled pixel features) and resize it to SAM's 64×64
embedding BEFORE the conv adapter, so the head costs O(64²) regardless of scene size — a big Thebe section's
1006-px pixfeat never materializes here (that 8× tensor is ~1 GB and OOMs the 5.67 GB GPU).

Vision-side only: the LM and the copy seam are untouched. SAM's decoder + prompt encoder are shared across
all SAM sizes, so ViT-B's checkpoint suffices even though we discard its ViT image encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SAM_CKPT = "hybrid/checkpoints/sam_vit_b.pth"


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update x·Aᵀ·Bᵀ·(α/r). B starts at 0 → the module is a no-op at
    init, so the pretrained decoder is unchanged until the adapter learns (SAG: adapters converge quickly)."""

    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                                  # the SAM prior — frozen
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))     # zero → Δ=0 at init
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def add_lora(module, r=4, alpha=8):
    """Recursively swap every nn.Linear under `module` for a LoRA-wrapped one (in place)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha))
        else:
            add_lora(child, r, alpha)


class SAGSegHead(nn.Module):
    """Reader pixel features + detection prompt -> SAM frozen decoder (+LoRA) -> fault mask logits."""

    def __init__(self, feat_dim=256, sam_ckpt=SAM_CKPT, sam_type="vit_b", lora_r=4, lora_alpha=8, query_dim=None):
        super().__init__()
        from segment_anything import sam_model_registry
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        sam.image_encoder = None                                     # drop the heavy natural-image ViT — SFM is our encoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad_(False)
        for p in self.mask_decoder.parameters():
            p.requires_grad_(False)                                  # freeze the whole decoder …
        add_lora(self.mask_decoder, lora_r, lora_alpha)              # … then inject the only trainable decoder params
        self.emb = self.prompt_encoder.image_embedding_size[0]       # 64 — dense-prompt + pos-enc are tied to this grid
        self.img = self.prompt_encoder.input_image_size[0]           # 1024 — the scale the prompt coords live in
        self.proj = nn.Sequential(                                   # SFM pixel features -> SAM's 256-ch image embedding
            nn.Conv2d(feat_dim, 256, 1), nn.GroupNorm(32, 256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1))
        # OPTIONAL content prompt: the reader's per-object query h_i (what the object IS) projected into SAM's
        # sparse-token space and appended to the point prompt (where it is). SAM sparse prompts are just
        # d-vectors, so this rides the same two-way attention as a point/box — no decoder change.
        self.query_proj = nn.Sequential(nn.Linear(query_dim or feat_dim, 256), nn.GELU(), nn.Linear(256, 256))
        self.query_type = nn.Parameter(torch.zeros(1, 1, 256))       # learned "this token is a content prompt" embed

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, memory, fhw, center=None, box=None, query=None, mask_prompt=None):
        """memory (1,N,C) reader grid tokens with N=fH*fW · fhw=(fH,fW) · center [row,col] in [0,1]
        (detection mu, optional — omit for a <SEG>-only / GLaMM-pure prompt) · box [x1,y1,x2,y2] in [0,1]
        (optional) · query (query_dim,) a content prompt (reader h_i OR LM <SEG> hidden), projected to a
        sparse token (optional) · mask_prompt (h,w) coarse mask LOGITS, e.g. the reader's footprint, fed to
        SAM's dense refine input (optional). Returns (256,256) mask logits.
        The grid is reshaped + resized to SAM's 64×64 embedding BEFORE the conv adapter, so the head's cost
        is O(64²) regardless of the reader's native resolution (no 8× pixfeat blow-up → fits 5.67 GB)."""
        dev = memory.device
        fH, fW = fhw
        grid = memory.transpose(1, 2).reshape(1, memory.shape[-1], fH, fW)   # (1,C,fH,fW) row-major = _grid order
        emb = F.interpolate(grid, size=(self.emb, self.emb), mode="bilinear", align_corners=False)
        emb = self.proj(emb)                                                 # (1,256,64,64) = SAM image embedding
        image_pe = self.prompt_encoder.get_dense_pe()                        # (1,256,64,64) frozen positional grid
        points = None
        if center is not None:                                               # SAM points are (x=col, y=row) in px
            r, c = float(center[0]), float(center[1])
            pts = torch.tensor([[[c * self.img, r * self.img]]], device=dev, dtype=torch.float)   # (1,1,2)
            lbl = torch.ones(1, 1, device=dev, dtype=torch.int)              # 1 = foreground point
            points = (pts, lbl)
        boxes = None if box is None else (box.to(dev).float() * self.img).view(1, 4)
        mp = None
        if mask_prompt is not None:                                          # (h,w) logits -> SAM's 4·emb dense mask input
            mp = F.interpolate(mask_prompt[None, None].float(), size=(4 * self.emb, 4 * self.emb),
                               mode="bilinear", align_corners=False)
        sparse, dense = self.prompt_encoder(points=points, boxes=boxes, masks=mp)
        if query is not None:                                                # append the reader's content prompt
            qt = self.query_proj(query.to(dev).view(1, -1)).view(1, 1, 256) + self.query_type
            sparse = torch.cat([sparse, qt], dim=1)
        low, _iou = self.mask_decoder(image_embeddings=emb, image_pe=image_pe,
                                      sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                      multimask_output=False)                # (1,1,256,256)
        return low[0, 0]


if __name__ == "__main__":                                                   # shape + param smoke test
    head = SAGSegHead()
    fH, fW = 40, 24                                                          # an anisotropic grid, N=fH*fW
    mem = torch.randn(1, fH * fW, 256)
    m = head(mem, (fH, fW), center=torch.tensor([0.4, 0.6]))
    mq = head(mem, (fH, fW), center=torch.tensor([0.4, 0.6]), query=torch.randn(256))       # + content prompt
    mm = head(mem, (fH, fW), center=torch.tensor([0.4, 0.6]), mask_prompt=torch.randn(fH, fW))   # + refine prompt
    hseg = SAGSegHead(query_dim=1536)                                                        # LM <SEG> hidden width
    ms = hseg(mem, (fH, fW), query=torch.randn(1536))                                        # <SEG>-only: NO point
    print("mask", tuple(m.shape), "· +query", tuple(mq.shape), "· +mask", tuple(mm.shape),
          "· <SEG>-only", tuple(ms.shape), "· trainable", f"{head.n_trainable()/1e6:.3f}M",
          "· total", f"{sum(p.numel() for p in head.parameters())/1e6:.3f}M")
