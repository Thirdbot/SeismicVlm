"""Seismic Foundation Model (SFM) encoder — a drop-in alternative to NcsEncoder.

SFM (Sheng et al. 2023, arXiv:2309.02791, MIT license) is an MAE-pretrained seismic ViT trained on
2.28M seismic images from 192 field volumes — a broader-corpus seismic foundation model than NCS-v1.
We load ONLY the encoder (the MAE decoder is discarded) into a timm VisionTransformer whose config is
READ FROM the checkpoint (embed_dim / depth / patch / in_chans / img_size), so Base/Large and 224/512
variants all self-configure — no per-variant code.

Interface MIRRORS NcsEncoder so stitch() consumes it unchanged:
    forward(img, return_spatial=True) -> (cls (B,D), spatial (B,D,gh,gw)).

Two SFM-specific things vs NCS:
  · Preprocessing = per-image zero-mean / unit-std on a SINGLE channel (SFM's scheme), NOT RGB/255.
  · Carries its own tile params (.tile_size/.grid/.patch/.fill) so stitch() tiles at the SFM
    resolution — a 512 model → 32x32 patch grid per tile (the finer-grid mask lever) vs NCS 14x14.
"""
import math
import os

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

device = torch.device("cuda")


def _unwrap(sd):
    """MAE/timm checkpoints are often nested under 'model'/'state_dict'."""
    for k in ("model", "state_dict", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            return sd[k]
    return sd


def _encoder_only(sd):
    """Keep ViT-encoder keys; drop the MAE decoder, the mask token, and any classifier head."""
    return {k: v for k, v in sd.items()
            if not (k.startswith("decoder") or k == "mask_token" or k.startswith("head."))}


class SfmEncoder(nn.Module):
    def __init__(self, ckpt, trainable_blocks=0, fill=True):
        super().__init__()
        # weights_only=False: the official SFM ckpt (MIT) pickles its training args (argparse.Namespace)
        # alongside the weights, which torch 2.6+ safe-load rejects. Trusted source → fine.
        sd = _encoder_only(_unwrap(torch.load(ckpt, map_location="cpu", weights_only=False)))
        pe = sd["patch_embed.proj.weight"]                       # [embed_dim, in_chans, P, P]
        embed_dim, in_chans, patch = pe.shape[0], pe.shape[1], pe.shape[2]
        depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        ntok = sd["pos_embed"].shape[1] - 1                      # drop the cls slot
        g = int(round(math.sqrt(ntok)))                          # pretrained token grid (32 for SFM-512)
        native_img = patch * g                                   # pretraining resolution in PIXELS (512)
        # ACUITY: optionally resample patch_embed to a SMALLER patch (FlexiViT-style bicubic on the conv
        # kernel). P=8 -> each token reads a real 8x8 block instead of 16x16 = 4x the cells at NATIVE
        # scale, with no invented pixels — the first true resolution increase (every earlier "resolution"
        # experiment only upsampled the input into the same 16px grid, which adds no information).
        # img_size stays patch*g so the checkpoint's pos_embed loads EXACTLY; dynamic_img_size then
        # interpolates it to whatever grid a real tile implies.
        want = int(os.environ.get("PATCH", patch))
        if want != patch:
            from timm.layers import resample_patch_embed
            sd["patch_embed.proj.weight"] = resample_patch_embed(pe, [want, want])
            patch = want
        img_size = patch * g
        self.model = VisionTransformer(
            img_size=img_size, patch_size=patch, in_chans=in_chans, embed_dim=embed_dim,
            depth=depth, num_heads=max(1, embed_dim // 64), num_classes=0,
            # dynamic_img_size: interpolate the position embedding to whatever grid the input implies, so
            # the encoder accepts NATIVE-ASPECT tiles. Without it every image had to be squashed into a
            # square 512, which applies a per-dataset geometric warp (synthetic 100x507 is stretched 5.1x,
            # so a 60deg fault presents as 84deg) — the dip head then learns a warp that does NOT hold on
            # real data with a different aspect, silently breaking syn->real transfer.
            global_pool="", class_token=True, dynamic_img_size=True)
        miss, unexp = self.model.load_state_dict(sd, strict=False)
        self.in_chans, self.embed_dim, self.patch, self.img_size = in_chans, embed_dim, patch, img_size
        # tile params consumed by stitch()/the loader — CONTEXT stays at the pretraining resolution in
        # PIXELS (512), independent of patch: shrinking the tile with the patch would buy acuity by
        # spending context, and a tile shorter than the image cuts ACROSS near-vertical faults.
        self.tile_size = int(os.environ.get("TILE", 0)) or native_img
        self.grid, self.fill = self.tile_size // patch, fill
        self.trainable_blocks = int(trainable_blocks)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        if self.trainable_blocks > 0:                            # optional last-N-block unfreeze (same as NCS)
            for blk in self.model.blocks[-self.trainable_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.model.norm.parameters():
                p.requires_grad = True
        print(f"[SFM] {ckpt}: ViT dim{embed_dim} depth{depth} patch{patch}"
              f"{f' (resampled from {pe.shape[2]})' if patch != pe.shape[2] else ''} in_chans{in_chans} "
              f"img{img_size} tile{self.tile_size} grid{self.grid} · trainable_blocks{self.trainable_blocks}"
              f" · loaded (missing {len(miss)}, unexpected {len(unexp)})", flush=True)
        if miss:
            print(f"[SFM] missing keys (first 6): {miss[:6]}", flush=True)

    def normalize(self, img):
        """RGB[0,1] -> single seismic channel -> per-IMAGE zero-mean/unit-std (SFM's preprocessing)."""
        if img.shape[1] > self.in_chans:
            img = img.mean(1, keepdim=True)                      # grayscale (seismic is single-channel)
        elif img.shape[1] < self.in_chans:
            img = img.repeat(1, self.in_chans, 1, 1)
        flat = img.reshape(img.shape[0], -1)
        m = flat.mean(1).view(-1, 1, 1, 1)
        s = flat.std(1).view(-1, 1, 1, 1) + 1e-6
        return (img - m) / s

    def forward(self, img, return_spatial=False):
        grad_ctx = torch.enable_grad() if self.trainable_blocks > 0 else torch.no_grad()
        with grad_ctx:
            x = self.normalize(img.float().to(device))
            tok = self.model.forward_features(x)                 # (B, 1+N, D), post-norm, cls kept
            cls = tok[:, 0, :]
            if not return_spatial:
                return cls
            pt = tok[:, 1:, :]
            gh, gw = x.shape[-2] // self.patch, x.shape[-1] // self.patch   # grid from the INPUT (non-square ok)
            if gh * gw != pt.shape[1]:
                return cls, None
            sp = pt.transpose(1, 2).reshape(pt.shape[0], self.embed_dim, gh, gw)
            return cls, sp
