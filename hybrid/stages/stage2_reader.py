"""Stage 2 (new) — the instance reader (facts + per-instance masks, one shared trunk).

Replaces the dense-seg + RANSAC front-end. `train_reader` fits the DETR set-prediction reader
(48 queries + Hungarian + ∅; class/centroid/dip/throw/area AND the per-instance mask head); `reader_facts`
adapts its detections to the digit-bridge fact dict; `reader_accuracy` reports held-out count/dip/class.
(The old LM-<SEG> mask decoder path was retired — masks come from the reader now.)
"""
import os
import torch
import torch.nn.functional as F
import copy
import random
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from hybrid.model.reader import RegionReader, scene_to_gt, FAULT, CLOSURE, SALT, ONLAP
from hybrid.model.registry import derived_facts
from hybrid.model.encoder import NcsEncoder

device = torch.device("cuda")


SFM_DEFAULT = "hybrid/checkpoints/SFM-Base-512.pth"


def _build_encoder(trainable_blocks=0):
    """SFM if its checkpoint is present (or SFM_CKPT set), else NCS; last-`trainable_blocks` blocks trainable.

    The default is resolved HERE, not in run_train: previously only run_train exported SFM_CKPT, so a
    standalone `python -m hybrid.eval.components` / `real_transfer` silently built an NCS-224 encoder for
    a reader TRAINED on SFM-512. Both are 768-dim so the checkpoint loaded without error and nothing
    warned — every standalone eval number was measured on a front-end the model never saw."""
    ck = os.environ.get("SFM_CKPT")
    if ck and not os.path.exists(ck):                          # a set-but-bad path is an ERROR, not a fallback
        raise FileNotFoundError(f"SFM_CKPT={ck} is set but the file does not exist.")
    ck = ck or (SFM_DEFAULT if os.path.exists(SFM_DEFAULT) else None)
    if ck:
        from hybrid.model.sfm_encoder import SfmEncoder
        enc = SfmEncoder(ck, trainable_blocks=trainable_blocks).to(device)
    elif os.environ.get("ALLOW_NCS") == "1":                    # NCS is legacy — opt in EXPLICITLY, loudly
        print("[enc] WARNING: SFM checkpoint absent — building NCS-224 (ALLOW_NCS=1). Numbers are NOT "
              "comparable to any SFM-512 result.", flush=True)
        enc = NcsEncoder(trainable_blocks=trainable_blocks).to(device)
    else:                                                      # NO SILENT FALLBACK: SFM-512 and NCS-224 are
        raise FileNotFoundError(                               # both 768-dim, so NCS would load clean and every
            f"SFM encoder checkpoint not found (SFM_CKPT unset and {SFM_DEFAULT} absent). Refusing to fall "
            f"back to NCS-224 silently — a reader trained on SFM-512 would be measured on a front-end it never "
            f"saw. Set SFM_CKPT=path/to/SFM-Base-512.pth, or ALLOW_NCS=1 to force NCS deliberately.")
    enc.eval()                                                 # eval = no dropout; grads still flow to unfrozen blocks
    # TUNED-ENCODER REUSE. set_encoder() keeps the encoder OUT of state_dict, so when the reader is
    # trained with unfrozen blocks the adapted weights live in reader_enc.pt — and every downstream
    # caller (finetune_real / real_transfer / eval.components) built a FRESH encoder and silently threw
    # them away, feeding the reader features it was never trained against. Same species as the SFM/NCS
    # mismatch: reload once, HERE, so no caller can forget. Shape mismatch (e.g. a PATCH=16 file against
    # a PATCH=8 encoder) raises LOUDLY rather than half-loading.
    tuned = "hybrid/checkpoints/reader_enc.pt"
    if int(os.environ.get("TRAINABLE_BLOCKS", 0)) > 0:
        if not os.path.exists(tuned):                          # NO SILENT FALLBACK: asked for unfrozen blocks,
            raise FileNotFoundError(                           # but the tuned weights live in reader_enc.pt — a
                f"TRAINABLE_BLOCKS>0 but {tuned} absent — the reader was trained with unfrozen ViT blocks "
                f"whose weights live there; a fresh encoder would silently feed the reader wrong features.")
        enc.load_state_dict(torch.load(tuned, map_location=device))
        print(f"[enc] loaded tuned encoder ← {tuned}", flush=True)
    return enc


def train_reader(scenes, epochs=200, lr=1e-4, save="hybrid/checkpoints/reader.pt", trainable_blocks=0,
                 cldice_w=0.0):
    """Clean fixed-epoch train: pixels -> encoder (frozen, or last-N unfrozen) -> DETR reader. NO val-split,
    NO early-stop, NO aug — that anti-overfit machinery UNDER-trained the reader (a val-loss early-stop
    stopped it ~ep17 → mask dice 0.08, while the clean overfit needs the full epochs to reach ~0.9). Keeps
    the LAST checkpoint. trainable_blocks>0 = unfreeze the last N ViT blocks (backbone at lr/10)."""
    net = RegionReader().to(device); net.cldice_w = cldice_w    # thin-structure loss on the mask (0 = off)
    net.set_encoder(_build_encoder(trainable_blocks))           # encoder IN the model: pixels -> grid (frozen or last-N unfrozen)
    groups = [{"params": list(net.parameters()), "lr": lr}]
    if trainable_blocks > 0:                                    # add the unfrozen encoder blocks (backbone 10× lower LR)
        groups.append({"params": [p for p in net.enc.parameters() if p.requires_grad], "lr": lr * 0.1})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    data = [(s, s.get("derived")) for s in scenes if s["objs"] or s.get("is_neg")]  # LAZY: scene meta only, masks load per step; keep negs (train ∅)
    random.Random(0).shuffle(data)
    npos = sum(1 for s, _ in data if s["objs"])
    print(f"[reader] {len(data)} scenes ({npos} pos) · {epochs} ep · "
          f"encoder {'unfrozen last '+str(trainable_blocks) if trainable_blocks else 'frozen'} · clDice {cldice_w}", flush=True)
    ebar = tqdm(range(epochs), desc="reader", unit="ep")
    for ep in ebar:
        net.train(); tot = 0.0
        for s, der in tqdm(data, desc=f"ep {ep}", unit="sc", leave=False):
            opt.zero_grad(); loss, _ = net(net.encode(s), scene_to_gt(s), der); loss.backward(); opt.step(); tot += float(loss)
        ebar.set_postfix(loss=f"{tot/len(data):.3f}")
        tqdm.write(f"[reader] ep {ep}/{epochs} loss {tot/len(data):.3f}")
    net.eval()
    if save:
        torch.save(net.state_dict(), save); print(f"[reader] saved (ep {epochs}) → {save}", flush=True)
        if trainable_blocks > 0:
            torch.save(net.enc.state_dict(), save.replace(".pt", "_enc.pt")); print("[reader] saved tuned encoder", flush=True)
    return net


@torch.no_grad()
def reader_facts(net, scene):
    """reader.detect -> the fact dict the digit bridge consumes. bbox/center are UN-NORMALIZED to
    the scene's PIXEL scale (matching region_metadata + the dataset evidence); the head stays normalized.
    Plus the scene-level tier-2 derived read (intersections + mode word)."""
    H, W = scene["hw"]
    def pxb(b):   # reader emits 0-100 → pixels
        return [int(b[0] / 100 * W), int(b[1] / 100 * H), int(b[2] / 100 * W), int(b[3] / 100 * H)]
    def pxc(c):   # centroid mu = [row, col] normalized → [x, y] pixels
        return [int(float(c[1]) * W), int(float(c[0]) * H)]
    faults, closures, salts, onlaps = [], [], [], []
    smap = net.encode(scene)                                   # tiled scene -> encoder -> stitched grid (in-model)
    for o in net.detect(smap):
        if o["cls"] == FAULT:
            faults.append({"dip": o["dip"], "bbox": pxb(o["bbox"]), "center": pxc(o["ctr"]), "throw": o["throw"]})
        elif o["cls"] == CLOSURE:
            closures.append({"area_pct": o["area"], "bbox": pxb(o["bbox"]), "center": pxc(o["ctr"]),
                             "derive": o.get("derive") or {}})     # object-scoped derived words (decoded)
        elif o["cls"] == SALT:
            salts.append({"area_pct": o["area"], "bbox": pxb(o["bbox"]), "center": pxc(o["ctr"])})
        elif o["cls"] == ONLAP:
            onlaps.append({"area_pct": o["area"], "bbox": pxb(o["bbox"]), "center": pxc(o["ctr"])})
    facts = {"faults": faults, "closures": closures, "salts": salts, "onlaps": onlaps,
             "derived": derived_facts(net.read_derived(smap))}
    return facts


def match_pred_gt(pred, gt):
    """Hungarian match detections↔GT by CENTRE distance → list of (pred, gt) pairs.

    THE METRIC FIX: class accuracy used to compare pred[i] vs gt[i] by ARBITRARY INDEX (pred is in
    query order, gt is x-sorted) — it measured nothing. Dip MAE paired SORTED pred dips against SORTED
    GT dips, a rank-match that structurally UNDER-states error (and never checks the right fault got
    the right dip). Both made runs incomparable. Now every attribute is scored on the SAME matched
    pairs, so count/class/dip move together and before/after comparisons are valid."""
    if not pred or not gt:
        return []
    pc = torch.tensor([[float(o["ctr"][0]), float(o["ctr"][1])] for o in pred])   # (P,2) normalized row,col
    gc = torch.stack([o["ctr"].cpu() for o in gt]).float()                        # (G,2)
    ri, ci = linear_sum_assignment(torch.cdist(pc, gc, p=1).numpy())
    return [(pred[i], gt[j]) for i, j in zip(ri, ci)]


@torch.no_grad()
def mask_dice(net, scenes, oracle=True):
    """Mean per-fault mask dice over a FIXED population (every GT fault in `scenes`).

    oracle=True  — teacher-forced (`tf_masks`): queries are matched using GT class+centroid, so this
                   measures the MASK DECODER in isolation and detection failures are INVISIBLE. Every
                   historical "mask dice" number in this project is this metric; it is not deployment.
    oracle=False — DEPLOYMENT: masks come from `detect()`, matched to GT by centroid; a GT fault with no
                   matched detection scores 0. This is what the model actually produces end-to-end.
    Both share the same denominator (all GT faults), so they are directly comparable and neither can be
    gamed by detecting less."""
    from hybrid.model.geometry import field_dice
    ds = []
    for s in tqdm(scenes, desc="dice", unit="sc", leave=False):
        gt = scene_to_gt(s)
        faults = [o for o in gt if o["cls"] == FAULT]
        if not faults:
            continue
        smap = net.encode(s)
        if oracle:
            ml = net.tf_masks(smap, gt)
            ds += [field_dice(ml[i], o["mask_full"].to(device)) for i, o in enumerate(gt) if o["cls"] == FAULT]
        else:
            pred, masks = net.detect(smap, want_masks=True)
            pairs = {id(g): p for p, g in zip(*_pair_masks(pred, masks, gt))} if pred else {}
            for o in faults:
                m = pairs.get(id(o))
                ds.append(0.0 if m is None else field_dice(
                    F.interpolate(m[None, None], size=o["mask_full"].shape, mode="bilinear",
                                  align_corners=False)[0, 0], o["mask_full"].to(device)))
    return (sum(ds) / len(ds), len(ds)) if ds else (None, 0)


def _pair_masks(pred, masks, gt):
    """Hungarian-match detections to GT by centroid; return (matched masks, matched gt objects)."""
    m = match_pred_gt(pred, gt)
    idx = {id(p): i for i, p in enumerate(pred)}
    return [masks[idx[id(p)]] for p, _ in m], [g for _, g in m]


@torch.no_grad()
def reader_accuracy(net, scenes):
    """Held-out reader metrics on MATCHED pairs (count MAE · dip MAE · class acc)."""
    cnt, dip, hit, tot = [], [], 0, 0
    for s in tqdm(scenes, desc="reader-eval", unit="sc", leave=False):
        gt = scene_to_gt(s); pred = net.detect(net.encode(s))
        cnt.append(abs(len(pred) - len(gt)))
        for p, g in match_pred_gt(pred, gt):
            tot += 1; hit += int(p["cls"] == g["cls"])
            if g["cls"] == FAULT and p["cls"] == FAULT and g.get("dip") is not None:
                dip.append(abs(p["dip"] - g["dip"]))

    def m(x): return (sum(x) / len(x), len(x)) if x else (None, 0)
    return dict(count=m(cnt), dip=m(dip), cls=(hit, tot))
