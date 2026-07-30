"""Stage 2 (new) — the instance reader (facts + per-instance masks, one shared trunk).

Replaces the dense-seg + RANSAC front-end. `train_reader` fits the autoregressive reader
(count/class/dip/throw AND the joint mask head); `reader_facts` adapts its detections to the
digit-bridge fact dict; `reader_accuracy` reports held-out count/dip/class.
(The old LM-<SEG> mask decoder path was retired — masks come from the reader now.)
"""
import torch
import copy
import random
from tqdm.auto import tqdm

from hybrid.model.reader import RegionReader, scene_to_gt, FAULT, CLOSURE, SALT, ONLAP
from hybrid.model.registry import derived_facts
from hybrid.model.encoder import NcsEncoder, stitch

device = torch.device("cuda")


def train_reader(scenes, epochs=200, lr=1e-4, save="hybrid/checkpoints/reader.pt", trainable_blocks=2,
                 val_frac=0.15, patience=18, aug_noise=0.05):
    # DETR set-prediction. trainable_blocks>0 (DEFAULT 2) = ENCODER-UNFREEZE: unfreeze the last N ViT
    # blocks and train them JOINTLY (re-encode in-graph per step, eval-mode → deterministic, backbone at
    # lr/10). Better features = the mask lever (+~0.19 ceiling). EARLY-STOPPING on a held-out val split
    # (keep the BEST-val checkpoint) + feature-noise aug prevent the ∅-collapse a fixed-epoch run hits
    # (the reader detects mid-training, then overfits back to ∅). Saves best reader.pt (+ tuned encoder as
    # <save>_enc.pt → inference/downstream re-encode with it via ENCODER_CKPT).
    net = RegionReader().to(device)
    enc = None
    groups = [{"params": list(net.parameters()), "lr": lr}]
    if trainable_blocks > 0:
        enc = NcsEncoder(trainable_blocks=trainable_blocks).to(device); enc.eval()   # eval = no dropout; grads still flow
        groups.append({"params": [p for p in enc.parameters() if p.requires_grad], "lr": lr * 0.1})  # backbone 10× lower LR
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    # KEEP negatives (empty gt → train ∅). Keep the SCENE dict (has img) for in-graph re-encode when unfrozen.
    data = [(s, scene_to_gt(s), s.get("derived")) for s in scenes if scene_to_gt(s) or s.get("is_neg")]
    random.Random(0).shuffle(data)
    vc = max(1, int(len(data) * (1 - val_frac))); tr_d, val_d = data[:vc], data[vc:]   # inner val for early-stop
    npos = sum(1 for _, g, _ in tr_d if g)
    enc_msg = f"UNFROZEN last {trainable_blocks} blocks (joint re-encode)" if enc else "frozen (cached smaps)"
    print(f"[reader] train {len(tr_d)} ({npos} pos) · val {len(val_d)} · ≤{epochs} ep · early-stop patience "
          f"{patience} · aug_noise {aug_noise} · encoder {enc_msg}", flush=True)

    def smap_of(s):
        return stitch(enc, s["img"])[0] if (enc and s.get("img")) else s["smap"]      # live encode when unfrozen

    @torch.no_grad()
    def val_loss():
        net.eval()
        return sum(float(net(smap_of(s), gt, der)[0]) for s, gt, der in val_d) / max(1, len(val_d))

    best_state, best_vl, best_ep, bad = None, 1e9, -1, 0
    ebar = tqdm(range(epochs), desc="reader", unit="ep")
    for ep in ebar:
        net.train()
        for s, gt, der in tqdm(tr_d, desc=f"ep {ep}", unit="sc", leave=False):
            sm = smap_of(s)
            if aug_noise:
                sm = sm + torch.randn_like(sm) * aug_noise                            # feature-noise aug (anti-overfit)
            opt.zero_grad(); loss, _ = net(sm, gt, der); loss.backward(); opt.step()
        vl = val_loss()
        star = vl < best_vl - 1e-4
        if star:
            best_vl, best_state, best_ep, bad = vl, copy.deepcopy(net.state_dict()), ep, 0
        else:
            bad += 1
        ebar.set_postfix(val=f"{vl:.3f}", best=str(best_ep))
        tqdm.write(f"[reader] ep {ep}/{epochs} val {vl:.3f}" + ("  *best*" if star else ""))
        if bad >= patience:
            print(f"[reader] EARLY STOP ep {ep} (best ep {best_ep}, val {best_vl:.3f})", flush=True); break
    net.load_state_dict(best_state); net.eval()
    if save:
        torch.save(net.state_dict(), save); print(f"[reader] saved best (ep {best_ep}) → {save}", flush=True)
        if enc:
            enc_path = save.replace(".pt", "_enc.pt")
            torch.save(enc.state_dict(), enc_path); print(f"[reader] saved tuned encoder → {enc_path}", flush=True)
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
    for o in net.detect(scene["smap"]):
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
             "derived": derived_facts(net.read_derived(scene["smap"]))}
    return facts


@torch.no_grad()
def reader_accuracy(net, scenes):
    cnt, dip, hit, tot = [], [], 0, 0
    for s in tqdm(scenes, desc="reader-eval", unit="sc", leave=False):
        gt = scene_to_gt(s); pred = net.detect(s["smap"])
        cnt.append(abs(len(pred) - len(gt)))
        gf = sorted(o["dip"] for o in gt if o["cls"] == FAULT and o.get("dip") is not None)
        pf = sorted(o["dip"] for o in pred if o["cls"] == FAULT)
        for i, gd in enumerate(gf):
            if i < len(pf):
                dip.append(abs(pf[i] - gd))
        for i in range(min(len(pred), len(gt))):
            tot += 1; hit += int(pred[i]["cls"] == gt[i]["cls"])

    def m(x): return (sum(x) / len(x), len(x)) if x else (None, 0)
    return dict(count=m(cnt), dip=m(dip), cls=(hit, tot))
