"""Stage 2 (new) — the instance reader (facts + per-instance masks, one shared trunk).

Replaces the dense-seg + RANSAC front-end. `train_reader` fits the autoregressive reader
(count/class/dip/throw AND the joint mask head); `reader_facts` adapts its detections to the
digit-bridge fact dict; `reader_accuracy` reports held-out count/dip/class.
(The old LM-<SEG> mask decoder path was retired — masks come from the reader now.)
"""
import torch
from tqdm.auto import tqdm

from hybrid.model.reader import InstanceReader, scene_to_gt, FAULT, CLOSURE, SALT, ONLAP
from hybrid.model.registry import derived_facts

device = torch.device("cuda")


def train_reader(scenes, epochs=150, lr=3e-4):
    net = InstanceReader().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    # KEEP negatives (empty gt): a K=0 scene trains the count head toward 0 (fault suppression). Drop
    # only scenes with neither objects nor a negative marker. `is_neg` = built by the real ungated CSV.
    data = [(s["smap"], scene_to_gt(s), s.get("derived")) for s in scenes
            if scene_to_gt(s) or s.get("is_neg")]
    npos = sum(1 for _, g, _ in data if g)
    print(f"[reader] training on {len(data)} scenes ({npos} with objects, {len(data)-npos} negatives) · "
          f"{epochs} epochs", flush=True)
    net.train()
    ebar = tqdm(range(epochs), desc="reader", unit="ep")           # epoch bar: rate + ETA over the run
    for ep in ebar:
        tot = 0.0
        for smap, gt, der in tqdm(data, desc=f"ep {ep}", unit="sc", leave=False):   # within-epoch bar
            opt.zero_grad(); loss, _ = net(smap, gt, der); loss.backward(); opt.step(); tot += loss.item()
        ebar.set_postfix(loss=f"{tot/len(data):.3f}")
        tqdm.write(f"[reader] ep {ep}/{epochs} loss {tot/len(data):.3f}")            # a line for logs too
    net.eval()
    return net


@torch.no_grad()
def reader_facts(net, scene):
    """reader.detect -> the fact dict the digit bridge consumes. bbox/center are UN-NORMALIZED to
    the scene's PIXEL scale (matching scene_facts + the dataset evidence); the head stays normalized.
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
