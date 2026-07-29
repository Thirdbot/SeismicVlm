"""Overfit — MASKED injection + L1-forced feature + <think> anchored with the real value.

The masked A/B showed: masking removes the copy short-circuit, but pure CE lets the model CONFABULATE
a plausible number, so the gate never opens. Fix (two losses):

  1. L1 on the GATED feature token — value_head(gate · feat_proj(h_i)) → value, supervised by smooth-L1
     against the GT number. A wrong number now HURTS (magnitude-aware), and because the head reads the
     GATED feature, the gate MUST leave zero for the loss to drop → the feature is forced to carry the
     value.  (This is the "make it hurt" pressure CE can't provide.)
  2. <think> anchored with the REAL value (a training target built from GT: "the fault at [c] has a dip
     of about D degrees ..."), so the model learns to VERBALIZE the L1-backed value in its reasoning →
     it anchors the answer.

Inject class + center only (bbox + values masked) + feature. A/B = feature ON vs OFF: value-in-think
(the anchor emerged) + the value_head L1 MAE (vs the reader's own MAE ceiling) + the gate.

Run:  python -m hybrid.stages.masked_feature_l1
"""
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from hybrid.model.captioner import Captioner, region_metadata, _region_objs, INSTRUCTION_ROLE
from hybrid.model.reader import RegionReader
from hybrid.model.text_metrics import _THINK
from hybrid.model.registry import MEASURE_SLOTS, MEASURE_SCALE
from hybrid.stages.stage3_answer import aligned_feats
from hybrid.data.schema import load_local_csv
from hybrid.data import synthetic

device = torch.device("cuda")
SEED = 42
LAMBDA_L1 = float(os.environ.get("LAMBDA_L1", 1.0))          # weight on the feature-value L1 term
_GT_KEY = {"dip": "dip", "throw": "throw", "area": "area_pct"}   # MEASURE slot -> facts key


def _obj_values(o):
    """(slot, gt_value) present on this object, matched to MEASURE_SLOTS order."""
    return [(slot, float(o[_GT_KEY[name]])) for slot, name in enumerate(MEASURE_SLOTS)
            if o.get(_GT_KEY[name]) is not None]


def think_anchor(facts):
    """A <think> body that STATES the real measured values — the anchor the model learns to verbalize.
    Built from GT (a legitimate training target, not faked model output)."""
    lines = []
    for f in facts.get("faults", []):
        c = f["center"]; s = f"the fault at [{c[0]},{c[1]}] has a dip of about {round(float(f['dip']),1):g} degrees"
        if f.get("throw") is not None:
            s += f" and a throw of about {round(float(f['throw']))} ms"
        lines.append(s + ".")
    for c in facts.get("closures", []) + facts.get("salts", []) + facts.get("onlaps", []):
        cc = c["center"]; lines.append(f"the region at [{cc[0]},{cc[1]}] covers about {round(float(c['area_pct']))} percent.")
    return " ".join(lines)


def train(nar, reader, value_head, data, epochs, lr=2e-4):
    nar.set_stage("s2"); nar.use_feature = True; nar.mask_digits = True; nar.train_mode()
    params = nar.trainable_params() + list(value_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    ebar = tqdm(range(epochs), desc="l1-anchor train", unit="ep")
    for ep in ebar:
        tot_ce = tot_l1 = 0.0
        for scene, facts, q, target in tqdm(data, desc=f"ep {ep}", unit="row", leave=False):
            feats = aligned_feats(reader, scene, facts)
            opt.zero_grad()
            ce = nar.ground_loss(facts, target, question=q, instruction=INSTRUCTION_ROLE, feats=feats)
            # L1 on the GATED feature token -> value (forces the gate open + the feature to carry value)
            l1 = torch.zeros((), device=device)
            for (cls, o), h in zip(_region_objs(facts), feats):
                if h is None:
                    continue
                gated = nar.feat_gate * nar.feat_proj(h)            # (lm_dim,)
                pred = value_head(gated)                            # (len MEASURE,)
                for slot, gt in _obj_values(o):
                    l1 = l1 + F.smooth_l1_loss(pred[slot], torch.tensor(gt / MEASURE_SCALE[slot], device=device))
            (ce + LAMBDA_L1 * l1).backward(); opt.step()
            tot_ce += ce.item(); tot_l1 += float(l1)
        ebar.set_postfix(ce=f"{tot_ce/len(data):.3f}", l1=f"{tot_l1/len(data):.3f}", gate=f"{float(nar.feat_gate):.4f}")
        tqdm.write(f"[l1-anchor] ep {ep}/{epochs} ce {tot_ce/len(data):.3f} · l1 {tot_l1/len(data):.3f} · gate {float(nar.feat_gate):.4f}")
    nar.eval_mode()


@torch.no_grad()
def evaluate(nar, reader, value_head, data, use_feature):
    """value-in-think (GT number appears in <think>) + value_head L1 MAE (per attr) + present."""
    nar.use_feature = use_feature; nar.mask_digits = True; nar.set_stage("s2"); nar.eval_mode()
    in_think = present = tot = 0
    abs_err = defaultdict(list)
    for scene, facts, q, _t in tqdm(data, desc=f"eval feat {'ON' if use_feature else 'OFF'}", unit="row", leave=False):
        feats = aligned_feats(reader, scene, facts) if use_feature else None
        gt_strs = [f"{round(float(x['dip']),1):g}" for x in facts["faults"]] + \
                  [f"{round(float(x['throw'])):g}" for x in facts["faults"] if x.get("throw") is not None]
        out = nar.caption(facts, question=q, instruction=INSTRUCTION_ROLE, feats=feats, max_new_tokens=120)
        tm = _THINK.search(out); think = tm.group(1) if tm else ""
        for v in gt_strs:
            tot += 1; present += int(bool(think)); in_think += (v in think)
        if use_feature:
            for (cls, o), h in zip(_region_objs(facts), aligned_feats(reader, scene, facts)):
                if h is None:
                    continue
                pred = value_head(nar.feat_gate * nar.feat_proj(h))
                for slot, gt in _obj_values(o):
                    abs_err[MEASURE_SLOTS[slot]].append(abs(float(pred[slot]) * MEASURE_SCALE[slot] - gt))
    mae = {k: (sum(v) / len(v), len(v)) for k, v in abs_err.items()}
    return in_think, tot, mae


def main():
    reader = RegionReader().to(device)
    reader.load_state_dict(torch.load("hybrid/checkpoints/reader.pt", map_location=device)); reader.eval()
    scenes = synthetic.scenes(max_scenes=int(os.environ.get("SCENES", 10_000)))
    scene_by_img = {s["img"]: s for s in scenes}

    nar = Captioner(); nar.dec.gradient_checkpointing_enable(); nar.dec.enable_input_require_grads()
    value_head = nn.Linear(nar.emb.embedding_dim, len(MEASURE_SLOTS)).to(device)

    data = []
    for r in load_local_csv(csv_path=synthetic.CSV):
        img = (r.get("image_paths") or [None])[0]; scene = scene_by_img.get(img)
        a = r.get("answer") or ""
        if scene is None or not a:
            continue
        from hybrid.model.captioner import row_region_metadata
        facts = row_region_metadata(r)
        if not any(_obj_values(o) for o in (facts["faults"] + facts.get("closures", []) + facts.get("salts", []) + facts.get("onlaps", []))):
            continue
        target = f"<think> {think_anchor(facts)} </think> <answer> {a} </answer>"
        data.append((scene, facts, r.get("question") or "", target))
    rng = random.Random(SEED); rng.shuffle(data)
    cut = int(len(data) * 0.75)
    tr = data[:cut][:int(os.environ.get("TRAIN_CAP", 150))]     # cap: fast overfit
    te = data[cut:][:int(os.environ.get("EVAL_CAP", 25))]       # cap: keep the generation eval short/safe
    print(f"[l1-anchor] valued rows {len(data)} · train {len(tr)} · test {len(te)}", flush=True)

    train(nar, reader, value_head, tr, epochs=int(os.environ.get("EPOCHS", 8)))
    it_on, n, mae_on = evaluate(nar, reader, value_head, te, use_feature=True)
    it_off, _, _ = evaluate(nar, reader, value_head, te, use_feature=False)
    print(f"[L1-ANCHOR A/B]  value-in-think ON {it_on}/{n}={it_on/max(1,n):.2f} · OFF {it_off}/{n}={it_off/max(1,n):.2f} · "
          f"gate {float(nar.feat_gate):.4f}", flush=True)
    print(f"[feature-value L1 MAE] " + " · ".join(f"{k} {v[0]:.1f}(n{v[1]})" for k, v in mae_on.items()) +
          "  (vs reader's own MAE = ceiling)", flush=True)
    print("Gate>0 + low L1 MAE + ON>>OFF value-in-think = the feature carries the value and it emerges in reasoning.", flush=True)
    print("L1_ANCHOR_DONE", flush=True)


if __name__ == "__main__":
    main()
