"""Overfit A/B — MASKED-INJECTION FEATURE test (values emerge in <think>, anchored by the feature).

The mechanism (NOT a probe head): inject only class + center; MASK bbox + every measured value to "?";
keep the <feature>_i soft token. The value is still needed for the answer but is absent from the
prompt, so the model must RECOVER it through its own generation — the hope is the number surfaces in
<think> (an anchor recovered from the feature) and then drives <answer>.

Trained with the masked-<think> fold (think in the masked prefix, answer supervised) + feature on, so
the reasoning emerges. A/B = feature ON vs OFF:
  · value-in-chain  : the GT number appears in the generated chain (recovered at all)
  · value-in-think  : the GT number appears in <think> (emerged as the anchor)
Feature OFF should be near-zero (no magnitude signal); a real ON>>OFF gap = the feature supplies the
masked value, generatively. Fast overfit — checks whether the mechanism CAN work.

Run:  python -m hybrid.stages.masked_feature_ab
"""
import os
import random
from collections import defaultdict

import torch
from tqdm.auto import tqdm

from hybrid.model.captioner import Captioner, region_metadata
from hybrid.model.reader import RegionReader
from hybrid.model.text_metrics import _THINK
from hybrid.stages.stage2_grounding import train_grounding
from hybrid.stages.stage3_answer import train_answer, generate_chain
from hybrid.data.schema import load_local_csv
from hybrid.data import synthetic

device = torch.device("cuda")
SEED = 42


def _gt_values(facts):
    """The measured numbers the answer would state, as rounded strings (same forms the copy uses)."""
    v = [f"{round(float(x['dip']), 1):g}" for x in facts["faults"]]
    v += [f"{round(float(x['throw'])):g}" for x in facts["faults"] if x.get("throw") is not None]
    v += [f"{round(float(c['area_pct'])):g}" for c in facts.get("closures", []) if c.get("area_pct") is not None]
    return v


@torch.no_grad()
def value_recovery_ab(nar, reader, scenes, use_feature):
    """Generate the masked chain; count GT values recovered anywhere in the chain and inside <think>."""
    nar.mask_digits = True
    in_chain = in_think = tot = 0
    for s in tqdm(scenes, desc=f"recover feat {'ON' if use_feature else 'OFF'}", unit="sc", leave=False):
        f = region_metadata(s)
        gt = _gt_values(f)
        if not gt:
            continue
        chain = generate_chain(nar, f, reader, s, use_feature=use_feature)   # masked prefix, think+answer
        tm = _THINK.search(chain); think = tm.group(1) if tm else ""
        for v in gt:
            tot += 1
            in_chain += (v in chain)
            in_think += (v in think)
    nar.mask_digits = False
    return in_chain, in_think, tot


def main():
    reader = RegionReader().to(device)
    reader.load_state_dict(torch.load("hybrid/checkpoints/reader.pt", map_location=device)); reader.eval()

    scenes = synthetic.scenes(max_scenes=int(os.environ.get("SCENES", 10_000)))
    from hybrid.model.captioner import objects_of
    scenes = [s for s in scenes if objects_of(s["objs"])]
    rng = random.Random(SEED); rng.shuffle(scenes)
    cut = int(len(scenes) * 0.75); tr, te = scenes[:cut], scenes[cut:]
    print(f"[masked-ab] scenes {len(scenes)} · train {len(tr)} · test {len(te)}", flush=True)

    rows_by_img = defaultdict(list)
    for r in load_local_csv(csv_path=synthetic.CSV):
        ip = (r.get("image_paths") or [None])[0]
        if ip:
            rows_by_img[ip].append(r)
    facts_by_img = {s["img"]: region_metadata(s) for s in tr}

    nar = Captioner()
    nar.dec.gradient_checkpointing_enable(); nar.dec.enable_input_require_grads()

    # MASKED grounding + MASKED fold (digit_dropout=1.0 → always masked) with the feature on. gate_reg
    # keeps the visual channel alive so it can carry the value the prompt no longer provides.
    nar.mask_digits = True
    train_grounding(nar, facts_by_img, epochs=int(os.environ.get("GROUND", 5)))
    nar.mask_digits = False
    train_answer(nar, reader, tr, rows_by_img, epochs=int(os.environ.get("ANSWER", 5)), rows_per=5,
                 use_feature=True, digit_dropout=1.0, gate_reg=1.0)

    on_c, on_t, n = value_recovery_ab(nar, reader, te, use_feature=True)
    off_c, off_t, _ = value_recovery_ab(nar, reader, te, use_feature=False)
    print(f"[MASKED-FEATURE A/B]  feature ON: value-in-chain {on_c}/{n}={on_c/max(1,n):.2f} · "
          f"value-in-think {on_t}/{n}={on_t/max(1,n):.2f}", flush=True)
    print(f"                      feature OFF: value-in-chain {off_c}/{n}={off_c/max(1,n):.2f} · "
          f"value-in-think {off_t}/{n}={off_t/max(1,n):.2f} · gate {float(nar.feat_gate):.4f}", flush=True)
    print("ON>>OFF (esp. value-in-think) = the feature supplies the masked value, emerging in reasoning.", flush=True)
    print("MASKED_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
