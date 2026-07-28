"""Train the main model end to end — reader + grounded narrator + reasoning.

Stage 1 : geology adapter (frozen; built once via stage1_geology) — the thinking capability.
Stage 2 : instance READER (facts = count/class/dip/throw + masks; frozen NCS) AND the grounding LoRA
          = EVIDENCE COPY (copies the facts into raw dataset evidence; target ENDS at </evidence>).
Stage 3 : the FUSE FOLD — fuse LoRA (geology + grounding FROZEN). Trains the grounded <answer> as the
          completion after a FULL, MASKED <think>: un-suppresses the think, gives the answer a trained
          home (no truncation), and protects the copy. </think> in the masked prefix; only <answer>
          supervised. Injection = digit tokens (measured+derived) + gated <feature>. (Replaces the
          failed STaR/reason-adapter path.)
Inference is a STAGE-SWITCH: evidence @ s2 (clean copy) -> think+answer @ s3 (fuse).
Then evaluate on HELD-OUT: reader count/dip · copy fidelity · per-object mask dice · reasoning chains.

Stage 1 (geology adapter) is built once via `python -m hybrid.stages.stage1_geology`.
Run:  python -m hybrid.stages.run_train
"""
import random
from pathlib import Path

import torch
from mpmath.math2 import INF
from tqdm.auto import tqdm

import hybrid.data.loader as sc
from hybrid.data.synthetic import CSV     # synthetic dataset path (unified schema)
SCENE_CAP = INF             # UNCAPPED — dataset = 406 imgs @100×507; all smaps ≈0.23GB GPU (feature maps are
                               # tiny). The old "200 OOM" was zombie-process contention, not smap memory. The one
                               # thing that scales is resident GT masks (~1.7GB @406). Fold rows capped in stage_fold.
sc.MAX_SCENES = SCENE_CAP
sc.CSV = CSV                # point the unified loader at the synthetic CSV

from hybrid.data.loader import build_scenes
from hybrid.model.narrator import (Narrator, objects_of, scene_facts, facts_to_kv,
                                   K_DIP, K_THROW, K_AREA)
from hybrid.stages.stage2_reader import train_reader, reader_accuracy, reader_facts
from hybrid.model.reader import InstanceReader, scene_to_gt
from hybrid.model.geometry import field_dice
from hybrid.stages.stage2_grounding import train_grounding
from hybrid.stages.stage3_fold import train_fold, fold_chain, fold_eval
from hybrid.data.schema import load_local_csv

device = torch.device("cuda")
SEED = 42
READER_EPOCHS = 200      # more data (406 scenes) + more epochs → better reader dip/throw/class/mask
CKPT = Path("hybrid/checkpoints")


def load_split():
    rng = random.Random(SEED)
    scenes = [s for s in build_scenes() if objects_of(s["objs"])]   # any object (fault OR closure)
    idx = list(range(len(scenes))); rng.shuffle(idx)
    cut = int(len(idx) * 0.75)
    return scenes, [scenes[i] for i in idx[:cut]], [scenes[i] for i in idx[cut:]]


def _fmt(mn): return f"{mn[0]:.2f}(n{mn[1]})" if mn and mn[0] is not None else "n0"


def main():
    scenes, tr, te = load_split()
    print(f"[train] scenes {len(scenes)} · train {len(tr)} · test {len(te)}", flush=True)

    # ---- Stage 2: instance reader (facts) ----
    reader_pt = CKPT / "reader.pt"
    reader = None
    if reader_pt.exists():                       # reuse a trained reader (delete reader.pt to retrain)
        try:
            reader = InstanceReader().to(device)
            reader.load_state_dict(torch.load(reader_pt, map_location=device))
            reader.eval()
            print("[train] loaded cached reader.pt (skip reader training)", flush=True)
        except RuntimeError as e:                # arch changed (e.g. registry heads) → cached ckpt stale
            print(f"[train] cached reader.pt incompatible with current arch → retraining ({str(e)[:80]}…)", flush=True)
            reader = None
    if reader is None:
        reader = train_reader(tr, epochs=READER_EPOCHS)
    for tag, sp in (("train", tr), ("test(held-out)", te)):
        a = reader_accuracy(reader, sp)
        dices = []
        for s in tqdm(sp, desc=f"dice/{tag}", unit="sc", leave=False):
            gt = scene_to_gt(s)
            if not gt:
                continue
            ml = reader.tf_masks(s["smap"], gt)
            dices += [field_dice(ml[i], o["mask_full"].to(device))
                      for i, o in enumerate(gt) if o["cls"] == 1]
        md = (sum(dices) / len(dices), len(dices)) if dices else (None, 0)
        print(f"[reader {tag}] count MAE {_fmt(a['count'])} · dip MAE {_fmt(a['dip'])}deg · "
              f"class {a['cls'][0]}/{a['cls'][1]} · mask dice {_fmt(md)}", flush=True)
    torch.save(reader.state_dict(), CKPT / "reader.pt")

    # ---- Stage 2 (LM): grounding LoRA = EVIDENCE COPY (set_stage s2). Copies the injected facts into
    # the raw dataset evidence text; this is the copy that the combined stage must NOT disturb. ----
    facts_by_img = {s["img"]: scene_facts(s) for s in tr}
    nar = Narrator()
    nar.dec.gradient_checkpointing_enable()      # recompute activations in backward -> fits the 5.67GB GPU
    nar.dec.enable_input_require_grads()
    import os                                     # DIAGNOSTIC (remove after): warm base to isolate port vs from-scratch
    if os.environ.get("WARM_BASE"):
        from hybrid.checkpoints import load_narrator
        load_narrator(nar, "stage34_narrator.pt"); print("[diag] loaded warm base stage34_narrator.pt", flush=True)
    train_grounding(nar, facts_by_img)   # grounding target ENDS at </evidence> (un-suppressed think)

    # ---- Stage 3 — the FUSE FOLD (set_stage s3; geology + grounding FROZEN). Trains the grounded
    # <answer> as the completion after a FULL, MASKED <think>: un-suppresses the think (fuse's delta
    # counteracts grounding's stop-bias), gives the answer a trained home (no truncation), and protects
    # the copy (grounding frozen). </think> in the masked prefix; only <answer> supervised. Injection =
    # digit tokens (measured+derived) + gated <feature>. Replaces the failed STaR/reason-adapter path. ----
    from collections import defaultdict
    rows_by_img = defaultdict(list)
    for r in load_local_csv(csv_path=CSV):
        ip = (r.get("image_paths") or [None])[0]
        if ip:
            rows_by_img[ip].append(r)
    # ---- copy fidelity helper — measured BEFORE and AFTER the fold to prove the fuse fold (grounding
    # frozen) did NOT tamper with the evidence copy ----

    def copy_score():
        nar.set_stage("s2"); nar.use_feature = False; nar.eval_mode(); hit = tot = 0   # evidence @ s2 (fuse OFF)
        for s in tqdm(te[:20], desc="copy-score", unit="sc", leave=False):
            facts = reader_facts(reader, s)
            if not (facts["faults"] or facts.get("closures")):
                continue
            vals = [v for k, v in facts_to_kv(facts) if k in (K_DIP, K_THROW, K_AREA)]  # any measured attr
            out = nar.generate(facts, question="", max_new_tokens=140)
            for d in vals:
                tot += 1; hit += (d in out)
        return hit, tot

    b_hit, b_tot = copy_score()
    print(f"[copy BEFORE fold] {b_hit}/{b_tot}", flush=True)

    train_fold(nar, reader, tr, rows_by_img, epochs=8, rows_per=5)   # fuse fold (grounding frozen)

    a_hit, a_tot = copy_score()
    print(f"[copy AFTER fold]  {a_hit}/{a_tot}  (must ~match BEFORE — proves fuse fold protects copy)", flush=True)

    # FEATURE A/B (reasoning): does the gated <feature>_i soft token help grounded reasoning? Same
    # held-out, feature ON vs OFF. gate ≈ 0 ⇒ ON≈OFF (digits suffice); gate opening + ON>OFF ⇒ it helps.
    m = fold_eval(nar, reader, te, use_feature=True)
    m0 = fold_eval(nar, reader, te, use_feature=False)
    print(f"[fold-eval feat ON ] present {m['present']:.2f} · clean {m['clean']:.2f} · grounded {m['grounded']:.2f} "
          f"· think {m['think']:.2f}  (n={m['n']})", flush=True)
    print(f"[fold-eval feat OFF] present {m0['present']:.2f} · clean {m0['clean']:.2f} · grounded {m0['grounded']:.2f} "
          f"· think {m0['think']:.2f}  · gate {float(nar.feat_gate):.4f}", flush=True)

    # ---- reasoning chains, one greedy pass via the stage-switch (evidence @ s2 -> think+answer @ s3),
    # printed with GT facts so grounding/faithfulness can be judged by eye ----
    shown = 0
    for s in te:
        f = scene_facts(s)
        if not (f["faults"] or f.get("closures")):
            continue
        dips = [round(float(x["dip"]), 1) for x in f["faults"]]
        areas = [round(float(c["area_pct"])) for c in f.get("closures", [])]
        print(f"\n[reason] FACTS {len(f['faults'])} faults dips={dips} · "
              f"{len(f.get('closures', []))} closures areas={areas} derived={f.get('derived')}",
              flush=True)
        print(f"[reason] {fold_chain(nar, f, reader, s).replace(chr(10), ' ')}", flush=True)  # mixed question (Q_MIX default)
        shown += 1
        if shown >= 8:
            break
    print("MAIN_MODEL_DONE", flush=True)


if __name__ == "__main__":
    main()
