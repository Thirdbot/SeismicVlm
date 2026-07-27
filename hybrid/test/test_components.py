"""Decoupled COMPONENT tests — run AFTER training, on the saved weights. Separates the two mechanisms
so a low number is unambiguous (copy failed vs. reader handed the LM a wrong fact):

  COPY-MECHANISM (GT)     : inject GROUND-TRUTH facts → does the LM reproduce them in the evidence? The
                            PURE copy test (comparable to the canonical ~0.89 / 54-61). Reader EXCLUDED.
  COPY-MECHANISM (reader) : inject READER facts → same. The deployment/pipeline number (reader-capped) —
                            shown alongside so the gap = the reader's contribution, not the copy path.
  READER-MECHANISM        : reader detect vs GT — count MAE / dip MAE / class / per-fault mask dice.

Run:  python -m hybrid.test.test_components                     (stage_fold_narrator.pt, 100 scenes)
      CKPT=foldfuse_narrator.pt SCENES=60 python -m hybrid.test.test_components
"""
import os
import random

import torch
from tqdm.auto import tqdm

import hybrid.model.scenes as sc
sc.MAX_SCENES = int(os.environ.get("SCENES", 10_000))  # match the training run's SCENE_CAP (uncapped) for the same split

from hybrid.model.scenes import build_scenes
from hybrid.model.narrator import Narrator, scene_facts, objects_of
from hybrid.model.reader import InstanceReader, scene_to_gt
from hybrid.model.segmenter import field_dice
from hybrid.train.stage_reader_mask import reader_accuracy, reader_facts
from hybrid.train.stage_fold import fold_evidence
from hybrid.checkpoints import load_narrator

device = torch.device("cuda")
CKPT = os.environ.get("CKPT", "stage_fold_narrator.pt")
SEED = 42


def held_out():
    """Held-out test scenes. REAL=1 → real-field windows (same scene format via real.real_scenes) for the
    before/after real-field eval; else synthetic (same split as train.py load_split, seed 42, 0.75 cut)."""
    if os.environ.get("REAL_CSV"):
        from hybrid.data.real_csv import real_csv_scenes            # ungated full-coverage panels (pos+neg)
        _, _, te = real_csv_scenes()
        return te
    if os.environ.get("REAL"):
        from hybrid.data.real import real_scenes                    # legacy fault-centred windows
        _, _, te = real_scenes()                                    # real windows, synthetic format
        return te
    rng = random.Random(SEED)
    scenes = [s for s in build_scenes() if objects_of(s["objs"])]   # any object
    idx = list(range(len(scenes))); rng.shuffle(idx)
    cut = int(len(scenes) * 0.75)
    return [scenes[i] for i in idx[cut:]]


@torch.no_grad()
def copy_test(nar, reader, scenes, use_reader):
    """Inject facts (GT or reader-measured), generate evidence @ s2, count EXACT reproduction of ANY
    measured attribute (fault dip/throw + closure area — multi-object). use_reader=False = pure copy
    mechanism (GT in); True = pipeline (reader in, reader-capped)."""
    hit = tot = 0
    for s in tqdm(scenes, desc=f"copy ({'reader' if use_reader else 'GT'})", unit="sc", leave=False):
        facts = reader_facts(reader, s) if use_reader else scene_facts(s)
        if not (facts["faults"] or facts.get("closures")):
            continue
        ev = fold_evidence(nar, facts)                 # evidence @ s2, feature off
        vals = [f"{round(float(x['dip']), 1):g}" for x in facts["faults"]]
        vals += [f"{round(float(x['throw'])):g}" for x in facts["faults"] if x.get("throw") is not None]
        vals += [f"{round(float(c['area_pct'])):g}" for c in facts.get("closures", [])
                 if c.get("area_pct") is not None]
        for v in vals:
            tot += 1
            hit += (v in ev)
    return hit, tot


@torch.no_grad()
def reader_attrs(reader, scenes):
    """Reader CLASS-DRIVEN attribute accuracy vs GT — the raw attributes that feed (and CAP) the copy
    pipeline: fault dip/throw + closure area, MAE by sorted-match within class. This is the test that
    'corresponds' to COPY-pipeline: once correspondence is fixed (evidence states each attr), this MAE
    is what separates COPY-GT (perfect facts) from COPY-pipeline (reader facts)."""
    dip, throw, area = [], [], []
    for s in tqdm(scenes, desc="reader-attrs", unit="sc", leave=False):
        gt = scene_to_gt(s); pred = reader.detect(s["smap"])
        for key, cls, acc in (("dip", 1, dip), ("throw", 1, throw), ("area", 2, area)):
            g = sorted(o[key] for o in gt if o["cls"] == cls and o.get(key) is not None)
            p = sorted(o[key] for o in pred if o["cls"] == cls)
            for i in range(min(len(g), len(p))):
                acc.append(abs(p[i] - g[i]))
    def mae(x):
        return (sum(x) / len(x), len(x)) if x else (None, 0)
    return mae(dip), mae(throw), mae(area)


def main():
    reader = InstanceReader().to(device)
    reader_pt = os.environ.get("READER", "hybrid/checkpoints/reader.pt")   # READER=…/reader_real.pt for AFTER
    sd = torch.load(reader_pt, map_location=device)
    if any(k.startswith("real_adapter") for k in sd):     # real-field ckpt = synthetic base + real adapter
        reader.add_real_adapter()                         # recreate the adapter module so keys match, then load
    reader.load_state_dict(sd)
    reader.eval()
    nar = Narrator(); nar.set_stage("s3"); load_narrator(nar, CKPT)
    nar.eval_mode()
    te = held_out()
    print(f"[test] narrator {CKPT} · reader {reader_pt} · {sc.MAX_SCENES} scenes · held-out {len(te)}", flush=True)

    # ---- COPY MECHANISM (the two yardsticks, side by side) ----
    g_hit, g_tot = copy_test(nar, reader, te, use_reader=False)
    r_hit, r_tot = copy_test(nar, reader, te, use_reader=True)
    print(f"[COPY-mechanism  GT ] {g_hit}/{g_tot} = {g_hit/max(1,g_tot):.2f}   <- PURE copy (inject GT), the 0.89-style test",
          flush=True)
    print(f"[COPY-pipeline reader] {r_hit}/{r_tot} = {r_hit/max(1,r_tot):.2f}   <- reader-capped (deployment)", flush=True)

    # ---- READER MECHANISM ----
    a = reader_accuracy(reader, te)
    dices = []
    for s in tqdm(te, desc="mask-dice", unit="sc", leave=False):
        gt = scene_to_gt(s)
        if not gt:
            continue
        ml = reader.tf_masks(s["smap"], gt)
        dices += [field_dice(ml[i], o["mask_full"].to(device)) for i, o in enumerate(gt) if o["cls"] == 1]
    md = sum(dices) / len(dices) if dices else 0.0
    cmae = a["count"][0] if a["count"] and a["count"][0] is not None else float("nan")
    dmae = a["dip"][0] if a["dip"] and a["dip"][0] is not None else float("nan")
    print(f"[READER-mechanism] count MAE {cmae:.2f} · class {a['cls'][0]}/{a['cls'][1]} · "
          f"mask dice {md:.2f}", flush=True)
    d, t, ar = reader_attrs(reader, te)                 # class-driven attribute MAE (caps COPY-pipeline)
    def fmt(m, u): return f"{m[0]:.1f}{u}(n{m[1]})" if m[0] is not None else "n0"
    print(f"[READER-attrs]  dip {fmt(d, 'deg')} · throw {fmt(t, 'ms')} · area {fmt(ar, '%')}", flush=True)
    print("COMPONENTS_DONE", flush=True)


if __name__ == "__main__":
    main()
