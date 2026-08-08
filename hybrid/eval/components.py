"""Decoupled COMPONENT tests — run AFTER training, on the saved weights. Separates the two mechanisms
so a low number is unambiguous (copy failed vs. reader handed the LM a wrong fact):

  COPY-MECHANISM (GT)     : inject GROUND-TRUTH facts → does the LM reproduce them in the evidence? The
                            PURE copy test (comparable to the canonical ~0.89 / 54-61). Reader EXCLUDED.
  COPY-MECHANISM (reader) : inject READER facts → same. The deployment/pipeline number (reader-capped) —
                            shown alongside so the gap = the reader's contribution, not the copy path.
  READER-MECHANISM        : reader detect vs GT — count MAE / dip MAE / class / per-fault mask dice.

Run:  python -m hybrid.eval.components                     (stage3_answer.pt, 100 scenes)
      CKPT=foldfuse_narrator.pt SCENES=60 python -m hybrid.eval.components
"""
import os
import random
import re

import torch
from tqdm.auto import tqdm

from hybrid.model.captioner import Captioner, region_metadata
from hybrid.model.reader import RegionReader, scene_to_gt
from hybrid.model.geometry import field_dice
from hybrid.model.text_metrics import _ANS
from hybrid.stages.stage2_reader import reader_accuracy, reader_facts
from hybrid.stages.stage3_answer import generate_evidence, generate_chain
from hybrid.eval.metrics import chair, bleu4, meteor, cider_d, map50, giou, box_iou
from hybrid.checkpoints import load_narrator

device = torch.device("cuda")
CKPT = os.environ.get("CKPT", "stage3_answer.pt")
SCENES = int(os.environ.get("SCENES", 10_000))       # cap (uncapped by default → same split as training)


def held_out():
    """Held-out test scenes via the dataset APIs (same schema): DATASET=smeaheia → real held-out;
    else synthetic held-out (seed-42, 0.75 image-level split)."""
    if os.environ.get("DATASET") == "smeaheia":
        from hybrid.data import smeaheia
        return smeaheia.scenes()[2]
    from hybrid.data import synthetic
    return synthetic.split(max_scenes=SCENES)[2]


@torch.no_grad()
def copy_test(nar, reader, scenes, use_reader):
    """Inject facts (GT or reader-measured), generate evidence @ s2, count TOLERANCE reproduction of ANY
    measured attribute (fault dip/throw + closure area): a value is "copied" if some evidence number is within
    ±2% of it (±0.5 absolute floor). Mirrors CHAIR's tolerance — NOT substring-inflated (a value's digits hiding
    inside a coordinate don't count) and NOT exact-match harsh (a faithful copy with minor precision drift still
    counts; only a genuine value change is a miss). use_reader=False = pure copy (GT in); True = pipeline (reader-capped)."""
    hit = tot = 0
    for s in tqdm(scenes, desc=f"copy ({'reader' if use_reader else 'GT'})", unit="sc", leave=False):
        facts = reader_facts(reader, s) if use_reader else region_metadata(s)
        if not (facts["faults"] or facts.get("closures")):
            continue
        ev = generate_evidence(nar, facts)                 # evidence @ s2, feature off
        vals = [float(x['dip']) for x in facts["faults"]]
        vals += [float(x['throw']) for x in facts["faults"] if x.get("throw") is not None]
        vals += [float(c['area_pct']) for c in facts.get("closures", []) if c.get("area_pct") is not None]
        ev_nums = [float(x) for x in re.findall(r"\d+\.?\d*", ev)]   # the numbers STATED in the evidence
        for v in vals:                                     # copied = some evidence number within ±2% (±0.5 floor) of v:
            tot += 1                                        # not substring-inflated (no coord-digit hits), not exact-harsh
            hit += any(abs(en - v) <= max(0.5, 0.02 * abs(v)) for en in ev_nums)
    return hit, tot


@torch.no_grad()
def reader_attrs(reader, scenes):
    """Reader CLASS-DRIVEN attribute accuracy vs GT — the raw attributes that feed (and CAP) the copy
    pipeline: fault dip/throw + closure area, MAE by sorted-match within class. This is the test that
    'corresponds' to COPY-pipeline: once correspondence is fixed (evidence states each attr), this MAE
    is what separates COPY-GT (perfect facts) from COPY-pipeline (reader facts).
    NOTE: the dip/throw MAE here pairs values SORTED/rank-matched within class (smallest-with-smallest),
    so it is an OPTIMISTIC lower bound, not a spatial match; the reported paper dip/throw is
    benchmark.py's centroid-matched number."""
    dip, throw, area = [], [], []
    for s in tqdm(scenes, desc="reader-attrs", unit="sc", leave=False):
        gt = scene_to_gt(s); pred = reader.detect(reader.encode(s))
        for key, cls, acc in (("dip", 1, dip), ("throw", 1, throw), ("area", 2, area)):
            g = sorted(o[key] for o in gt if o["cls"] == cls and o.get(key) is not None)
            p = sorted(o[key] for o in pred if o["cls"] == cls)
            for i in range(min(len(g), len(p))):
                acc.append(abs(p[i] - g[i]))
    def mae(x):
        return (sum(x) / len(x), len(x)) if x else (None, 0)
    return mae(dip), mae(throw), mae(area)


def _reference_answers():
    """{image_path: reference answer text} from the synthetic CSV (grounded-captioning references).
    Empty for real (no narration GT) → narration-overlap metrics are skipped there."""
    if os.environ.get("DATASET") == "smeaheia":
        return {}
    from hybrid.data.schema import load_local_csv
    from hybrid.data import synthetic
    ref = {}
    for r in load_local_csv(csv_path=synthetic.CSV):
        ip = (r.get("image_paths") or [None])[0]
        ans = (r.get("answer") or "").strip()
        if ip and ans and ip not in ref:
            ref[ip] = ans
    return ref


@torch.no_grad()
def academic_table(nar, reader, te):
    """Region-conditioned text generation metrics: generate the narration per held-out scene, then
    CHAIR_I (faithfulness — both datasets) + BLEU-4/METEOR/CIDEr-D vs the dataset answer (synthetic)."""
    ref_by_img = _reference_answers()
    hyps, refs, chairs, meteors = [], [], [], []
    for s in tqdm(te, desc="narrate", unit="sc", leave=False):
        facts = reader_facts(reader, s)
        if not (facts["faults"] or facts.get("closures")):
            continue
        chain = generate_chain(nar, facts, reader, s)
        m = _ANS.search(chain); ans = m.group(1).strip() if m else chain
        c, n = chair(ans, facts)
        if n:
            chairs.append(c)
        ref = ref_by_img.get(s["img"])
        if ref:
            hyps.append(ans); refs.append(ref); meteors.append(meteor(ans, ref))
    ch = sum(chairs) / len(chairs) if chairs else float("nan")
    print(f"[FAITHFULNESS] CHAIR_I {ch:.3f} (n={len(chairs)}; 0 = every stated number is a measured fact)", flush=True)
    if hyps:
        print(f"[NARRATION] BLEU-4 {bleu4(hyps, refs):.3f} · METEOR {sum(meteors)/len(meteors):.3f} · "
              f"CIDEr-D {cider_d(hyps, refs):.3f}  (vs dataset answer, n={len(hyps)})", flush=True)


@torch.no_grad()
def reader_spatial(reader, te):
    """Box precision: mAP@0.5 (VOC) + mean GIoU of matched detections. Reader bboxes are 0-100 →
    normalized to 0-1 to match the scene's normalized GT boxes; class-aware, greedy IoU matching."""
    preds, gts, gious = [], [], []
    for s in tqdm(te, desc="boxes", unit="sc", leave=False):
        img = s["img"]
        # GT via scene_to_gt (NOT raw s["objs"]) so boxes are scored against the SAME object population
        # as every other metric — raw objs still contain the degenerate masks scene_to_gt drops.
        gt = [(o["bbox"], int(o["cls"])) for o in scene_to_gt(s)]
        pr = [([b / 100.0 for b in o["bbox"]], int(o["cls"])) for o in reader.detect(reader.encode(s))]
        for b, c in gt:
            gts.append((b, c, img))
        for b, c in pr:
            preds.append((b, 1.0, c, img))                       # no per-object confidence → score 1.0
            same = [gb for gb, gc in gt if gc == c]
            if same:
                gious.append(max(giou(b, gb) for gb in same))
    m, aps = map50(preds, gts)
    mg = sum(gious) / len(gious) if gious else float("nan")
    return m, mg


def main():
    reader = RegionReader().to(device)
    reader_pt = os.environ.get("READER", "hybrid/checkpoints/reader.pt")   # READER=…/reader_real.pt for AFTER
    sd = torch.load(reader_pt, map_location=device)
    if any(k.startswith("real_adapter") for k in sd):     # real-field ckpt = synthetic base + real adapter
        reader.add_real_adapter()                         # recreate the adapter module so keys match, then load
    reader.load_state_dict(sd)
    reader.eval()
    from hybrid.stages.stage2_reader import _build_encoder
    reader.set_encoder(_build_encoder())               # encoder in-model (pixels -> grid)
    nar = Captioner(); nar.set_stage("s3"); load_narrator(nar, CKPT)
    nar.eval_mode()
    te = held_out()
    ds = os.environ.get("DATASET", "synthetic")
    print(f"[test] narrator {CKPT} · reader {reader_pt} · dataset {ds} · held-out {len(te)}", flush=True)

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
        ml = reader.tf_masks(reader.encode(s), gt)
        dices += [field_dice(ml[i], o["mask_full"].to(device)) for i, o in enumerate(gt) if o["cls"] == 1]
    md = sum(dices) / len(dices) if dices else 0.0
    cmae = a["count"][0] if a["count"] and a["count"][0] is not None else float("nan")
    dmae = a["dip"][0] if a["dip"] and a["dip"][0] is not None else float("nan")
    print(f"[READER-mechanism] count MAE {cmae:.2f} · class {a['cls'][0]}/{a['cls'][1]} · "
          f"mask dice {md:.2f}", flush=True)
    d, t, ar = reader_attrs(reader, te)                 # class-driven attribute MAE (caps COPY-pipeline)
    def fmt(m, u): return f"{m[0]:.1f}{u}(n{m[1]})" if m[0] is not None else "n0"
    print(f"[READER-attrs]  dip {fmt(d, 'deg')} · throw {fmt(t, 'ms')} · area {fmt(ar, '%')}", flush=True)
    mAP, mGIoU = reader_spatial(reader, te)             # box precision (spatial suite)
    print(f"[READER-boxes]  mAP@0.5 {mAP:.3f} · mean GIoU {mGIoU:.3f}  (score=1.0 → single PR point)", flush=True)

    # ---- ACADEMIC TABLE (region-conditioned text generation): faithfulness + narration overlap ----
    academic_table(nar, reader, te)
    print("COMPONENTS_DONE", flush=True)


if __name__ == "__main__":
    main()
