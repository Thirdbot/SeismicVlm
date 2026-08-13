"""Qualitative inference viz — the full presentation package per held-out scene.

For a few held-out scenes: run the end-to-end pipeline (reader measures facts + masks -> LM copies them
into grounded narration) and emit, as ONE composite PNG, the five things needed to read the result:
seismic image · GROUND-TRUTH segmentation · PREDICTED overlay · the question · the LM response
(evidence / think / answer). Also prints the chain and tallies narration malformation.

  DATASET=synthetic|thebe|cracks|smeaheia · READER=…/reader.pt|reader_joint_full.pt · CKPT=…narrator.pt
  N=3 · OUT=<dir> · QUESTION="…"  (set QUESTION to override the default grounding/locating/describing/geology set)
Run:  DATASET=synthetic python -m hybrid.eval.inference
      DATASET=smeaheia READER=hybrid/checkpoints/reader_joint_full.pt python -m hybrid.eval.inference
"""
import os

import numpy as np
import torch
from tqdm.auto import tqdm

from hybrid.model.captioner import Captioner, region_metadata
from hybrid.model.reader import RegionReader
from hybrid.model.registry import ID_CLASS
from hybrid.model.text_metrics import _ANS, _THINK
from hybrid.stages.stage2_reader import reader_facts
from hybrid.stages.stage3_answer import generate_evidence, generate_think_answer, _clean
from hybrid.eval.metrics import malform
from hybrid.checkpoints import load_narrator
from hybrid.eval.viz import overlay_classes, overlay_gt, panel

device = torch.device("cuda")
DATASET = os.environ.get("DATASET", "synthetic")     # synthetic | smeaheia | thebe | cracks
N = int(os.environ.get("N", 3))
SCENES = int(os.environ.get("SCENES", 10_000))
MIN_F = int(os.environ.get("MIN_F", 0))              # keep scenes with ≥ MIN_F GT faults (MIN_F=1 → skip closure-only)
MAX_F = int(os.environ.get("MAX_F", 10 ** 9))        # … and ≤ MAX_F (MAX_F=1 → single-fault scenes for cleaner narration)
IMG = os.environ.get("IMG", "")                      # pin to scenes whose image basename contains this (e.g. a specific tile)
BOXES = os.environ.get("BOXES", "1").lower() not in ("0", "false", "no")   # BOXES=0 → clean mask-only overlays (paper figures)
OUT = os.environ.get("OUT", "hybrid/inference")      # output dir for rendered chains + overlays (override with OUT=…)

_Q = os.environ.get("QUESTION", "")
QUESTIONS = ([("QUESTION", _Q)] if _Q else [                 # a spread of grounding / locating / describing / geology
    ("GROUNDING",  "Where is the fault in this section?"),
    ("LOCATING",   "Which part of the section does the fault occupy — top, middle, or bottom?"),
    ("DESCRIBING", "Describe the main fault visible in this section."),
    ("GEOLOGY",    "What does the fault's dip indicate about the deformation — is it a steep normal fault?"),
])


def held_out():
    if DATASET == "synthetic":
        from hybrid.data import synthetic
        return synthetic.split(max_scenes=SCENES)[2]
    import importlib                                             # any real survey: thebe | cracks | smeaheia
    return importlib.import_module(f"hybrid.data.{DATASET}").scenes()[2]


def main():
    reader = RegionReader().to(device)
    rpt = os.environ.get("READER", "hybrid/checkpoints/reader.pt")
    sd = torch.load(rpt, map_location=device)
    is_real = any(k.startswith("real_adapter") for k in sd)
    if is_real:
        reader.add_real_adapter()
    reader.load_state_dict(sd); reader.eval()
    from hybrid.stages.stage2_reader import _build_encoder
    reader.set_encoder(_build_encoder())               # encoder in-model (pixels -> grid)
    nar = Captioner(); nar.set_stage("s3")
    load_narrator(nar, os.environ.get("CKPT", "stage3_answer.pt")); nar.eval_mode()

    te = held_out()
    os.makedirs(OUT, exist_ok=True)                                # ensure the overlay output dir exists
    print(f"[infer] DATASET={DATASET} · reader={rpt} · held-out {len(te)} · showing {N}", flush=True)
    shown = 0
    mf = []                                                        # narration malformation tally (degenerate-language check)
    for s in tqdm(te, desc="inference", unit="sc", leave=False):
        if IMG and IMG not in os.path.basename(s["img"]):                  # pin to a specific scene when requested
            continue
        gf = region_metadata(s)
        gt_faults = [o for o in s.get("objs", []) if int(o["cls"]) == 1]   # raw fault objs incl. mask-only
        if not (gt_faults or gf.get("closures")):                          # (CRACKS: dip gated → absent from gf["faults"])
            continue
        if not (MIN_F <= len(gt_faults) <= MAX_F):                         # scene-complexity filter (viz only)
            continue
        facts = reader_facts(reader, s)                            # deployment: reader measures the facts
        if is_real:
            facts["derived"] = {}                                  # strip derived tier on real (relations reasoned, not asserted)
        objs, masks = reader.detect(reader.encode(s), want_masks=True)    # predicted instances + masks

        ev = generate_evidence(nar, facts)                        # LM copies the measured numbers, once
        qas, chains = [], []
        for tag, q in QUESTIONS:                                   # then reasons + answers per question
            chain = _clean(generate_think_answer(nar, facts, ev, question=q))
            chains.append(chain)
            th, an = _THINK.search(chain), _ANS.search(chain)
            qas.append({"tag": tag, "q": q, "think": th.group(1).strip() if th else "",
                        "answer": an.group(1).strip() if an else chain.strip()})
        mf.append(malform(" ".join(chains)))

        base = os.path.splitext(os.path.basename(s["img"]))[0]
        gt_png = os.path.join(OUT, f"infer_{DATASET}_{shown}_{base}_gt.png")
        pred_png = os.path.join(OUT, f"infer_{DATASET}_{shown}_{base}_pred.png")
        overlay_gt(s, gt_png, boxes=BOXES)                         # GROUND-TRUTH: mask (+ bbox + class if BOXES)
        overlay_classes(s["img"], objs, masks, pred_png, boxes=BOXES)   # PREDICTION: mask (+ bbox + class if BOXES)
        f0 = (facts.get("faults") or [{}])[0]
        mline = f"{len(objs)} object(s) · classes {[ID_CLASS.get(o['cls']) for o in objs]}"
        if f0.get("dip") is not None:
            thr = f" · throw {f0['throw']:.0f} ms" if f0.get("throw") is not None else ""
            mline += f" · fault dip {f0['dip']:.1f}°{thr} · center {f0.get('center')}"
        comp = os.path.join(OUT, f"infer_{DATASET}_{shown}_{base}.png")
        panel(s["img"], gt_png, pred_png, [("measured", mline), ("evidence", ev.strip())],
              qas, comp, title=f"{DATASET} · {base}")

        gt_dips = [round(float(o["meas"][0]), 1) for o in gt_faults if float(o["mmask"][0]) > 0]   # dips only where valid (CRACKS: none)
        print(f"\n=== {DATASET} #{shown}  img={base} ===", flush=True)
        print(f"[GT]     {len(gt_faults)} faults dips={gt_dips} · {len(gf.get('closures', []))} closures", flush=True)
        print(f"[reader] {len(objs)} objects, classes={[o['cls'] for o in objs]}, "
              f"dips={[round(o['dip'], 1) for o in objs if o['cls'] == 1]}", flush=True)
        print(f"[evidence] {ev.strip()}", flush=True)
        for qa in qas:
            print(f"  [{qa['tag']}] {qa['q']}\n     answer: {qa['answer']}", flush=True)
        print(f"[panel]  {comp}", flush=True)
        shown += 1
        if shown >= N:
            break
    if mf:                                                        # aggregate degenerate-language tally (lower = cleaner)
        agg = {k: float(np.mean([d[k] for d in mf])) for k in ("unclosed", "keyleak", "confab", "truncated", "length")}
        print(f"\n[MALFORM {DATASET}] n={len(mf)} · unclosed {agg['unclosed']:.2f} · keyleak {agg['keyleak']:.2f} · "
              f"confab {agg['confab']:.2f} · truncated {agg['truncated']:.2f} · length {agg['length']:.0f} "
              f"(baseline before fix: unclosed 2.0 · keyleak 3.0 · confab 2.4; lower=cleaner)", flush=True)
    print("INFER_DONE", flush=True)


if __name__ == "__main__":
    main()
