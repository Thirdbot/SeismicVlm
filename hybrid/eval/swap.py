"""Causal SWAP test — the seam-faithfulness probe behind tab:faith's flip/baseline rows.

The copy seam is non-differentiable: the narration can only restate numbers injected as markers
(hybrid/model/captioner.py _obj_markers). This test PROVES the dependence is causal by intervening
on the injected values and checking the narration follows — over EVERY numeric attribute of EVERY
injected object of EVERY class (not just one fault):

  channels : dip, throw (faults) · area (closure/salt/onlap) · center, bbox (all objects)
  tally     : one instance PER (object, attribute), aggregated across all scenes — so n is the real
              count of injected attributes, not a per-scene sample.
  baseline : inject facts UNCHANGED → does the narration state the ORIGINAL value? (control = the
             attribute's natural statement rate, NOT a ceiling on flip)
  flip     : swap that ONE channel on ALL objects that carry it, regenerate → does the narration
             follow the SWAPPED value?

Scope = ALL objects, UNCAPPED — the same denominator copy_test uses — so the swap BASELINE for
dip/throw/area is directly comparable to the copy score (GT 0.77 / reader 0.89): a cross-check that the
test itself is sound. Objects past MAX_OBJ are never injected, so they surface as honest baseline+flip
misses (the narration-budget cost copy_test also pays), reported as a note.
A value is matched with copy_test's tolerance (±2%, ±0.5 floor) on the value AS INJECTED (dip→1dp,
throw/area→int, center/bbox→int); multi-number attributes (center=2, bbox=4) count as a hit only if
ALL their component numbers land — a stricter, honest bar that exposes which channels are cleanly
isolable (the paper left area/bbox as "not isolable → future work"; this measures them).

Reuses the deployment injection + generation paths; trains nothing, edits no existing file. Because
the original tab:faith run was an ad-hoc, never-committed script, these reproducible numbers may
differ from the paper's 0.92/0.78 — if so, adopt THESE (committed reproducer) and update the paper.

Run:  DATASET=synthetic READER=hybrid/checkpoints/reader.pt CKPT=stage3_narrator.pt python -m hybrid.eval.swap
      USE_READER=1 …   # swap reader-measured facts instead of GT (default 0 = GT-injected, clean isolate)
"""
import os
import re

import torch
from tqdm.auto import tqdm

from hybrid.model.captioner import Captioner, region_metadata, _region_objs
from hybrid.model.reader import RegionReader
from hybrid.stages.stage2_reader import reader_facts, _build_encoder
from hybrid.stages.stage3_answer import generate_evidence
from hybrid.checkpoints import load_narrator
from hybrid.eval.components import held_out, CKPT

device = torch.device("cuda")
USE_READER = os.environ.get("USE_READER", "0").lower() not in ("0", "false", "no")

CHANNELS = ["dip", "throw", "area", "center", "bbox"]
BUCKETS = (("faults", "fault"), ("closures", "closure"), ("salts", "salt"), ("onlaps", "onlap"))
_NUM = re.compile(r"\d+\.?\d*")


def _all_objs(facts):
    """EVERY object of every class, UNCAPPED — matches copy_test's uncapped denominator so the swap
    baseline is directly comparable to the copy score. Objects past MAX_OBJ are never injected, so they
    show as honest baseline+flip misses (the narration-budget cost), exactly as copy_test counts them."""
    return [(cls, o) for bk, cls in BUCKETS for o in facts.get(bk, [])]


def _nums(text):
    return [float(x) for x in _NUM.findall(text)]


def _hit(v, nums):
    """copy_test tolerance: some stated number within ±2% of v (±0.5 absolute floor)."""
    return any(abs(n - v) <= max(0.5, 0.02 * abs(v)) for n in nums)


def _hit_attr(val, nums):
    """A (possibly multi-number) attribute is copied iff ALL its component numbers land."""
    comps = val if isinstance(val, (tuple, list)) else (val,)
    return all(_hit(float(c), nums) for c in comps)


def _attr(ch, o):
    """The attribute value AS INJECTED (rounded exactly as _obj_markers does), or None if absent."""
    if ch == "dip":
        return round(float(o["dip"]), 1) if o.get("dip") is not None else None
    if ch == "throw":
        return float(round(float(o["throw"]))) if o.get("throw") is not None else None
    if ch == "area":
        return float(round(float(o["area_pct"]))) if o.get("area_pct") is not None else None
    if ch == "center":
        b = o.get("bbox") or [0, 0, 0, 0]
        c = o.get("center") or [int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)]
        return (int(c[0]), int(c[1]))
    if ch == "bbox":
        b = o.get("bbox")
        return tuple(int(x) for x in b) if b else None
    return None


def _swap(ch, val):
    """FIXED off-distribution swap, delta >> tolerance, stays plausible."""
    if ch == "dip":
        return round((val + 30) if val <= 55 else (val - 30), 1)     # in ~[5,85]
    if ch == "throw":
        return float(round(val + 60))                                # +60 ms
    if ch == "area":
        return float(round((val + 25) if val <= 50 else (val - 25))) # in ~[0,75]
    if ch == "center":
        return (int(val[0]) + 41, int(val[1]) + 41)                  # +41 px each
    if ch == "bbox":
        return tuple(int(v) + 29 for v in val)                       # +29 px each corner


def _mod(facts, ch):
    """New facts with channel `ch` swapped on every object that carries it. Shallow (new object dicts,
    tensors shared by ref, never mutated). When swapping bbox, PIN center to its injected value so the
    bbox swap doesn't also move the (bbox-derived) center — keeps the two channels isolated."""
    nf = dict(facts)
    for bk, _cls in BUCKETS:
        if bk not in facts:
            continue
        out = []
        for o in facts[bk]:
            no = dict(o)
            cur = _attr(ch, o)
            if cur is not None:
                if ch == "center":
                    no["center"] = list(_swap(ch, cur))
                elif ch == "bbox":
                    no["center"] = list(_attr("center", o))          # pin center (isolate bbox)
                    no["bbox"] = list(_swap(ch, cur))
                elif ch == "dip":
                    no["dip"] = _swap(ch, cur)
                elif ch == "throw":
                    no["throw"] = _swap(ch, cur)
                elif ch == "area":
                    no["area_pct"] = _swap(ch, cur)
            out.append(no)
        nf[bk] = out
    return nf


@torch.no_grad()
def swap_test(nar, reader, scenes, use_reader):
    flip = {c: [0, 0] for c in CHANNELS}     # [hit, total]
    base = {c: [0, 0] for c in CHANNELS}
    dropped = 0                              # objects past MAX_OBJ (injected-budget, out of seam scope)
    tag = "reader" if use_reader else "GT"
    for s in tqdm(scenes, desc=f"swap ({tag})", unit="sc", leave=False):
        facts = reader_facts(reader, s) if use_reader else region_metadata(s)
        objs = _all_objs(facts)                                    # EVERY object, uncapped (matches copy_test)
        if not objs:
            continue
        dropped += max(0, len(objs) - len(_region_objs(facts)))    # objects past MAX_OBJ (never injected)
        n0 = _nums(generate_evidence(nar, facts))                  # baseline evidence (un-swapped)

        for _cls, o in objs:                                       # BASELINE: every injected attribute
            for ch in CHANNELS:
                v = _attr(ch, o)
                if v is None:
                    continue
                base[ch][1] += 1
                base[ch][0] += _hit_attr(v, n0)

        for ch in CHANNELS:                                        # FLIP: one swap-gen per channel
            present = [o for _c, o in objs if _attr(ch, o) is not None]
            if not present:
                continue
            ns = _nums(generate_evidence(nar, _mod(facts, ch)))
            for o in present:
                new = _swap(ch, _attr(ch, o))
                flip[ch][1] += 1
                flip[ch][0] += _hit_attr(new, ns)
    return flip, base, dropped


def main():
    reader = RegionReader().to(device)
    reader_pt = os.environ.get("READER", "hybrid/checkpoints/reader.pt")
    sd = torch.load(reader_pt, map_location=device)
    if any(k.startswith("real_adapter") for k in sd):
        reader.add_real_adapter()
    reader.load_state_dict(sd); reader.eval()
    reader.set_encoder(_build_encoder())
    nar = Captioner(); nar.set_stage("s3"); load_narrator(nar, CKPT); nar.eval_mode()

    te = held_out()
    ds = os.environ.get("DATASET", "synthetic")
    print(f"[swap] narrator {CKPT} · reader {reader_pt} · dataset {ds} · facts={'reader' if USE_READER else 'GT'} "
          f"· held-out {len(te)} · MAX_OBJ={os.environ.get('MAX_OBJ', 3)}", flush=True)

    flip, base, dropped = swap_test(nar, reader, te, USE_READER)
    print("[SWAP] per (object,attribute) over ALL injected objects/classes. flip = narration FOLLOWS the "
          "swapped value; baseline = states the UN-swapped value (control, not a ceiling).", flush=True)
    for ch in CHANNELS:
        fh, ft = flip[ch]; bh, bt = base[ch]
        fr = fh / ft if ft else float("nan")
        br = bh / bt if bt else float("nan")
        print(f"  {ch:8} flip {fh}/{ft} = {fr:.2f}   ·   baseline {bh}/{bt} = {br:.2f}", flush=True)
    # cross-check vs copy_test, SCOPE-MATCHED (copy numbers differ by class scope!):
    #   all-classes (dip+throw+area) ≈ eval_language.sh with ACTIVE_CLASSES unset (GT 0.77 / reader 0.67)
    #   fault-only  (dip+throw)      ≈ eval_language.sh ACTIVE_CLASSES=fault    (GT ~1.00 / reader 0.89)
    # If either lands ~1.00 where the reference isn't, the test is under-counting.
    def agg(chs):
        h = sum(base[c][0] for c in chs); t = sum(base[c][1] for c in chs)
        return f"{h}/{t} = {h/t:.2f}" if t else "n/a"
    print(f"  [cross-check] all-class baseline {agg(('dip','throw','area'))}  "
          f"(≈ copy {'reader 0.67' if USE_READER else 'GT 0.77'}, ACTIVE_CLASSES unset)", flush=True)
    print(f"  [cross-check] fault-only baseline {agg(('dip','throw'))}  "
          f"(≈ copy {'reader 0.89' if USE_READER else 'GT ~1.00'}, ACTIVE_CLASSES=fault)", flush=True)
    if dropped:
        print(f"[note] {dropped} object(s) past MAX_OBJ were never injected (budget limit, not a seam test).",
              flush=True)
    print("SWAP_DONE", flush=True)


if __name__ == "__main__":
    main()
