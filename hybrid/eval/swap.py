"""Causal SWAP test — the seam-faithfulness probe behind tab:faith's flip/baseline rows.

The copy seam is non-differentiable: the narration can only restate numbers injected as markers
(hybrid/model/captioner.py _obj_markers). This test PROVES the dependence is causal by intervening
on the injected value and checking the narration follows:

  baseline : inject the measured facts UNCHANGED, generate evidence → does the narration state the
             ORIGINAL value? (a control = the channel's natural statement rate, NOT a ceiling on flip)
  flip     : inject the SAME facts with ONE channel's value SWAPPED to an off-distribution value,
             regenerate → does the narration follow the SWAPPED value?

Channels: dip, throw, centroid (fault-scoped, the seam's measured numbers). A value is matched with
the SAME tolerance as copy_test (±2%, ±0.5 absolute floor), on the value AS INJECTED (dip→1 dp,
throw→int, centroid→int), so nothing here is exact-match harsh or substring-inflated.

Reuses the exact injection + generation paths (region_metadata / reader_facts / generate_evidence)
that deployment uses — this is a read-only probe over saved weights, it trains nothing and edits no
existing file. Swap rules are FIXED + documented below; because the original tab:faith run was an
ad-hoc script (never committed), these reproducible numbers may differ slightly from the paper's
0.92/0.78 — if so, adopt THESE (they have a committed reproducer) rather than the un-reproducible ones.

Run:  DATASET=synthetic READER=hybrid/checkpoints/reader.pt CKPT=stage3_narrator.pt python -m hybrid.eval.swap
      USE_READER=1 …   # swap reader-measured facts instead of GT (default 0 = GT-injected, clean seam isolate)
"""
import os
import re

import torch
from tqdm.auto import tqdm

from hybrid.model.captioner import Captioner, region_metadata
from hybrid.model.reader import RegionReader
from hybrid.stages.stage2_reader import reader_facts, _build_encoder
from hybrid.stages.stage3_answer import generate_evidence
from hybrid.checkpoints import load_narrator
from hybrid.eval.components import held_out, CKPT

device = torch.device("cuda")
USE_READER = os.environ.get("USE_READER", "0").lower() not in ("0", "false", "no")

_NUM = re.compile(r"\d+\.?\d*")


def _nums(text):
    """Numbers STATED in a generated evidence string (same extraction as copy_test)."""
    return [float(x) for x in _NUM.findall(text)]


def _hit(v, nums):
    """Copy_test tolerance: some stated number within ±2% of v (±0.5 absolute floor)."""
    return any(abs(n - v) <= max(0.5, 0.02 * abs(v)) for n in nums)


# ---- FIXED swap rules: off-distribution, delta >> tolerance, stays plausible ----
def _swap_dip(v):    return round((v + 30) if v <= 55 else (v - 30), 1)   # in ~[5,85], delta 30
def _swap_throw(v):  return float(round(v + 60))                          # delta 60 ms
def _swap_xy(x, y):  return int(x) + 41, int(y) + 41                      # delta 41 px each


def _center_of(o):
    """The center AS INJECTED (o['center'], else derived from bbox — mirrors _obj_markers)."""
    c = o.get("center")
    if c:
        return int(c[0]), int(c[1])
    b = o.get("bbox") or [0, 0, 0, 0]
    return int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)


def _mod_faults(facts, fn):
    """Shallow copy that swaps a scalar channel on every fault WITHOUT deep-copying tensors: new dicts
    for the fault entries (so masks/tensors are shared by reference and never mutated)."""
    nf = dict(facts)
    nf["faults"] = [dict(f) for f in facts.get("faults", [])]
    for f in nf["faults"]:
        fn(f)
    return nf


@torch.no_grad()
def swap_test(nar, reader, scenes, use_reader):
    flip = {"dip": [0, 0], "throw": [0, 0], "centroid": [0, 0]}   # [hit, total]
    base = {"dip": [0, 0], "throw": [0, 0], "centroid": [0, 0]}
    tag = "reader" if use_reader else "GT"
    for s in tqdm(scenes, desc=f"swap ({tag})", unit="sc", leave=False):
        facts = reader_facts(reader, s) if use_reader else region_metadata(s)
        faults = facts.get("faults", [])
        if not faults:
            continue
        f0 = faults[0]                                            # probe the primary fault per scene
        n0 = _nums(generate_evidence(nar, facts))                # baseline evidence (un-swapped)

        # dip
        if f0.get("dip") is not None:
            orig = round(float(f0["dip"]), 1)
            base["dip"][1] += 1; base["dip"][0] += _hit(orig, n0)
            new = _swap_dip(float(f0["dip"]))
            def sd(f, new=new):
                if f.get("dip") is not None:
                    f["dip"] = new
            ns = _nums(generate_evidence(nar, _mod_faults(facts, sd)))
            flip["dip"][1] += 1; flip["dip"][0] += _hit(new, ns)

        # throw
        if f0.get("throw") is not None:
            orig = float(round(float(f0["throw"])))
            base["throw"][1] += 1; base["throw"][0] += _hit(orig, n0)
            new = _swap_throw(float(f0["throw"]))
            def st(f, new=new):
                if f.get("throw") is not None:
                    f["throw"] = new
            ns = _nums(generate_evidence(nar, _mod_faults(facts, st)))
            flip["throw"][1] += 1; flip["throw"][0] += _hit(new, ns)

        # centroid (weaker probe — coordinates are copied wherever narrated)
        ox, oy = _center_of(f0)
        base["centroid"][1] += 1; base["centroid"][0] += (_hit(ox, n0) and _hit(oy, n0))
        def sc(f):
            cx, cy = _center_of(f)
            nx, ny = _swap_xy(cx, cy)
            f["center"] = [nx, ny]
        ns = _nums(generate_evidence(nar, _mod_faults(facts, sc)))
        nx, ny = _swap_xy(ox, oy)
        flip["centroid"][1] += 1; flip["centroid"][0] += (_hit(nx, ns) and _hit(ny, ns))
    return flip, base


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
          f"· held-out {len(te)}", flush=True)

    flip, base = swap_test(nar, reader, te, USE_READER)
    print("[SWAP] flip = fraction the narration FOLLOWS the swapped value; "
          "baseline = fraction stating the UN-swapped value (control, not a ceiling)", flush=True)
    for ch in ("dip", "throw", "centroid"):
        fh, ft = flip[ch]; bh, bt = base[ch]
        fr = fh / ft if ft else float("nan")
        br = bh / bt if bt else float("nan")
        print(f"  {ch:9} flip {fh}/{ft} = {fr:.2f}   ·   baseline {bh}/{bt} = {br:.2f}", flush=True)
    print("SWAP_DONE", flush=True)


if __name__ == "__main__":
    main()
