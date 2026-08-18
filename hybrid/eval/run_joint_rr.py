"""C — COMPLEMENTARY-JOINT run via ROUND-ROBIN rotation (equal turns → no forgetting, extensible), on the
FULL UNGATED data (every survey's extracted attributes are stored, so A/B can score them and the RESULT
decides validity — no asserted gate). Then benchmark WITHIN each survey (never pooled) → the {joint} rows of
the {alone}-vs-{joint} complementarity grid (compare to the per-domain baselines already captured).

Design (all settled + tested upstream):
  · round-robin feeder (test_round_robin.py, all-pass): one scene per survey per cycle, small sets recycle.
  · attributes UNGATED by default: Thebe/CRACKS apparent dip (mask-derived → learnability) + Smeaheia 3-D
    dip/throw (independent → accuracy) all train under TRAIN_MEASURE=1; throw only where present (Smeaheia);
    derive OFF. Opt-out per survey at the data level (e.g. CRACKS_GATE_DIP=1) if a run must be mask-only.
  · loss: TVERSKY + POS_WEIGHT_MAX via env (the step-2 winner), Dice+Tversky additive.

  TVERSKY=0.4,0.6,1.0 POS_WEIGHT_MAX=15 TOTAL_STEPS=10000 JOINT_EPOCHS=1 \
    TRAIN_CLASS=1 TRAIN_MEASURE=1 python -m hybrid.eval.run_joint_rr
"""
import os
os.environ.setdefault("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")
import importlib
import torch

from hybrid.stages.finetune_vision import finetune_real
from hybrid.data.round_robin import weighted_round_robin

device = torch.device("cuda")
DATASETS = os.environ.get("DATASETS", "thebe,cracks,smeaheia").split(",")
WEIGHTS = {kv.split(":")[0]: int(kv.split(":")[1])                       # WEIGHTED turns (large survey more slots
           for kv in os.environ.get("WEIGHTS", "thebe:8,cracks:1,smeaheia:1").split(",") if kv}   # so it trains
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 10000))                  # adequately; smalls stay refreshed)
JOINT_EPOCHS = int(os.environ.get("JOINT_EPOCHS", 1))
TRAIN_CLASS = os.environ.get("TRAIN_CLASS", "1") == "1"
TRAIN_MEASURE = os.environ.get("TRAIN_MEASURE", "1") == "1"
TRAIN_DERIVED = os.environ.get("TRAIN_DERIVED", "0") == "1"     # OFF — relations reasoned, not asserted
JOINT_SAVE = os.environ.get("JOINT_SAVE", "hybrid/checkpoints/reader_joint_rr.pt")
DRY = os.environ.get("DRY", "0") == "1"                         # data-prep only, no training (GPU-free check)

# Attributes are stored UNGATED by default (each survey keeps whatever its build extracts); scene_to_gt
# attaches an attribute wherever its slot is present, and validity is decided by the RESULT (dip MAE vs the
# constant baseline), not a pre-gate. The runner carries NO attribute gate — the head toggles
# (TRAIN_CLASS/MEASURE/DERIVED) are the only experimental switch, kept separate from the (ungated) data.
# A survey can still opt OUT at the data level (e.g. CRACKS_GATE_DIP=1) for a deliberately mask-only run.


def main():
    scenes_by_ds, tests = {}, {}
    for name in DATASETS:
        _, tr, te = importlib.import_module(f"hybrid.data.{name}").scenes()
        scenes_by_ds[name] = tr; tests[name] = te
        print(f"[C] {name}: train {len(tr)} · test {len(te)}", flush=True)

    # WEIGHTS is the authoritative training mix: a survey NOT named in WEIGHTS gets weight 0 (NOT trained),
    # not a silent default of 1. Else WEIGHTS=thebe:1 (an "alone" run) would train cracks+smeaheia at weight 1
    # too — a 1:1:1 joint masquerading as alone. To train a survey it must appear in WEIGHTS.
    seq = weighted_round_robin(scenes_by_ds, {n: WEIGHTS.get(n, 0) for n in DATASETS}, total_steps=TOTAL_STEPS)
    train = [sc for _, sc in seq]
    from collections import Counter
    print(f"[C] weighted round-robin: {len(train)} scenes · weights {WEIGHTS} · turns {dict(Counter(n for n,_ in seq))} · "
          f"{JOINT_EPOCHS} ep · loss TVERSKY={os.environ.get('TVERSKY','off')} POS_WEIGHT_MAX={os.environ.get('POS_WEIGHT_MAX','50')} · "
          f"heads[class={TRAIN_CLASS} measure={TRAIN_MEASURE} derived={TRAIN_DERIVED}]", flush=True)
    if DRY:
        print("[C] DRY run — data prep OK, no training.", flush=True); print("C_DRY_DONE", flush=True); return

    finetune_real(train, epochs=JOINT_EPOCHS, save=JOINT_SAVE,
                  train_class=TRAIN_CLASS, train_measure=TRAIN_MEASURE, train_derived=TRAIN_DERIVED)

    # {joint} rows: benchmark each survey's held-out on the tested metrics
    import hybrid.eval.benchmark as B
    from hybrid.model.reader import RegionReader
    from hybrid.stages.stage2_reader import _build_encoder
    B.N_TEST = int(os.environ.get("N_TEST", 60))
    r = RegionReader().to(device); r.add_real_adapter()
    r.load_state_dict(torch.load(JOINT_SAVE, map_location=device)); r.eval(); r.set_encoder(_build_encoder())
    print("\n[C — JOINT benchmark, per survey · Dice=oracle(tf)/deploy(detect) · dip = fault-trace geometry "
          "(smeaheia dip ALSO from independent projected sticks → an extra cross-check)]", flush=True)
    for name in DATASETS:
        d = B.bench(r, name)
        P, R, _ = d["ppr"]
        dipsrc = " (sticks)" if name == "smeaheia" else ""   # smeaheia dip has an independent source; others read trace geometry
        print(f"  {name:9s} | Dice {d['tdice']:.3f}/{d['ddice']:.3f} (n{d['n_inst']}) · pixP {P:.3f}/R {R:.3f} · "
              f"tol-F1 {d['tolf']:.3f} · detF1 {d['det'][2]:.3f} · dip {d['dip']:.2f}(c{d['dip_const']:.2f}){dipsrc} · "
              f"throw {d['throw']:.2f}(c{d['throw_const']:.2f})", flush=True)
    print("C_JOINT_DONE", flush=True)


if __name__ == "__main__":
    main()
