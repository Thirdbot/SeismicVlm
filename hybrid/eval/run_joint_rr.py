"""C — COMPLEMENTARY-JOINT run via ROUND-ROBIN rotation (equal turns → no forgetting, extensible), with
VALIDITY-GATED attributes (each survey trains only its geologically-valid GT) and the validated mask loss.
Then benchmark WITHIN each survey (never pooled) → the {joint} rows of the {alone}-vs-{joint} complementarity
grid (compare to the per-domain baselines already captured).

Design (all settled + tested upstream):
  · round-robin feeder (test_round_robin.py, all-pass): one scene per survey per cycle, small sets recycle.
  · validity gating: CRACKS dip degenerate → null mmask[dip] so it can't corrupt the SHARED dip head.
    Thebe/Smeaheia dip kept; throw only where present (Smeaheia). derive OFF (train_derived=False).
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

# per-dataset attribute slots to NULL (degenerate/invalid GT) — MEASURE_SLOTS = [dip, throw, area]
GATE_OFF = {"cracks": [0]}                                      # CRACKS dip degenerate (12% pinned at 90°)


def gate_off(scenes, slots):
    n = 0
    for s in scenes:
        for o in s.get("objs", []):
            if int(o["cls"]) == 1:
                for sl in slots:
                    if float(o["mmask"][sl]) > 0:
                        o["mmask"][sl] = 0.0; n += 1
    return n


def main():
    scenes_by_ds, tests = {}, {}
    for name in DATASETS:
        _, tr, te = importlib.import_module(f"hybrid.data.{name}").scenes()
        g = gate_off(tr, GATE_OFF.get(name, []))
        scenes_by_ds[name] = tr; tests[name] = te
        print(f"[C] {name}: train {len(tr)} · test {len(te)} · gated-off {g} attr-values", flush=True)

    seq = weighted_round_robin(scenes_by_ds, {n: WEIGHTS.get(n, 1) for n in DATASETS}, total_steps=TOTAL_STEPS)
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
    print("\n[C — JOINT benchmark, per survey · Dice=oracle(tf)/deploy(detect) · dip~circ = mask-derived GT (not "
          "independently claimable; only smeaheia dip is, from sticks)]", flush=True)
    for name in DATASETS:
        d = B.bench(r, name)
        P, R, _ = d["ppr"]
        dipc = "" if name == "smeaheia" else "~circ"       # dip GT independent only for smeaheia (projected sticks)
        print(f"  {name:9s} | Dice {d['tdice']:.3f}/{d['ddice']:.3f} (n{d['n_inst']}) · pixP {P:.3f}/R {R:.3f} · "
              f"tol-F1 {d['tolf']:.3f} · detF1 {d['det'][2]:.3f} · dip{dipc} {d['dip']:.2f}(c{d['dip_const']:.2f}) · "
              f"throw {d['throw']:.2f}(c{d['throw_const']:.2f})", flush=True)
    print("C_JOINT_DONE", flush=True)


if __name__ == "__main__":
    main()
