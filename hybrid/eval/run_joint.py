"""JOINT real-field finetune + PER-DATASET benchmark.

WHY JOINT: the mask decoder is ONE shared copy that finetune_real RETRAINS (not adapts), so a
sequential Thebe→CRACKS finetune overwrites Thebe. Concatenate the real train splits and run ONE
adapter-isolation finetune over all of them (they share the image·mask·regions contract).
Reference: reference_thebe_transfer_result.md.

WHY BENCHMARK SEPARATELY (never pooled): the datasets are incommensurable — Thebe has ~thousands of
instances vs Smeaheia's 144 (a pooled per-instance mean is ~all Thebe); CRACKS dip is degenerate
(a constant beats the model 4×); Thebe/Smeaheia class is single-class (trivially 100%). So we loop
and score each held-out split on its OWN population, reporting only the metrics each dataset supports.

Composes FROZEN hybrid.* — changes nothing in main. Env knobs:
  DATASETS=thebe,cracks,smeaheia   JOINT_EPOCHS=20   JOINT_CAP=0(uncapped)   RETRAIN_READER=0
  READER_EPOCHS=80   SFM_CKPT=...   N_CHUNKS=2 (thebe)
"""
import os
os.environ.setdefault("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")
import importlib
import torch

from hybrid.data import synthetic
from hybrid.model.reader import RegionReader
from hybrid.stages.stage2_reader import _build_encoder, train_reader, mask_dice
from hybrid.stages.finetune_vision import finetune_real
from hybrid.eval.real_transfer import evaluate

device = torch.device("cuda")


def ft(t):                                               # format a (value, n) metric tuple
    return f"{t[0]:.3f}(n{t[1]})" if t and t[0] is not None else "n0"

DATASETS = os.environ.get("DATASETS", "thebe,cracks,smeaheia").split(",")
JOINT_EPOCHS = int(os.environ.get("JOINT_EPOCHS", 20))
JOINT_CAP = int(os.environ.get("JOINT_CAP", 0))          # >0 caps each dataset's TRAIN (smoke only)
RETRAIN_READER = os.environ.get("RETRAIN_READER", "0") == "1"
READER_EPOCHS = int(os.environ.get("READER_EPOCHS", 80))
BASE = "hybrid/checkpoints/reader.pt"
JOINT = "hybrid/checkpoints/reader_joint.pt"


def load_real(name):
    ds = importlib.import_module(f"hybrid.data.{name}")
    alls, tr, te = ds.scenes()
    if JOINT_CAP:
        tr = tr[:JOINT_CAP]
    return tr, te


def eval_row(tag, reader, te, name):
    m = evaluate(reader, te)                              # count/class/dip/throw/dice (dice = per-instance oracle tf_masks)
    dd = mask_dice(reader, te, oracle=False)             # deployment dice (detect(); misses=0)
    return (f"  {name:9s} {tag:7s} | dice(oracle) {ft(m['dice'])} · dice(deploy) "
            f"{ft(dd)} | class {m['cls'][0]}/{m['cls'][1]} | count+ {ft(m['count_pos'])} "
            f"count- {ft(m['count_neg'])} | dip {ft(m['dip'])} | throw {ft(m['throw'])}")


def build_reader(ckpt, real_adapter):
    r = RegionReader().to(device)
    if real_adapter:
        r.add_real_adapter()
    r.load_state_dict(torch.load(ckpt, map_location=device))
    r.eval(); r.set_encoder(_build_encoder())
    return r


def main():
    # 1) synthetic base reader (reuse reader.pt or retrain uncapped)
    if RETRAIN_READER:
        _, str_, ste = synthetic.split(max_scenes=10**9)
        print(f"[joint] retrain synthetic reader: train {len(str_)} · {READER_EPOCHS} ep", flush=True)
        train_reader(str_, epochs=READER_EPOCHS)         # saves reader.pt
    base = build_reader(BASE, real_adapter=False)

    # 2) build the JOINT real train set (concatenate the train splits) + keep each test split
    tests = {}
    joint_train = []
    for name in DATASETS:
        tr, te = load_real(name)
        tests[name] = te
        joint_train += tr
        print(f"[joint] {name}: train {len(tr)} · test {len(te)}", flush=True)
    print(f"[joint] JOINT train total {len(joint_train)} scenes · {JOINT_EPOCHS} ep", flush=True)

    # 3) BEFORE: synthetic-base zero-shot on every held-out (incl. synthetic itself)
    _, _, synth_te = synthetic.split(max_scenes=10**9)
    print("\n[BEFORE — synthetic base, zero-shot]", flush=True)
    print(eval_row("BEFORE", base, synth_te, "synthetic"), flush=True)
    for name in DATASETS:
        print(eval_row("BEFORE", base, tests[name], name), flush=True)

    # 4) JOINT finetune (adapter isolation over ALL real train at once)
    finetune_real(joint_train, epochs=JOINT_EPOCHS, save=JOINT)

    # 5) AFTER: per-dataset benchmark on the joint model
    after = build_reader(JOINT, real_adapter=True)
    print("\n[AFTER — joint real adapter]", flush=True)
    print(eval_row("AFTER", after, synth_te, "synthetic"), flush=True)   # synthetic mask expected to collapse (mask decoder retrained)
    for name in DATASETS:
        print(eval_row("AFTER", after, tests[name], name), flush=True)
    print("JOINT_DONE", flush=True)


if __name__ == "__main__":
    main()
