"""Real-field before/after on the UNGATED CSV — encode once, evaluate synthetic reader vs real-adapted.

BEFORE = synthetic reader on real held-out. AFTER = adapter + mask decoder trained on the real TRAIN
split (positives + negatives), evaluated on the SAME held-out. Reports reader detection/measure/mask
AND count-MAE on NEGATIVE panels (the false-fault-suppression signal the negatives are there for).

Run:  python -m hybrid.test.real_before_after
"""
import torch
from tqdm.auto import tqdm

from hybrid.data.real_csv import real_csv_scenes
from hybrid.model.reader import InstanceReader, scene_to_gt, FAULT
from hybrid.model.segmenter import field_dice
from hybrid.train.stage_realfield import finetune_real

device = torch.device("cuda")

REAL_EPOCHS = 200
@torch.no_grad()
def evaluate(reader, scenes):
    """Reader metrics on real held-out: count MAE (pos & neg separately), class, dip/throw MAE, dice."""
    cpos, cneg, dip, throw, dices, cls_hit, cls_tot = [], [], [], [], [], 0, 0
    for s in tqdm(scenes, desc="real-eval", unit="sc", leave=False):
        gt = scene_to_gt(s); pred = reader.detect(s["smap"])
        (cneg if s.get("is_neg") else cpos).append(abs(len(pred) - len(gt)))
        gf = sorted(o["dip"] for o in gt if o["cls"] == FAULT and o.get("dip") is not None)
        pf = sorted(o["dip"] for o in pred if o["cls"] == FAULT)
        for i in range(min(len(gf), len(pf))):
            dip.append(abs(pf[i] - gf[i]))
        gtt = sorted(o["throw"] for o in gt if o["cls"] == FAULT and o.get("throw") is not None)
        ptt = sorted(o["throw"] for o in pred if o["cls"] == FAULT)
        for i in range(min(len(gtt), len(ptt))):
            throw.append(abs(ptt[i] - gtt[i]))
        for i in range(min(len(pred), len(gt))):
            cls_tot += 1; cls_hit += int(pred[i]["cls"] == gt[i]["cls"])
        if gt:
            ml = reader.tf_masks(s["smap"], gt)
            dices += [field_dice(ml[i], o["mask_full"].to(device)) for i, o in enumerate(gt) if o["cls"] == FAULT]

    def m(x): return (sum(x) / len(x), len(x)) if x else (None, 0)
    return dict(count_pos=m(cpos), count_neg=m(cneg), dip=m(dip), throw=m(throw),
                cls=(cls_hit, cls_tot), dice=m(dices))


def fmt(d):
    def f(x, u=""): return f"{x[0]:.2f}{u}(n{x[1]})" if x[0] is not None else "n0"
    return (f"count+ {f(d['count_pos'])} · count- {f(d['count_neg'])} · class {d['cls'][0]}/{d['cls'][1]} · "
            f"dip {f(d['dip'],'deg')} · throw {f(d['throw'],'ms')} · dice {f(d['dice'])}")


def main():
    alls, tr, te = real_csv_scenes()
    print(f"[real-ba] all {len(alls)} · train {len(tr)} · test {len(te)}", flush=True)

    reader = InstanceReader().to(device)
    reader.load_state_dict(torch.load("hybrid/checkpoints/reader.pt", map_location=device)); reader.eval()
    print(f"[BEFORE] {fmt(evaluate(reader, te))}", flush=True)

    finetune_real(tr, epochs=REAL_EPOCHS)                                   # adapter + mask decoder on real train
    reader2 = InstanceReader().to(device); reader2.add_real_adapter()
    reader2.load_state_dict(torch.load("hybrid/checkpoints/reader_real.pt", map_location=device)); reader2.eval()
    print(f"[AFTER ] {fmt(evaluate(reader2, te))}", flush=True)
    print("REAL_BA_DONE", flush=True)


if __name__ == "__main__":
    main()
