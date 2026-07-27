"""Stage 4 — REAL-FIELD finetune via ADAPTER ISOLATION.

Only the reader's REAL ADAPTER trains; the entire synthetic reader (all classes + attributes) and the
whole LM stay FROZEN. This adapts vision to real seismic *appearance* without forgetting the synthetic
multi-object knowledge — the shared trunk is never touched (unlike freezing only the absent-class
heads, which lets the trunk drift). Real GT is sparse (e.g. Smeaheia ~59/462 lines), so adapter
isolation (not a full finetune) is what avoids overfit + catastrophic forgetting.

The LM is untouched by this stage — grounded narration/reasoning is domain-agnostic (it copies whatever
the reader measures), so only the reader needs to adapt to real seismic.

Provide `real_scenes` as [{smap, ...}] with GT (see hybrid/data/real.py). Optionally pass `rehearse`
(synthetic multi-object scenes) to further protect the synthetic classes.

Run (once real scenes are built):  python -m hybrid.train.stage_realfield
"""
import torch
from tqdm.auto import tqdm

from hybrid.model.reader import InstanceReader, scene_to_gt

device = torch.device("cuda")


def finetune_real(real_scenes, reader_pt="hybrid/checkpoints/reader.pt", epochs=20, lr=1e-4,
                  save="hybrid/checkpoints/reader_real.pt", rehearse=None):
    """Load the synthetic reader, FREEZE it, add a real adapter, train ONLY the adapter on real scenes.
    rehearse = optional synthetic scenes mixed in (rehearsal) to protect the synthetic classes further."""
    reader = InstanceReader().to(device)
    reader.load_state_dict(torch.load(reader_pt, map_location=device))   # synthetic base
    params = reader.add_real_adapter()                                   # freeze base + zero-init adapter
    opt = torch.optim.AdamW(params, lr=lr)

    def prep(scenes):
        # KEEP negatives (empty gt, is_neg) — a K=0 scene trains the adapter to SUPPRESS false faults
        # on real (the ungated CSV's whole point); drop only truly empty scenes.
        return [(s["smap"], scene_to_gt(s), s.get("derived")) for s in scenes
                if scene_to_gt(s) or s.get("is_neg")]

    data = prep(real_scenes) + (prep(rehearse) if rehearse else [])
    if len(data) < 2:
        print("[real] too few real scenes — abort", flush=True); return reader
    n_adapt = sum(p.numel() for p in params)
    print(f"[real] {len(data)} scenes (real{' + rehearsal' if rehearse else ''}) · "
          f"adapter params {n_adapt} · base FROZEN", flush=True)
    reader.train()
    ebar = tqdm(range(epochs), desc="real-adapter", unit="ep")
    for ep in ebar:
        tot = 0.0
        for smap, gt, der in tqdm(data, desc=f"ep {ep}", unit="sc", leave=False):
            opt.zero_grad()
            loss, _ = reader(smap, gt, der)
            loss.backward(); opt.step(); tot += loss.item()
        ebar.set_postfix(loss=f"{tot/max(1, len(data)):.3f}")
        tqdm.write(f"[real] ep {ep}/{epochs} loss {tot/max(1, len(data)):.3f}")
    reader.eval()
    torch.save(reader.state_dict(), save)
    print(f"[real] saved {save} (synthetic base + real adapter)", flush=True)
    return reader


if __name__ == "__main__":
    # Real scenes come from the real-field loader (hybrid/data/real.py) — wire it here when ready:
    #   from hybrid.data.real import build_real_scenes
    #   finetune_real(build_real_scenes(), rehearse=build_scenes()[:50])
    print("[real] provide real_scenes via hybrid/data/real.py, then call finetune_real(...)", flush=True)
