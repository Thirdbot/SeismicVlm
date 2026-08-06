"""Stage 4 — REAL-FIELD finetune via ADAPTER ISOLATION.

Only the reader's REAL ADAPTER trains; the entire synthetic reader (all classes + attributes) and the
whole LM stay FROZEN. This adapts vision to real seismic *appearance* without forgetting the synthetic
multi-object knowledge — the shared trunk is never touched (unlike freezing only the absent-class
heads, which lets the trunk drift). Real GT is sparse (e.g. Smeaheia ~59/462 lines), so adapter
isolation (not a full finetune) is what avoids overfit + catastrophic forgetting.

The LM is untouched by this stage — grounded narration/reasoning is domain-agnostic (it copies whatever
the reader measures), so only the reader needs to adapt to real seismic.

Provide `real_scenes` as a list of scene dicts with GT — from any dataset's `scenes()` (e.g.
`hybrid.data.smeaheia.scenes()[1]`). Optionally pass `rehearse` (synthetic scenes) to protect the
synthetic classes.

Driven by the joint runner / scripts (they call finetune_real): `scripts/joint.sh`, `hybrid.eval.run_joint_rr`.
"""
import os

import torch
from tqdm.auto import tqdm

from hybrid.model.reader import RegionReader, scene_to_gt

device = torch.device("cuda")


def finetune_real(real_scenes, reader_pt="hybrid/checkpoints/reader.pt", epochs=20, lr=1e-4,
                  save="hybrid/checkpoints/reader_real.pt", rehearse=None,
                  train_class=False, train_measure=False, train_derived=False):
    """Load the synthetic reader, FREEZE it, add a real adapter, train ONLY the adapter (+ mask decoder)
    on real scenes. rehearse = optional synthetic scenes mixed in (rehearsal) to protect synthetic classes.
    train_class/measure/derived TOGGLE per-domain head unfreezing (add_real_adapter) so the reader learns
    to emit DOMAIN-CORRECT values → the frozen LM gets clean input (fixes OOD narration at the source, not
    by blinding it). Only enable a head where this domain has GT to supervise it."""
    reader = RegionReader().to(device)
    reader.load_state_dict(torch.load(reader_pt, map_location=device))   # synthetic base
    from hybrid.stages.stage2_reader import _build_encoder
    reader.set_encoder(_build_encoder())                                 # encoder in-model (frozen; pixels -> grid)
    params = reader.add_real_adapter(train_class=train_class, train_measure=train_measure,
                                     train_derived=train_derived)        # freeze base + zero-init adapter + toggled heads
    opt = torch.optim.AdamW(params, lr=lr)

    def prep(scenes):
        # KEEP negatives (empty gt, is_neg) — a K=0 scene trains the adapter to SUPPRESS false faults
        # on real (the ungated CSV's whole point); drop only truly empty scenes.
        return [(s, s.get("derived")) for s in scenes         # LAZY: scene meta only; masks load per step (RAM-safe on big real sets)
                if s["objs"] or s.get("is_neg")]

    data = prep(real_scenes) + (prep(rehearse) if rehearse else [])
    if len(data) < 2:
        print("[real] too few real scenes — abort", flush=True); return reader
    n_adapt = sum(p.numel() for p in params)
    print(f"[real] {len(data)} scenes (real{' + rehearsal' if rehearse else ''}) · "
          f"adapter params {n_adapt} · base FROZEN", flush=True)
    ckpt_every = int(os.environ.get("CKPT_EVERY", 0))      # >0 → overwrite `save` every N steps (crash-safe long runs)
    step = 0
    reader.train()
    ebar = tqdm(range(epochs), desc="real-adapter", unit="ep")
    for ep in ebar:
        tot = 0.0
        for s, der in tqdm(data, desc=f"ep {ep}", unit="sc", leave=False):
            opt.zero_grad()
            loss, _ = reader(reader.encode(s), scene_to_gt(s), der)
            loss.backward(); opt.step(); tot += loss.item()
            step += 1
            if ckpt_every and step % ckpt_every == 0:      # latest-wins checkpoint; a crash keeps the newest state
                reader.eval(); torch.save(reader.state_dict(), save); reader.train()
                tqdm.write(f"[real] checkpoint @ step {step} → {save}")
        ebar.set_postfix(loss=f"{tot/max(1, len(data)):.3f}")
        tqdm.write(f"[real] ep {ep}/{epochs} loss {tot/max(1, len(data)):.3f}")
    reader.eval()
    torch.save(reader.state_dict(), save)
    print(f"[real] saved {save} (synthetic base + real adapter)", flush=True)
    return reader


if __name__ == "__main__":
    # finetune_real is driven by the joint runner + scripts, which supply real scenes from each dataset's
    # scenes(). Run those, not this module directly:
    #   scripts/joint.sh   (or)   WEIGHTS=... TOTAL_STEPS=... python -m hybrid.eval.run_joint_rr
    print("[real] use scripts/joint.sh or `python -m hybrid.eval.run_joint_rr` (they call finetune_real).", flush=True)
