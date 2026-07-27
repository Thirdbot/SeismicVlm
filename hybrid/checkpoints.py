"""Explicit per-stage checkpoints.

Saves the TRAINABLE weights of each stage so the model is LOADABLE (standalone
inference) and RETRAINABLE (resume a stage) without redoing earlier stages.
Stage 1 (geology) is the cached adapter dir; stages 2-3 save here. Only trainable
params are saved (grounding+fuse LoRA; facts inject as named text, no learned marker)
and loaded with strict=False,
so the frozen 4-bit base reloads fresh each time. (The reader saves separately as
reader.pt via train.py.)
"""
import torch
from pathlib import Path

CKPT = Path("hybrid/checkpoints")
device = torch.device("cuda")


def _lora_state(dec, names=("grounding", "fuse", "reason")):
    return {k: v.detach().cpu() for k, v in dec.state_dict().items()
            if "lora_" in k and any(nm in k for nm in names)}


def save_narrator(nar, name="stage3_narrator.pt"):
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": _lora_state(nar.dec)}, CKPT / name)
    print(f"[ckpt] narrator -> {CKPT / name}", flush=True)


def load_narrator(nar, name="stage3_narrator.pt"):
    d = torch.load(CKPT / name, map_location=device)
    nar.dec.load_state_dict(d["lora"], strict=False)   # grounding+fuse LoRA only
    return nar
