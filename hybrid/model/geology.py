"""Stage-1 geology adapter — cached loader for the main model.

`load_geology_adapter` returns the cached Stage-1 LoRA adapter dir (built once by
`hybrid.stages.stage1_geology`, out of band because Unsloth must be imported before
transformers/peft). The decoder loads this frozen geology adapter, then adds the
trainable grounding adapter on top.
"""
import hashlib
import json
import os
from pathlib import Path

CKPT = Path("hybrid/checkpoints")

# Stage-1 config — the hash of this dict names the cached adapter dir, so it must
# stay identical to whatever produced the cache. Consumed by both this loader and
# the `stage1_geology` training script. batch_size/grad_accum are COMPUTE-LAYOUT
# (env-tunable per GPU) and are EXCLUDED from the identity hash below, so changing
# them for a bigger GPU never forces a re-train. Default effective batch = 8.
GEOLOGY_CFG = dict(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    max_train_samples=None, max_eval_samples=64, max_seq_length=1024,   # None = train on ALL GeoGPT-CoT-QA rows
    lora_r=16, lora_alpha=16, num_epochs=int(os.environ.get("GEOLOGY_EPOCHS", 1)), learning_rate=2e-5,
    batch_size=int(os.environ.get("GEOLOGY_BATCH", 8)),                 # 24 GB GPU → real parallel batch (was 1)
    grad_accum=int(os.environ.get("GEOLOGY_GRAD_ACCUM", 1)),           # effective batch = batch × accum (default 8×1)
    save_mode="adapter",
    evidence_placeholder=False,   # geology supplies only <think>/<answer>; later stages add <evidence> in front
)

_IDENTITY_SKIP = ("batch_size", "grad_accum")   # compute-layout, not model identity → not hashed


def _hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def adapter_dir(geology_cfg: dict = GEOLOGY_CFG) -> Path:
    """Cache dir this config hashes to (built or not). Excludes compute-layout keys so batch tuning
    doesn't rename the adapter / force a re-train."""
    ident = {k: v for k, v in geology_cfg.items() if k not in _IDENTITY_SKIP}
    return CKPT / f"stage1_{_hash(ident)}"


def load_geology_adapter(geology_cfg: dict = GEOLOGY_CFG) -> str:
    """Return the cached geology adapter dir (raise if it hasn't been built)."""
    merged = adapter_dir(geology_cfg)
    if (merged / "stage1_config.json").exists():
        print(f"[geology] cache hit -> {merged}", flush=True)
        return str(merged)
    raise RuntimeError(
        f"Geology adapter not cached at {merged}; build it first: "
        f"python -m hybrid.stages.stage1_geology")
