"""SINGLE CONFIG SURFACE — every knob for the whole pipeline in one place.

The project has always been env-driven (no argparse): each module reads `os.environ.get("NAME", default)`
at import. This file is the CATALOG of all of those knobs — model · train · eval · inference — so you can
see (and set) everything from one screen. It reads the SAME env vars the modules read, with the SAME
defaults, so importing it changes nothing; setting a var here (or in the shell / scripts/config.sh) flows to
the module that owns it. `python -m hybrid.config` prints the resolved values, and flags the knobs that
actively CAP model capability (⚠) so throttles are never hidden.

    NAME=value python -m hybrid.<entry>       # override any knob per run (unchanged mechanism)
    python -m hybrid.config                    # dump the resolved config + capability flags
"""
import os


def _b(name, default):  # bool env
    return os.environ.get(name, str(default)).lower() not in ("0", "false", "no", "")


def _i(name, default):
    return int(os.environ.get(name, default))


def _f(name, default):
    return float(os.environ.get(name, default))


def _s(name, default):
    return os.environ.get(name, default)


# ================================ MODEL ================================
# Encoder + native tiling (tied to the frozen SFM ViT-B/16 @512; do not change without changing the encoder).
SFM_CKPT        = _s("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")   # frozen encoder (REQUIRED, hard error if absent)
ALLOW_NCS       = _b("ALLOW_NCS", False)          # permit the NCS fallback encoder when SFM ckpt is missing
TILE            = _i("TILE", 512)                 # native-aspect tile size fed to the encoder
PATCH           = _i("PATCH", 16)                 # encoder patch (grid = TILE/PATCH). Pretrained at 16; ⚠ PATCH=8 is OOD
# Instance reader (DETR set prediction).
N_QUERIES       = _i("N_QUERIES", 48)             # ⚠ object slots — MUST exceed the densest scene, else instances are dropped
DET_THRESH      = _f("DET_THRESH", 0.9)           # ⚠ detection objectness gate. 0.9 = swept default; LOWER → more recall
ACTIVE_CLASSES  = _s("ACTIVE_CLASSES", "")        # ⚠ restrict to a subset of classes ("fault"→fault-only); "" = all classes
# Language decoder (Qwen 4-bit QLoRA) — see hybrid/data/config.py for the model name / LoRA rank.
LORA_R          = _i("LORA_R", 8)
MAX_OBJ         = _i("MAX_OBJ", 3)                # ⚠ objects PER CLASS injected/stated to the LM; ALSO filters training/eval
                                                 #    rows with more than this many faults/closures. Caps dense scenes.

# ================================ TRAIN ================================
# Synthetic curriculum (run_train).
READER_EPOCHS   = _i("READER_EPOCHS", 40)         # ⚠ reader-from-scratch epochs. 40 assumed a ~10× bigger set; verify vs the
                                                 #    ACTUAL scene count (see hybrid.config check) — too few → undertrained
GROUND_EPOCHS   = _i("GROUND_EPOCHS", 20)         # grounding (evidence-copy) LoRA epochs
ANSWER_EPOCHS   = _i("ANSWER_EPOCHS", 15)         # fuse-fold (answer) epochs
CLDICE_W        = _f("CLDICE_W", 1.0)             # thin-structure centerline loss weight on the reader mask
TRAINABLE_BLOCKS = _i("TRAINABLE_BLOCKS", 0)      # frozen SFM by default; >0 unfreezes top blocks (experiment only)
SCENE_CAP       = os.environ.get("SCENE_CAP")     # smoke cap on scenes; unset = uncapped (the correct default)
RETRAIN_READER  = _b("RETRAIN_READER", True)      # False reuses a cached reader.pt (⚠ makes epoch/loss knobs no-ops)
SKIP_GROUNDING  = _b("SKIP_GROUNDING", False)     # re-run only the fold on a cached grounding ckpt
# Real-field finetune + weighted round-robin joint (finetune_vision / run_joint_rr).
TRAIN_CLASS     = _b("TRAIN_CLASS", True)         # unfreeze the class head on real (scoped by ACTIVE_CLASSES)
TRAIN_MEASURE   = _b("TRAIN_MEASURE", True)       # unfreeze dip/throw/area heads on real (0 = keep synthetic heads frozen)
TRAIN_DERIVED   = _b("TRAIN_DERIVED", False)      # derived/relational tier (needs real relational GT; off by default)
WEIGHTS         = _s("WEIGHTS", "thebe:4,cracks:3,smeaheia:3")   # round-robin per-survey turn weighting
TOTAL_STEPS     = _i("TOTAL_STEPS", 100000)       # round-robin steps (⚠ < thebe_pool/weight → Thebe not fully covered)
JOINT_EPOCHS    = _i("JOINT_EPOCHS", 1)
CKPT_EVERY      = _i("CKPT_EVERY", 10000)         # checkpoint interval; RESUME=1 warm-restarts
REAL_CAP        = _i("REAL_CAP", 100000)          # ⚠ cap on real panels loaded (lower it only for smoke runs)
# Mask loss (validated by sweep — do not change a default without re-validating).
TVERSKY         = _s("TVERSKY", "0.4,0.6,1.0")    # Dice + Focal-Tversky(α,β,γ); β>α penalizes over-prediction
POS_WEIGHT_MAX  = _f("POS_WEIGHT_MAX", 15)        # BCE positive-weight clamp (over-prediction control)
DILATE_R        = _i("DILATE_R", 0)               # ⚠ mask-target dilation. 0 = PURE thin-line GT (never inflate for metrics)

# ================================ EVAL =================================
CKPT            = _s("CKPT", "hybrid/checkpoints/reader.pt")   # checkpoint under test
DATASETS        = _s("DATASETS", "synthetic,thebe,cracks,smeaheia")
N_TEST          = _i("N_TEST", 300)               # ⚠ held-out cap; ≥ split ⇒ uncapped (the honest setting). ALWAYS uncapped for real numbers
DET_TAU         = _f("DET_TAU", 0.1)              # detection-F1 centroid-match gate (τ); wider → easier detF1

# ============================= INFERENCE ==============================
EVIDENCE_TOKENS = _i("EVIDENCE_TOKENS", 320)      # ⚠ evidence generation budget; too low truncates multi-object copy
ANSWER_TOKENS   = _i("ANSWER_TOKENS", 512)        # ⚠ think+answer budget; too low truncates the answer
REPETITION_PENALTY = _f("REPETITION_PENALTY", 1.3)
READER          = _s("READER", "hybrid/checkpoints/reader.pt")   # reader for infer/eval (reader_real_<survey>.pt on real)
NARRATOR        = _s("NARRATOR", "stage3_answer.pt")
IMAGE           = _s("IMAGE", "")                 # single-image inference target
QUESTION        = _s("QUESTION", "")
OUT             = _s("OUT", "hybrid/inference")   # inference output dir
BOXES           = _b("BOXES", True)               # overlay bbox+class (0 = clean mask-only, paper figures)

# Knobs flagged ⚠ above are the ones that actively CAP capability — the first place to look when the model
# seems throttled: DET_THRESH (recall), N_QUERIES / MAX_OBJ (instance count), READER_EPOCHS (undertraining),
# ACTIVE_CLASSES (class set), *_TOKENS (truncation), N_TEST/REAL_CAP (eval coverage), DILATE_R (honest mask).
_CAPABILITY_FLAGS = ["DET_THRESH", "N_QUERIES", "MAX_OBJ", "READER_EPOCHS", "ACTIVE_CLASSES",
                     "EVIDENCE_TOKENS", "ANSWER_TOKENS", "N_TEST", "REAL_CAP", "DILATE_R", "PATCH"]


def resolved():
    """All knobs as a flat {name: value} dict (current env applied)."""
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith("_")}


def main():
    r = resolved()
    print("=== resolved config (env applied) ===")
    for k in sorted(r):
        flag = "  ⚠ caps capability" if k in _CAPABILITY_FLAGS else ""
        print(f"  {k:18} = {r[k]!r}{flag}")
    print("\n⚠ = a knob that can suppress model capability; check these first if the model seems throttled.")


if __name__ == "__main__":
    main()
