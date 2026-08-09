"""VLM-style inference — give it a seismic IMAGE (+ an optional QUESTION), get a GROUNDED answer.

The vision reader MEASURES the geology (fault count, dip, throw, location, per-fault mask); the language
model COPIES those numbers across the non-differentiable "digit seam" and answers in words. The LM can
never invent a number the vision stack did not measure.

  IMAGE=path/to/section.png python -m hybrid.infer
  IMAGE=path/to/section.png QUESTION="Where is the fault and what does its dip imply?" python -m hybrid.infer
  IMAGE=... READER=hybrid/checkpoints/reader_real_thebe.pt python -m hybrid.infer   # a survey-specific reader

Env knobs (no argparse — the project's config style):
  IMAGE     (required)  path to a 2-D seismic section (PNG/JPG, any size — tiled natively)
  QUESTION  (optional)  free-text question; default asks for a full description
  READER    checkpoint  vision reader (default reader.pt = synthetic base; use reader_real_<survey>.pt or
                        reader_joint_full.pt for real data)
  NARRATOR  checkpoint  language narrator (default stage3_answer.pt)
  OUT       PNG path    predicted-mask overlay ("" = skip)
"""
import os
os.environ.setdefault("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")
import torch
from PIL import Image

from hybrid.model.reader import RegionReader
from hybrid.model.captioner import Captioner
from hybrid.stages.stage2_reader import reader_facts, _build_encoder
from hybrid.stages.stage3_answer import generate_evidence, generate_think_answer, _clean
from hybrid.model.text_metrics import _ANS, _THINK
from hybrid.checkpoints import load_narrator
from hybrid.eval.viz import overlay_classes

device = torch.device("cuda")
IMAGE = os.environ.get("IMAGE", "")
QUESTION = os.environ.get("QUESTION", "Describe the faults: their location, dip, and throw.")
READER = os.environ.get("READER", "hybrid/checkpoints/reader.pt")
NARRATOR = os.environ.get("NARRATOR", "stage3_answer.pt")
OUT = os.environ.get("OUT", "hybrid/inference/vlm_overlay.png")
BOXES = os.environ.get("BOXES", "1").lower() not in ("0", "false", "no")   # BOXES=0 → clean mask-only overlay


def _load_reader():
    r = RegionReader().to(device)
    sd = torch.load(READER, map_location=device)
    is_real = any(k.startswith("real_adapter") for k in sd)
    if is_real:
        r.add_real_adapter()
    r.load_state_dict(sd); r.eval(); r.set_encoder(_build_encoder())
    return r, is_real


@torch.no_grad()
def answer(image, question):
    if not image or not os.path.exists(image):
        raise SystemExit(f"IMAGE not found: {image!r} — set IMAGE=path/to/section.png")
    W, H = Image.open(image).size
    scene = {"img": image, "hw": (H, W)}                   # raw-image scene — encode() tiles it lazily; no GT needed

    reader, is_real = _load_reader()
    nar = Captioner(); nar.set_stage("s3"); load_narrator(nar, NARRATOR); nar.eval_mode()

    facts = reader_facts(reader, scene)                    # 1) VISION MEASURES the geology
    if is_real:
        facts["derived"] = {}                              # relations reasoned, not asserted, on real
    faults = facts.get("faults", [])
    objs, masks = reader.detect(reader.encode(scene), want_masks=True)   # objs[i] ↔ masks[i] (class-colored overlay)

    print(f"\n=== {os.path.basename(image)} ({W}x{H}) · reader={os.path.basename(READER)} ===", flush=True)
    print(f"[measured] {len(faults)} fault(s):", flush=True)
    for i, f in enumerate(faults):
        thr = f" · throw {f['throw']:.0f} ms" if f.get("throw") is not None else ""
        print(f"   fault {i}: dip {f['dip']:.1f}° · center {f.get('center')}{thr}", flush=True)
    if not faults:
        print("[answer] No fault is visible in this section.", flush=True)
        return

    ev = generate_evidence(nar, facts)                     # 2) LM COPIES the numbers, then reasons + answers
    chain = _clean(generate_think_answer(nar, facts, None, ev, question=question, use_feature=False))
    think, ans = _THINK.search(chain), _ANS.search(chain)
    print(f"\n[Q] {question}", flush=True)
    if think:
        print(f"[reasoning] {think.group(1).strip()}", flush=True)
    print(f"[answer] {ans.group(1).strip() if ans else chain}", flush=True)
    if OUT and masks:
        overlay_classes(image, objs, masks, OUT, boxes=BOXES); print(f"[overlay] {OUT}", flush=True)


if __name__ == "__main__":
    answer(IMAGE, QUESTION)
