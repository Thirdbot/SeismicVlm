"""Train the COMPLETE grounded seismic VLM, end to end.

ARCHITECTURE (vision measures · language copies+reasons · masks grounded):
  Encoder   : SFM (Seismic Foundation Model — frozen ViT-B/16, MAE-pretrained) → dense feature map at
              the finer 512-tile grid. NCS is the fallback when the SFM ckpt is absent.
  Reader    : instance READER over the map — DETR set-prediction (queries + Hungarian + ∅), a learned
              2-D positional grid, registry class-driven heads → facts (count/class/dip/throw/area +
              centroid) and per-instance masks (BCE + dice + clDice thin-structure loss).
  Grounding : grounding LoRA = EVIDENCE COPY — copies the reader's MEASURED facts into the raw evidence
              text across the non-differentiable DIGIT-COPY seam (target ENDS at </evidence>).
  Fuse fold : geology + grounding FROZEN. Trains the grounded <answer> after a FULL, MASKED <think> —
              un-suppresses the think, gives the answer a trained home, protects the copy.
Inference is a STAGE-SWITCH: evidence @ s2 (clean copy) → think+answer @ s3 (fuse).
Masks are the reader's (DETR) instance masks; evaluation is separate (see hybrid.eval.benchmark).

PREPARE (once, in order):
  1. data    — the synthetic CSV (hybrid.data.synthetic.CSV) must exist.
  2. encoder — hybrid/checkpoints/SFM-Base-512.pth (else falls back to NCS).
  3. geology — python -m hybrid.stages.stage1_geology   (builds the frozen geology adapter)
RUN:  python -m hybrid.run_train
  env knobs: READER_EPOCHS · GROUND_EPOCHS · ANSWER_EPOCHS · CLDICE_W · TRAINABLE_BLOCKS · SCENE_CAP · SFM_CKPT
"""
import os
import random
from pathlib import Path

import torch
from tqdm.auto import tqdm

import hybrid.data.loader as sc
from hybrid.data.synthetic import CSV     # synthetic dataset path (unified schema)
SCENE_CAP = (int(os.environ["SCENE_CAP"]) if os.environ.get("SCENE_CAP") else float("inf"))  # env cap for smoke; else uncapped
_SCENE_CAP_DOC = float("inf")        # UNCAPPED — dataset = 406 imgs @100×507; all smaps ≈0.23GB GPU (feature maps are
                               # tiny). The old "200 OOM" was zombie-process contention, not smap memory. The one
                               # thing that scales is resident GT masks (~1.7GB @406). Fold rows capped in stage_fold.
sc.MAX_SCENES = SCENE_CAP
sc.CSV = CSV                # point the unified loader at the synthetic CSV

from hybrid.data.loader import build_scenes
from hybrid.model.captioner import (Captioner, objects_of, region_metadata, region_markers,
                                   K_DIP, K_THROW, K_AREA)
from hybrid.stages.stage2_reader import train_reader, reader_accuracy, reader_facts, mask_dice
from hybrid.model.reader import RegionReader, scene_to_gt
from hybrid.model.geometry import field_dice
from hybrid.stages.stage2_grounding import train_grounding
from hybrid.stages.stage3_answer import train_answer
from hybrid.data.schema import load_local_csv

device = torch.device("cuda")
SEED = 42
READER_EPOCHS = int(os.environ.get("READER_EPOCHS", 40))        # env-tunable. 40 suits the ~2.6k-scene HF synthetic
                                                               # (was 200 for the old ~276-scene set); 10x more data
                                                               # needs far fewer epochs for comparable exposure.
                                                               # Raise it back for a tiny dataset / a full-report run.
GROUND_EPOCHS = int(os.environ.get("GROUND_EPOCHS", 20))
ANSWER_EPOCHS = int(os.environ.get("ANSWER_EPOCHS", 15))
TRAINABLE_BLOCKS = int(os.environ.get("TRAINABLE_BLOCKS", 0))    # frozen default — full SFM finetune is laptop-impractical
                                                                # (re-encode in-graph per step); unfreeze = experiment only
CLDICE_W = float(os.environ.get("CLDICE_W", 1.0))               # thin-structure (centerline) loss weight on the reader mask
CKPT = Path("hybrid/checkpoints")
SFM_CKPT = os.environ.get("SFM_CKPT", str(CKPT / "SFM-Base-512.pth"))   # DEFAULT ENCODER = SFM (finer grid + better dip)
if os.path.exists(SFM_CKPT):                                    # loader auto-uses it; unset SFM_CKPT to fall back to NCS
    os.environ["SFM_CKPT"] = SFM_CKPT


def load_split():
    rng = random.Random(SEED)
    scenes = [s for s in build_scenes() if objects_of(s["objs"])]   # any object (fault OR closure)
    idx = list(range(len(scenes))); rng.shuffle(idx)
    cut = int(len(idx) * 0.75)
    return scenes, [scenes[i] for i in idx[:cut]], [scenes[i] for i in idx[cut:]]


def _fmt(mn): return f"{mn[0]:.2f}(n{mn[1]})" if mn and mn[0] is not None else "n0"


def main():
    print(f"[prep] encoder {'SFM' if os.environ.get('SFM_CKPT') else 'NCS'} · clDice {CLDICE_W} · "
          f"reader epochs {READER_EPOCHS} · trainable_blocks {TRAINABLE_BLOCKS}", flush=True)
    scenes, tr, te = load_split()
    print(f"[train] scenes {len(scenes)} · train {len(tr)} · test {len(te)}", flush=True)

    # ---- Stage 2: instance reader (facts) ----
    reader_pt = CKPT / "reader.pt"
    reader = None
    # RETRAIN_READER=0 to reuse a cached reader. Default is RETRAIN, because silently reusing reader.pt
    # made READER_EPOCHS / CLDICE_W / TRAINABLE_BLOCKS / the encoder into NO-OPS — the run then reported
    # metrics for the PREVIOUS config's reader and re-saved it, erasing the evidence.
    if reader_pt.exists() and os.environ.get("RETRAIN_READER", "1") == "0":
        try:
            reader = RegionReader().to(device)
            reader.load_state_dict(torch.load(reader_pt, map_location=device))
            reader.eval()
            print("[train] loaded cached reader.pt (skip reader training)", flush=True)
        except RuntimeError as e:                # arch changed (e.g. registry heads) → cached ckpt stale
            print(f"[train] cached reader.pt incompatible with current arch → retraining ({str(e)[:80]}…)", flush=True)
            reader = None
    if reader is None:
        reader = train_reader(tr, epochs=READER_EPOCHS, trainable_blocks=TRAINABLE_BLOCKS, cldice_w=CLDICE_W)
    if reader.enc is None:                           # cached reader (loaded from reader.pt) has no encoder — attach it
        from hybrid.stages.stage2_reader import _build_encoder
        reader.set_encoder(_build_encoder(TRAINABLE_BLOCKS))
        enc_ck = CKPT / "reader_enc.pt"
        if TRAINABLE_BLOCKS > 0 and enc_ck.exists():
            reader.enc.load_state_dict(torch.load(enc_ck, map_location=device))   # tuned encoder for the unfrozen run
    for tag, sp in (("train", tr), ("test(held-out)", te)):
        a = reader_accuracy(reader, sp)
        md = mask_dice(reader, sp, oracle=True)      # mask decoder in isolation (GT-matched)
        dd = mask_dice(reader, sp, oracle=False)     # DEPLOYMENT: masks from detect(), misses score 0
        print(f"[reader {tag}] count MAE {_fmt(a['count'])} · dip MAE {_fmt(a['dip'])}deg · "
              f"class {a['cls'][0]}/{a['cls'][1]} · mask dice {_fmt(md)} (oracle) / {_fmt(dd)} (deployed)",
              flush=True)
    torch.save(reader.state_dict(), CKPT / "reader.pt")

    # ---- Stage 2 (LM): grounding LoRA = EVIDENCE COPY (set_stage s2). Copies the injected facts into
    # the raw dataset evidence text; this is the copy that the combined stage must NOT disturb. ----
    facts_by_img = {s["img"]: region_metadata(s) for s in tr}
    nar = Captioner()
    nar.dec.gradient_checkpointing_enable()      # recompute activations in backward -> fits a ~6 GB consumer GPU
    nar.dec.enable_input_require_grads()
    from hybrid.checkpoints import save_narrator, load_narrator
    if os.environ.get("WARM_BASE"):
        load_narrator(nar, "stage34_narrator.pt"); print("[diag] loaded warm base stage34_narrator.pt", flush=True)
    if os.environ.get("SKIP_GROUNDING") == "1" and (CKPT / "stage2_grounding.pt").exists():
        load_narrator(nar, "stage2_grounding.pt")              # STAGE INDEPENDENCE: re-run ONLY the fold (stage 3)
        print("[stage2] SKIP_GROUNDING: loaded grounding checkpoint → re-running only the fold", flush=True)
    else:
        train_grounding(nar, facts_by_img, epochs=GROUND_EPOCHS)   # grounding target ENDS at </evidence>
        save_narrator(nar, name="stage2_grounding.pt")        # post-grounding checkpoint → fold is independently re-runnable
        print("[stage2] grounding checkpoint → stage2_grounding.pt", flush=True)

    # ---- Stage 3 — the FUSE FOLD (set_stage s3; geology + grounding FROZEN). Trains the grounded
    # <answer> as the completion after a FULL, MASKED <think>: un-suppresses the think (fuse's delta
    # counteracts grounding's stop-bias), gives the answer a trained home (no truncation), and protects
    # the copy (grounding frozen). </think> in the masked prefix; only <answer> supervised. Injection =
    # digit tokens (measured+derived). Replaces the failed STaR/reason-adapter path. ----
    from collections import defaultdict
    rows_by_img = defaultdict(list)
    for r in load_local_csv(csv_path=CSV):
        ip = (r.get("image_paths") or [None])[0]
        if ip:
            rows_by_img[ip].append(r)
    # ---- copy fidelity helper — measured BEFORE and AFTER the fold to prove the fuse fold (grounding
    # frozen) did NOT tamper with the evidence copy ----

    def copy_score():
        nar.set_stage("s2"); nar.eval_mode(); hit = tot = 0                            # evidence @ s2 (fuse OFF)
        for s in tqdm(te[:20], desc="copy-score", unit="sc", leave=False):
            facts = reader_facts(reader, s)
            if not (facts["faults"] or facts.get("closures")):
                continue
            vals = [v for k, v in region_markers(facts) if k in (K_DIP, K_THROW, K_AREA)]  # any measured attr
            out = nar.generate(facts, question="", max_new_tokens=140)
            for d in vals:
                tot += 1; hit += (d in out)
        return hit, tot

    b_hit, b_tot = copy_score()
    print(f"[copy BEFORE fold] {b_hit}/{b_tot}", flush=True)

    train_answer(nar, reader, tr, rows_by_img, epochs=ANSWER_EPOCHS, rows_per=5)   # fuse fold (grounding frozen)

    a_hit, a_tot = copy_score()
    print(f"[copy AFTER fold]  {a_hit}/{a_tot}  (must ~match BEFORE — proves fuse fold protects copy)", flush=True)
    save_narrator(nar)                            # secondary copy → stage3_narrator.pt (canonical is stage3_answer.pt, saved by train_answer)
    print("MAIN_MODEL_DONE", flush=True)
    # Evaluation is SEPARATE (train here, eval there) — run after this finishes:
    #   python -m hybrid.eval.benchmark      # component metrics: pooled IoU + detF1 (the paper numbers)
    #   python -m hybrid.eval.inference      # qualitative reasoning chains (evidence→think→answer)


if __name__ == "__main__":
    main()
