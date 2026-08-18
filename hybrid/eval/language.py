"""Standalone LANGUAGE re-eval — copy fidelity + faithfulness (CHAIR), nothing else.

The language half (grounding + fuse-fold narrator) is trained INDEPENDENTLY of the vision reader, so
this scores JUST the narrator on the saved weights, without re-running any training:

  [COPY-mechanism  GT ]   inject GROUND-TRUTH facts → does the narration reproduce every injected number?
                          (the pure copy test — reader EXCLUDED)
  [COPY-pipeline reader]   inject READER-measured facts → same (the deployment number; reader-capped)
  [FAITHFULNESS] answer CHAIR_I   fraction of numbers in <answer> NOT backed by an injected marker
                          (0 = every stated number is measured; FULL marker set — count, bbox, center,
                           dip/throw/area, derived — not just dip/throw/area)
  [FAITHFULNESS] think  CHAIR_I   same test on the free-reasoning <think> (expected >= answer — the
                          think is where CoT confabulates before the answer re-grounds; reported, not hidden)
  [FAITHFULNESS] think->answer consistency   fraction of <answer> numbers that also appear in <think>
                          (1 = every stated number was reasoned to; <1 = the answer introduces/flips numbers)

  [OVERLAP] BLEU-1..4 / ROUGE-L / CIDEr-D / METEOR — the standard captioning-overlap metrics, reported
                          for FIELD COMPARABILITY only. The narration is free-generated and does not
                          template-match the dataset answer, so overlap is a LOWER BOUND, NOT the
                          faithfulness axis (that's copy/CHAIR/swap above). Kept so the paper answers the
                          "where are the standard caption metrics?" question without conceding the framing.

No reader-accuracy / mask / box metrics here. Reuses the exact copy/CHAIR code paths from
hybrid.eval.components; overlap metrics live in hybrid.eval.caption_metrics (dependency-free).

  CKPT=stage3_answer.pt READER=hybrid/checkpoints/reader.pt python -m hybrid.eval.language
  DATASET=smeaheia CKPT=stage3_answer.pt READER=hybrid/checkpoints/ab_experiment/B_joint.pt \
      python -m hybrid.eval.language        # language on a real survey (reader = a real-adapter ckpt)

Env: CKPT (narrator, default stage3_answer.pt) · READER (default reader.pt) · DATASET (synthetic|smeaheia)
     · SCENES (cap; uncapped by default = same held-out split as training).
"""
import os
os.environ.setdefault("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")   # frozen encoder (no silent fallback)
import torch

from hybrid.model.captioner import Captioner
from hybrid.model.reader import RegionReader
from hybrid.checkpoints import load_narrator
from hybrid.stages.stage2_reader import _build_encoder
from hybrid.eval.components import copy_test, academic_table, overlap_table, held_out, CKPT

device = torch.device("cuda")


def main():
    reader = RegionReader().to(device)
    reader_pt = os.environ.get("READER", "hybrid/checkpoints/reader.pt")
    sd = torch.load(reader_pt, map_location=device)
    if any(k.startswith("real_adapter") for k in sd):      # real-field ckpt = synthetic base + real adapter
        reader.add_real_adapter()
    reader.load_state_dict(sd); reader.eval()
    reader.set_encoder(_build_encoder())
    nar = Captioner(); nar.set_stage("s3"); load_narrator(nar, CKPT); nar.eval_mode()

    te = held_out()
    ds = os.environ.get("DATASET", "synthetic")
    print(f"[language] narrator {CKPT} · reader {reader_pt} · dataset {ds} · held-out {len(te)}", flush=True)

    g_hit, g_tot = copy_test(nar, reader, te, use_reader=False)
    r_hit, r_tot = copy_test(nar, reader, te, use_reader=True)
    print(f"[COPY-mechanism  GT ] {g_hit}/{g_tot} = {g_hit/max(1,g_tot):.2f}   <- PURE copy (inject GT)", flush=True)
    print(f"[COPY-pipeline reader] {r_hit}/{r_tot} = {r_hit/max(1,r_tot):.2f}   <- reader-capped (deployment)", flush=True)

    academic_table(nar, reader, te)                        # [FAITHFULNESS] CHAIR_I (full marker set) — PRIMARY
    overlap_table(nar, reader, te)                         # [OVERLAP] BLEU/ROUGE-L/CIDEr-D/METEOR — comparability
    print("LANGUAGE_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
