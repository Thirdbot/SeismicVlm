"""Stage 3 — grounded narration (fuse alignment; segmentor frozen).

Trains ONLY the fuse combiner (geology + grounding frozen) to produce the dataset's
grounded narration — the tagged <evidence>...</evidence> <answer>...</answer> chain —
with every number COPIED from the injected role-tagged digit facts. Reader is NOT
touched here (masks come from the reader; mask generalization is the real-field stage's
job). The LM narrates from the injected numbers only — decoupled.

No <think> here: this stage sets up evidence + tag-wrapped answer; the combined stage
(stage4_star.train_reasoning) opens <think> after </evidence> and fills the reasoning via
STaR (completion-only, evidence masked). Target = raw dataset evidence + <answer>{answer}
</answer>; every number is a copied fact, nothing about the answer is templated.
"""
import torch
from mpmath.math2 import INF

from hybrid.data.dataset import load_local_csv
from hybrid.model.scenes import CSV
from hybrid.model.narrator import narration_target, MAX_OBJ, row_facts

LM_EPOCHS = 150
MAX_ROWS = INF
ROWS_PER_IMG = INF              # cover every training image with a few rows (diversity) — NOT 500 rows
                             # clustered on a handful of images. This is the real speed lever.


def narration_rows(facts_by_img):
    """(injected facts, target, question) — Q+A on the RAW dataset evidence + answer, capped to a
    few rows per image so every training scene is represented without over-sampling a few."""
    from collections import defaultdict
    seen = defaultdict(int)
    out = []
    for r in load_local_csv(csv_path=CSV):
        img = (r.get("image_paths") or [None])[0]
        if img not in facts_by_img:           # train/test split filter (no leakage)
            continue
        facts = row_facts(r)                  # 1:1 with THIS row's evidence (not the image union)
        if not facts["faults"]:
            continue
        if len(facts["faults"]) > MAX_OBJ or len(facts["closures"]) > MAX_OBJ:
            continue
        if seen[img] >= ROWS_PER_IMG:
            continue
        seen[img] += 1
        ev, an, q = r.get("evidence") or "", r.get("answer") or "", r.get("question") or ""
        instr = r.get("instruction") or ""
        out.append((facts, narration_target(ev, an), q, instr))
        if len(out) >= MAX_ROWS:
            break
    return out


def train_narrator(nar, facts_by_img, epochs=LM_EPOCHS):
    """Stage 3: train the fuse (geology+grounding frozen) on the grounded narration."""
    nar.set_stage("s3")
    data = narration_rows(facts_by_img)
    opt = torch.optim.AdamW(nar.trainable_params(), lr=1e-4)
    nar.train_mode()
    print(f"[narrator] grounded narration on {len(data)} rows", flush=True)
    for ep in range(epochs):
        tot = 0.0
        for facts, target, question, instr in data:
            opt.zero_grad()
            loss = nar.ground_loss(facts, target, question=question, instruction=instr)   # chatml Q+A
            loss.backward(); opt.step(); tot += loss.item()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[narrator] ep {ep} loss {tot/max(1, len(data)):.3f}", flush=True)
