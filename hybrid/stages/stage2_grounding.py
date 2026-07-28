"""Stage 2 (language half) — evidence-copy grounding.

Trains the grounding adapter to COPY injected fact values into the dataset's
evidence text, grounding the shared linguistic latent (per the staging + fuse
design). geology frozen, fuse inactive. Grounding is frozen afterwards, when
stage 3 switches the decoder to train the fuse combiner.
"""
import re

import torch
from mpmath.math2 import INF
from tqdm.auto import tqdm

from hybrid.data.schema import load_local_csv
from hybrid.data.synthetic import CSV
from hybrid.model.narrator import INSTRUCTION_ROLE,MAX_OBJ, row_facts

GROUND_EPOCHS = 150          # more epochs → better multi-object enumeration (copy); un-suppress holds
MAX_ROWS = INF


def evidence_rows(facts_by_img):
    """(injected facts, raw-evidence target). Inject facts_to_kv (the SAME structure inference
    injects); target = the row's RAW dataset evidence (grounding_target → raw_narrative, 1:1 with
    this row's facts), so every injected marker has a home AND the latent stays seismic."""
    from collections import defaultdict
    seen = defaultdict(int)
    out = []
    for r in load_local_csv(csv_path=CSV):
        img = (r.get("image_paths") or [None])[0]
        if img not in facts_by_img:           # train/test split filter (no leakage)
            continue
        facts = row_facts(r)                  # 1:1 with THIS row's evidence
        if not (facts["faults"] or facts["closures"]):   # any object (fault OR closure)
            continue
        if len(facts["faults"]) > MAX_OBJ or len(facts["closures"]) > MAX_OBJ:
            continue
        if seen[img] >= 5:                    # more rows/image → more multi-fault enumeration coverage
            continue
        seen[img] += 1
        out.append((facts,r.get("evidence")or ""))
        if len(out) >= MAX_ROWS:
            break
    return out


def train_grounding(nar, facts_by_img, epochs=GROUND_EPOCHS):
    nar.set_stage("s2")
    data = evidence_rows(facts_by_img)
    opt = torch.optim.AdamW(nar.trainable_params(), lr=1e-4)
    print(f"[grounding] training on {len(data)} rows · {epochs} epochs", flush=True)
    nar.train_mode()
    ebar = tqdm(range(epochs), desc="grounding", unit="ep")
    for ep in ebar:
        tot = 0.0
        for facts, target in tqdm(data, desc=f"ep {ep}", unit="row", leave=False):
            opt.zero_grad()
            loss = nar.ground_loss(facts, target, question=None, instruction=INSTRUCTION_ROLE)
            loss.backward(); opt.step(); tot += loss.item()
        ebar.set_postfix(loss=f"{tot/max(1, len(data)):.3f}")
        tqdm.write(f"[grounding] ep {ep}/{epochs} loss {tot/max(1, len(data)):.3f}")
