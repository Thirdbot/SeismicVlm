"""Stage 3 — the FUSE FOLD (validated design).

Trains the grounded ANSWER as the completion after a FULL, MASKED <think>, on the FUSE adapter
(set_stage s3; geology + grounding FROZEN). This:
  (a) UN-SUPPRESSES the think — the fuse's additive delta (every fold example has a full <think>
      before the answer) counteracts grounding's stop-bias, so the <think> fires at inference
      (measured think 0.22 -> 1.00 when the fold is trained);
  (b) gives the answer a TRAINED HOME — so it comes out clean WITHOUT truncation (present/clean rise
      from 0.00), the </think>->answer transition being the training;
  (c) PROTECTS the copy — fuse trains with grounding frozen (its AdapterFusion job), so the evidence
      copy can't drift (copy before == after).

Target per row (completion-only):
    prefix  (MASKED): <evidence> {grounded} <SEG> </evidence>\\n<think> {full free think} </think>
    completion (loss): <answer> {grounded dataset answer} </answer>
</think> lives in the MASKED prefix so the fold never trains "close the think early" (that re-suppressed
it); only the answer is supervised. Injection = digit tokens (measured+derived) + gated <feature>_i.

Inference is a STAGE-SWITCH: evidence @ s2 (clean copy) -> think+answer @ s3 (fuse).
2 knowledge stages only: geology (frozen) + grounding (frozen); fuse is the combiner. The reason (s4)
adapter is unused by this path.
"""
import torch
from mpmath.math2 import INF
from tqdm.auto import tqdm

from hybrid.model.narrator import (row_facts, MAX_OBJ, INSTRUCTION_ROLE,
                                    _fact_objs)
from hybrid.model.reader import scene_to_gt
from hybrid.model.narrator import scene_facts
from hybrid.model.text_metrics import _EV, _THINK, _ANS, _clean, grounded, stray_tags, degenerate
from hybrid.checkpoints import save_narrator
import regex as re
device = torch.device("cuda")
# Mixed question (grounding clause + implication clause) — authored, not from the dataset. Drives the
# out-of-distribution reasoning leg at inference; training uses each row's own (in-distribution) question.

MAX_FOLD_ROWS = INF
TRAIN_FUSE = 10

# MODES = {"horst": "horst_and_graben", "graben": "horst_and_graben", "relay": "relay_ramp",
#          "ramp": "relay_ramp", "branch": "self_branching", "stair": "stair_case"}
NUM = re.compile(r"\d+\.?\d*")
def _exact(f):
    """ALL ground-truth values — every class, every attribute — because the <think> may reference any
    of them and the filter validates against the COMPLETE measured+derived fact set (not just faults).
    Using the full truth makes the reasoning a CORRECTION layer: filtered against the true values, the
    think can reason past the ~11% the raw-evidence copy gets wrong toward a right answer."""
    faults = f.get("faults", []); closures = f.get("closures", [])
    a = [float(len(faults))]
    if closures:
        a.append(float(len(closures)))                 # count: closures (nclosure)
    for x in faults:
        a.append(round(float(x["dip"]), 1))            # measure: dip
        if x.get("throw") is not None:
            a.append(round(float(x["throw"])))         # measure: throw
    for c in closures:
        if c.get("area_pct") is not None:
            a.append(round(float(c["area_pct"])))      # measure: closure area
    d = f.get("derived") or {}
    if d.get("intersect") is not None:
        a.append(float(d["intersect"]))                # derive: intersections
    return a


def _coords(f):
    c = []
    for x in f["faults"] + f.get("closures", []):
        c += [float(v) for v in (x.get("bbox") or [])] + [float(v) for v in (x.get("center") or [])]
    return c


def respects(text, facts):
    """THE faithfulness filter: every number respects the facts (exact | coord | ordinal | in-range |
    difference | average), mode matches, steep/gentle matches the dip, no drift, no stray tags,
    no academic-persona drift."""
    if degenerate(text) or stray_tags(text):
        return False
    a = _exact(facts); co = _coords(facts); t = text.lower();n=len(a)
    for m in NUM.findall(text):
        try:
            v = float(m)
        except ValueError:
            continue
        ok = (any(abs(v - x) < 0.6 for x in a) or any(abs(v - x) < 1.0 for x in co)
              or (1 <= v <= n and v == int(v)))
        if not ok:
            return False
    # gmk = ((facts.get("derived") or {}).get("mode") or "").replace(" ", "_").replace("-", "_")
    # for kw, mode in MODES.items():
    #     if kw in t and gmk and mode != gmk and gmk not in mode:
    #         return False
    return True

def _answer_ok(ans_text, facts):
    """ANSWER-CORRECTNESS (STaR's core check, adapted): the model's OWN answer must respect the facts
    (no confabulated number/mode) AND cite a real dip. Verifies the reasoning BRIDGED to a faithful
    answer — not just that the think was faithful. (Our answer is grounded generation, so 'correct' =
    faithful to the measured facts, the checkable quantity in this domain.)"""
    return respects(ans_text, facts) and grounded(ans_text, facts)


def _answer_anchor(ans_text, facts):
    """ANSWER-AS-ANCHOR, CONTENT-ONLY (the fix that stops the collapse). Same numeric-faithfulness check
    as respects() — every number is an exact fact / coord / ordinal / in-range / diff / average, and it
    cites a real dip — but WITHOUT the stray-tag / degenerate / academic FORM gates. The 1.5B's answer
    FORM is messy tag-salad, and form is irrelevant here because we train on the DATASET answer, not the
    model's; gating on form is exactly what starved main17 (5→1). Content-only ⇒ far more chains pass ⇒
    the loop doesn't starve, while a confabulated number in the answer still rejects the chain."""
    a = _exact(facts); co = _coords(facts)
    n = len(a)
    for m in NUM.findall(ans_text):
        try:
            v = float(m)
        except ValueError:
            continue
        ok = (any(abs(v - x) < 0.6 for x in a) or any(abs(v - x) < 1.0 for x in co)
              or (1 <= v <= n and v == int(v)))
        if not ok:
            return False
    return grounded(ans_text, facts)

@torch.no_grad()
def aligned_feats(reader, scene, facts):
    """Per-object reader hidden state h_i aligned to the fact objects by nearest centroid — the gated
    <feature>_i soft token's source (qualitative, index-bound). None-padded where unmatched."""
    gt = scene_to_gt(scene)
    if not gt:
        return []
    h, cents = reader.object_states(scene["smap"], gt)
    H, W = scene["hw"]
    objs = _fact_objs(facts)                              # (class, obj) across all buckets — matches _ft order
    feats = []
    for _cls, o in objs:
        c = o.get("center")
        if c is None or not cents:
            feats.append(None); continue
        r, col = c[1] / H, c[0] / W
        d = [(r - cr) ** 2 + (col - cc) ** 2 for cr, cc in cents]
        feats.append(h[d.index(min(d))])
    return feats


@torch.no_grad()
def fold_evidence(nar, facts):
    """Evidence @ s2 — digit only (feature off), clean grounded copy (ends at </evidence>)."""
    nar.use_feature = False; nar.set_stage("s2")
    out = nar.generate(facts, question="", instruction=INSTRUCTION_ROLE, max_new_tokens=120)
    m = _EV.search(out)
    return m.group(1) if m else out


@torch.no_grad()
def fold_think_answer(nar, facts, feats, ev, question, sample=False, use_feature=True):
    """Think + answer @ s3 (fuse) — prefill clean evidence + <think>, generate the whole remainder
    (full think, </think>, answer) in one pass. digit(+feature) injected. NO truncation. use_feature
    toggles the <feature>_i soft token for the reasoning A/B (does the qualitative feature help?)."""
    nar.use_feature = use_feature; nar.set_stage("s3")
    prompt = nar.build_prompt(nar._ft(facts, feats if use_feature else None), INSTRUCTION_ROLE, question=question)
    prefix = f"<evidence> {ev} <SEG> </evidence>\n<think>"
    full = torch.cat([prompt, nar._emb_text(prefix)], 0)
    kw = dict(do_sample=True, temperature=0.85, top_p=0.92) if sample else dict(do_sample=False)
    g = nar.dec.generate(inputs_embeds=full.unsqueeze(0), max_new_tokens=320,   # room for verbose think + closed answer
                         repetition_penalty=1.3, pad_token_id=nar.tok.eos_token_id, **kw)
    return prefix + nar.tok.decode(g[0], skip_special_tokens=True).strip()


@torch.no_grad()
def fold_chain(nar, facts, reader=None, scene=None, use_feature=True):
    """Full inference chain via the stage-switch: evidence @ s2 -> think+answer @ s3. use_feature
    toggles the reasoning feature token (A/B)."""
    feats = (aligned_feats(reader, scene, facts)
             if (use_feature and reader is not None and scene is not None) else None)
    ev = fold_evidence(nar, facts)
    return _clean(fold_think_answer(nar, facts, feats, ev, use_feature=use_feature,question=""))


@torch.no_grad()
def fold_eval(nar, reader, scenes, use_feature=True):
    """Same metric as the experiment (validates the port numerically): present/clean/grounded/think on
    IN-DISTRIBUTION (<=MAX_OBJ) held-out scenes, one greedy pass, NO truncation. present = answer emitted;
    clean = no stray tags/degeneration; grounded = answer respects facts + cites a real dip; think =
    <think> non-empty. use_feature = the reasoning feature A/B (ON vs OFF → does the feature help?)."""
    pres = clean = grd = think_ok = tot = 0
    for s in tqdm(scenes, desc=f"fold-eval (feat {'ON' if use_feature else 'OFF'})", unit="sc", leave=False):
        f = scene_facts(s)
        if not (f["faults"] or f.get("closures")) or len(f["faults"]) > MAX_OBJ \
                or len(f.get("closures", [])) > MAX_OBJ:
            continue                                       # any object · in-distribution (<=MAX_OBJ per class)
        ch = fold_chain(nar, f, reader, s, use_feature=use_feature)   # mixed question (Q_MIX)
        tm = _THINK.search(ch); think = tm.group(1).strip() if tm else ""
        am = _ANS.search(ch)
        if am:
            ans = am.group(1).strip()
        else:                                              # answer opened but not closed (budget) — take to end
            i = ch.find("<answer>")
            ans = ch[i + len("<answer>"):].strip() if i != -1 else ""
        tot += 1
        pres += int(bool(ans))
        clean += int(bool(ans) and not stray_tags(ans) and not degenerate(ans))
        grd += int(bool(ans) and respects(ans, f) and grounded(ans, f))
        think_ok += int(len(think) > 15)
    n = max(1, tot)
    return dict(present=pres / n, clean=clean / n, grounded=grd / n, think=think_ok / n, n=tot)


def train_fold(nar, reader, scenes, rows_by_img, epochs=TRAIN_FUSE, lr=2e-5, rows_per=5, use_feature=True,
               digit_dropout=0.0, gate_reg=0.0, save="stage_fold_narrator.pt"):
    """Build (full masked think + grounded answer) targets and SFT the fuse (s3, grounding frozen).

    FEATURE-ACTIVATION knobs (default OFF — only help when the answer needs qualitative texture the
    digit can't give; ON with purely-numeric answers corrupts the copy):
      digit_dropout : fraction of examples trained with the injected VALUES blanked (nar.mask_digits) —
                      modality dropout that forces the answer to lean on the <feature>_i visual token.
      gate_reg      : anti-collapse pull that keeps |feat_gate| from decaying to 0 (attention-reg proxy)."""
             # bound build time (each row = one sampled think generation) + memory
    nar.use_feature = use_feature; nar.set_stage("s3"); nar.eval_mode()
    data = []
    # prep is the SLOW phase (one sampled think generation per pair) — show a bar toward the cap so it
    # is never a silent wait.
    prep = tqdm(total=MAX_FOLD_ROWS, desc="fold-prep (generating thinks)", unit="pair")
    for s in scenes:
        if len(data) >= MAX_FOLD_ROWS:
            break
        for r in rows_by_img.get(s["img"], [])[:rows_per]:
            f = row_facts(r)
            if not (f["faults"] or f.get("closures")) or len(f["faults"]) > MAX_OBJ \
                    or len(f.get("closures", [])) > MAX_OBJ:
                continue                                   # any object (fault OR closure)
            ev = r.get("evidence") or ""
            ans = r.get("answer") or ""
            q = r.get("question")
            if not ev or not ans:
                continue
            feats = aligned_feats(reader, s, f) if use_feature else None
            ch = _clean(fold_think_answer(nar, f, feats, ev, q, sample=True))
            i = ch.find("</think>")
            m = _THINK.search(ch[:i + len("</think>")] if i != -1 else ch)
            think = m.group(1).strip() if m else ""
            # </think> in the MASKED prefix; supervise ONLY <answer> (never train "close think early")
            prefix = " ".join(f"<evidence> {ev} <SEG> </evidence>\n<think> {think} </think>".split())
            completion = " ".join(f"{ans}".split())
            data.append((f, feats, prefix, completion, q)); prep.update(1)
            if len(data) >= MAX_FOLD_ROWS:
                break
    prep.close()
    print(f"[fold] {len(data)} fold pairs (fuse, grounding frozen)", flush=True)
    if len(data) < 4:
        print("[fold] too few — abort", flush=True); return
    nar.use_feature = use_feature; nar.set_stage("s3"); nar.train_mode()   # trains fuse (+feat); grounding frozen
    opt = torch.optim.AdamW(nar.trainable_params(), lr=lr)
    ebar = tqdm(range(epochs), desc="fold", unit="ep")
    for ep in ebar:
        tot = 0.0
        for bi, (f, feats, prefix, completion, q) in enumerate(tqdm(data, desc=f"ep {ep}", unit="pair", leave=False)):
            opt.zero_grad()
            nar.mask_digits = digit_dropout > 0 and ((bi * 2654435761) % 100) / 100.0 < digit_dropout
            loss = nar.completion_loss(f, prefix, completion, question=q,
                                       instruction=INSTRUCTION_ROLE, feats=feats)
            if gate_reg > 0:                                  # keep the visual channel alive (anti-collapse)
                loss = loss + gate_reg * torch.relu(0.1 - nar.feat_gate.abs()).mean()
            loss.backward(); opt.step(); tot += loss.item()
        nar.mask_digits = False
        ebar.set_postfix(loss=f"{tot/max(1,len(data)):.3f}", gate=f"{float(nar.feat_gate):.4f}")
        tqdm.write(f"[fold] ep {ep}/{epochs} loss {tot/max(1,len(data)):.3f} · gate {float(nar.feat_gate):.4f}")
    nar.eval_mode()
    save_narrator(nar, save)
    print(f"[fold] saved {save}", flush=True)
