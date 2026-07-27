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

from hybrid.model.narrator import (row_facts, raw_narrative, MAX_OBJ, INSTRUCTION_S2, INSTRUCTION_ROLE,
                                    _fact_objs)
from hybrid.model.reader import scene_to_gt
from hybrid.model.narrator import scene_facts
from hybrid.train.stage4_star import _EV, _THINK, _ANS, _clean, respects, grounded, stray_tags, degenerate
from hybrid.checkpoints import save_narrator

device = torch.device("cuda")
# Mixed question (grounding clause + implication clause) — authored, not from the dataset. Drives the
# out-of-distribution reasoning leg at inference; training uses each row's own (in-distribution) question.

MAX_FOLD_ROWS = INF

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
    out = nar.generate(facts, question="", instruction=INSTRUCTION_S2, max_new_tokens=120)
    m = _EV.search(out)
    return raw_narrative(m.group(1) if m else out)


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


def train_fold(nar, reader, scenes, rows_by_img, epochs=8, lr=2e-5, rows_per=5, use_feature=True,
               save="stage_fold_narrator.pt"):
    """Build (full masked think + grounded answer) targets and SFT the fuse (s3, grounding frozen)."""
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
            ev = raw_narrative(r.get("evidence") or ""); ans = r.get("answer") or ""
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
            completion = " ".join(f"<answer> {ans} </answer>".split())
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
        for f, feats, prefix, completion, q in tqdm(data, desc=f"ep {ep}", unit="pair", leave=False):
            opt.zero_grad()
            loss = nar.completion_loss(f, prefix, completion, question=q,
                                       instruction=INSTRUCTION_ROLE, feats=feats)
            loss.backward(); opt.step(); tot += loss.item()
        ebar.set_postfix(loss=f"{tot/max(1,len(data)):.3f}", gate=f"{float(nar.feat_gate):.4f}")
        tqdm.write(f"[fold] ep {ep}/{epochs} loss {tot/max(1,len(data)):.3f} · gate {float(nar.feat_gate):.4f}")
    nar.eval_mode()
    save_narrator(nar, save)
    print(f"[fold] saved {save}", flush=True)
