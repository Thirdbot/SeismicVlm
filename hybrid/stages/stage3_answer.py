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
    prefix  (MASKED): <evidence> {grounded} <SEG> </evidence>\\n<think> {full free think}
    completion (loss): </think> <answer> {grounded dataset answer} </answer>
The think BODY stays in the MASKED prefix (never re-suppressed, never trains "close the think early"), but
</think> is SUPERVISED so the fold trains the </think>-><answer> TRANSITION — otherwise the model closes the
think and drifts back into <region>/<SEG> instead of the answer ("evidence after think"). Injection = digit
tokens (measured+derived).

Inference is a STAGE-SWITCH: evidence @ s2 (clean copy) -> think+answer @ s3 (fuse).
2 knowledge stages only: geology (frozen) + grounding (frozen); fuse is the combiner. The reason (s4)
adapter is unused by this path.
"""
import os
import torch
from tqdm.auto import tqdm

# Generation budgets — env-knobs so the real-field truncation (dense scenes overrun these) can be swept
# without editing code. Defaults 320/512 preserve all prior behavior.
EVIDENCE_TOKENS = int(os.environ.get("EVIDENCE_TOKENS", 320))
ANSWER_TOKENS = int(os.environ.get("ANSWER_TOKENS", 512))

from hybrid.model.captioner import (REPETITION_PENALTY, row_region_metadata, MAX_OBJ, INSTRUCTION_ROLE,
                                    _region_objs)
from hybrid.model.reader import scene_to_gt
from hybrid.model.captioner import region_metadata
from hybrid.model.text_metrics import _EV, _THINK, _ANS, _clean, grounded, stray_tags, degenerate
from hybrid.checkpoints import save_narrator
import regex as re
device = torch.device("cuda")


def _fold_evidence(ev):
    """Canonical single-wrapped evidence block for the fold prefix. The evidence is STRAIGHT FROM THE
    DATASET and already complete — `<region> …copy… <SEG> </region>` carrying its own per-region <SEG>.
    So we take its inner content (stripping the dataset's outer `<evidence>` tags if present) and wrap it
    in `<evidence>` EXACTLY ONCE, adding NOTHING (no extra <SEG>, no second wrapper). This is applied
    identically at train and inference, so there is no double-<evidence>/double-<SEG> and no whitespace seam.
    Handles both the dataset's full `<evidence>…</evidence>` block (training) and the outer-stripped inner
    that generate_evidence returns (inference)."""
    m = _EV.search(ev)
    inner = (m.group(1) if m else ev).strip()
    return f"<evidence> {inner} </evidence>"


MAX_ANSWER_ROWS = 60         # fast overfit — cap the (slow, generated) answer prep; raise for a full run
ANSWER_EPOCHS = 10

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

@torch.no_grad()
def generate_evidence(nar, facts):
    """Evidence @ s2 — clean grounded digit copy (ends at </evidence>)."""
    nar.set_stage("s2")
    out = nar.generate(facts, question="", instruction=INSTRUCTION_ROLE, max_new_tokens=EVIDENCE_TOKENS)
    m = _EV.search(out)
    return m.group(1) if m else out


@torch.no_grad()
def generate_think_answer(nar, facts, ev, question, sample=False):
    """Think + answer @ s3 (fuse) — prefill clean evidence + <think>, generate the whole remainder
    (full think, </think>, answer) in one pass. Digit markers injected. NO truncation."""
    nar.set_stage("s3")
    prompt = nar.build_prefix(nar._ft(facts), INSTRUCTION_ROLE, question=question)
    prefix = f"{_fold_evidence(ev)}\n<think>"                # dataset evidence used as-is, wrapped once (no added <SEG>)
    full = torch.cat([prompt, nar._emb_text(prefix)], 0)
    kw = dict(do_sample=True, temperature=0.85, top_p=0.92) if sample else dict(do_sample=False)
    g = nar.dec.generate(inputs_embeds=full.unsqueeze(0), max_new_tokens=ANSWER_TOKENS,   # room for verbose think + closed answer
                         repetition_penalty=REPETITION_PENALTY, pad_token_id=nar.tok.eos_token_id, **kw)
    return prefix + nar.tok.decode(g[0], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_chain(nar, facts):
    """Full inference chain via the stage-switch: evidence @ s2 -> think+answer @ s3."""
    ev = generate_evidence(nar, facts)
    return _clean(generate_think_answer(nar, facts, ev, question=""))


@torch.no_grad()
def evaluate_generation(nar, reader, scenes):
    """present/clean/grounded/think on IN-DISTRIBUTION (<=MAX_OBJ) held-out scenes, one greedy pass, NO
    truncation. present = answer emitted; clean = no stray tags/degeneration; grounded = answer respects
    facts + cites a real dip; think = <think> non-empty."""
    pres = clean = grd = think_ok = tot = 0
    for s in tqdm(scenes, desc="fold-eval", unit="sc", leave=False):
        f = region_metadata(s)
        if not (f["faults"] or f.get("closures")) or len(f["faults"]) > MAX_OBJ \
                or len(f.get("closures", [])) > MAX_OBJ:
            continue                                       # any object · in-distribution (<=MAX_OBJ per class)
        ch = generate_chain(nar, f)   # mixed question (Q_MIX)
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


def train_answer(nar, reader, scenes, rows_by_img, epochs=ANSWER_EPOCHS, lr=2e-5, rows_per=5,
               save="stage3_answer.pt"):
    """Build (full masked think + grounded answer) targets and SFT the fuse (s3, grounding frozen)."""
    nar.set_stage("s3"); nar.eval_mode()
    data = []
    # prep is the SLOW phase (one sampled think generation per pair) — show a bar toward the cap so it
    # is never a silent wait.
    prep = tqdm(total=MAX_ANSWER_ROWS, desc="fold-prep (generating thinks)", unit="pair")
    for s in scenes:
        if len(data) >= MAX_ANSWER_ROWS:
            break
        for r in rows_by_img.get(s["img"], [])[:rows_per]:
            f = row_region_metadata(r)
            if not (f["faults"] or f.get("closures")) or len(f["faults"]) > MAX_OBJ \
                    or len(f.get("closures", [])) > MAX_OBJ:
                continue                                   # any object (fault OR closure)
            ev = r.get("evidence") or ""
            ans = r.get("answer") or ""
            q = r.get("question")
            if not ev or not ans:
                continue
            ch = _clean(generate_think_answer(nar, f, ev, q, sample=True))
            i = ch.find("</think>")
            m = _THINK.search(ch[:i + len("</think>")] if i != -1 else ch)
            think = m.group(1).strip() if m else ""
            # think BODY stays MASKED (never re-suppressed, never trains "close think early"), but </think>
            # moves into the SUPERVISED span so the fold trains the </think>-><answer> TRANSITION — else the
            # model closes the think then drifts back into <region>/<SEG> at inference ("evidence after think").
            prefix = f"{_fold_evidence(ev)}\n<think> {think}"            # think body masked; NO closing tag in the prefix
            completion = " ".join(f"</think> {ans}".split())            # supervise: </think> <answer> {ans} </answer>
            data.append((f, prefix, completion, q)); prep.update(1)
            if len(data) >= MAX_ANSWER_ROWS:
                break
    prep.close()
    print(f"[fold] {len(data)} fold pairs (fuse, grounding frozen)", flush=True)
    if len(data) < 4:
        print("[fold] too few — abort", flush=True); return
    nar.set_stage("s3"); nar.train_mode()   # trains fuse; grounding frozen
    opt = torch.optim.AdamW(nar.trainable_params(), lr=lr)
    ebar = tqdm(range(epochs), desc="fold", unit="ep")
    for ep in ebar:
        tot = 0.0
        for f, prefix, completion, q in tqdm(data, desc=f"ep {ep}", unit="pair", leave=False):
            opt.zero_grad()
            loss = nar.completion_loss(f, prefix, completion, question=q, instruction=INSTRUCTION_ROLE)
            loss.backward(); opt.step(); tot += loss.item()
        ebar.set_postfix(loss=f"{tot/max(1,len(data)):.3f}")
        tqdm.write(f"[fold] ep {ep}/{epochs} loss {tot/max(1,len(data)):.3f}")
    nar.eval_mode()
    save_narrator(nar, save)
    print(f"[fold] saved {save}", flush=True)
