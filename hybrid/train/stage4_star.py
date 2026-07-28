"""Combined stage 3-4 — STaR: bootstrap faithful, grounded, IN-ORDER reasoning into the narrator.

Produces the full skeleton `<evidence>{grounded facts} <SEG> </evidence> <think>{grounded reasoning}
</think> <answer>{answer}</answer>`. The finalize is COMPLETION-ONLY (`train_reasoning`): the evidence
+ open <think> is a MASKED prefix (stage-2's evidence copy is never touched); only the think body,
closing tags, and tag-wrapped dataset answer are supervised. The think is generated + trained under
the SAME prompt as the earlier stages (fact_ft facts, raw evidence, dataset question, INSTRUCTION_S34
— the reasoning role in the instruction), so there is no train/serve seam. STaR: sample K chains, KEEP
the well-formed ones (</think>+<answer>, respect the facts, stay grounded), SFT on them. respects() is
the faithfulness filter (the strict exact-match check was an artifact — see reference_star_probe): a
number is OK if it's an exact fact, a bbox/center coord, a fault ordinal, in the dip range, a
difference of two dips, or the average; plus mode/steep-gentle consistency and no non-ASCII drift.
grounded() = cites a real dip. Saves stage34_narrator.pt.
"""
import re
import random
from collections import Counter

import torch

_FAIL = Counter()          # set-build fail-mode tally (diagnostic)
_DBG = {"printed": 0}      # cap on raw-chain prints

from hybrid.model.narrator import MAX_OBJ, INSTRUCTION_ROLE
from hybrid.checkpoints import save_narrator


def _steep(d):
    return "steeply dipping" if d >= 55 else ("gently dipping" if d <= 40 else "moderately dipping")


def build_think(facts):
    """OPTION A — CONSTRUCT a faithful <think> directly from the ground-truth facts, instead of
    generating-and-filtering with the confabulating 1.5B. Every number comes from a fact (0%
    confabulation by construction), the prose is clean (no stray tags / HTML / brackets), and it's
    built for EVERY row (not the ~15% STaR keeps). Qualitative reasoning about the true values —
    steep/gentle, steepest/gentlest, intersections, mode, closure. Single- and multi-fault."""
    faults = facts.get("faults", [])[:MAX_OBJ]
    closures = facts.get("closures", [])
    der = facts.get("derived") or {}
    n = len(faults)
    dips = [round(float(x["dip"]), 1) for x in faults]
    parts = []
    if n == 1:
        d = dips[0]
        parts.append(f"There is one fault dipping at about {d:g} degrees, which is {_steep(d)}.")
        if faults[0].get("throw") is not None:
            parts.append(f"Its throw of about {round(float(faults[0]['throw']))} ms measures the "
                         f"vertical displacement across the fault.")
    else:
        parts.append(f"There are {n} faults, dipping at " + ", ".join(f"{d:g}" for d in dips)
                     + " degrees respectively.")
        if max(dips) != min(dips):
            i_s = dips.index(max(dips)); i_g = dips.index(min(dips))
            parts.append(f"Fault {i_s + 1} at {max(dips):g} degrees is the steepest and Fault "
                         f"{i_g + 1} at {min(dips):g} degrees the gentlest.")
        overall = ("steeply dipping" if min(dips) >= 55 else
                   "gently dipping" if max(dips) <= 40 else "of mixed steepness")
        parts.append(f"Overall the faults are {overall}.")
    if der.get("intersect") is not None:
        k = int(der["intersect"])
        parts.append(f"The faults intersect, with {k} intersection{'s' if k != 1 else ''} in this section.")
    if der.get("mode"):
        parts.append(f"This pattern is characteristic of a {der['mode']} structure.")
    if closures:
        parts.append(f"A closure covering about {round(float(closures[0]['area_pct']))} percent of the "
                     f"section is also present, a potential trap.")
    return " ".join(parts)

device = torch.device("cuda")
# Reasoning role now lives in INSTRUCTION_S34 (system turn) so the prompt STRUCTURE matches the earlier
# stages exactly — the user turn stays the dataset question. A neutral fallback question is used only
# when a scene has no dataset question (e.g. scene_facts eval paths).
Q_FALLBACK = "Describe the fault structure and what it implies for the subsurface."




def degenerate(t):
    return any(ord(ch) > 127 for ch in t)


# Strip base-model artifacts that make FAITHFUL reasoning fail the filter (the 1.5B habitually wraps
# numbers in formatting tags and leaks template placeholders): non-skeleton tags (<mathematical_expression>,
# <inline>, …) → keep inner text; bracketed placeholders ([dip], [center_0], …) → drop. The skeleton
# tags (evidence/think/answer/seg) are preserved, so genuinely broken structure still fails.
_STRAY_TAG = re.compile(r"</?(?!(?:evidence|think|answer|seg)\b)[a-zA-Z][^>]*>", re.I)
_PLACEHOLDER = re.compile(r"\[[^\]\n]*\]")


def _clean(ch):
    return _PLACEHOLDER.sub("", _STRAY_TAG.sub("", ch))


_TAG = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)")
_OK_TAGS = {"evidence", "think", "answer", "seg"}


def stray_tags(t):
    """True if the chain has any tag outside the expected skeleton — catches ASCII garbage
    (<solution>, <br>, <Evaluation>, </p>) that the non-ASCII degeneracy check misses."""
    return any(m.lower() not in _OK_TAGS for m in _TAG.findall(t))





def grounded(t, f):
    """Cites ANY real measured value — fault dip/throw OR closure area (multi-object, not fault-only)."""
    vals = [f"{round(float(x['dip']), 1):g}" for x in f.get("faults", [])]
    vals += [f"{round(float(x['throw'])):g}" for x in f.get("faults", []) if x.get("throw") is not None]
    vals += [f"{round(float(c['area_pct'])):g}" for c in f.get("closures", []) if c.get("area_pct") is not None]
    return any(v in t for v in vals)



_EV = re.compile(r"<evidence>(.*?)</evidence>", re.S)


@torch.no_grad()
def _model_evidence(nar, facts, question, instruction=INSTRUCTION_ROLE):
    """The model's OWN raw-narrative evidence — what Stage 2/3 was trained to emit (copy 1.00), tags/
    answer stripped — so the reasoning is seeded on evidence the model actually produces. Greedy so
    the seed evidence is stable across the retry loop. Same prompt as the earlier stages."""
    out = nar.generate(facts, question=question, instruction=instruction, max_new_tokens=120)
    m = _EV.search(out)
    return m.group(1) if m else out


@torch.no_grad()
def _sample(nar, facts, question, evidence=None, instruction=INSTRUCTION_ROLE, max_new=160):
    """Generate the IN-ORDER structure: prefill '<evidence>{raw grounded narration} <SEG></evidence>
    \\n<think>' then sample the reasoning that closes </think> and reaches <answer>. The prompt is
    IDENTICAL to the earlier stages — fact_ft word+index facts in the system turn, the dataset question
    in the user turn (the reasoning role lives in the instruction) — so nothing about stage-2's evidence
    copy is disturbed. evidence = the DATASET raw narration (train) or the model's own (inference).
    max_new sized to reach past </think> into <answer> when answer-correctness is checked."""
    nar.eval_mode()
    if evidence is None:
        evidence = _model_evidence(nar, facts, question, instruction)
    prompt = nar.build_prompt(nar.fact_ft(facts), instruction, question=question)
    prefix = f"<evidence> {evidence} <SEG> </evidence>\n<think>"
    full = torch.cat([prompt, nar._emb_text(prefix)], 0)
    g = nar.dec.generate(inputs_embeds=full.unsqueeze(0), max_new_tokens=max_new, do_sample=True,
                         temperature=0.85, top_p=0.92, repetition_penalty=1.3,
                         pad_token_id=nar.tok.eos_token_id)
    return prefix + nar.tok.decode(g[0], skip_special_tokens=True).strip()


def well_formed(ch, facts):
    return ("</think>" in ch and "<answer>" in ch and ch, facts and grounded(ch, facts))


@torch.no_grad()
def reason(nar, facts, question=None, instruction=INSTRUCTION_ROLE, tries=6):
    """INFERENCE reasoning. Structured generation (grounded evidence → <think> → answer) with rejection
    sampling: retry until the chain passes the filter. Used as a serving fallback; the finalized model
    (train_reasoning) closes the tags in a single greedy pass. Returns the first passing chain."""
    q = question or Q_FALLBACK
    ev = _model_evidence(nar, facts, q, instruction)   # greedy → stable; re-sample only the <think>
    last = ""
    for _ in range(tries):
        ch = _sample(nar, facts, q, evidence=ev, instruction=instruction)
        last = ch
        if well_formed(ch, facts):
            return ch
    return last


_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def _gen_think(nar, facts, evidence, question, answer=None, tries=4, instruction=INSTRUCTION_ROLE):
    """STaR rationale generation with OUR faithfulness filter. Returns (think, rationalized) or
    (None, False).

    Filter is the FULL chain (well_formed): this is the LAST stage, so we keep only samples where the
    model produced the COMPLETE structure — </think> AND <answer> — and it respects the facts + is
    grounded. The generation budget (_sample max_new_tokens=320) is sized to cover reasoning + </think>
    + <answer>…</answer> so a good chain isn't false-failed for running out of tokens. Only the <think>
    body is kept for the target (the DATASET answer is stapled on at training); requiring the full chain
    here just validates the model can close all the tags end to end.

    Pass 1 — SPONTANEOUS: sample under the standard prompt; keep if the full chain is well-formed.
    Pass 2 — RATIONALIZATION (STaR step 3): if pass 1 fails, give the DATASET ANSWER as a generation
    hint (in the instruction only) and reason toward it — kept ONLY if it STILL passes well_formed, so
    the hint can't smuggle in confabulated numbers. The hint is generation-only; training uses the
    un-hinted INSTRUCTION_S34, so the model learns to reason unaided (matching STaR)."""
    def _ok(ch):
        ch = _clean(ch)                                  # strip stray tags + placeholder leaks
        i = ch.find("</think>")                          # keep ONLY up to the think — the 1.5B degenerates
        if i != -1:                                      # into GitHub-HTML garbage after it; answer = dataset
            ch = ch[:i + len("</think>")]
        if _DBG["printed"] < 4:                          # eyeball the first few (cleaned) thinks
            print(f"[dbg-chain] {ch[:420].replace(chr(10), ' ')}", flush=True)
            _DBG["printed"] += 1
        _DBG["att"] = _DBG.get("att", 0) + 1             # periodic fail-mode tally (so we see it live)
        if _DBG["att"] % 12 == 0:
            print(f"[dbg-fail] {_DBG['att']} attempts · fails {dict(_FAIL)}", flush=True)
        m = _THINK.search(ch)
        if not m:
            _FAIL["no_close_think"] += 1; return None     # ran out of tokens before </think>
        if not (ch, facts):
            _FAIL["respects"] += 1; return None           # think stated a number/mode not in the facts
        if not grounded(ch, facts):
            _FAIL["grounded"] += 1; return None           # think cited no real dip
        return m.group(1).strip()
    for _ in range(tries):
        th = _ok(_sample(nar, facts, question, evidence=evidence, instruction=instruction))
        if th:
            return th, False
    if answer:
        hinted = (instruction + f" The established conclusion is: {answer} "
                  "Reason step by step from the measured evidence toward that conclusion.")
        for _ in range(tries):
            th = _ok(_sample(nar, facts, question, evidence=evidence, instruction=hinted))
            if th:
                return th, True
    return None, False


_ANS = re.compile(r"<answer>(.*?)</answer>", re.S)




_MEAS = re.compile(r"\d+\.?\d*\s*(?:degrees?|°|deg|ms|percent|%|km|\bm)\b", re.I)
_DECIMAL = re.compile(r"\b\d+\.\d+\b")
_HEDGE = re.compile(r"\b(?:at|of)?\s*about\b|\bapproximately\b|\baround\b|\bnear(?:ly)?\b", re.I)


def _qualify(think):
    """Make the model's OWN <think> QUALITATIVE by stripping the measurement numbers (they're redundant —
    already protected in the evidence copy — and the ONLY thing the think can confabulate). This is NOT a
    constructed template: the model's free prose, clause structure, and geological reasoning are kept
    verbatim; only the digits+units and their 'about/approximately' hedges are removed. What survives is
    pure qualitative reasoning (steeper/gentler, intersecting, pattern, implication) that cannot state a
    wrong value. Bare small integers (Fault 1, counts) are kept as ordinals/anchors."""
    t = _clean(think)                                  # drop stray tags + [placeholder] leaks first
    t = _MEAS.sub("", t)                               # "79.9 degrees" / "227 ms" / "12 percent" → ""
    t = _DECIMAL.sub("", t)                            # bare decimals "79.9" → "" (dips/throws)
    t = _HEDGE.sub("", t)                              # dangling "at about" / "approximately"
    t = re.sub(r"\s+([.,;:])", r"\1", t)               # tidy space-before-punct
    t = re.sub(r"([.,;:])\s*(?=[.,;:])", "", t)        # collapse orphaned punctuation runs
    t = re.sub(r"\bdips?\b(?=\s*[.,;])", "is present", t, flags=re.I)  # "Fault 1 dips ." → readable
    t = re.sub(r"\s{2,}", " ", t).strip(" ,;:.")
    return t


_QUAL_WORDS = re.compile(r"steep|gentl|intersect|dip|fault|horst|graben|ramp|branch|stair|thrust|"
                         r"pattern|structure|imply|implies|suggest|closure|trap|displac", re.I)


def _star_qual_target(nar, facts, evidence, answer, question, K):
    """Sample up to K chains at the CURRENT reason adapter; keep the FIRST whose ANSWER passes the
    content-only anchor and whose reasoning stayed INSIDE <think> tags. Returns the qualified (number-
    stripped) think, or None. The generation prompt is INSTRUCTION_QUAL (qualitative role) so there is no
    train/serve seam. Evidence is the fixed DATASET narration (clean prefill) — only think+answer sampled."""
    for _ in range(K):
        ch = _sample(nar, facts, question, evidence=evidence,
                     instruction=INSTRUCTION_ROLE, max_new=180)
        am = _ANS.search(ch)
        if not am:   # ANSWER-ONLY validation (content)
            continue
        tm = _THINK.search(ch)
        if not tm:                                             # reason MUST be bounded in <think></think>
            continue
        qth = _qualify(tm.group(1))
        if len(qth) < 25 or degenerate(qth) or stray_tags(qth) or not _QUAL_WORDS.search(qth):
            continue                                           # skip empty / drifted / non-reasoning thinks
        return qth
    return None


def train_reasoning_star(nar, scenes, rows_by_img, rounds=2, K=3, epochs=3, lr=2e-5, rows_per=2,
                         save="stage34_narrator.pt"):
    """STAGE 3b — MULTI-ROUND STaR, ANSWER-ANCHORED + QUALITATIVE (user's design, the anti-collapse variant).
    Each round: sample K chains per row at the CURRENT reason adapter → KEEP by the ANSWER anchor ONLY
    (content-faithful answer + reason bounded in <think>) → strip the think to QUALITATIVE (confab-proof) →
    SFT the reason adapter on {qual think} </think> <answer>{DATASET answer}</answer> (evidence masked +
    fuse frozen ⇒ copy protected) → REPEAT. Watch kept-count climb round-over-round (rising = bootstrapping;
    flat = still the 1.5B ceiling). Differs from main17: the filter is answer-CONTENT-only (no well_formed
    FORM gate — that's what starved 5→1), and the think is number-stripped so it can't confabulate."""
    from hybrid.model.narrator import row_facts
    rng = random.Random(0)
    pool = []
    for s in scenes:
        for r in rows_by_img.get(s["img"], [])[:rows_per]:
            f = row_facts(r)
            if not f["faults"] or len(f["faults"]) > MAX_OBJ:
                continue
            ev = r.get("evidence") or ""
            ans = r.get("answer") or ""
            q = r.get("question") or Q_FALLBACK
            if ev and ans:
                pool.append((f, ev, ans, q))
    print(f"[star-qual] pool {len(pool)} rows · rounds {rounds} · K {K}", flush=True)
    if len(pool) < 4:
        print("[star-qual] pool too small — abort", flush=True); return
    for rd in range(rounds):
        nar.set_stage("s4"); nar.eval_mode()
        data = []
        for (f, ev, ans, q) in pool:
            qth = _star_qual_target(nar, f, ev, ans, q, K)
            if qth is None:
                continue
            prefix = " ".join(f"<evidence> {ev} <SEG> </evidence>\n<think>".split())
            completion = " ".join(f"{qth} </think> <answer> {ans} </answer>".split())
            data.append((f, prefix, completion, q))
        print(f"[star-qual round {rd}] kept {len(data)}/{len(pool)}", flush=True)
        if len(data) < 4:
            print(f"[star-qual round {rd}] too few kept — stop", flush=True); break
        nar.set_stage("s4"); nar.train_mode()
        opt = torch.optim.AdamW(nar.trainable_params(), lr=lr)
        for ep in range(epochs):
            rng.shuffle(data); tot = 0.0
            for f, prefix, completion, q in data:
                opt.zero_grad()                          # supervise think+close+answer; INSTRUCTION_QUAL = gen prompt
                loss = nar.completion_loss(f, prefix, completion, question=q, instruction=INSTRUCTION_ROLE)
                loss.backward(); opt.step(); tot += loss.item()
            print(f"[star-qual round {rd}] ep {ep} loss {tot/len(data):.3f}", flush=True)
    nar.eval_mode()
    save_narrator(nar, save)
    print(f"[star-qual] saved {save}", flush=True)


def train_reasoning(nar, scenes, rows_by_img, epochs=8, lr=2e-5, rows_per=3,
                    save="stage34_narrator.pt"):
    """STAGE 3b — ANSWER-SUPERVISED reasoning (user's design). NO filter, NO reasoning-text target (which
    is what made every STaR variant confabulate/degrade). For each row, ONCE: generate a FREE <think>
    anchored on the dataset answer (rationalization — so it bridges to it), keep it UNFILTERED. Then
    train with loss ONLY on the <answer>: the answer's gradient propagates BACK through the (masked,
    in-sequence) <think> tokens, so the reason adapter is shaped to reason TOWARD the answer WITHOUT the
    reasoning being pinned to a text. The answer loss is computed on the CLEAN prompt (no answer injected)
    so the answer is a real target (not a copy) and the gradient actually reaches the reasoning. Evidence
    masked + fuse frozen ⇒ copy protected. EVERY row trains ⇒ no yield collapse / tiny-set overfit.
    Trains the reason adapter (s4)."""
    from hybrid.model.narrator import row_facts, INSTRUCTION_ROLE
    rng = random.Random(0)
    nar.set_stage("s4"); nar.eval_mode()
    data = []
    for s in scenes:
        for r in rows_by_img.get(s["img"], [])[:rows_per]:
            f = row_facts(r)
            if not f["faults"] or len(f["faults"]) > MAX_OBJ:
                continue
            ev = r.get("evidence") or ""
            ans = r.get("answer") or ""
            q = r.get("question") or Q_FALLBACK
            # Encourage the reasoning PROCESS toward the answer (not just hand it over — that made the
            # think empty by short-circuiting). Ask it to EXPLAIN how the evidence leads to the answer.
            hinted = (INSTRUCTION_ROLE + f" The established conclusion is: {ans} Explain step by step, "
                      "inside <think> tags, how the measured evidence leads to that conclusion.")
            ch = _clean(_sample(nar, f, q, evidence=ev, instruction=hinted, max_new=180))
            i = ch.find("</think>")
            m = _THINK.search(ch[:i + len("</think>")] if i != -1 else ch)
            think = m.group(1).strip() if m else ""       # FREE reasoning, unfiltered (may be terse)
            prefix = " ".join(f"<evidence> {ev} <SEG> </evidence>\n<think> {think}".split())
            completion = " ".join(f"</think> <answer> {ans} </answer>".split())   # 3b learns to CLOSE + answer
            data.append((f, prefix, completion, q))
    print(f"[reason-train] {len(data)} answer-supervised targets (no filter, free reasoning)", flush=True)
    if len(data) < 4:
        print("[reason-train] too few — abort", flush=True); return
    nar.train_mode()
    opt = torch.optim.AdamW(nar.trainable_params(), lr=lr)
    for ep in range(epochs):
        rng.shuffle(data); tot = 0.0
        for f, prefix, completion, q in data:
            opt.zero_grad()                              # CLEAN instruction (no answer) → answer is a real target
            loss = nar.completion_loss(f, prefix, completion, question=q, instruction=INSTRUCTION_ROLE)
            loss.backward(); opt.step(); tot += loss.item()
        print(f"[reason-train] ep {ep} loss {tot/len(data):.3f}", flush=True)
    nar.eval_mode()
    save_narrator(nar, save)
    print(f"[reason-train] saved {save}", flush=True)
