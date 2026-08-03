"""The facts-bridge narrator — the main model's language half.

Detector facts (count + per-fault dips) become DIGIT-TOKEN embeddings, prepended
to the prompt; the LM copies the exact numbers. The decoder is the stacked-adapter
FUSE: stage 2 trains the grounding adapter to copy facts into evidence text
(grounding the shared latent), then it freezes; stage 3 trains the fuse combiner
to align the detector facts into narration on top of both frozen adapters.

Proven (copy path): held-out copy 1.00, faithfulness swap 16/16.
"""

import os

import torch
import torch.nn as nn
from hybrid.model.geology import load_geology_adapter, GEOLOGY_CFG
from hybrid.model.decoder import GroundedDecoder
from hybrid.model.registry import (derived_facts, object_derived_facts, CLASS_ID,
                                    SECTION_DERIVED)

device = torch.device("cuda")
K_COUNT, K_DIP, K_EVID, K_NCLOSURE, K_AREA, K_BBOX, K_THROW = 0, 1, 2, 3, 4, 5, 6
MAX_OBJ = 3           # cap objects per scene injected/stated (bounds LM sequence → GPU memory)
# A COPY model must be allowed to repeat. 1.3 penalised exactly the tokens the target repeats most —
# a coordinate like [91,400.5] is restated up to 4x in the evidence, so its digits carried the largest
# accumulated penalty and got suppressed, leaving the "the fault at ___ appears" gap. Digits are copied
# from injected facts, so repetition here is CORRECT, not degeneration.
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", 1.0))

# Unified chatml prompt: facts live in the SYSTEM turn (vision supplies them — a real
# ANSWER-SUPERVISED reasoning (user's design): geology role-play prompt — read everything injected,
# reason FREELY, then answer. The reasoning is NOT given a text target; only the answer is supervised, so
# the reasoning is free (no confabulation pressure) and shaped only by the answer's backprop.
INSTRUCTION_ROLE = ("You are a geophysicist analysing a seismic section. Read the measured evidence and "
                    "the reported facts, reason step by step inside <think> tags about the fault "
                    "structure and what it implies for the subsurface, then answer the question. "
                    "Reference specific objects and state the measured values directly; do not add facts "
                    "unsupported by the evidence.")


def faults_of(scene_objs):
    """GT per-fault dips (cls==1 with a dip present), in region order."""
    return [float(o["meas"][0]) for o in scene_objs
            if int(o["cls"]) == 1 and float(o["mmask"][0]) > 0]


def objects_of(scene_objs):
    """GT valid objects of ANY class (fault/closure/salt/onlap) — MULTI-CLASS scene selection. NOT
    measurement-gated: an object with only class+mask (e.g. onlap, which carries no tier-1 measure)
    still trains the reader's detection/class/mask heads; tier-1/derived losses are present-gated
    downstream, and the LM copy stages self-select scenes that actually carry measured facts. (Was
    `cls in (1,2) and mmask[0]>0` — a DIP-fault gate that dropped 406 scenes to 51.)"""
    return [o for o in scene_objs if int(o["cls"]) in CLASS_ID.values()]


def region_metadata(scene):
    """GT facts: per-fault {dip, bbox, center, throw?}, per-closure {area_pct, bbox, center}.
    bbox/center are UN-NORMALIZED to the image's PIXEL scale (x·W, y·H) so the injected digits
    match the dataset evidence's own pixel coordinates (the head stays normalized; only injection
    un-normalizes). Round-trip is exact: build_scenes stored x/W,y/H → here ·W,·H → original pixels."""
    H, W = scene["hw"]
    faults, closures, salts, onlaps = [], [], [], []
    for o in scene["objs"]:
        x1, y1, x2, y2 = o["bbox"]                        # normalized 0-1
        cx, cy = o["center"]
        bbox = [int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)]   # pixels
        center = [int(cx * W), int(cy * H)]
        cls = int(o["cls"])
        if cls == 1 and float(o["mmask"][0]) > 0:
            f = {"dip": float(o["meas"][0]), "bbox": bbox, "center": center}
            if float(o["mmask"][1]) > 0:
                f["throw"] = float(o["meas"][1])
            faults.append(f)
        elif cls in (2, 3, 4) and float(o["mmask"][2]) > 0:
            obj = {"area_pct": float(o["meas"][2]), "bbox": bbox, "center": center}
            if cls == 2:
                obj["derive"] = object_derived_facts(o.get("derive"))   # closure fluid/intersects words
                closures.append(obj)
            else:
                (salts if cls == 3 else onlaps).append(obj)
    return {"faults": faults, "closures": closures, "salts": salts, "onlaps": onlaps,
            "derived": derived_facts(scene.get("derived"))}


def row_region_metadata(row):
    """Facts from a SINGLE dataset ROW's regions — 1:1 with the row's question-scoped evidence.
    THE FIX for the union mismatch: inject exactly what THIS row's evidence states, so the LM learns
    to enumerate ALL injected facts (the 2-fault rows teach multi-object). bbox/center are the
    region's PIXEL coords, dataset-native (no normalization round-trip). Used for narrator training;
    the reader still trains on the whole-image union (it must detect all)."""
    faults, closures, salts, onlaps = [], [], [], []
    der = {marker: None for _i, _k, marker, _kd, _l in SECTION_DERIVED}
    for reg in (row.get("regions") or []):
        cid = CLASS_ID.get(reg.get("object_type"))
        if cid is None:
            continue
        b = reg.get("bbox") or [0, 0, 0, 0]
        c = reg.get("center") or [int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)]
        bbox = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]; center = [int(c[0]), int(c[1])]
        v = reg.get("values") or {}; mv = v.get("measure") or {}; dv = v.get("derive") or {}
        for _i, key, marker, kind, labels in SECTION_DERIVED:      # section-scoped (registry-driven)
            if der[marker] is not None or dv.get(key) is None:
                continue
            if kind == "cat":
                if dv[key] in labels:
                    der[marker] = labels.index(dv[key])
            elif kind == "bool":
                der[marker] = bool(dv[key])
            else:
                der[marker] = float(dv[key])
        if cid == 1 and mv.get("dip_deg") is not None:
            f = {"dip": float(mv["dip_deg"]), "bbox": bbox, "center": center}
            if mv.get("throw") is not None:
                f["throw"] = float(mv["throw"])
            faults.append(f)
        elif cid in (2, 3, 4) and mv.get("area_pct") is not None:
            obj = {"area_pct": float(mv["area_pct"]), "bbox": bbox, "center": center}
            if cid == 2:
                obj["derive"] = object_derived_facts(dv)          # closure fluid/intersects words
                closures.append(obj)
            else:
                (salts if cid == 3 else onlaps).append(obj)
    return {"faults": faults, "closures": closures, "salts": salts, "onlaps": onlaps,
            "derived": derived_facts(der)}


def _region_objs(facts):
    """Ordered (class_word, obj) across ALL buckets, each capped at MAX_OBJ — the single object
    sequence the injection iterates (fault → closure → salt → onlap). feats align to this order."""
    return ([("fault", o) for o in facts.get("faults", [])[:MAX_OBJ]]
            + [("closure", o) for o in facts.get("closures", [])[:MAX_OBJ]]
            + [("salt", o) for o in facts.get("salts", [])[:MAX_OBJ]]
            + [("onlap", o) for o in facts.get("onlaps", [])[:MAX_OBJ]])


def region_markers(facts):
    """Detector facts -> role-tagged kv: count · per-fault dip/throw · per-closure area."""
    faults = facts.get("faults", []); closures = facts.get("closures", [])
    kv = [(K_COUNT, f"{len(faults)}")]
    for f in faults:
        kv.append((K_DIP, f"{round(float(f['dip']), 1):g}"))
        if "throw" in f and f["throw"] is not None:
            kv.append((K_THROW, f"{round(float(f['throw']))}"))
    if closures:
        kv.append((K_NCLOSURE, f"{len(closures)}"))
        for c in closures:
            kv.append((K_AREA, f"{round(float(c['area_pct']))}"))
    for c in facts.get("salts", []) + facts.get("onlaps", []):   # area-measured objects (same role)
        kv.append((K_AREA, f"{round(float(c['area_pct']))}"))
    return kv






def caption_target(evidence, answer):
    """Stage-3 target: RAW evidence (tags stripped) + <SEG> + the TAG-WRAPPED dataset answer. No
    <think> (the combined stage fills reasoning between </evidence> and <answer>). The answer is
    wrapped in <answer> … </answer> so the tag skeleton matches the combined stage + geology + the
    well_formed check. All numbers backed; the LM learns placement + phrasing."""
    return " ".join(
        f"{evidence}{answer}".split())


class Captioner:
    """Stacked-adapter decoder (geology + grounding + fuse) + digit-token bridge.

    Stage flow: `set_stage('s2')` + `ground_loss` train the grounding adapter on
    evidence-copy; `set_stage('s3')` + `loss` train the fuse combiner on the
    detector-facts narration with grounding+geology frozen."""
    def __init__(self, lora_r=8, lora_alpha=16, prompt="Describe the faults: ", reader_d=256):
        adapter = load_geology_adapter(GEOLOGY_CFG)
        self.model = GroundedDecoder(adapter_dir=adapter, lora_r=lora_r,
                                     lora_alpha=lora_alpha).to(device)
        self.dec, self.tok = self.model.decoder, self.model.tokenizer
        self.emb = self.dec.get_input_embeddings()
        # <feature>_i path: project the reader's per-object h_i into LM space, GATED (init 0 → the
        # model STARTS as the raw-evidence narrator and opens the gate only if the feature earns it).
        # LayerNorm keeps the soft token in-distribution. h_i arrives DETACHED → gradient trains
        # feat_proj + feat_gate (+ fuse/LoRA), never the reader (the seam holds). use_feature toggles A/B.
        lm_dim = self.emb.embedding_dim
        self.feat_proj = nn.Sequential(nn.Linear(reader_d, lm_dim), nn.LayerNorm(lm_dim)).to(device)
        self.feat_gate = torch.zeros(1, device=device, requires_grad=True)
        self.use_feature = False
        # MODALITY DROPOUT knob (feature activation): when True, the injected tier-1 VALUES are blanked
        # ("dip_0 ?") while the markers + <feature>_i soft token stay — so the answer cannot be served by
        # copying the digit and must lean on the visual feature. Only helps when the answer needs
        # qualitative info the digit can't give; default False (else it corrupts the numeric copy).
        self.mask_digits = False

    def set_stage(self, stage):
        self.model.set_stage(stage)

    def trainable_params(self):
        ps = [q for q in self.dec.parameters() if q.requires_grad]
        if self.use_feature:                              # feature A/B: also train projection + gate (low-lr)
            ps = ps + list(self.feat_proj.parameters()) + [self.feat_gate]
        return ps

    def _emb_text(self, s):
        ids = self.tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        return self.emb(ids).squeeze(0)

    def build_prefix(self, ft, instruction, question=None):
        """Chatml prompt with the measured facts (ft) spliced into the SYSTEM turn — vision
        supplies facts, the user only asks. question=None -> no user turn (S2 grounding). Ends
        at '<|im_start|>assistant\\n' so geology's trained <think> trigger is present."""
        pre = self._emb_text(f"<|im_start|>system\n{instruction}\nMeasured facts: ")
        if question:
            post = self._emb_text(f"<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n"
                                  f"<|im_start|>assistant\n")
        else:
            post = self._emb_text("<|im_end|>\n<|im_start|>assistant\n")
        return torch.cat([pre, ft, post], 0)

    def _lm_loss(self, prompt_emb, target_str):
        tgt = self.tok(target_str + "<|im_end|>", add_special_tokens=False,
                       return_tensors="pt").input_ids.to(device)
        inp = torch.cat([prompt_emb, self.emb(tgt).squeeze(0)], 0).unsqueeze(0)
        labels = torch.cat([torch.full((prompt_emb.shape[0],), -100, device=device),
                            tgt.squeeze(0)], 0).unsqueeze(0)                    # prompt masked
        return self.dec(inputs_embeds=inp, labels=labels).loss

    def _obj_markers(self, i, cls, o):
        """WORD+INDEX markers for one object — every field carries its index _i so the LM binds all
        _i fields to object i by NAME. When mask_digits (the masked-injection experiment): keep ONLY
        class_i + center_i (identity + location), and BLANK bbox + every measured/derived value to "?"
        — so extent/size/magnitude must come from the <feature>_i soft token (forces the gate open)."""
        b = o.get("bbox") or [0, 0, 0, 0]
        c = o.get("center") or [int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)]
        blank = self.mask_digits                                     # masked injection: hide extent + values
        bbox = "? ? ? ?" if blank else f"{int(b[0])} {int(b[1])} {int(b[2])} {int(b[3])}"
        p = [f"class_{i} {cls}",
             f"bbox_{i} {bbox}",
             f"center_{i} {int(c[0])} {int(c[1])}"]
        if cls == "fault":
            dip = "?" if blank else f"{round(float(o['dip']), 1):g}"
            p.append(f"dip_{i} {dip}")
            if o.get("throw") is not None:
                throw = "?" if blank else round(float(o["throw"]))
                p.append(f"throw_{i} {throw}")
        else:
            area = "?" if blank else round(float(o["area_pct"]))
            p.append(f"area_{i} {area}")
        for marker, val in (o.get("derive") or {}).items():          # object-scoped derived, index-bound
            p.append(f"{marker}_{i} {'?' if blank else val}")
        return p

    def _derived_tail(self, facts):
        """SECTION-scoped derived markers (scene-level; same copy rail). nclosure is already stated as
        the closure count, so it's not repeated here."""
        der = facts.get("derived") or {}
        out = []
        for marker in ("intersect", "mode", "nonlap", "salt"):
            if der.get(marker) is not None:
                out.append(f"{marker} {der[marker]}")
        return out

    def soft_prompt(self, facts):
        """WORD+INDEX marker injection across ALL object classes (fault/closure/salt/onlap):
          "count 2  class_0 fault  bbox_0 …  dip_0 62  class_1 closure  area_1 18  fluid_1 gas  …".
        The word carries pretrained meaning (transfer); bbox_i is the object's spatial identity;
        VALUES are measured/derived, copied from vision."""
        objs = _region_objs(facts)
        parts = [f"count {len(facts.get('faults', [])[:MAX_OBJ])}"]
        nclo = len(facts.get("closures", [])[:MAX_OBJ])
        if nclo:
            parts.append(f"nclosure {nclo}")
        for i, (cls, o) in enumerate(objs):
            parts += self._obj_markers(i, cls, o)
        parts += self._derived_tail(facts)
        return self._emb_text("  ".join(parts))

    def soft_prompt_feat(self, facts, feats):
        """Like soft_prompt, but INTERLEAVES a soft <feature>_i token (gated projection of the reader's
        h_i) right after each object's markers — object-anchored, index-bound. feats = list of h_i
        (reader_d,) aligned with _region_objs order (faults, closures, salts, onlaps) or None."""
        objs = _region_objs(facts)
        nclo = len(facts.get("closures", [])[:MAX_OBJ])
        head = f"count {len(facts.get('faults', [])[:MAX_OBJ])}" + (f"  nclosure {nclo}" if nclo else "")
        segs = [self._emb_text(head + "  ")]
        for i, (cls, o) in enumerate(objs):
            segs.append(self._emb_text("  ".join(self._obj_markers(i, cls, o)) + "  "))
            hi = feats[i] if (feats and i < len(feats)) else None
            if hi is not None and self.use_feature:                  # gated soft token (gate init 0)
                segs.append((self.feat_gate * self.feat_proj(hi)).unsqueeze(0))
                segs.append(self._emb_text("  "))
        tail = self._derived_tail(facts)
        if tail:
            segs.append(self._emb_text("  ".join(tail)))
        return torch.cat(segs, 0)

    def _ft(self, facts, feats):
        """Pick the injection: feature-interleaved when use_feature + feats given, else plain markers."""
        if feats is not None and self.use_feature:
            return self.soft_prompt_feat(facts, feats)
        return self.soft_prompt(facts)

    def ground_loss(self, facts, target, question=None, instruction=None, feats=None):
        """Inject the named facts into the SYSTEM turn; supervise the assistant target. feats (per-object
        h_i) enable the <feature>_i tokens when use_feature. S2 grounding: instruction=INSTRUCTION_S2,
        question=None. S3 QA: dataset instruction + question."""
        return self._lm_loss(
            self.build_prefix(self._ft(facts, feats), instruction, question), target)

    def completion_loss(self, facts, prefix, completion, question=None, instruction=None, feats=None):
        """Supervise ONLY the completion after a GIVEN prefix. The prefix — e.g.
        '<evidence>{grounded} <SEG> </evidence>\\n<think>' — is folded into the (loss-masked) prompt as
        given context, so gradient never touches the evidence the model already produces (copy 1.00);
        only the think body + closing tags + answer are learned. This is the joint Stage 3+4 objective:
        Stage 2/3 opens evidence + <think>, the joint stage completes and closes the remaining tags.
        Mirrors inference exactly (prefill prefix → generate) so there's no train/serve seam."""
        prompt = self.build_prefix(self._ft(facts, feats), instruction, question)
        prompt = torch.cat([prompt, self._emb_text(prefix)], 0)
        return self._lm_loss(prompt, completion)

    @torch.no_grad()
    def caption(self, facts, question=None, instruction=None, max_new_tokens=160, feats=None):
        """Inference: inject the named (detected/GT) facts into the system turn, ask the question
        in the user turn, generate the grounded chain freely — the LM copies each number into its
        NAMED object phrase. feats add the <feature>_i soft tokens when use_feature."""
        prompt = self.build_prefix(self._ft(facts, feats), instruction, question)
        g = self.dec.generate(inputs_embeds=prompt.unsqueeze(0), max_new_tokens=max_new_tokens,
                              do_sample=False, repetition_penalty=REPETITION_PENALTY,
                              pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(g[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate(self, facts, max_new_tokens=160, question=None, instruction=None, feats=None):
        """Grounded narration/answer from the named-facts bridge, optionally question-conditioned."""
        return self.caption(facts, question=question, instruction=instruction,
                            max_new_tokens=max_new_tokens, feats=feats)

    def train_mode(self):
        self.dec.train()

    def eval_mode(self):
        self.dec.eval()
