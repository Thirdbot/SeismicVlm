# Hybrid Grounded Seismic VLM — Project Overview

## 1. Why this needs to exist

Seismic interpretation is a dual task. A geoscientist looking at a seismic section needs
two things at once:

1. **Natural-language geological reasoning** — *"this is a normal fault produced by
   extensional stress; the closure to its left is gas-charged and may leak across the
   juxtaposition."*
2. **Precise, quantitative localization** — *where* the objects are (bbox, pixel mask) and
   *how much* (dip ≈ 65°, throw ≈ 68 ms, closure area, fluid type).

No existing model class does both. Generic VLMs are fluent but hallucinate measurements as
free text. Detection/segmentation models localize but can't reason. Grounded LMMs (GLaMM,
LISA, Kosmos-2) link phrases to masks but are built for natural images and have **no
mechanism for typed domain measurements** — they structurally cannot output "throw = 68 ms."

This project builds a **hybrid grounded VLM** for the seismic domain under a **~6 GB
consumer-GPU** budget. The one idea that makes it work: **vision MEASURES the numbers, the
language model COPIES them** across a deliberate **non-differentiable seam**. Every figure the
model states is a real measurement, not a hallucination, and the number lives *inside the
language* where the model can reason about it.

## 2. Problem statement

**Input:** a seismic image (2-D section) + a natural-language question.
**Output:** tagged narration plus per-object masks:

```
<evidence> per-object measured evidence, each region ending in <SEG> </evidence>
<think>    grounded geological reasoning </think>
<answer>   final grounded answer </answer>
```

It is **multi-object / multi-class**. Object types come from the synthoseis label factory:
**fault** (dip, throw), **closure** (area + derived fluid & intersects), **salt** (area),
**onlap** (area). Numbers are grounded by **copying** measured/derived values (not regressed);
masks by a content-prompted `<SEG>` decoder.

## 3. Architecture

```
 seismic image  ──tile + stitch (mult-of-32 pad)──▶  FROZEN NCS encoder (seismic 2-D ViT)
                                                            │
                                                            ▼
 INSTANCE READER (reader.py — autoregressive, query-free, class-driven)
   · detects objects of ANY class · measures class-appropriate attrs
   · dip from SPATIAL footprint (2nd moment, never a pooled scalar) · throw/area pooled
   · emits per-object hidden state h_i  · per-instance mask head
                                                            │
                                          ┌─────────────────┴─────────────────┐
                                          ▼                                   ▼
                              ONE derived head (query-conditioned)     stitched spatial map
                                 · SECTION scope (ctx = pool):              (for masks)
                                     intersect · mode · nclosure · salt
                                 · OBJECT scope (ctx = h_i):
                                     closure fluid · intersects_*
                                          │
                                          ▼
 ┌─ NON-DIFFERENTIABLE BRIDGE (vision MEASURES/DERIVES → LM COPIES) ──────────────────┐
 │  digit tokens (fact_ft: word+index markers) — MEASURED count/dip/throw/area/bbox/  │
 │    center + SECTION-DERIVED intersect/mode/nclosure/salt        → LM copies EXACT   │
 │  object-derived WORDS — fluid_i gas · intersects_fault_i yes (adjacent to object i) │
 │  <feature>_i — gated soft token from h_i (qualitative texture, index-bound)         │
 └────────────────────────────────────────────────────────────────────────────────────┘
                                          ▼
 Qwen2.5-1.5B decoder (4-bit QLoRA) · stacked LoRA (freeze ladder)
   ├─ geology   (FROZEN)  — CoT thinking capability
   ├─ grounding (s2)      — evidence COPY (facts → raw evidence text)
   └─ fuse      (s3)      — the ANSWER FOLD (geology + grounding frozen)
                                          │                    <SEG> hidden + stitched map
                                          ▼                              ▼
 <evidence>…<SEG></evidence> <think>…</think> <answer>…</answer>  → mask decoder → per-object mask
```

**Components**
- **Frozen NCS encoder** (`NorskRegnesentralSTI/NCS-v1-2d-base`) — domain-pretrained seismic
  ViT. Never trains. Feeds the reader (facts/masks) and the stitched map (mask decoder).
- **Instance reader** (`reader.py`) — replaces the old dense-seg + slot-regression front-end.
  Autoregressive / query-free / cap-free (emit-until-stop), **class-driven**: one detector
  over object types, measuring class-appropriate attributes (fault → dip from spatial
  footprint + throw; closure/salt/onlap → area). Emits per-object `h_i` (source of `<feature>`
  and the mask prompt). Frozen/GT-trained; the LM only *reads* its facts.
- **The attribute registry** (`registry.py`) — two tiers, so the head count is FIXED, not
  per-attribute. **Tier-1 primitives** (dip/throw/area/count) have their own reader heads.
  **Tier-2 derived** flow through **ONE query-conditioned head** (`derived_head.py::DerivedHead`)
  conditioned on a per-attribute query embedding; the only difference between **section**
  scope (context = section pool → intersect/mode/nclosure/salt) and **object** scope (context
  = `h_i` → closure fluid/intersects) is which context is fed. Adding a derived attribute is a
  registry row, never a new head. Categorical attrs (fluid, mode) emit a **word** token,
  scalars emit **digits** — both ride the same copy seam.
- **Non-differentiable digit-token bridge** (`fact_ft`) — facts injected as **word+index**
  tokens (`dip_0 82.6`, `throw_0 …`, `fluid_1 gas`). The LM reproduces them exactly; because
  no gradient crosses the seam, it can only *copy* what vision measured/derived → structurally
  cannot confabulate a figure. This replaced the slot-regression heads (which collapsed to a
  constant prior, ignoring vision).
- **`<feature>_i`** — a gated soft token from `h_i` carrying *qualitative* texture the digits
  can't (index-bound). Currently **dormant** (gate ≈ 0); has **two** activation paths (§8).
- **Stacked-LoRA decoder** — Qwen2.5-1.5B (4-bit) with a **freeze ladder**: geology (frozen
  CoT) + grounding (s2 copy) + fuse (s3 answer fold). Each stage freezes the earlier adapters.
- **Mask decoder** (from-scratch, full-res) — a `<SEG>` hidden state cross-attends the
  stitched map → full-res mask (BCE + Dice). Over SAM because seismic faults are ~1-px
  curvilinear structures SAM's blob prior can't trace. Masks train in Stage 2 (reader),
  independent of the reasoning fold.

## 4. Training curriculum

Two **knowledge** stages (frozen after) + a **fuse combiner** + a real-field stage. No RL/STaR.

| stage | trains | objective |
|-------|--------|-----------|
| **1. Geology CoT** | geology LoRA (Unsloth SFT, once) | `<evidence><think><answer>` skeleton + geological reasoning — frozen thereafter |
| **2a. Reader** | instance reader + derived head (frozen NCS) | image → per-object facts (measure + derive) + masks |
| **2b. Grounding** | grounding LoRA (s2) | facts → raw-evidence COPY; target **ends at `</evidence>`** (un-suppress fix); 1:1 `row_facts` injection |
| **3. Fuse fold** | fuse LoRA (s3; geology+grounding FROZEN) | grounded `<answer>` as the completion after a **full, MASKED** `<think>`; `</think>` in the masked prefix, only `<answer>` gets loss |
| **4. Real-field** | reader **real adapter** (all else FROZEN) | adapt vision to real seismic via **adapter isolation** (§7) |
| ~~RL / STaR~~ | — | dropped (7 STaR variants failed; the fold replaced it) |

**The Stage-3 fold trick** — `</think>` in the masked prefix + supervise only `<answer>` does
three things at once: (1) **un-suppresses the think** (the fuse's additive delta counteracts
grounding's stop-bias), (2) gives the answer a trained home → **no truncation**, (3) **protects
the copy** (grounding frozen → evidence copy can't drift, proven copy-before==after every run).

**Inference = stage-switch:** evidence at **s2** (grounding, feature off → clean copy) →
think+answer at **s3** (fuse, digit+feature). Two passes, one model.

## 5. Evaluation — decoupled component tests

`hybrid/test/test_components.py` grades **each stage of the copy chain independently**, so a
low number is unambiguous (copy vs. reader vs. reasoning vs. correspondence). Env-switchable:
`CKPT=…`, `SCENES=…`, `REAL=1`.

- **COPY-GT** — inject **GT** facts → exact reproduction, **per attribute** (dip/throw/area).
  The pure copy test. *Isolates the copy mechanism + evidence correspondence, reader excluded.*
- **COPY-pipeline** — inject **reader** facts → reader-capped (deployment). Gap vs COPY-GT =
  the reader's contribution.
- **READER-mechanism** — count MAE · class · **mask dice**.
- **READER-attrs** — **dip / throw / area MAE** (class-driven measurement) — the test that
  caps COPY-pipeline once correspondence is fixed.
- **Fold-eval** (`train.py`) — `present · clean · grounded · think` (one greedy pass, no
  truncation) + **copy-before == copy-after** (proves the fold protects the copy).

**Diagnostic logic:** COPY-GT low → copy/correspondence; COPY-pipeline < COPY-GT → reader is
the cap; READER-attrs bad → reader can't measure it; grounded low but copy high → reasoning
confabulation; dice → segmentation quality; feature ON > OFF → the feature earns its keep.

**Key finding (correspondence gradient):** copy reproduces only what the evidence STATES —
dip (always stated) copies ~0.73, throw (~25% stated) partial, area (never stated) 0. The copy
mechanism is fine; the lever is **per-object per-attribute evidence coverage** in the dataset.

Split is **group-wise by image** (whole images held out — no row-level leakage).

## 6. Real-field transfer (adapter isolation)

The wall is **vision** (real seismic *looks* different; held-out dice ~0.2), not language.
So real-field is a **vision-only** stage with the whole LM frozen.

- **Format unification** (`data/real.py::real_scenes`) — Smeaheia windows (SEG-Y read → 3D
  fault-stick projection → mask + dip (RANSAC) + throw (horizon offset)) are emitted in the
  **same scene format** as synthetic, so **one pipeline/tester** handles both. Verified:
  `scene_facts(real_scene)` yields `{dip, throw, bbox, center}` in the standard fact form.
- **Adapter isolation** (`reader.add_real_adapter` + `train/stage_realfield.py::finetune_real`)
  — FREEZE the entire synthetic reader (trunk + all heads → every synthetic class/attribute
  preserved) and add a zero-init **residual real adapter** on the grid features (16.7k params,
  starts as identity). Only the adapter trains. This is stronger than freezing just the
  absent-class heads (which lets the shared trunk drift). Real GT is sparse (~59–143 windows),
  so isolation is what avoids overfit + catastrophic forgetting. The **before/after** flow:
  `REAL=1 test_components` → `stage_realfield` → `REAL=1 CKPT=reader_real.pt test_components`.

## 7. Key contributions (for the paper)

1. **Digit-token copy bridge** — the LM *copies* measured numbers as tokens across a
   non-differentiable seam instead of a head *regressing* them: faithful (swap follows the
   fact), any attribute (no per-attribute head), and the number lives in the language.
2. **The fuse fold** — one placement (`</think>` in the masked prefix, supervise only
   `<answer>`) that simultaneously un-suppresses reasoning, gives the answer a truncation-free
   home, and freezes the grounded copy.
3. **Two-tier attribute registry** — one query-conditioned derived head for all second-order
   attributes, **section- or object-scoped** (context = pool vs. `h_i`); adding an attribute
   never adds a head.
4. **Stacked-adapter freeze ladder** + **adapter isolation for real transfer** — parameter-
   isolated continual learning end to end (geology→grounding→fuse; synthetic reader→real adapter).
5. **Stage-switch inference** — decouples clean number-copy (s2) from reasoning (s3).
6. **~6 GB-GPU-feasible** grounded VLM.

## 8. Implementation status (live / scaffolded / design)

**Live in `hybrid/` (working, validated):** frozen NCS encoder; **fault + closure** reader;
section-derived head (intersect, mode); digit bridge (measured + section-derived); `<feature>`
**wired but dormant**; geology/grounding/fuse decoder; stage-switch; the fuse fold; multi-object
selection + metrics; the **component tester** (`hybrid/test/test_components.py`); **real-field**
end to end (loader + format unification + adapter isolation + finetune stage + `REAL=1`).

**Wired in code, pending new-dataset train (the scaffolding flip, done):** **4 classes**
(fault/closure/**salt**/**onlap**) — `CLASS_ID` merged, reader `class_head`/`obj_cls` 3→5,
`scene_to_gt`/`scenes.py`/`scene_facts`/`row_facts`/`reader_facts` route all four (area is
class-agnostic). **Object-scoped derived** — the ONE `DerivedHead` now serves BOTH scopes
(section ctx = pool, object ctx = h_i); `registry.DERIVED` unified to 9 attrs across scopes;
closures carry per-object `fluid`/`intersects_*`, supervised from `h_i` and injected as
index-bound marker words (`fluid_1 gas`, `intersects_fault_1 yes`). These change the reader
state-dict → the **old `reader.pt` no longer loads; a retrain on the new dataset is required**
(present-gating makes every new attr a no-op on the old data, so nothing breaks meanwhile).

**Design (still to build):** **feature activation** — two independent paths: the **answer path**
(a comparative/qualitative answer opens the gate via answer-loss) and the **SEG path**
(referring-seg — "segment this one" — opens it via the dense mask/Dice reward, since `<feature>`
and `<SEG>` share `h_i`). The class/object-derived scaffolding above is now wired (was here);
what remains is training it on the new dataset and confirming the derived attributes copy.

## 9. Limitations & roadmap

- **Multi-object enumeration** — multi-object scenes drop/confabulate the 2nd/3rd object's
  attribute; partly the reader under-detecting (a separate lever from the fold).
- **Evidence correspondence** — copy only reproduces stated attributes; the new dataset must
  state each attribute **per object** (dip *and* throw for faults, area/fluid/intersects for
  closures) or those values won't be copied.
- **Vision generalization** — held-out dice ~0.2; fix = real-field (adapter isolation) + the
  GN1101 3D cube for data volume (2D lines cap at ~59–143 windows).
- **`<feature>` activation** — dormant until the new dataset's answers need non-digitizable
  texture, or a referring-seg task provides the dense mask reward.

**Roadmap:** new dataset → activate the scaffolding (classes + object-derived) → retrain
(`rm reader.pt; python -m hybrid.train.train`) → test (`python -m hybrid.test.test_components`)
→ real-field before/after (adapter isolation).
