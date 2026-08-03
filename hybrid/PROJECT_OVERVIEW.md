# Hybrid Grounded Seismic VLM — Project Overview

*Last updated 2026-08-02. Architecture is settled; the open work is data, not design.*

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
 seismic image ──native-aspect TILING (encoder_tiling: TILE 512 / PATCH 16, never pad)──▶
                        FROZEN SFM encoder (seismic ViT, dim768/depth12/patch16, grid 32)
                        [falls back to NCS-2d-base when no SFM checkpoint]
                                                            │
                                                            ▼
 INSTANCE READER (reader.py — DETR set prediction, class-driven)
   · N_QUERIES=48 learned slots + Hungarian matching + ∅ (no over-detection train/infer mismatch)
   · Mask2Former MASKED ATTENTION in the decoder (query attends only to its own occupancy)
   · measures class-appropriate attrs · dip from SPATIAL footprint (never a pooled scalar)
   · emits per-object hidden h_i · pixel decoder → per-instance mask head
                                                            │
                                          ┌─────────────────┴─────────────────┐
                                          ▼                                   ▼
                              ONE derived head (query-conditioned)     pixel features
                                 · SECTION scope (ctx = pool):         (_mask_features)
                                     intersect · mode · nclosure · salt        │
                                 · OBJECT scope (ctx = h_i):                   │
                                     closure fluid · intersects_*              │
                                          │                                    │
                                          ▼                                    │
 ┌─ NON-DIFFERENTIABLE BRIDGE (vision MEASURES/DERIVES → LM COPIES) ──────────┐│
 │  digit tokens (fact_ft: word+index markers) — MEASURED count/dip/throw/    ││
 │    area/bbox/center + SECTION-DERIVED    → LM copies EXACT                 ││
 │  object-derived WORDS — fluid_i gas · intersects_fault_i yes               ││
 │  <feature>_i — gated soft token from h_i  ⚠️ CLOSED, see §8                ││
 └────────────────────────────────────────────────────────────────────────────┘│
                                          ▼                                    │
 Qwen2.5-1.5B decoder (4-bit QLoRA) · stacked LoRA (freeze ladder)              │
   ├─ geology   (FROZEN)  — CoT thinking capability                            │
   ├─ grounding (s2)      — evidence COPY (facts → raw evidence text)          │
   └─ fuse      (s3)      — the ANSWER FOLD (geology + grounding frozen)        │
                                          │         <SEG> hidden ──────────────┘
                                          ▼                    ▼
 <evidence>…<SEG></evidence> <think>…</think> <answer>…</answer>   SegMaskHead → per-object mask
```

**Two mask paths exist and are measured against each other** (identical metric + population):
the **reader** DETR mask head (`tf_masks`) and the **referring** path (LM `<SEG>` hidden →
`SegMaskHead` → query over the reader's pixel features, `stages/seg_mask.py`). The referring
path wins (0.104 vs 0.062 held-out), but see §8 — they share the pixel substrate, so combining
them is a proven dead end.

**Components**
- **Frozen vision encoder** — **SFM-Base-512** by default (`model/sfm_encoder.py`, reads
  dim/depth/patch/in_chans/img_size *from the checkpoint*; supports FlexiViT `PATCH` resampling
  and `dynamic_img_size`), NCS-v1-2d-base as fallback. Never trains (`TRAINABLE_BLOCKS=0`).
  **One resolver** (`stage2_reader._build_encoder`) so no caller can silently build the wrong
  front-end; **one tiling resolver** (`data/loader.encoder_tiling`) so loader and reader can
  never disagree.
- **Instance reader** (`model/reader.py`) — **DETR set prediction**: fixed learned queries,
  Hungarian matching to GT, ∅ for unmatched. This replaced an autoregressive/emit-until-stop
  reader that had a train/inference over-detection mismatch. Class-driven: one detector over
  object types, measuring class-appropriate attributes. Emits `h_i`; per-instance mask head
  over a pixel-decoder trunk.
- **The attribute registry** (`model/registry.py`) — two tiers, so head count is FIXED.
  **Tier-1 primitives** (dip/throw/area/count) have reader heads. **Tier-2 derived** flow
  through **ONE query-conditioned head**; **section** scope (ctx = pool) and **object** scope
  (ctx = `h_i`) differ only in the context fed. Adding a derived attribute is a registry row,
  never a new head. Categorical attrs emit a **word**, scalars emit **digits** — same copy seam.
- **Non-differentiable digit-token bridge** (`fact_ft`) — facts injected as **word+index**
  tokens (`dip_0 82.6`, `fluid_1 gas`). No gradient crosses the seam, so the LM can only *copy*
  what vision measured → structurally cannot confabulate a figure. Replaced slot-regression
  heads (which collapsed to a constant prior).
- **Stacked-LoRA decoder** — Qwen2.5-1.5B (4-bit) with a **freeze ladder**: geology (frozen
  CoT) + grounding (s2 copy) + fuse (s3 answer fold).
- **Mask decoder** — from-scratch, over the reader's pixel features. Chosen over SAM because
  seismic faults are ~1-px curvilinear structures SAM's blob prior can't trace (confirmed:
  SAM2/DINOv2 localizers failed on thin+sparse line-faults).

## 4. Training curriculum

Two **knowledge** stages (frozen after) + a **fuse combiner** + real-field + referring-seg.
No RL/STaR.

| stage | trains | objective |
|-------|--------|-----------|
| **1. Geology CoT** | geology LoRA (Unsloth SFT, once; cached) | `<evidence><think><answer>` skeleton + geological reasoning — frozen thereafter |
| **2a. Reader** | instance reader + derived head (frozen encoder) | image → per-object facts (measure + derive) + masks |
| **2b. Grounding** | grounding LoRA (s2) | facts → raw-evidence COPY; target **ends at `</evidence>`** (un-suppress fix); 1:1 `row_facts` injection |
| **3. Fuse fold** | fuse LoRA (s3; geology+grounding FROZEN) | grounded `<answer>` as the completion after a **full, MASKED** `<think>`; `</think>` in the masked prefix, only `<answer>` gets loss |
| **4. Real-field** | reader **real adapter** (all else FROZEN) | adapt vision to real seismic via **adapter isolation** (§6) |
| **5. Referring seg** | `SegMaskHead` only (LM + reader FROZEN) | LM `<SEG>` hidden → per-instance mask over reader pixel features |
| ~~RL / STaR~~ | — | dropped (7 STaR variants failed; the fold replaced it) |

**The Stage-3 fold trick** — `</think>` in the masked prefix + supervise only `<answer>` does
three things at once: (1) **un-suppresses the think**, (2) gives the answer a trained home →
**no truncation**, (3) **protects the copy** (grounding frozen). Verified every run by
`copy-before == copy-after` instrumentation in `run_train.py`.

**Inference = stage-switch:** evidence at **s2** (grounding, feature off → clean copy) →
think+answer at **s3** (fuse). Two passes, one model. Generation budgets: evidence 320,
think+answer 512 tokens (raised 2026-08-02 — the old 120/320 truncated multi-object answers
mid-enumeration; reader-facts copy rose 0.70 → 0.84 from the bump alone).

**Entry points (no argparse — env knobs, fixed config):**
`run_train.py` (main curriculum) · `run_cracks.py` (`DATASET=cracks|thebe` real-field runner) ·
`run_eval.py` · `eval/components.py` (decoupled component tests) · `stages/seg_mask.py`.

## 5. Evaluation — decoupled component tests

`hybrid/eval/components.py` grades **each stage of the copy chain independently**, so a low
number is unambiguous. Env-switchable: `CKPT=…`, `SCENES=…`, `REAL=1`.

- **COPY-GT** — inject **GT** facts → exact reproduction per attribute. *Isolates the copy
  mechanism, reader excluded.*
- **COPY-pipeline** — inject **reader** facts (deployment). Gap vs COPY-GT = reader's contribution.
- **READER-mechanism** — count MAE · class · **mask dice** (`oracle=True` teacher-forced =
  mask decoder in isolation; `oracle=False` = deployment, misses score 0).
- **READER-attrs** — dip / throw / area MAE.
- **Fold-eval** — `present · clean · grounded · think` + **copy-before == copy-after**.
- **Referring-seg** — `seg_mask.eval_seg_dice`, deliberately the **same** `field_dice` over the
  **same** population as `mask_dice(oracle=True)` so the two mask paths are directly comparable.

Split is **group-wise by image** (no row-level leakage). Real datasets use a **contiguous**
split (adjacent crosslines are near-duplicates; a random split would leak).

> ⚠️ **Always evaluate UNCAPPED.** `SCENE_CAP` < dataset size with a cached `reader.pt` puts
> training scenes into the "held-out" set: the same reader scores **0.58** capped vs **0.06**
> uncapped. Capped numbers are contaminated, not good news.

## 6. Real-field transfer (adapter isolation)

The wall is **vision** (real seismic looks different), not language — so real-field is a
**vision-only** stage with the whole LM frozen.

- **Format unification** — every real dataset is converted to the same CSV contract as
  synthetic (`image · mask · regions`) so the identical loader/stages/eval consume both:
  - **Smeaheia** (`data/smeaheia/`) — SEG-Y, 3D fault-stick projection → mask + dip (RANSAC) +
    throw (horizon offset). Small (144 faults) but the **only source with throw**.
  - **CRACKS** (`data/cracks/`) — 397 sections / 7,648 fault instances. Masks only (dip is
    degenerate — the annotations are strokes, not traces).
  - **Thebe** (`data/thebe/`) — largest public real fault-segmentation set (1803 crosslines,
    expert pixel labels), auto-downloads by chunk from Harvard Dataverse, streamed so RAM never
    holds more than one sub-volume. Meaningful dip (transposed so faults are near-vertical);
    no throw.
  - Unlabeled volumes for future SSL: **f3**, **penobscot** (downloaded); poseidon/volve scripts.
- **Adapter isolation** (`reader.add_real_adapter` + `stages/finetune_vision.finetune_real`) —
  FREEZE the entire synthetic reader and add a zero-init **residual real adapter** on the grid
  features (starts as identity). Only the adapter trains. Avoids catastrophic forgetting on
  sparse real GT. Result on Smeaheia: **class 29% → 100%, dip → 20°** — detection/values
  transfer well; **mask stayed 0.07** (why the bigger mask datasets exist).

## 7. Key contributions (for the paper)

1. **Digit-token copy bridge** — the LM *copies* measured numbers across a non-differentiable
   seam instead of a head *regressing* them: faithful, any attribute, number lives in language.
2. **The fuse fold** — one placement that simultaneously un-suppresses reasoning, gives the
   answer a truncation-free home, and freezes the grounded copy.
3. **Two-tier attribute registry** — one query-conditioned derived head for all second-order
   attributes, section- or object-scoped; adding an attribute never adds a head.
4. **Stacked-adapter freeze ladder** + **adapter isolation for real transfer** — parameter-
   isolated continual learning end to end.
5. **Stage-switch inference** — decouples clean number-copy (s2) from reasoning (s3).
6. **~6 GB-GPU-feasible** grounded VLM.

## 8. Status — what is settled, and what the measurements say

**SETTLED / VALIDATED (architecture is not the open question):**
frozen SFM encoder + native tiling · DETR reader with Hungarian matching · two-tier registry ·
digit copy bridge · geology/grounding/fuse ladder · stage-switch inference · the fold (copy
protected every run) · adapter isolation · referring-seg path · uncapped/contiguous eval
discipline.

**Measured (2026-08-02, uncapped synthetic held-out unless noted):**

| quantity | value |
|---|---|
| copy — GT facts / reader facts | **0.86 / 0.84** |
| fold-eval present · clean · think | **1.00 · 1.00 · 1.00** |
| reader mask dice (oracle) | **0.06 – 0.11** (run-to-run variance is large) |
| referring `<SEG>` mask dice | **0.104** (vs reader head 0.062) |
| **oracle-query ceiling** (best mask ANY query can paint from the frozen substrate) | **0.418** |
| reader class acc · count MAE · dip MAE | **48–58% · 1.2–1.3 · 5.5–9.8°** |
| zero-shot on real Thebe (n=2033) | class 23% · dip 6.9° · **dice 0.03** |

**CLOSED — do not re-try (each has a measurement):**
- **`<feature>` as an LM channel** — inert for numbers *and* qualitative text (ON == OFF
  byte-identical; gate ≈ 0). The `value_head` proves it's encodable, but the LM won't read
  across the non-differentiable seam. Numbers stay **copied**.
- **Fusing the two mask heads** — oracle *selector* `max(seg, reader)` = 0.111 vs 0.104 today
  (+0.007). Errors correlate +0.64; **56/100 faults are co-missed**. They share the pixel
  substrate, so fusion recombines queries but can't manufacture evidence.
- **Naive `PATCH=8`** — oracle ceiling *drops* 0.418 → 0.269. The backbone was pretrained at
  patch-16; finer sampling is off-distribution. Only viable folded into SSL at patch-8.
- **clDice on the seg-head** — 0.103 (baseline) vs 0.102 (clDice). A topology loss refines
  roughly-right masks; ours are mostly *absent*, so there is nothing to connect.
- **RL / STaR**, **copy-lock / constrained decoding**, **stage-3 co-refine** — all previously
  closed.

**THE ONE OPEN PROBLEM — vision generalization (data, not design).**
The oracle-query probe is decisive: **0% of held-out faults are unrepresented** in the frozen
features (every one reaches ≥0.1 when a query is fit to it), yet a head that must *generalize*
reaches only 0.104. That is a **4× generalization gap**, plus a **substrate gap** above 0.418
(only 8% of faults can be made crisp). Loss tweaks and resolution tricks cannot close either —
both were tested and failed. The levers are **real data volume** (Thebe/CRACKS for masks,
Smeaheia for values) and **SSL on real seismic** (f3/penobscot) for the substrate.

## 9. Limitations & roadmap

- **Vision generalization** — the binding constraint (above). Mask 0.06–0.11 held-out.
- **Multi-object enumeration** — grounded/enum improve with budget (grounded 0.2 → 0.5–0.67
  after the rebuild + token bump) but remain capped; end-to-end is further hurt by reader
  **over-detection** (count MAE 1.3; ~3 false objects on fault-free real panels).
- **Evidence correspondence** — copy only reproduces attributes the evidence *states*.
- **Reader class accuracy** — 88% on seen data vs ~49% held-out: the same generalization
  disease as the mask, not a separate bug.

**Roadmap:** real-data mask training (Thebe → CRACKS, contiguous split, uncapped eval) →
multi-source combination → SSL on f3/penobscot for the substrate → re-audit enumeration on a
reader that generalizes. Experiments live in gitignored `hybrid/experiments/` and import the
frozen `hybrid.*` modules; main code changes stay surgical.
