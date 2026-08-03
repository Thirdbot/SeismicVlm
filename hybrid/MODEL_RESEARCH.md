# Measured, Not Hallucinated: A Grounded Seismic VLM with a Non-Differentiable Number Seam

**Research write-up.** Companion documents: `PROJECT_OVERVIEW.md` (system narrative),
`ARCHITECTURE.md` (shape/loss specification), `RESULTS_full_run.md` (single-run log).
This file is the *research* framing: claims, positioning, experimental design, evidence,
negative results, and threats to validity. Status as of **2026-08-02**.

---

## Abstract

Vision–language models applied to scientific imagery are asked to do two incompatible things:
produce fluent domain reasoning, and report *quantities* that must be correct. Generic VLMs
emit numbers as free text and therefore hallucinate them; detectors and segmenters produce
quantities but cannot reason; grounded LMMs link phrases to regions but have no mechanism for
typed domain measurements. We present a hybrid architecture for seismic interpretation in which
**vision measures every quantity and the language model only copies it**, across a deliberately
**non-differentiable seam**. Measurements are injected as word+index *digit tokens*; because no
gradient crosses the seam, the LM is structurally unable to invent a figure — faithfulness
becomes an architectural property rather than a training objective. Around this seam we add a
two-tier attribute registry (adding an attribute never adds a head), a stacked-adapter freeze
ladder with a novel *answer fold* that un-suppresses reasoning while freezing the numeric copy,
and adapter-isolated real-field transfer. The whole system trains and runs on a **5.67 GB
consumer GPU**.

We report a working language/grounding stack (copy fidelity 0.86 with ground-truth facts, 0.84
with reader facts; answer well-formedness 1.00) and a vision stack that does **not** yet
generalize (held-out fault-mask Dice 0.06–0.11). Our main empirical contribution is a
**diagnostic that localizes where such a system is actually failing**: an *oracle-query probe*
that fits a mask query directly to each held-out instance over the frozen pixel features,
establishing the best mask any query could paint. It shows that 0% of held-out faults are
absent from the features while a generalizing head reaches only 0.104 — decomposing the
shortfall into a **4× generalization gap** and a separate **substrate gap**. Guided by this
probe we report five *negative* results with the measurements that closed them, including that
fusing two mask heads is worthless when they share a substrate (+0.007 over the better head).

---

## 1. Problem and motivation

Seismic interpretation requires simultaneous qualitative and quantitative output. A useful
statement is *"a normal fault dipping ≈ 76.3° with ≈ 215 ms of throw offsets the closure to its
left, which is gas-charged"* — it interleaves geological reasoning with figures that must be
measured, not guessed. This is the general shape of scientific VLM tasks (radiology, materials,
remote sensing): **the prose is worthless if the numbers are wrong**.

Three failure modes motivate the design:

1. **Free-text numbers hallucinate.** An LM trained to emit "76.3" as text has no mechanism
   tying that token to pixel evidence.
2. **Regression heads collapse.** Our earlier slot-regression design predicted attributes with
   dedicated heads; they converged to a constant prior and ignored the image (dip-swap test
   0/8 — the prediction did not follow the fact). Continuous-valued regression from pooled
   features was the failure, and digit tokenization fixed it (8/8).
3. **Per-attribute heads do not scale.** Every new measurable attribute would add a head, a
   loss term, and a supervision requirement.

## 2. Positioning w.r.t. related work

- **Grounded LMMs** — LISA (Lai et al., 2023) introduced a `<SEG>` token whose hidden state
  prompts a mask decoder; GLaMM (Rasheed et al., 2024) generates grounded conversations;
  Kosmos-2 (Peng et al., 2023) grounds phrases to boxes. We inherit the `<SEG>` mechanism but
  these systems ground *phrases to regions*; none carries **typed numeric measurements**. Our
  contribution is orthogonal and composable: the copy seam is what they lack.
- **Detection/segmentation** — DETR (Carion et al., 2020) set prediction and Mask2Former
  (Cheng et al., 2022) masked attention are used inside our reader. We adopted DETR
  specifically to remove a train/inference over-detection mismatch in an earlier
  autoregressive reader.
- **Promptable segmentation** — SAM/SAM2 (Kirillov et al., 2023) and DINOv2 (Oquab et al.,
  2023) features were evaluated and **rejected** for this domain: seismic faults are ~1-px
  curvilinear structures, and blob-prior promptable segmenters failed on thin, sparse line
  structures.
- **Seismic deep learning** — synthetic-first fault segmentation (e.g. Wu et al., 2019) is
  standard; our synthetic corpus comes from a physical forward model (Shell *synthoseis*) with
  a full label factory. Real benchmarks used here: **Thebe** (An et al., 2021,
  doi:10.7910/DVN/YBYGBK, CC-BY-4.0) and **CRACKS**.
- **Parameter-efficient continual learning** — LoRA/QLoRA (Hu et al., 2021; Dettmers et al.,
  2023) and adapter composition (Pfeiffer et al., 2021, AdapterFusion) underpin our freeze
  ladder and our real-field adapter isolation.
- **Reasoning-elicitation methods we tried and rejected** — STaR-style bootstrapping
  (Zelikman et al., 2022) and GRPO-style RL (Shao et al., 2024) both failed here (§7).

## 3. Method

Full specification in `ARCHITECTURE.md`; the research-relevant claims are these four.

### 3.1 The non-differentiable copy seam (primary contribution)

Vision measures attributes; they are serialized as **word+index digit tokens**
(`dip_0 82.6`, `throw_0 215.26`, `fluid_1 gas`) and prefixed to the LM context. The LM's task
for numbers is *transcription*, not estimation. Because the seam is non-differentiable:

- the LM **cannot** fabricate a figure that vision did not measure (structural faithfulness);
- no gradient flows back, so language loss cannot corrupt the measurement path;
- **any** attribute is expressible without a new output head — scalars become digits,
  categoricals become words, both riding one mechanism;
- attribution is testable: swapping an injected fact must swap the stated number.

This inverts the usual arrangement, in which faithfulness is *encouraged* by a loss or reward.
Here it is enforced by the absence of a path.

### 3.2 The answer fold

Naively supervising `<evidence><think><answer>` teaches the model to stop early, producing
empty thinks; supervising the whole sequence lets answer loss drift the numeric copy. The
**fold** places `</think>` inside a **masked prefix** and supervises **only** `<answer>`, with
the geology and grounding adapters frozen. One placement achieves three effects: the think is
un-suppressed (the fuse adapter's additive delta counteracts the grounding stop-bias), the
answer acquires a trained home (no truncation), and the copy is provably untouched. We verify
the last claim with `copy-before == copy-after` instrumentation on **every** run.

### 3.3 Two-tier attribute registry

Tier-1 primitives (dip, throw, area, count) have reader heads. All tier-2 derived attributes
(fault intersections, fault mode, closure fluid, adjacency) pass through **one
query-conditioned head**, where *section* scope and *object* scope differ only in the context
vector supplied (global pool vs. per-object `h_i`). Adding an attribute is a registry row.

### 3.4 Adapter-isolated real transfer

Real seismic GT is sparse. We freeze the entire synthetic reader and add a zero-initialized
residual adapter on the grid features (identity at init); only the adapter trains. This is
strictly stronger than freezing only the heads of absent classes, which still lets the shared
trunk drift.

## 4. Experimental setup

| | |
|---|---|
| Hardware | RTX 3060, **5.67 GB** VRAM · 15 GB RAM |
| LM | Qwen2.5-1.5B-Instruct, 4-bit QLoRA, stacked adapters (geology / grounding / fuse) |
| Vision | SFM-Base-512 (frozen; ViT dim 768, depth 12, patch 16, grid 32); NCS-2d fallback |
| Reader | DETR set prediction, N_QUERIES = 48, Hungarian matching + ∅, Mask2Former masked attention |
| Synthetic | synthoseis-derived, 369–406 scenes, 4 classes; group-wise split by image (0.75), seed 42 |
| Real | Thebe (5,600 panels / 8,134 instances from 2 chunks), CRACKS (397 sections / 7,648 instances), Smeaheia (144 faults, the only source with throw) |

**Metric discipline (necessary for any of the below to mean anything):**

- Mask Dice is reported **per instance**, never as an area-weighted global union — a union
  metric is far easier and is not comparable across paths. An earlier headline result of ours
  was invalidated on exactly this ground and is retracted (§7).
- The two mask paths are scored with the **same** function over the **same** population, so
  their comparison is exact.
- `oracle=True` (teacher-forced query↔GT matching) measures the mask decoder in isolation;
  `oracle=False` is deployment. Both share the GT-fault denominator so neither can be gamed by
  detecting less.
- Real datasets use a **contiguous** (not random) split: adjacent crosslines are near-duplicates
  and a random split leaks.
- **Evaluation must be uncapped.** Capping scenes while reusing a cached reader silently places
  training scenes in the "held-out" set: the identical reader scores **0.58 capped vs 0.062
  uncapped**. This is reported as a methodological warning — it is an easy way to publish a
  5–9× overstatement by accident.

## 5. Results

### 5.1 Language / grounding stack (works)

| quantity | value |
|---|---|
| Copy fidelity, GT facts injected | **0.86** |
| Copy fidelity, reader facts injected | **0.84** |
| Answer present / clean / think non-empty | **1.00 / 1.00 / 1.00** |
| Copy before vs. after fold | **identical every run** (e.g. 51/75 → 51/75) |
| Grounded answer (GT-inject, K=1 / K=2) | 0.50 / 0.67 |

The fold behaves as designed. Numbers survive the reasoning stage untouched.

**Generation budget is a confound worth reporting.** With a 120-token evidence budget, evidence
closed properly only **8%** of the time and reader-fact copy scored 0.70; raising the budget
lifted closure to 42% and copy to **0.84**, converging on the GT-inject ceiling of 0.86. A
substantial part of an apparent *model* deficiency was a *decoding* artifact — the reader
over-detects, producing more objects, which overruns a fixed budget. Multi-object enumeration
studies that do not control the token budget will mis-attribute this.

### 5.2 Vision stack (does not generalize)

| quantity | value |
|---|---|
| Reader mask Dice (oracle, uncapped held-out) | **0.06 – 0.11** |
| Referring `<SEG>` mask Dice | **0.104** (reader head: 0.062) |
| Class accuracy — seen data vs held-out | **88% → 49%** |
| Count MAE | 1.2 – 1.3 |
| Dip MAE | 5.5 – 9.8° |
| Zero-shot on real Thebe (n = 2033) | class 23%, dip 6.9°, **Dice 0.03** |

Two observations with mechanistic content: **dip transfers far better than masks** (6.9° on
real data zero-shot) because dip is read from mask *orientation*, a property that survives
domain shift even when mask quality does not; and **class accuracy collapses off-distribution
in lockstep with mask Dice**, indicating one disease (generalization) rather than two bugs.

### 5.3 The oracle-query probe (methodological contribution)

For each held-out instance we fit a query vector **directly to that instance's ground-truth
mask** over the frozen pixel features, using the same BCE+Dice objective the real head uses.
This upper-bounds any method of the form `mask = query · features`, independent of the LM,
the reader, and the training data.

| | Dice |
|---|---|
| Reader head (must generalize) | 0.062 |
| Referring `<SEG>` head (must generalize) | 0.104 |
| **Oracle query (fit to the answer)** | **0.431** (median 0.393; 0/100 below 0.1; only 8/100 above 0.7) |

The decomposition:

```
0.104  ────────────────▶  0.431  ────────────────▶  1.0
       generalization gap          substrate gap
        (≈4×; data-limited)      (thin faults weakly encoded)
```

**Every held-out fault is represented in the frozen features** (nothing falls below 0.1 under
the oracle), so the head's failure is *not* missing evidence — it is inability to learn a query
that finds evidence which is demonstrably present. Simultaneously, even with the answer in
hand, the mean is only 0.431 and just 8% of instances can be made crisp: the substrate itself
limits attainable sharpness. We recommend this probe generally: it separates "the feature
extractor cannot see it" from "the head cannot learn to find it", which are routinely conflated
and imply opposite remedies.

## 6. Ablations

| ablation | result | interpretation |
|---|---|---|
| Referring `<SEG>` head vs. reader mask head | 0.104 vs 0.062 | LM-conditioned query beats the visual query |
| `<feature>` soft token ON vs. OFF | byte-identical outputs; gate ≈ 0 | the LM does not read the soft channel (§7) |
| Evidence budget 120 → 480 | copy 0.70 → 0.84; closure 8% → 42% | decoding artifact, not model deficiency |
| Encoder patch 16 → 8 (FlexiViT resample) | oracle ceiling 0.418 → **0.269** | resolution without matched pretraining *hurts* |
| clDice added to seg-head loss | 0.103 → 0.102 | topology loss needs masks that already exist |
| Mask-head fusion (oracle selector) | 0.104 → 0.111 (+0.007) | shared substrate ⇒ correlated errors |

## 7. Negative results (reported deliberately)

Negative results with measurements are the most transferable output of this project so far.

1. **A soft visual token is not a usable LM channel.** `<feature>_i`, a gated projection of the
   reader's per-object hidden state, is **inert**: with it ON versus OFF the generated text is
   byte-identical for qualitative content, and for numbers the model confabulates identically.
   A probing `value_head` recovers the quantity from the same hidden state, proving the
   information *is* encodable — the LM simply will not read across a non-differentiable seam
   it was never trained to consult. **Implication:** the copy seam works *because* it is
   symbolic; re-introducing a continuous side-channel does not extend it.
2. **Fusing two mask heads is futile when they share a substrate.** Both heads compute
   `query · pixel_features` and differ only in query origin. An *oracle selector* choosing the
   better head per instance yields 0.111 vs 0.104 (+0.007); a union ensemble 0.107; error
   correlation +0.64; **56/100 instances are missed by both**. Fusion recombines queries but
   cannot manufacture evidence.
3. **Naive resolution increase harms.** Resampling the patch embedding 16→8 (FlexiViT-style)
   drops the oracle ceiling from 0.418 to 0.269: the backbone was pretrained at patch-16 and
   the finer token grid is off-distribution. Resolution is only a lever *jointly* with
   pretraining at that resolution.
4. **clDice does not help masks that are absent.** Adding a soft-skeleton connectivity loss
   changed nothing (0.103 → 0.102) and its own loss term barely descended. Topology losses
   refine roughly-correct masks; at Dice ≈ 0.1 there is no structure to keep connected.
5. **RL and STaR failed for faithfulness.** GRPO with a faithfulness reward and seven STaR
   variants degraded or failed to improve grounding; the architectural fold replaced them.
   Consistent with our thesis: make faithfulness structural, not rewarded.

**Retraction.** An earlier internal result ("the referring path generalizes 4× better",
0.24 vs 0.06) was an artifact of comparing a **union/semantic** Dice against a **per-instance**
Dice. It is void. The corrected, matched comparison is 0.104 vs 0.062. We report this because
the failure mode — a metric mismatch flattering a favored method — is easy to commit and hard
to detect after the fact.

## 8. Threats to validity

- **Small evaluation sets.** Synthetic held-out yields n = 100 fault instances; in-distribution
  fold-eval can be n = 3–16. Real Thebe evaluation is far better (n = 2033) and should be
  preferred for any published claim.
- **Run-to-run variance.** Two identical 80-epoch reader retrains produced mask Dice **0.07**
  and **0.11**. Any claimed improvement below ~0.05 absolute on this metric is noise. Results
  here are single-seed; multi-seed reporting is required before publication.
- **Split contamination.** Documented in §4; discovered mid-study and corrected. All numbers in
  §5 are uncapped.
- **Synthetic→real gap.** Synthetic scenes come from a forward model whose noise statistics are
  not those of field data; zero-shot real Dice of 0.03 quantifies the gap.
- **Oracle-probe interpretation.** The probe upper-bounds *linear query* decoding over frozen
  features. A different decoder family (e.g. a trained convolutional refiner) could exceed it;
  the probe bounds the architecture class in use, not all possible methods.
- **Single-domain evidence.** All conclusions are seismic-specific; the copy-seam claim should
  be replicated in another measurement-bearing domain before it is stated generally.

## 9. Current position and future work

The architecture is settled; **the open problem is data, and we have measurements that say so
rather than intuitions.** The generalization gap (0.104 → 0.431) is addressable with data
volume; the substrate gap (0.431 → 1.0) requires better features.

1. **Real-data mask training at volume** — Thebe (8,134 expert-labeled instances) and CRACKS
   (7,648), contiguous splits, uncapped evaluation. Directly attacks the generalization gap.
   *(In progress at time of writing; zero-shot baseline Dice 0.03 established.)*
2. **Multi-source training** — combining Thebe + CRACKS guards against locking onto a single
   annotation style, a distinct risk from data volume alone.
3. **Self-supervised pretraining on unlabeled field volumes** (F3, Penobscot) — the principled
   attack on the substrate gap, and the only route by which higher resolution (patch-8) becomes
   viable, per §7.3.
4. **Enumeration under a generalizing reader** — multi-object answer completeness is currently
   confounded by reader over-detection (~3 false objects on fault-free real panels); it should
   be re-audited once detection is sound.
5. **Domain replication of the copy seam** — the strongest path to a general claim.

## 10. Reproducibility

- Entry points are fixed-config (no argparse), environment-variable driven:
  `run_train.py` (curriculum), `run_cracks.py` (`DATASET=cracks|thebe` real-field),
  `run_eval.py`, `eval/components.py` (decoupled component tests), `stages/seg_mask.py`.
- Real datasets auto-download and convert to a single CSV contract (`image · mask · regions`)
  shared with synthetic, so identical loaders/stages/metrics consume both.
- Diagnostic probes (oracle-query ceiling, fusion headroom, copy/budget decomposition,
  split-contamination check) live in `hybrid/experiments/`, import the frozen `hybrid.*`
  modules, and modify nothing in the main path.
- Seeds fixed (42) for splits; multi-seed training runs are **not** yet done (§8).

---

*Honest summary: the language half of the thesis is demonstrated — numbers are measured, copied,
and provably preserved through reasoning. The vision half is not yet demonstrated at useful
accuracy, and we have localized precisely why. The negative results and the oracle-query probe
are, at present, the most valuable exportable findings.*
