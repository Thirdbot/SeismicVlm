# MODEL_RESEARCH.md — Grounded Seismic VLM: model and experiments source document

Forensic source document for the model/experiments half of a CVPR-format paper. Every number is
exact and carries a provenance parenthetical: a `file.py:LINE`, a checkpoint, an eval script, or a
run. Unknowns are marked, never filled. All values verified against the code state at commit
`9981f21` (2026-08-03). Parameter counts were obtained by instantiating the modules and calling
`.numel()`; they are exact, not estimated.

Provenance conventions: `(reader.py:349)` = code; `(measured: param-count script)` = counted by
instantiation this session; `(run: DATASET=thebe run_cracks, 2026-08-03)` = a training/eval run;
`(experiments/oracle_ceiling.py)` = a probe script. `UNKNOWN — not set in code` and `NOT RUN` are
used verbatim.

---

## 1. Summary and budget

The model reads a 2-D seismic section and emits grounded interpretation in which **every numeric
quantity is measured by the vision stack and only transcribed by the language model**, across a
deliberately non-differentiable "digit seam". A frozen ViT encoder feeds a DETR-style instance
reader that measures per-object class, dip, throw, area and per-instance masks; these measurements
are serialized as plain-text digit tokens and prefixed to a 4-bit Qwen2.5-1.5B LM whose stacked
LoRA adapters copy them into grounded evidence, reason over them, and answer. Because no gradient
crosses the seam, the LM is structurally unable to fabricate a figure the vision stack did not
measure. The system trains and runs on a single **RTX 3060, 5.67 GB VRAM, 15 GB host RAM**
(project_current_stack.md; hardware fixed throughout).

### Parameter budget (all counts measured by instantiation this session)

| Component | Parameters | Trained in | Frozen after |
|---|---|---|---|
| SFM-Base-512 encoder (ViT-B/16) | **86,041,344** | — (never trained here) | always frozen |
| Qwen2.5-1.5B-Instruct base (4-bit nf4) | nominal **1.54 B** (4-bit packed numel 934,778,368) | — | always frozen |
| Geology LoRA (r=16) | **18,464,768** | Stage 1 | yes |
| Grounding LoRA (r=8) | **9,232,384** | Stage 2-LM | during Stage 3 |
| Fuse LoRA (r=8) | **9,232,384** | Stage 3 | — |
| Instance reader (`RegionReader`) | **9,498,510** | Stage 2-vision | during real finetune (except mask decoder) |
| `feat_proj` + `feat_gate` (feature path) | **397,824 + 1** | Stage 3 / seg, only if `use_feature` | — |
| `SegMaskHead` (referring seg) | **459,264** | referring-seg stage | — |
| Real-field adapter (net-new params) | **16,672** | real finetune | — |

Notes. The geology LoRA is r=16 (`geology.py:21`), hence ~2× the r=8 grounding/fuse adapters — the
measured 18,464,768 = 2 × 9,232,384 confirms this. A `reason` LoRA (9,232,384) is instantiated
(`decoder.py`) but is **not trained in the current curriculum** (its Stage-4 STaR/RL scheme was
abandoned, §7-abandoned); it is excluded from the current model. `add_real_adapter` reports
3,886,880 trainable, of which only **16,672** are net-new (the zero-init grid adapter); the
remaining 3,870,208 are the reader's own `mask_q` (65,792) + `mask_up` (3,804,416) mask decoder,
deliberately unfrozen for retraining (measured: param-count script; `finetune_vision.py`).

Reader internal breakdown (measured): `proj` 196,864 · `pixdec` 1,579,520 · `dec` 3,160,320 ·
`class_head` 1,285 · `foot_q` 65,792 · `occ_q` 65,792 · `measure_heads` 17,795 · `mask_q` 65,792 ·
`mask_up` 3,804,416 · `derived` 135,174 · `derived_section_ctx` 131,328 · `query` 12,288 ·
`pos` 262,144.

### Memory and wall-clock

| Quantity | Value | Provenance |
|---|---|---|
| Reader train step, peak VRAM | **0.565 GB** allocated / 0.623 GB reserved | measured this session (`torch.cuda.max_memory_allocated`, batch=1) |
| Reader train step, peak host RSS | **2.005 GB** | measured this session |
| Frozen-encoder eval, VRAM | ~0.877 GB observed | measured this session |
| Real-field finetune step, host RSS | ~2.36 GB observed | measured this session |
| LM stage (grounding/fuse) peak VRAM | **UNKNOWN — not instrumented**; runs within the 5.67 GB budget with 4-bit + gradient checkpointing | `run_train.py:124` |
| Inference peak VRAM | **UNKNOWN — not separately instrumented**; runs within 5.67 GB | — |
| Reader 80 ep / 276 scenes | ~17 min (12.6–15 s/ep) | run log this session |
| Reader 200 ep / 276 scenes | ~50 min (15 s/ep) | run log this session |
| Referring-seg 15 ep / 272 scenes | ~9–11 min (36–46 s/ep) | run log this session |
| CRACKS real finetune, 30 ep / 297 sections | ~55 min | run log this session |
| Joint real finetune, 20 ep / 4,759 panels | **~6.3 h** (≈19 min/ep) | measured (`run_joint`, 2026-08-04) |
| Thebe-only real finetune, 60 ep / 4,200 panels | **UNKNOWN — not timed** (different session) | reference_thebe_transfer_result.md |
| Stage-1 geology, 5,000 samples × 1 ep | ~1–2 h observed | run log this session |

---

## 2. Vision encoder

**SFM-Base-512** — the Seismic Foundation Model, an MAE-pretrained ViT-B loaded encoder-only
(`sfm_encoder.py`). Configuration is read *from the checkpoint* at load, not hard-coded
(`sfm_encoder.py:47-52`):

| Property | Value | Provenance |
|---|---|---|
| Architecture | ViT-B/16, encoder only (MAE decoder + mask token + head dropped) | `sfm_encoder.py:34-37` |
| Hidden dim | 768 | read from `patch_embed.proj.weight`; `vdim=768` default (`reader.py:130`) |
| Depth | 12 | `1 + max(block index)` (`sfm_encoder.py:49`) |
| Patch size | 16 | from conv-kernel shape (`sfm_encoder.py:47-48`) |
| Input channels | 1 (single-channel seismic; per-image zero-mean/unit-std) | `sfm_encoder.py:98-107` |
| Native resolution | 512 (`patch × g = 16 × 32`) | `sfm_encoder.py:50-52` |
| Output token grid | 32 × 32 at native tile | `sfm_encoder.py:80` |
| Heads | `max(1, 768//64) = 12` | `sfm_encoder.py:67` |
| Parameters | **86,041,344** | measured |
| Frozen? | **Yes**, `TRAINABLE_BLOCKS=0` (0 trainable params measured) | `run_train.py:58`, measured |
| Pretraining corpus/objective | masked-autoencoding on a large seismic image corpus (see §19; SFM paper) | — |

`dynamic_img_size=True` (`sfm_encoder.py:73`) lets the position embedding interpolate to any input
grid, so native-aspect tiles are accepted without squashing. The downstream **feature tensor
consumed by the reader** is `(B, 768, gh, gw)` with `gh = H//16, gw = W//16` (`sfm_encoder.py:118-121`).
Tiling is native-aspect: `PATCH=16, TILE=512, DILATE_R=3` (`data/loader.encoder_tiling`); tiles are
scattered onto a stitched grid consumed by the positional grid, DETR decoder, and mask decoder.

**Fallback encoder.** When the SFM checkpoint is absent, an NCS ViT is built (224 input, patch 16,
14×14 grid; `config.py:24-30`). It is a legacy fallback only; all reported numbers use SFM-512. An
audited bug (fixed) was that standalone eval scripts silently built the NCS fallback for an
SFM-trained reader — now both go through one resolver `_build_encoder` (`stage2_reader.py`).

An optional FlexiViT patch-resample (`resample_patch_embed`, `sfm_encoder.py:59-63`, env `PATCH`)
was tested and **rejected** (§13-patch8); it is not part of the current model.

---

## 3. Object reader (DETR set prediction)

`RegionReader` (`reader.py`), a set-prediction detector over the frozen feature grid.

| Property | Value | Provenance |
|---|---|---|
| Object queries `N_QUERIES` | **48** (env-overridable) | `reader.py:59` |
| Decoder layers | 3 | `reader.py:130` |
| Attention heads | 8 | `reader.py:130` |
| Model dim `d` | 256 | `reader.py:130` |
| Pixel decoder | 2-layer `TransformerEncoder`, ffn = 4·d = 1024 | `reader.py:139-140` |
| Learned positional grid | `Parameter(1, 256, 32, 32)`, bicubic-interpolated to `(fH,fW)` | `reader.py:141` |
| Query init | `randn(1, 48, 256) × 0.02` | `reader.py:152` |
| Mask2Former masked attention | implemented, **off** by default | `reader.py:153` |
| Classes | `{NO_OBJ 0, fault 1, closure 2, salt 3, onlap 4}`, `N_CLASS=5` | `reader.py:58`, `registry.py:18` |

**Hungarian matching cost** (`reader.py:349`):

$$\mathcal{C}(i,j) = -\,p_i(\text{cls}_j) \;+\; \lVert \mu_i - c_j \rVert_1$$

Class term weight 1.0 (negative softmax class probability), centroid L1 term weight 1.0. **No mask
term** in the matching cost. Unmatched queries are assigned ∅.

**No-object handling.** ∅ = class 0; the class-loss weight vector is `ones(5)` with `cw[0]=0.1`
(**eos_coef = 0.1**, `reader.py:356`), applied inside `F.cross_entropy`. Detection thresholds
*objectness* `conf = 1 − p(∅) > 0.9`, not max-class prob, so the under-trained ∅ class cannot
deflate the threshold (`reader.py:514, 528-530`).

**Reader loss** (every term with its coefficient; `reader.py:333-421`):

$$
\mathcal{L} = \underbrace{\mathrm{CE}(cls, cls^{*}; cw)}_{1.0}
+ \underbrace{5.0\,\lVert \mu_{\text{row}} - c^{*}\rVert_1}_{\text{centroid}}
+ \underbrace{\big[\mathrm{BCE}_{pw=8}(\text{foot}, g) + \mathrm{Dice}(\text{occ}, g)\big]}_{\text{footprint}}
+ \sum_{m}\underbrace{\mathrm{SmoothL1}(\text{meas}_m, g_m/s_m)}_{1.0}
+ \underbrace{\big[\mathrm{BCE}_{pw}(\text{mask}, g) + \mathrm{Dice}\big]}_{\text{mask}}
+ \underbrace{w_{cl}\,\mathrm{clDice}}_{w_{cl}=0}
+ \sum \underbrace{\mathcal{L}_{\text{derived}}}_{1.0}
$$

- Class CE weight 1.0, with `cw` above (`reader.py:357`).
- Centroid L1 weight **5.0** on matched queries (`reader.py:366-367`).
- Footprint: `BCE_with_logits(pos_weight=8.0)` + per-instance Dice, combined weight 1.0
  (`reader.py:369-373`).
- Measures (dip/throw/area): SmoothL1 on `meas/scale`, gated so only classes that declare a measure
  and whose GT is present contribute (`reader.py:376-382`).
- Mask: `BCE_with_logits` with **adaptive** `pos_weight = ((1−r)/r).clamp(1,50)`,
  `r = mask.mean().clamp(1e-4, 0.5)`; plus per-instance Dice (`reader.py:394-397`).
- clDice: weight `self.cldice_w`, **default 0.0** (`reader.py:398-399`, `156`).
- Derived (§4): section-scoped on the pooled context, object-scoped on matched `h_i`, weight 1.0
  each (`reader.py:404-420`).

**Per-instance Dice** (`_dice_loss`, `reader.py:68-76`), smoothing constant 1, mean over the K
instances — *not* an area-weighted union:

$$\mathrm{Dice} = \frac{1}{K}\sum_k \Big(1 - \frac{2\sum_p g + 1}{\sum p + \sum g + 1}\Big)$$

This per-instance form (vs a global union) was itself an audited fix; a union Dice inflates the
number and is not comparable across mask paths (§11, §13-retraction).

---

## 4. Attribute heads (two-tier registry)

Attributes are declared in a registry (`registry.py`); adding one is a row, not a head.

**Tier-1 measures** — one head per measure, shared across the classes that declare it
(`registry.py:28-35`):

| measure | kind | scale `s` | dataset key | carried by classes |
|---|---|---|---|---|
| dip | spatial (reads 8-dim stats) | 90.0 | dip_deg | fault |
| throw | pooled (reads d=256 feature) | 500.0 | throw | fault |
| area | spatial | 100.0 | area_pct | closure, salt, onlap |

Measure-head architecture: `Linear(8 if spatial else 256, 64) → GELU → Linear(64, 1)`
(`reader.py:163-165`). "spatial" heads read an **8-dim geometry-stats vector** derived from the
*supervised occupancy map* (not the pooling map): `[μrow, μcol, cov_rr, cov_cc, cov_rc, μrow−.5,
μcol−.5, occ_frac]` (`reader.py:305-306`). Dip is therefore read from mask *orientation* (the
covariance), a property that transfers across domains even when mask quality does not (§12). All
measure heads together: **17,795 params** (measured).

**Tier-2 derived** — a single query-conditioned head (`DerivedHead`, `heads.py:25-41`) serves all
9 derived attributes; scope is resolved by the context vector supplied, not by a separate head:

- `query = Embedding(9, 256)` selects the attribute; `trunk = Linear(512,256)+GELU`;
  output heads `cat = Linear(256, MAX_CAT=4)`, `scalar = Linear(256,1)`, `boolean = Linear(256,1)`
  (`heads.py`).
- **Section scope** (context = pooled section, via `derived_section_ctx = Linear(512,256)`):
  number_fault_intersections (scalar), fault_mode (cat, 4 modes), number_hc_closures (scalar),
  number_onlap_episodes (scalar), salt_inserted (bool) — `registry.py:111-121`.
- **Object scope** (context = matched query `h_i`, closures only): fluid (cat, {gas,oil,brine}),
  intersects_fault/_salt/_onlap (bool).
- `NUM_DERIVED=9`, `MAX_CAT=4`; scalar scales `{intersect:50, nclosure:10, nonlap:10}` else 1
  (`registry.py:122-126`). `DerivedHead` + `derived_section_ctx`: **135,174 + 131,328 params**
  (measured).

---

## 5. The copy seam

**Serialization.** Reader measurements are serialized to a plain-text marker string, parts joined
by two spaces, each per-object marker carrying an index `_i` (`captioner.py:225-273`). A concrete
one-fault example (bbox `[12,34,56,78]`, center `[34,56]`, dip 62, throw 100), character for
character:

```
count 1  class_0 fault  bbox_0 12 34 56 78  center_0 34 56  dip_0 62  throw_0 100
```

Closures append `area_i` plus derived markers (`fluid_1 gas`, `intersects_fault_1 …`); the section
tail appends `intersect N`, `mode WORD`, `nonlap N`, `salt yes/no`; `nclosure` follows `count`
(`captioner.py:250-268`).

**Numeric tokenization.** Values are formatted as ordinary decimal strings — dip
`f"{round(dip,1):g}"`, throw/area `round(...)`, bbox/center `int(...)` — then tokenized with the
base tokenizer and embedded (`captioner.py:142-147, 201-203, 233-245`). There is **no** learned
scalar projection; a number is literally the LM's own digit tokens. This is the design that fixed
an earlier head-regression that ignored the image (dip-swap 0/8 → 8/8 with digit tokenization;
§16 / MODEL_RESEARCH prior draft).

**Why non-differentiable, and what it buys.** The reader→format→tokenize→embed path contains a
`round`/string step and detached embeddings, so no gradient flows LM→reader. Consequences: (i) the
LM cannot fabricate a figure vision did not measure — faithfulness is structural, not rewarded;
(ii) language loss cannot corrupt the measurement path; (iii) any attribute is expressible without
a new output head. **Across the seam:** measured digits + a soft `<feature>_i` token flow *into*
the LM context; **nothing flows back** — no gradient, and (measured, §13) the LM does not read the
soft `<feature>` channel at all.

---

## 6. Language model and adapters

| Property | Value | Provenance |
|---|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct | `config.py:17` |
| Hidden size | 1536 | measured (`emb.embedding_dim`) |
| Quantization | 4-bit, `nf4`, double-quant, compute dtype bfloat16 | `decoder.py:43-45`, `config.py:18` |
| LoRA rank / alpha / dropout | r=8 / α=16 / dropout=0.1 (geology r=16) | `config.py:20-22`, `decoder.py:59`, `geology.py:21` |
| LoRA bias | none | `decoder.py:60` |
| LoRA target modules | q,k,v,o,gate,up,down `_proj` | `decoder.py:61-62` |
| Task | CAUSAL_LM | `decoder.py:60` |
| `MAX_OBJ` (objects injected/stated) | 3 | `captioner.py:23` |
| Generation budgets | evidence 320 tok · think+answer 512 tok | stage3_answer.py (§16) |

**Adapters:** `geology` (18,464,768; frozen after Stage 1), `grounding` (9,232,384),
`fuse` (9,232,384); `reason` (9,232,384) exists but is not trained in the current curriculum.

**Freeze ladder** (`decoder.py:84-96`; verified by counting trainable params per stage, measured):

| Stage | active adapters | trainable | trainable params |
|---|---|---|---|
| s2 (grounding) | geology, grounding | grounding only | 9,232,384 |
| s3 (fuse) | geology, grounding, fuse | fuse only | 9,232,384 |
| (s4 reason) | +reason | reason only | 9,232,384 — **not used** |

Base LM and all non-active adapters are frozen at every stage. `feat_proj` (Linear 256→1536 +
LayerNorm, 397,824) and `feat_gate` (scalar, init 0) are trained only when `use_feature=True`
(`captioner.py:182-185`), which is off in the current model (§13).

---

## 7. Output structure and the answer fold

**Output grammar** (skeleton tags `_OK_TAGS`, `text_metrics.py:11-13, 29`):

```
<evidence> <region> …markers… <SEG> </region> … </evidence>
<think> …reasoning… </think>
<answer> …grounded answer… </answer>
```

Special tokens: `<evidence> <think> <answer> <SEG> <region>` (skeleton) plus ChatML
`<|im_start|>{system,user,assistant} <|im_end|>` (`captioner.py:209-214`). `<feature>_i` is a soft
embedding (`feat_gate · feat_proj(h_i)`), not a vocabulary token (`captioner.py:286-288`).

**The fold (loss masking).** The fold trains only `<answer>` while `</think>` sits inside a
**masked prefix** (`stage3_answer.py:241-242`, `completion_loss` `captioner.py:307-316`):

- **Masked prefix (labels −100):** `<evidence> {ev} <SEG> </evidence>\n<think> {think} </think>`
- **Supervised completion (loss):** the grounded `answer` span only.

One placement achieves three effects with geology + grounding frozen: the think is un-suppressed
(the fuse adapter's additive delta counteracts grounding's stop bias), the answer gets a trained
home (no truncation), and the numeric copy is provably untouched.

**Invariant + check.** The invariant is *the fuse fold must not alter the copy*. It is verified on
every run by measuring copy fidelity before and after the fold: e.g. **51/75 → 51/75, identical**
(project_current_stack.md; `run_train` prints `copy BEFORE fold` / `copy AFTER fold`). This session's
rebuild reproduced identical before/after (`run: run_train, 2026-08-02`).

---

## 8. Mask decoder

**Features.** Both mask paths decode the same substrate — the reader's pixel features
`pixfeat = mask_up(pixel-decoder(memory))`, shape `(256, H', W')`, an 8× upsample built from
`Conv2d(256,256,3) → GroupNorm(8) → GELU`, then 3× `ConvTranspose2d(256,256,4,stride2) → GN → GELU`,
then `Conv2d(256,256,1)` (`reader.py:170-173`).

**Mask formulation** (`reader.py:386`): for query/hidden `q_k` (reader `mask_q(h_k)` or LM
`<SEG>` hidden), the mask logits are the inner product with the pixel features:

$$M_k(h,w) = \big\langle q_k,\; \text{pixfeat}[:,h,w]\big\rangle \quad(\texttt{einsum "kd,dhw->khw"})$$

**Training objective:** `BCE_with_logits(pos_weight = ((1−r)/r).clamp(1,50), r = mask.mean().clamp(1e-4,.5))`
+ per-instance `_dice_loss(sigmoid, g)`, each weight 1.0 (`reader.py:394-397`; identical form in the
referring `SegMaskHead`, `seg_mask.py:135-142`).

**Binarization threshold:** 0.5 on the sigmoid for Dice scoring (`geometry.field_dice`).

**Matching at inference.** Reader head: predictions from `detect()` (objectness `1−p(∅) > 0.9`) are
Hungarian-matched to GT by centroid; a GT fault with no matched detection scores 0
(`stage2_reader.mask_dice(oracle=False)`). Referring head: at training, teacher-forced by an
**ordering contract** (one `<region><SEG></region>` per object in canonical order faults→closures→
salts→onlaps, so region *i* ↔ object *i* ↔ mask *i*; `seg_mask.py:60-125`); at true inference the
model's own generated `<SEG>` anchors (token id `SEG_ANCHOR = 78759`, hidden read at anchor+1) are
used (`seg_mask.py:44, 114-124`). `SegMaskHead`: `Linear(1536,256) → GELU → Linear(256,256)`,
**459,264 params** (measured).

---

## 9. Real-field transfer

**Mechanism: adapter isolation** (`finetune_vision.finetune_real`). The entire synthetic reader is
frozen; a **zero-initialized residual grid adapter** (identity at init, **16,672 net-new params**,
measured) is added on the feature grid, and the mask decoder (`mask_q` + `mask_up`, 3,870,208
params) is unfrozen for retraining. The LM and encoder stay frozen. Trained trainable set =
**3,886,880 params** (measured). Rationale: freezing only the heads of absent classes still lets the
shared trunk drift; freezing the whole trunk and adding a zero-init residual is strictly stronger.
Verified: with the adapter off, count/class/dip return **bit-identical** to the synthetic baseline
(reference_thebe_transfer_result.md) — the finetune moved only the mask path.

**Consequence (measured, and a design constraint):** because the mask decoder is *retrained*, a
second sequential real finetune overwrites the first (Thebe mask → synthetic mask collapsed 0.106 →
0.010, and the adapter-off model does not recover it, so the damage is in the shared mask decoder,
not the adapter; reference_thebe_transfer_result.md). Multi-source real data must therefore be
trained **jointly** (concatenated CSVs), not sequentially — a data-loading change, not architecture.

**Datasets and the shared contract.** All datasets convert to one CSV contract
`image · mask · regions`, where `regions[].values = {measure:{dip_deg?,throw?,area_pct?}, derive:{}}`,
so identical loaders/stages/metrics consume synthetic and real alike.

All counts below are **read live from the CSVs on disk** (datasets sub-agent) unless marked (code).
The apparent-dip derivation is shared across the three real datasets: RANSAC line-fit → apparent dip
in degrees (`geometry.line_dip`, `inlier_dist=3.0`, `iters=300`).

| dataset | instances (disk) | attributes labeled | fault instancing → mask | split |
|---|---|---|---|---|
| Synthetic (synthoseis) | 369 images / 1,729 objects (fault 625, onlap 598, closure 351, salt 155) | dip, throw, area, all derived | forward-model label; skeletonize→dilate r=3 (~7 px) | **image-level random shuffle**, seed 42, 0.75 → 276 train / 93 held-out |
| Thebe (An et al.) | 2 chunks = 5,600 panels (2,107 fault / 3,493 bg) / 8,134 inst | fault mask only (+dip from geometry) | expert pixel labels; CC → instances | **contiguous by crossline** @ 150 → 4,200 train / 1,400 test (n=2,033 GT inst) |
| CRACKS (OLIVES) | 397 sections (all positive on disk) / 7,648 inst | fault mask only; dip present but **degenerate** (see below); confidence stored unused | connected-component strokes, MIN_AREA 12 | **contiguous by section** @ 299 → 297 train / 100 test |
| Smeaheia (Equinor/Gassnova) | **144 faults** across 84 positive panels (3,468 bg panels) | dip (144) **and throw (114)** | fault-stick projection (MATCH 150 m) → rasterized polyline | **line-level** (group by source line), seed 42 → 345 / 116 lines |

Code-vs-disk discrepancies worth flagging (datasets sub-agent, live reads): CRACKS code comment says
400 sections but disk has **397** (3 skipped for a missing expert label) and **0** came out
background despite a negative-row code path; Thebe was built at `N_CHUNKS=2` (200 crosslines), not the
full 18-chunk (1,803-crossline) volume; Smeaheia comment says ~462 lines vs **461** on disk, and throw
is present on only **114/144** instances.

**Data caveats that bound claims (reference_realfield_data_audit.md):** CRACKS dip is unusable — 95.1%
of labels fall in 75–90°, std 5.6°, and a constant predictor (MAE 4.46°) beats the model (18.56°) by
4× — because CRACKS "instances" are ~40 px annotation *strokes*, not fault traces (vertical closing
does not link them: instances/section 18.9→18.4). Use CRACKS as mask data only. Smeaheia is the only
source whose numbers are non-circular: its dip comes from the interpreted fault-stick polyline (std
26.1°, geological), and its **114 throws** (from horizon offset) are the only quantity in the project
not derived from the mask the model also predicts.

---

## 10. Training curriculum

Extracted verbatim from code (config sub-agent, full `file:line` provenance below). **No PyTorch-loop
stage sets a scheduler, warmup, or gradient clipping; all use constant LR and effective batch size 1
(one `opt.step` per scene/row, no accumulation).** Weight decay is set only in the reader. AdamW
`betas`/`eps` are never set → `UNKNOWN — not set in code` (Torch defaults apply but are not recorded).

### Stage 1 — Geology CoT (`stage1_geology.py`, `geology.py`)
Purpose: teach domain chain-of-thought + the `<think>` trigger; produces the frozen geology LoRA.
Data: `GeoGPT-Research-Project/GeoGPT-CoT-QA`, ≤5,000 rows (`geology.py:20`).

| field | value | prov |
|---|---|---|
| optimizer | `adamw_8bit` | `stage1_geology.py:99` |
| LR | 2e-5 | `geology.py:21` |
| schedule / warmup | cosine / warmup_ratio 0.03 | `stage1_geology.py:97-98` |
| weight decay | UNKNOWN — not set in code | — |
| grad clip | UNKNOWN — not set in code | — |
| epochs | 1 | `geology.py:21` |
| batch × accum | 1 × 8 (effective 8) | `geology.py:22` |
| precision | 4-bit; gradient checkpointing "unsloth" | `stage1_geology.py:63-74` |
| seed | 42 | `stage1_geology.py:31` |
| LoRA | r16, α16, dropout 0.0, targets q,k,v,o,gate,up,down | `geology.py:21`, `stage1_geology.py:69-72` |
| trainable | 18,464,768 | measured |

### Stage 2 — Reader (`stage2_reader.train_reader`)
Purpose: measure class/dip/throw/area + per-instance masks. Data: synthetic train split (276).

| field | value | prov |
|---|---|---|
| optimizer | AdamW (betas/eps UNKNOWN) | `stage2_reader.py:64` |
| LR | 1e-4 (encoder blocks, if unfrozen, 1e-5) | `stage2_reader.py:53, 62-63` |
| schedule / warmup | constant / none | — |
| weight decay | 1e-4 | `stage2_reader.py:64` |
| grad clip | UNKNOWN — not set in code | — |
| epochs | 200 (run_train) / 80 (run_cracks) | `run_train.py:55`, `run_cracks.py:35` |
| batch × accum | 1 × 1 | `stage2_reader.py:73-74` |
| precision | fp32 reader; frozen encoder eval; no AMP | `stage2_reader.py:72-74` |
| seed | data shuffle `Random(0)`; split seed 42 | `stage2_reader.py:66`, `run_train.py:54` |
| cldice_w | 1.0 (run_train) / 0.0 (run_cracks) | `run_train.py:60`, `run_cracks.py:37` |
| trainable | 9,498,510 | measured |

### Stage 2-LM — Grounding / evidence copy (`stage2_grounding.train_grounding`)
Purpose: copy measured facts into evidence; target ends at `</evidence>`.

| field | value | prov |
|---|---|---|
| optimizer / LR | AdamW / 1e-4 | `stage2_grounding.py:50` |
| schedule/warmup/wd/clip | constant / none / UNKNOWN / UNKNOWN | — |
| epochs | 20 (run_train) / fn-default 5 | `run_train.py:56` |
| batch × accum | 1 × 1 | `stage2_grounding.py:56-59` |
| precision | 4-bit LM + gradient checkpointing | `run_train.py:124-125` |
| seed | UNKNOWN — not set in code (deterministic row order) | — |
| rows/image cap | 5 | `stage2_grounding.py:38` |
| trainable | 9,232,384 (grounding LoRA) | measured |

### Stage 3 — Fuse fold / grounded answer (`stage3_answer.train_answer`)

| field | value | prov |
|---|---|---|
| optimizer / LR | AdamW / 2e-5 | `stage3_answer.py:207, 251` |
| schedule/warmup/wd/clip | constant / none / UNKNOWN / UNKNOWN | — |
| epochs | 15 (run_train) / fn-default 10 | `run_train.py:57` |
| batch × accum | 1 × 1 | `stage3_answer.py:255-262` |
| MAX_ANSWER_ROWS / rows_per | 60 / 5 | `stage3_answer.py:38, 207` |
| digit_dropout / gate_reg | 0.0 / 0.0 | `run_train.py:67-68` |
| seed | UNKNOWN — not set in code | — |
| trainable | 9,232,384 (fuse LoRA) [+397,825 feat path if use_feature] | measured |

### Referring seg (`seg_mask.train_seg_mask`)

| field | value | prov |
|---|---|---|
| optimizer / LR | AdamW / 1e-3 | `seg_mask.py:145, 155` |
| schedule/warmup/wd/clip | constant / none / UNKNOWN / UNKNOWN | — |
| epochs | 12 (run_train) / 15 (standalone) | `run_train.py:196`, `seg_mask.py:145` |
| batch × accum | 1 × 1 | `seg_mask.py:171-176` |
| use_feature | False (run_train) | `run_train.py:196` |
| seed | UNKNOWN — not set in code | — |
| trainable | 459,264 (SegMaskHead; LM + reader frozen) | measured |

### Real-field finetune (`finetune_vision.finetune_real`)

| field | value | prov |
|---|---|---|
| optimizer / LR | AdamW / 1e-4 | `finetune_vision.py:25, 34` |
| schedule/warmup/wd/clip | constant / none / UNKNOWN / UNKNOWN | — |
| epochs | 60 (run_cracks) / fn-default 20 | `run_cracks.py:36` |
| batch × accum | 1 × 1 | `finetune_vision.py:52-55` |
| seed | UNKNOWN — not set in code | — |
| trainable | 3,886,880 (16,672 net-new + mask decoder 3,870,208) | measured |

**Number of optimizer steps** per stage = epochs × (#scenes or #rows), batch 1. Exact scene/row
counts are data-dependent and not code literals; synthetic train = 276 (project_current_stack.md),
grounding/fold rows are capped (≤5/image; fold ≤60 pairs).

---

## 11. Evaluation protocol

| metric | definition | population / aggregation | denominator | capped? | split |
|---|---|---|---|---|---|
| Mask Dice (oracle) | per-instance soft Dice, GT-matched queries (`tf_masks`) | every GT fault, **per-instance mean** | all GT faults | **uncapped** | held-out |
| Mask Dice (deployment) | per-instance Dice, `detect()`-matched; misses = 0 | every GT fault | all GT faults | uncapped | held-out |
| Class accuracy | matched-pair `pred.cls == gt.cls` after Hungarian on centroids | matched pairs | matched pairs | uncapped | held-out |
| Count MAE | `|#pred − #gt|`, pos/neg panels separately | per scene | scenes | uncapped | held-out |
| Dip / throw MAE | `|pred − gt|` on matched fault pairs | matched pairs with GT value | matched pairs | uncapped | held-out |
| Copy fidelity | fraction of injected values appearing (substring) in generated evidence | injected values | injected values | — | held-out |
| Grounded / present / clean / think | answer cites a real value / emitted / no stray tags / think non-empty | in-distribution fold-eval scenes | scenes | — | held-out |

Protocol choices that would change the numbers if a comparison used different ones:

- **Per-instance vs union Dice.** We report per-instance; a union/area-weighted Dice is far easier
  and not comparable (an earlier headline was retracted on exactly this, §13).
- **Oracle vs deployment.** `oracle=True` teacher-forces query↔GT matching (mask decoder in
  isolation, detection failures invisible); `oracle=False` is deployment. Both share the GT-fault
  denominator so neither is gamed by detecting less.
- **Uncapped.** Capping scenes while reusing a cached reader leaks train into "held-out": the
  identical `reader.pt`+`mask_dice` gives **0.576 at SCENE_CAP=100 vs 0.062 uncapped** — a ~9×
  overstatement (reference_split_contamination.md). All numbers below are uncapped.
- **Contiguous real splits.** Adjacent crosslines are near-duplicates; a random split leaks.
- **Same function, same population** for both mask paths.

---

## 12. Results

Single RTX 3060; **all vision numbers are single-seed** unless noted. Measured noise floor: two
identical 80-epoch reader retrains gave mask Dice **0.07 and 0.11** (project_current_stack.md), so
any mask-Dice delta below **~0.05 absolute is indistinguishable from noise**.

### Language / grounding (works)

| quantity | value | n | provenance |
|---|---|---|---|
| Copy fidelity, GT facts | 0.86 | — | `run_train` copy-score |
| Copy fidelity, reader facts | 0.84 | reader-facts eval | `run_train` copy-score |
| Answer present / clean / think | 1.00 / 1.00 / 1.00 | fold-eval | `run_train`, 2026-08-02 |
| Copy before vs after fold | identical (51/75 → 51/75) | 75 injected vals | `run_train` (fold invariant) |
| Grounded, feature OFF | 0.54 | n=24 fold-eval | `run_train`, 2026-08-02 |
| Grounded, feature ON | 0.42 | n=24 fold-eval | `run_train`, 2026-08-02 |
| Grounded (GT-inject, K=1 / K=2) | 0.50 / 0.67 | small | project_current_stack.md |

### Vision (does not generalize on synthetic; moves on real)

| quantity | value | n | provenance |
|---|---|---|---|
| Reader mask Dice, oracle (synthetic held-out) | 0.06 – 0.11 | 100 fault inst | reference_split_contamination.md; noise floor applies |
| Reader mask Dice, deployment | 0.054 – 0.062 | 100 | reference_split_contamination.md |
| Referring `<SEG>` Dice (per-instance, matched) | 0.104 | 100 | `stages/seg_mask.py`, 2026-08-02 |
| Reader mask head, same population | 0.062 | 100 | `stages/seg_mask.py`, 2026-08-02 |
| Class accuracy, in-cap vs uncapped held-out | 88% → 49% | 51 / 153 | reference_split_contamination.md |
| Count MAE | 1.2 – 1.3 | 93 scenes | project_current_stack.md |
| Dip MAE | 5.5 – 9.8° | 4–44 | project_current_stack.md |
| **Thebe zero-shot** (real) | class 23%, dip 6.9°, Dice **0.03** | 2,033 | reference_thebe_transfer_result.md (single seed) |
| **Thebe after real-adapter finetune** | Dice **0.25**, bg false-objects 3.34 → **0.00** | 2,033 | reference_thebe_transfer_result.md (single seed) |

Two mechanistic observations: dip transfers far better than masks (6.9° zero-shot on real) because
it is read from mask *orientation*; and class accuracy collapses off-distribution in lockstep with
mask Dice (88→49%), i.e. one disease (generalization from ~276 scenes), not two bugs.

### Joint real-field benchmark (`eval/run_joint.py`, 2026-08-04, single seed)

ONE adapter-isolation finetune over the concatenated real train splits (Thebe 4,200 + CRACKS 297 +
Smeaheia 262 = 4,759 panels), 20 epochs (train loss 1.436 → 0.694, ~19 min/epoch, ≈ 6.3 h), then each
held-out split benchmarked **separately** (never pooled — the sets are incommensurable, §9). oracle =
teacher-forced `tf_masks`; deploy = `detect()`-matched with misses = 0.

| dataset | BEFORE Dice (oracle/deploy) | AFTER Dice (oracle/deploy) | class | count− (bg over-detect) | n |
|---|---|---|---|---|---|
| synthetic | 0.084 / 0.055 | 0.050 / 0.032 | 84/173 | — | 100 inst |
| **Thebe** | 0.037 / 0.018 | **0.246 / 0.217** | 1777/1777 | 3.62 → **0.00** | 2,033 inst |
| CRACKS | 0.022 / 0.001 | 0.036 / 0.035 | 1459/1459 | — | 1,679 inst |
| **Smeaheia** | 0.015 / 0.009 | **0.080 / 0.025** | 16/16 | 3.26 → **0.02** | 35 inst |

Findings: (i) **joint > sequential is demonstrated** — one shared mask decoder learned all three real
datasets in a single finetune (Thebe 0.246, Smeaheia 0.080 = 5.3×, CRACKS 0.036 = 1.6×), whereas a
sequential finetune retrains the decoder and would retain only the last dataset (reference:
reference_thebe_transfer_result.md). (ii) **No cost to the dominant set** — Thebe matched its *solo*
finetune (0.246 vs 0.25) despite two datasets added. (iii) **Over-detection eliminated on all real
sets** (count− → 0.00/0.02). (iv) **Dominance cost, quantified** — CRACKS reached only 0.036 vs its
*solo* 0.10 because it is 6% of the joint set and its stroke-style masks were swamped by Thebe's; the
fix is to oversample the small sets (§18). (v) Synthetic mask collapsed 0.084 → 0.050 as designed (the
mask decoder is retrained for real appearance; count also over-detects more, 1.23 → 3.01). Class = 100%
on the single-class real sets is degenerate (§17). Deploy Dice is only meaningful on Thebe (0.217);
CRACKS/Smeaheia deploy (0.035/0.025) shows detection has not caught up with their annotation styles.

---

## 13. Ablations

| ablation | before | after | Δ | > noise floor? | conclusion | provenance |
|---|---|---|---|---|---|---|
| Referring `<SEG>` vs reader mask head | 0.062 | 0.104 | +0.042 | yes (~0.05 is borderline; single-seed) | LM-conditioned query beats the visual query on the same substrate | seg_mask.py, 2026-08-02 |
| `<feature>` soft token ON vs OFF (grounding) | 0.54 (OFF) | 0.42 (ON) | −0.12 grounded | yes | the soft channel is inert-to-harmful; gate stays ≈0 | `run_train` fold-eval |
| Evidence budget 140 → 480 (reader facts) | copy 0.70, closed 8% | copy 0.84, closed 42% | +0.14 copy | yes | a decoding artifact, not a model deficiency | reference_copy_budget.md (n=12/28) |
| clDice added to seg-head loss (α=1, 8 iters) | 0.103 | 0.102 | −0.001 | **no** | topology loss needs masks that already exist | experiments/seg_cldice.py (n=100) |
| Mask-head fusion (oracle selector) | 0.104 | 0.111 | +0.007 | **no** | shared substrate ⇒ correlated errors (corr +0.64) | experiments/seg_headroom.py |
| Encoder patch 16 → 8 (FlexiViT resample) | oracle 0.418 | 0.269 | −0.149 | yes | resolution without matched pretraining *hurts* | experiments/oracle_ceiling.py |

**Retraction (a metric-mismatch ablation).** An earlier internal "referring path generalizes 4×
better" (0.24 vs 0.06) compared a **union/semantic** Dice against a **per-instance** Dice and is
void. The corrected, matched comparison is **0.104 vs 0.062** (§12). Reported because the failure
mode — a metric mismatch flattering a favored method — is easy to commit and hard to detect after.

---

## 14. Sweeps

| swept factor | grid | status | held fixed | retrain? | conclusion |
|---|---|---|---|---|---|
| Encoder patch size | {16, 8} | RUN | encoder ckpt, reader | reused frozen ckpt (probe) | patch-8 **hurts** (0.418→0.269); §13 |
| Evidence token budget | {140, 480} (main set 320/512) | RUN | model | no (decode only) | budget-limited, not model-limited; §13 |
| clDice weight | {0, 1.0} | RUN | seg harness | seg-head retrain | null (0.103↔0.102); §13 |
| Noise / SNR band | **NOT RUN** | — | — | — | intended grid: SNR ∈ {∞, 20, 10, 5} dB added to synthetic; retrain reader per point |
| Questions-per-scene vs scene count (fixed row budget) | **NOT RUN** | — | — | — | intended grid: at fixed ≈1,380 rows, {1,3,5} Q/scene × {276,460,690…} scenes |
| Query count `N_QUERIES` | {12, 24, 48, 96} | **RUN** | encoder, held-out 93 | reader retrain per point (80 ep) | **null** — oracle dice 0.082–0.096, within noise; 48 nominally best; see below |
| Training scene count | {69, 138, 276} | **RUN** | held-out 93 fixed | reader retrain per point (80 ep) | mask dice/class **flat within noise**; only **dip** scales with volume; see below |
| Adapter rank `r` | {4, 8, 16, 32} | **NOT RUN** | — | — | intended grid; retrains grounding+fuse per point (~1 h each), language-side only |
| Noise / SNR band | {∞,20,10,5} dB | **NOT RUN** | — | — | synthetic corpus is externally generated; cannot inject calibrated field noise without regenerating it |
| Questions-per-scene vs scene count (fixed row budget) | {1,3,5} Q/scene | **NOT RUN** | — | — | needs a grounding-data restructuring |

**Query-count sweep** (`experiments/sweep_configs.py`, 80-ep reader retrain per point, fixed 93-scene
held-out, single seed):

| N_QUERIES | oracle Dice | deploy Dice | class | count MAE | dip MAE |
|---|---|---|---|---|---|
| 12 | 0.082 | 0.063 | 95/167 (57%) | 1.10 | 11.5° |
| 24 | 0.088 | 0.029 | 49/111 (44%) | 1.46 | 4.2° |
| **48** | **0.096** | **0.081** | 93/158 (59%) | 1.15 | 6.5° |
| 96 | 0.083 | 0.064 | 91/170 (54%) | 1.55 | 9.7° |

Conclusion: **null / negative result.** Oracle Dice spans 0.082–0.096 (range 0.014, inside the ~0.05
noise floor), so query count does not move mask Dice on synthetic held-out. 48 is nominally best on
every column but not distinguishable from noise; it is retained because CRACKS has up to 42 faults per
section (`registry`/data), so N_QUERIES must exceed 42.

**Training-scene-count sweep** (subsample TRAIN, held-out 93 fixed — avoids the split-contamination
trap; 80-ep retrain per point, single seed):

| train scenes | oracle Dice | deploy Dice | class | count MAE | dip MAE |
|---|---|---|---|---|---|
| 69 | 0.111 | 0.060 | 74/136 (54%) | 1.22 | **27.0°** |
| 138 | 0.042 | 0.036 | 81/146 (55%) | 1.19 | 14.4° |
| 276 | 0.095 | 0.067 | 81/159 (55%) | 1.25 | **11.7°** |

Conclusion: **mask Dice is non-monotonic (0.111 → 0.042 → 0.095) — noise-dominated, no volume trend —
and class is flat (~54–55%)**, so more of the *same* synthetic data does not improve mask
generalization or class. The one clean signal is **dip MAE, which improves monotonically with volume
(27.0° → 14.4° → 11.7°)**: a straightforward supervised regression head benefits from more scenes even
where the mask/class generalization does not. This is the sweep-level evidence that the mask problem is
generalization/substrate (attack with *different* real data), not synthetic volume.

---

## 15. Diagnostic probes

**Oracle-query ceiling** (`experiments/oracle_ceiling.py`). Procedure: for each held-out instance,
fit a query vector **directly to that instance's GT mask** over the *frozen* pixel features, using
the same BCE+Dice objective the real head uses. Upper-bounds any method of the form
`mask = query · features`, independent of the LM, reader, and training data. Result: **mean 0.431,
median 0.393, 0/100 below 0.1, only 8/100 above 0.7** (reference_mask_levers_closed.md). Licenses:
every held-out fault *is* represented in the frozen features (0% blank), so the head's failure is
inability to *learn* a query, not missing evidence — decomposing the shortfall into a **~4×
generalization gap** (0.104 → 0.431) plus a **substrate gap** (0.431 → 1.0). Does **not** license
claims beyond the *linear-query-over-frozen-features* method class; a different decoder family could
exceed 0.431.

**Split-contamination check** (reference_split_contamination.md). Same `reader.pt`+`mask_dice` gives
0.576 (SCENE_CAP=100, held-out 25) vs 0.062 (uncapped, held-out 93); cause is train leakage, not an
easier population (mean fault area 4.95% vs 5.22%). Licenses the "always evaluate uncapped" rule.

**Copy/budget decomposition** (`experiments/copy_budget.py`, n=12 scenes / 28 values). GT@140 86%
(closed 58%), GT@480 86% (closed 75%), reader@140 70% (closed **8%**), reader@480 **84%** (closed
42%). Licenses: "reader-capped copy" was largely budget truncation + over-detection, not LM
under-copy; the residual flat 0.86 GT ceiling is ~14% genuine under-copy or metric string-match
rounding (unresolved).

---

## 16. Measured failures on the current model (with the fix delta)

- **Coordinate strip in the answer chain.** The `_clean` post-processor deleted *every* bracketed
  span, erasing measured coordinates from the deployed chain (`The fault at [59,356.5] …` →
  `The fault at  …`). It also contaminated fold training (think prefixes built from cleaned text).
  Fix: strip only digit-free brackets (`_PLACEHOLDER = r"\[(?![^\]\n]*\d)[^\]\n]*\]"`,
  `text_metrics.py:27`) + a fold retrain. **Before:** coordinates absent in evidence/think/answer.
  **After:** present in all three, dip matched to GT exactly, multi-object distinct per region
  (`run: run_train, 2026-08-02`).
- **Evidence budget truncation.** Reader-fact copy 0.70 → **0.84** by raising the evidence budget
  120→320 (and think+answer 320→512); closure 8% → 42% (reference_copy_budget.md).
- **Real-panel over-detection.** Background false objects 3.34 → **0.00** after Thebe real finetune
  (reference_thebe_transfer_result.md).
- **Unfixed limitation — synthetic mask generalization.** Held-out mask Dice 0.06–0.11 vs train
  0.75–0.96; measured, not fixed on synthetic (the fix is real-data volume; Thebe 0.25).
- **Unfixed limitation — CRACKS dip label quality.** A constant predictor beats the model 4×; not a
  model failure but a data limitation, quantified (§9).

---

## 17. Threats to validity

- **Small evaluation sets.** Synthetic held-out n=100 fault instances; in-distribution fold-eval
  n=3–24. Real Thebe (n=2,033) is far stronger and should be preferred for any published claim.
- **Single-seed.** Every vision number is single-seed; the measured mask-Dice noise floor is ~0.05
  absolute (two retrains: 0.07, 0.11). Multi-seed reporting is required before publication.
- **Protocol sensitivity.** Per-instance vs union Dice, oracle vs deployment, capped vs uncapped
  each move the headline by 2–9× (§11); comparisons across papers must fix these.
- **Domain gap.** Synthetic zero-shot real Dice 0.03 quantifies it; CRACKS is one survey (F3), so its
  contiguous split tests across-section, not across-field, generalization (only Thebe tests a
  different basin).
- **Circular measurements.** Dip on both CRACKS and Thebe is read from mask/stick geometry — it
  cannot independently corroborate the mask. Only Smeaheia throw (n=114) is non-circular.
- **Oracle-probe scope.** Bounds the linear-query-over-frozen-features class, not all methods.
- **Single domain.** All conclusions are seismic-specific; the copy-seam claim needs replication in
  another measurement-bearing domain.
- **Degenerate class accuracy on single-class real data.** Thebe class 2019/2019 = 100% is an
  artifact of single-class data, not evidence the 49% synthetic class problem is solved.

---

## 18. Not yet done (specified but unrun)

- The five sweeps in §14 marked NOT RUN (noise/SNR, Q-per-scene vs scene count, query count, adapter
  rank, training scene count).
- Multi-seed training for every load-bearing vision number.
- ~~Joint Thebe + CRACKS real finetune~~ — **DONE** (§12 joint benchmark, 2026-08-04). Remaining:
  **oversample the small sets** (CRACKS/Smeaheia) in the joint mix — straight concatenation is
  Thebe-dominated (88%), capping CRACKS at 0.036 vs its solo 0.10.
- Deployment (generated-evidence, Hungarian-matched, misses=0) evaluation of the referring `<SEG>`
  path; its 0.104 is oracle-matched only.
- Data augmentation (h-flip, gain/contrast, noise, elastic) — the one untested no-new-data lever
  hypothesized to lift both mask and class.
- SSL pretraining on unlabeled field volumes (F3, Penobscot, on disk) — the substrate-gap attack and
  the only route by which patch-8 could become viable.
- Wall-clock/peak-VRAM instrumentation for the LM stages and inference (§1 unknowns).

---

## 19. Citations

Standard-method citations (verified from knowledge):

- Qwen Team (Alibaba), "Qwen2.5 Technical Report," arXiv:2412.15115, 2024. **VERIFIED** (base LM
  Qwen2.5-1.5B-Instruct).
- Carion et al., "End-to-End Object Detection with Transformers (DETR)," ECCV 2020. **VERIFIED**.
- Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation (Mask2Former),"
  CVPR 2022. **VERIFIED**.
- Lai et al., "LISA: Reasoning Segmentation via Large Language Model," CVPR 2024 (arXiv 2023).
  **VERIFIED**.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022 (arXiv 2021). **VERIFIED**.
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023. **VERIFIED**.
- Pfeiffer et al., "AdapterFusion: Non-Destructive Task Composition for Transfer Learning," EACL 2021.
  **VERIFIED**.
- Shit et al., "clDice — a Novel Topology-Preserving Loss Function for Tubular Structure
  Segmentation," CVPR 2021. **VERIFIED**.
- Kirillov et al., "Segment Anything (SAM)," ICCV 2023. **VERIFIED**. Ravi et al., "SAM 2," 2024
  (arXiv). **VERIFIED**.
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," TMLR 2024. **VERIFIED**.
- Peng et al., "Kosmos-2: Grounding Multimodal Large Language Models to the World," ICLR 2024
  (arXiv 2023). **VERIFIED**. Rasheed et al., "GLaMM: Pixel Grounding Large Multimodal Model," CVPR
  2024. **VERIFIED**.
- Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning," NeurIPS 2022. **VERIFIED**.
  Shao et al., "DeepSeekMath (GRPO)," arXiv 2024. **VERIFIED**.

Domain-specific (verify exact venue/id before submission):

- SFM / Seismic Foundation Model — Sheng et al. (lead Hanlin Sheng, with Xinming Wu et al.), "Seismic
  Foundation Model (SFM): a next generation deep-learning model in geophysics," arXiv:2309.02791, 2023
  (MIT license). arXiv id **VERIFIED** from `sfm_encoder.py:3`; full author list **UNVERIFIED**.
- Thebe fault dataset — An et al., "Deep convolutional neural network for automatic fault recognition
  from 3D seismic datasets," Computers & Geosciences, 2021; dataset DOI 10.7910/DVN/YBYGBK, CC-BY-4.0.
  DOI **VERIFIED** (`thebe/build_csv.py:4`); exact title/authors **UNVERIFIED — confirm before citing**.
- CRACKS dataset — Georgia Tech OLIVES (AlRegib et al.), "CRACKS: Crowdsourcing Resources for Analysis
  and Categorization of Key Subsurface faults," arXiv:2408.11185, 2024 (HF `gOLIVES/CRACKS`). arXiv id
  **VERIFIED** (`cracks/build_csv.py:2`); exact title/authors **UNVERIFIED**.
- Smeaheia dataset — © Equinor & Gassnova, via CO2DataShare (modified CC-BY-4.0; attribution
  required). **VERIFIED source/license from dataset README; no academic paper cite**.
- synthoseis — Shell open-source synthetic-seismic forward model. Formal citation **UNVERIFIED**
  (software; cite repository/URL).

---

*Document status: all architecture, loss, parameter-count, and config values are code-exact as of
commit `9981f21`. Results are single-seed unless noted, with the measured ~0.05 mask-Dice noise
floor stated. The largest remaining gaps are multi-seed vision numbers and the five unrun sweeps
(§14, §18).*
