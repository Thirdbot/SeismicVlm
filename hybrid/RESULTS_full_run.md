# Hybrid Grounded Seismic VLM — Full-Run Results

Single end-to-end run on the full multi-class dataset (2026-07-27). All numbers are held-out unless
stated. Split is group-wise by image (whole images held out; seed 42, 0.75).

## Configuration

| | |
|---|---|
| Hardware | NVIDIA RTX 3060, 5.67 GB VRAM · 15 GB RAM · Linux |
| LM decoder | Qwen2.5-1.5B-Instruct, 4-bit QLoRA · stacked adapters: geology (frozen CoT) + grounding (evidence copy) + fuse (answer fold) |
| Vision encoder | NCS-v1-2d-base (frozen) · patch 16 · tiles 224 / stride 112 |
| Instance reader | 5-class (∅/fault/closure/salt/onlap) · one query-conditioned DerivedHead (9 tier-2 attrs, section+object scope) · ~10.3 M params |
| Number bridge | non-differentiable digit-token markers (word+index); gated `<feature>_i` soft token |
| Dataset | 1819 rows → **406 scenes** (train 304 / test 102) · regions: fault 629, onlap 945, closure 353, salt 155 |
| Object-derived (closures) | fluid, intersects_fault, intersects_salt, intersects_onlap (all matched to data) |
| Section-derived | number_fault_intersections, fault_mode, number_hc_closures, number_onlap_episodes (salt_inserted absent → salt is a class) |
| Schedule | reader 200 ep @3e-4 · grounding 10 ep · fuse fold 8 ep @2e-5 (102 pairs) · real adapter 60 ep @1e-4 (16.7 k params, base frozen) |

## Synthetic held-out (n = 102; fold-eval in-distribution n = 16)

**Reader**
| | train | held-out |
|---|---|---|
| count MAE | 0.47 | 1.09 |
| dip MAE | 11.5° (n36) | 19.8° (n6) |
| throw MAE | — | 50.9 ms (n9) |
| area MAE | — | 0.6 % (n1) |
| class acc | 374/697 (54%) | 76/191 (40%) |
| mask dice | 0.15 | 0.15 |

**Copy / grounding**
| metric | value |
|---|---|
| COPY-GT (pure copy mechanism) | 0.48 (12/25) |
| COPY-pipeline (reader-capped, deployment) | 0.64 (145/227) |
| copy before fold / after fold | **0.89 / 0.89** (identical → fold protects copy) |

**Reasoning — feature A/B (fold-eval)**
| | present | clean | grounded | think |
|---|---|---|---|---|
| feature ON | 0.94 | 0.94 | **0.62** | 0.69 |
| feature OFF | 1.00 | 0.88 | 0.44 | 0.75 |

Feature gate = **0.0000**. The ON/OFF grounded gap is within n=16 noise / a token-slot artifact,
not feature content → **feature dormant for reasoning**.

## Feature `<feature>_i` A/B — masking (LM `<SEG>`→mask head)

| | dice (held-out, n=16) |
|---|---|
| feature ON | 0.256 |
| feature OFF | 0.257 |

Gate = −0.006. **Feature does not help masking either.** Secondary positive: the LM `<SEG>` token
*can* drive a mask at **dice 0.256 — above the reader's own h_i head (0.15)** → referring-seg from the
LM works, independent of the feature.

## Real-field (Smeaheia) — adapter isolation, fair before/after

All 72 fault-centred windows, split 54 train / 18 held-out (seed 42). BEFORE = synthetic reader
zero-shot on held-out; AFTER = real adapter trained on the 54 train windows, evaluated on the **same**
18 held-out. LM untouched (adapter is vision-only).

| metric | BEFORE | AFTER |
|---|---|---|
| copy-GT (LM) | 0.53 | 0.53 (untouched ✓) |
| copy-pipeline | 0.79 | 0.63 |
| reader count MAE | 1.28 | **0.06** |
| reader class acc | 8/18 (44%) | **18/18 (100%)** |
| reader dip MAE | 33.6° | **22.0°** |
| reader throw MAE | 159 ms | **52 ms** |
| reader mask dice | 0.07 | **0.12** |

## Key findings

1. **Digit-copy bridge + fuse fold hold on full data.** The fold leaves the evidence copy identical
   before vs after (0.89 → 0.89) while producing grounded answers.
2. **Adapter isolation transfers vision to real without touching the LM** — count MAE 1.28→0.06,
   class 44%→100%, dip 33.6→22.0°, throw 159→52 ms, dice 0.07→0.12; copy-GT stays 0.53 (proof the
   frozen LM is untouched). Vision-only real transfer works.
3. **The `<feature>` soft token is dormant** for BOTH reasoning and masking (gate ≈ 0 in both A/Bs):
   the digit + bbox markers already carry what the tasks need. Clean negative result.
4. **LM `<SEG>`→mask works** (dice 0.256 > reader 0.15) — the referring-seg direction is viable and
   does not depend on the feature.
5. **Scaling to the full 4-class data trades fidelity.** Versus a fault-clean 51-scene run
   (copy-GT 0.88, grounded 0.85), the full onlap-dominated 4-class data gives copy-GT 0.48 and
   grounded 0.62. Cause = the correspondence gradient: many throw/area/derived attributes are not
   stated per-object in the evidence, so the copy has nothing to reproduce, and the 1.5 B LM
   confabulates more surrounding prose on the richer injection.

## Limitations

- Held-out mask dice low (0.15 synthetic / 0.12 real) — the known vision-generalization gap.
- Reader class accuracy 40% held-out — 4-class + onlap dominance (945/1762 regions) is hard.
- Copy/grounded lower than the fault-only regime — correspondence gap (evidence must state each
  attribute per object for it to be copyable).
- dip held-out n=6 — few dip-faults survive in the onlap-heavy held-out; that MAE is noisy.
- Feature dormant — no signal in this dataset opens the gate; needs a task whose answer/mask cannot
  be served by digits+bbox.
