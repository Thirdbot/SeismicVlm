"""Seismic VLM — GROUNDED CAPTIONING / REGION-CONDITIONED TEXT GENERATION over 2-D sections.

Academic framing (standard CV / multimodal terminology):
  · Task       — Grounded Captioning / Region-Conditioned Text Generation (the reader supplies
                 per-region measurements; the captioner writes text conditioned on them).
  · Input      — Multimodal Prefix Tuning: the measurements are SOFT PROMPTS (word+index digit tokens
                 + the <feature> soft token) prefixed to the LM sequence (RegionReader.soft_prompt).
  · Goal state — Contextual Cross-Modal Alignment: the captioner strictly uses the (vision-measured)
                 region info instead of guessing from text priors.

    data/       unified loader (schema · loader) + dataset folders (synthetic · smeaheia · …)
                common vision contract across datasets = image · mask · regions; synthetic ALSO
                carries the LM columns (instruction/question/answer/evidence) → real is vision-only
    model/      encoder · reader (RegionReader) · registry · heads · captioner · decoder · geology
                · geometry · text_metrics
    stages/     stage1_geology · stage2_reader · stage2_grounding · stage3_answer · finetune_vision
    eval/       metrics · components · real_transfer · inference · benchmark · schema_check
    run_train.py / run_eval.py   top-level orchestrators (config vars at top, no argparse)

Pipeline: image -> frozen NCS encoder -> RegionReader (class-driven measurement + masks) ->
multimodal-prefix soft prompts (vision MEASURES, captioner COPIES by index) -> Qwen decoder (frozen
geology + grounding + answer combiner) -> grounded caption with the exact copied numbers + masks.

The copy-seam is deliberate: because the injected numbers are measured by vision and copied (not
regressed) by the captioner, the text cannot fabricate figures — faithfulness is structural, measured
with the CHAIR hallucination metric (see eval/metrics.py).
"""
