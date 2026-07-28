"""Grounded seismic VLM — region-conditioned text generation over seismic sections.

    data/       unified loader (schema · loader) + dataset folders (synthetic · smeaheia · …)
                common vision contract across datasets = image · mask · regions; synthetic ALSO
                carries the LM columns (instruction/question/answer/evidence) → real is vision-only
    model/      encoder · reader · registry · heads · narrator · decoder · geology · geometry · text_metrics
    stages/     stage1_geology · stage2_reader · stage2_grounding · stage3_fold · finetune_vision · seg_mask
    eval/       metrics · components · real_transfer · inference · schema_check
    run_train.py / run_eval.py   top-level orchestrators (config vars at top, no argparse)

Pipeline: image -> frozen NCS encoder -> instance reader (class-driven measurement + masks) ->
non-differentiable digit-token bridge (vision MEASURES, LM COPIES by index) -> Qwen decoder (frozen
geology + grounding + fuse) -> grounded narration with the exact copied numbers, plus per-object masks.

The copy-seam is deliberate: because the injected numbers are measured by vision and copied (not
regressed) by the LM, the narration cannot fabricate figures — faithfulness is structural, and is
measured with a CHAIR-style hallucination metric (see eval/metrics.py).
"""
