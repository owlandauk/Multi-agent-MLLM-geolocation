---
title: "Behavioral Safety Audit of Vision-Language Models: Confidence on Wrong Answers"
authors: (authors)
venue: arXiv
year: 2026
arxiv: 2608.02790
topics: [calibration, overconfidence, uncertainty, VLM audit]
base_model: multiple VLMs
benchmarks: [calibration benchmarks]
role: STRONGEST evidence that low sampling-variance != correctness
---

# Behavioral Safety Audit of VLMs (2608.02790)

Audit finding: VLM **mean confidence on WRONG answers is 0.82–0.97**, with 33–46% high-confidence errors; the most-accurate model is the most overconfident on its errors.

## Relation to this thesis
- STRONGEST support for the thesis's [[sl-module]] failure diagnosis: SL assumes "unsure → high variance", but Qwen-7B is **confidently-wrong → low variance**, so SL's variance shrinkage does nothing to catch the error.
- Justifies disabling SL variance-shrink at coarse levels (`SL_DISABLE_LEVELS`) and motivates a hard semantic-entropy gate instead of soft variance shrink.
- Links to [[semantic-entropy]], [[consistency-uncertainty]], [[consensus-entropy]].
