---
title: "Qwen2.5-VL Technical Report"
authors: Qwen Team, Alibaba
venue: arXiv technical report
year: 2025
arxiv: 2502.13923
topics: [vision-language model, MLLM, base model, grounding, document parsing]
base_model: Qwen2.5-VL (3B / 7B / 72B)
benchmarks: [general VLM benchmarks]
role: BASE MODEL used as the MLLM verifier across the thesis pipeline
---

# Qwen2.5-VL

Qwen2.5-VL is the vision-language model family used as the **verifier MLLM** in [[geobayes]] and in this thesis's [[thesis-pure-mllm-pipeline]]. Strong at grounding, OCR/text-script reading, and spatial reasoning — capabilities the pipeline exploits for [[country-cue-prompting]] (text script, driving-side, license-plate cues).

## Relation to this thesis
- The 7B variant is the workhorse verifier ([[geobayes]] baseline and thesis pipeline both run Qwen2.5-VL-7B).
- CORRECTED (2026-09-09, see [[geobayes]]): the old "bare 68.8@750 → GeoBayes 55.8, raw model knows more" claim was a threshold misread — 68.8 is baseline @2500 (continent). Same-threshold: baseline @750 ≈ 50.7 → GeoBayes @750 = 55.8 (retrieval HELPS +5pt at country).
- Contrast with [[clip]]-based retrieval encoders used by [[geoclip]], [[pigeon]], [[georanker]].
- Known limitation: not geolocation-SOTA (Qwen2.5-72B ~36% R@1km) and exhibits [[regional-bias-vlm]] North-America overprediction.
