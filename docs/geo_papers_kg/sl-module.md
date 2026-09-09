---
title: "SL Module (Sampling-Likelihood)"
authors: (this thesis)
venue: Master's thesis
year: 2026
arxiv: n/a
topics: [thesis module, sampling, uncertainty shrinkage]
base_model: n/a (pipeline component)
benchmarks: n/a
role: THESIS MODULE — sampling-likelihood with variance shrinkage
---

# SL Module

`SLModule` (geo_pipeline/modules/sl.py): samples the MLLM N times per evidence item and **shrinks the likelihood toward neutral when sample variance is high** (assumes high variance = unsure).

## Diagnosis
- FAILURE MODE: [[qwen25-vl]]-7B is often **confidently-wrong → low variance**, so SL's shrinkage does nothing to catch the error (see [[vlm-safety-audit]], [[consistency-uncertainty]]).
- FIX direction: replace soft variance-shrink with a **hard [[semantic-entropy]] / [[consensus-entropy]] gate** at coarse levels; keep soft shrink at city/street. Disable knob: `SL_DISABLE_LEVELS`. The strongest fix is a **[[vision-conditioned-semantic-entropy]]** gate (UniVRSE/VASE): perturb the image and compare answer distributions, catching the confidently-wrong (ignores-pixels) case that plain semantic entropy misses.
- Uncertainty-theory chain: [[kuhn-semantic-uncertainty]] (origin of semantic entropy), [[kadavath-know]] + [[xiong-verbalized-confidence]] (verbalized-confidence calibration behind GeoBayes' c-score), [[semantic-entropy-probes]] + [[kernel-language-entropy]] (efficient / fine-grained variants).
- Part of [[thesis-pure-mllm-pipeline]] alongside [[dst-module]], [[pomdp-module]].
