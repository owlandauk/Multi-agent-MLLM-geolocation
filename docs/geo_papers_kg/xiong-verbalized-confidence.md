---
title: "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation"
authors: Miao Xiong, et al.
venue: ICLR
year: 2024
arxiv: 2306.13063
topics: [verbalized confidence, confidence elicitation, calibration]
base_model: LLMs (GPT-family)
benchmarks: [confidence elicitation]
role: verbalized-confidence elicitation benchmark
---

# Xiong 2024 (Verbalized Confidence)

Systematically benchmarks **confidence elicitation** methods (verbalized scores, consistency, hybrid), finding verbalized confidence is often poorly calibrated and overconfident.

## Relation to this thesis
- Directly informs how [[geobayes]]/[[sl-module]] should treat the MLLM's stated confidence — don't trust it raw; combine with sampling.
- Extends [[kadavath-know]]; complements [[kuhn-semantic-uncertainty]], [[vlm-safety-audit]].
