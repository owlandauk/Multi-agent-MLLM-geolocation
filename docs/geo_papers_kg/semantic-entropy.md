---
title: "Detecting Hallucinations in Large Language Models Using Semantic Entropy"
authors: Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal
venue: Nature
year: 2024
arxiv: (Nature 2024)
topics: [semantic entropy, hallucination detection, uncertainty quantification]
base_model: LLMs
benchmarks: [QA uncertainty]
role: THEORETICAL basis for semantic-entropy uncertainty gate
---

# Semantic Entropy (Nature 2024)

Introduces **semantic entropy**: cluster sampled generations by meaning, then compute entropy over meaning-clusters — a measure that catches confident hallucinations that token-level or lexical-variance measures miss.

## Relation to this thesis
- The theoretical basis for replacing [[sl-module]]'s soft variance-shrink with a **hard semantic-entropy gate** at coarse levels (the not-yet-coded SL upgrade).
- Directly addresses the [[vlm-safety-audit]] failure mode (low lexical variance ≠ correct).
- Links to [[consensus-entropy]], [[consistency-uncertainty]].
- Lineage: originated in [[kuhn-semantic-uncertainty]] (ICLR 2023); efficient/fine-grained variants [[semantic-entropy-probes]], [[kernel-language-entropy]]; **vision-grounded** branch [[vision-conditioned-semantic-entropy]] (UniVRSE/VASE) — the one most relevant to a VLM geolocation pipeline.
