---
title: "Vision-Conditioned Semantic Entropy (UniVRSE / VASE)"
authors: Zehui Liao, Shishuai Hu, Ke Zou, Huazhu Fu, Liangli Zhen, Yong Xia (VASE); UniVRSE same group
venue: arXiv (UniVRSE) / MICCAI 2025 (VASE)
year: 2025
arxiv: 2503.20504 (UniVRSE)
topics: [semantic entropy, vision grounding, hallucination detection, uncertainty quantification]
base_model: medical VLMs (transferable to any VLM)
benchmarks: [medical open-ended VQA]
role: THEORETICAL upgrade path for the SL module's confidently-wrong failure
---

# Vision-Conditioned Semantic Entropy

Two related methods that fix a blind spot in plain (text-only) [[semantic-entropy]]: it clusters generations by meaning but never checks whether the answer is actually **grounded in the image**. A VLM can be *consistently wrong* (low text entropy, high confidence) when it ignores the pixels.

- **UniVRSE** (Unified Vision-conditioned Response Semantic Entropy, arXiv 2503.20504): conditions the semantic-entropy estimate on the visual input so that visually-ungrounded consistency no longer looks "certain".
- **VASE** (Vision-Amplified Semantic Entropy, MICCAI 2025): samples the semantic predictive distribution under **weak** image transforms (preserve validity), then contrasts with the distribution from a **distorted** image to amplify the visual-grounding signal. Final entropy of the contrastive distribution = VASE score; higher = higher hallucination risk. Beats prior hallucination detectors on two medical open-ended VQA sets.

## Relation to this thesis
- DIRECT fix for the [[sl-module]] failure mode: [[qwen25-vl]]-7B is often **confidently-wrong → low sample variance**, so SL's variance-shrink does nothing. Perturbing the image and comparing answer distributions catches exactly the "consistent but wrong" case that plain [[semantic-entropy]] misses.
- Upgrade direction: replace SL's soft variance-shrink at coarse levels with a **hard vision-conditioned semantic-entropy gate**; keep soft shrink at city/street. Ties to knob `SL_DISABLE_LEVELS`.
- Uncertainty-theory lineage: [[kuhn-semantic-uncertainty]] → [[semantic-entropy]] → [[semantic-entropy-probes]] / [[kernel-language-entropy]]; this is the vision-grounded branch.
- Evidence of the failure it addresses: [[vlm-safety-audit]], [[consistency-uncertainty]].
