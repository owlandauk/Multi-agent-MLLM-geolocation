---
title: "Pure-MLLM Country Bias (US Over-Prediction)"
authors: (this thesis)
venue: Master's thesis
year: 2026
arxiv: n/a
topics: [thesis diagnosis, distribution bias, US over-prediction]
base_model: Qwen2.5-VL-7B
benchmarks: [YFCC4K]
role: THESIS ROOT-CAUSE DIAGNOSIS
---

# Pure-MLLM Country Bias

Root cause of the thesis pipeline's coarse-level gap: **North-America / US over-prediction**. v6b: 825 NA false positives, 575 European images misrouted to US. Empirical [[yfcc4k]] prior confirms US=0.343 dominant.

## Relation
- The failure [[coarse-subtraction-hypothesis]] tries to fix; surfaced in [[thesis-pure-mllm-pipeline]].
- External evidence: [[where-vlms-fail]] (near-perfect in NA, degrades elsewhere), [[vlms-geoguessr-masters]] (FairLocator: −12.5% less-developed, −17% sparse, over-predicts Sydney), [[regional-bias-vlm]], [[ai-sees-location]].
- Attempted fixes: `COUNTRY_PRIOR_DEBIAS_GAMMA` (divide by empirical freq), [[country-cue-prompting]] positive cues. NOTE: v7 prompt-suppression of US OVER-corrected (street crashed) — lesson: positive cues, not suppression.
