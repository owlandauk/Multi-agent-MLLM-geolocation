---
title: "Where Do VLMs Fail? Regional Failure Analysis for Image Geolocation"
authors: (EarthVision authors)
venue: CVPR Workshop (EarthVision)
year: 2026
arxiv: (CVPR 2026 EarthVision)
topics: [regional bias, failure analysis, VLM geolocation, North America overprediction]
base_model: multiple VLMs
benchmarks: [worldwide geoloc, per-region]
role: KEY EVIDENCE for the US-overprediction bias the thesis fights
---

# Where Do VLMs Fail (EarthVision CVPR 2026)

Regional failure analysis showing VLMs are **near-perfect in North America but degrade sharply elsewhere** — direct evidence of the geographic distribution bias.

## Relation to this thesis
- PRIMARY external evidence for the thesis's root cause: **North-America / US over-prediction bias** at country level (see [[coarse-subtraction-hypothesis]] and [[pure-mllm-country-bias]]).
- Motivates reporting per-region country accuracy, not just overall.
- Companion bias papers: [[regional-bias-vlm]], [[ai-sees-location]], [[geo-privacy]].
