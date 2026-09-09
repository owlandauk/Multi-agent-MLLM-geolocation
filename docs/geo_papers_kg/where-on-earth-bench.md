---
title: "Where on Earth? A Vision-Language Benchmark for Probing Model Geolocation Skills Across Scales"
authors: (under review)
venue: OpenReview (ICLR 2026 submission)
year: 2025
arxiv: n/a (OpenReview id=Lr0Pe2Nxty)
topics: [benchmark, multi-scale geolocation, reasoning-chain scoring, web search]
base_model: 12 SOTA VLMs
benchmarks: [WhereCountry (500 MCQ + panoramas), WhereStreet (310 street-level)]
role: benchmark — multi-scale, reasoning-aware evaluation
---

# Where on Earth? (WhereBench)

810 globally-distributed images across two scales: **WhereCountry** (500 country-level MCQ with panoramas) and **WhereStreet** (310 fine-grained street-level, optional web search). Scores coordinate accuracy (Acc@k), hierarchical path scores for text localization, and — notably — **intermediate reasoning chains** against human-verified clues with a Shapley-reweighted "thinking score". Evaluates 12 SOTA VLMs with web-search tools.

## Relation to this thesis
- Multi-scale framing (country → street) mirrors the thesis's coarse-to-fine [[thesis-pure-mllm-pipeline]]; its reasoning-chain scoring is a way to audit *why* a prediction is right, not just whether.
- Panorama + web-search setting connects to retrieval/search-augmented work ([[geosearch]], [[geoagent]]).
- Benchmark family: [[imageo-bench]], [[systematic-benchmark-geoloc]], [[geobench-vlm]], [[llmgeo]].
