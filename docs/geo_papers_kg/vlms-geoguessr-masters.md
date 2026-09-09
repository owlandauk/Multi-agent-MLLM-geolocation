---
title: "VLMs as GeoGuessr Masters: Fairness / Geographic Bias in VLM Geolocation"
authors: Jingyuan Huang, Jen-tse Huang, Ziyi Liu, Xiaoyuan Liu, Wenxuan Wang, Jieyu Zhao
venue: arXiv
year: 2025
arxiv: 2502.11163
topics: [geographic fairness, regional bias, VLM geolocation, over-prediction]
base_model: multiple VLMs
benchmarks: [FairLocator]
role: KEY EVIDENCE quantifying the geographic bias the thesis fights
---

# VLMs as GeoGuessr Masters (FairLocator)

Fairness audit of VLM image geolocation. Headline numbers: ~**53.8% city-level accuracy** overall, but **−12.5%** on less-developed regions and **−17.0%** on sparsely-populated areas; models **over-predict a few dominant cities** (e.g. Sydney). GitHub: uscnlp-lime/FairLocator.

## Relation to this thesis
- QUANTITATIVE external evidence for the thesis root cause: geographic distribution bias / over-prediction of dominant locations (see [[pure-mllm-country-bias]], [[coarse-subtraction-hypothesis]]).
- Complements [[where-vlms-fail]] (near-perfect NA, degrades elsewhere) with a fairness framing and a named benchmark (FairLocator).
- Companion bias papers: [[regional-bias-vlm]], [[ai-sees-location]].
- Supports the argument that the winnable battleground is OOD / under-represented regions, not YFCC4K head classes.
