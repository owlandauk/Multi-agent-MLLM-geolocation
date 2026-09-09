---
title: "GeoCoT / GeoComp: Chain-of-Thought Geolocation with a Human-Comparison Benchmark"
authors: (GeoCoT authors)
venue: arXiv
year: 2025
arxiv: 2502.13759
topics: [chain-of-thought geolocation, benchmark, coarse-to-fine reasoning]
base_model: MLLM
benchmarks: [GeoComp, Im2GPS3K]
role: coarse-to-fine CoT reasoning + benchmark reference
---

# GeoCoT / GeoComp

GeoCoT structures geolocation as an explicit **coarse-to-fine chain of thought** (continent → country → region → city), and GeoComp provides a human-comparison benchmark for it.

## Relation to this thesis
- The coarse-to-fine CoT ordering is the literature basis for the thesis's **climate-first fixed-order** degradation of [[pomdp-module]] at coarse levels (subtract adaptivity where myopic EIG regressed).
- Shares the reasoning paradigm with [[georeasoner]], [[geoagent]], [[gre-suite]].
