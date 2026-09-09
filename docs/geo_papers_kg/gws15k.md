---
title: "GWS15k: Global Worldwide Street-View Benchmark"
authors: (introduced in G3 / related)
venue: (benchmark)
year: 2024
arxiv: (via G3)
topics: [OOD benchmark, worldwide street view, distribution shift]
base_model: n/a (dataset)
benchmarks: [GWS15k]
role: OUT-OF-DISTRIBUTION benchmark — thesis's identified winnable battleground
---

# GWS15k

A globally-distributed street-view benchmark, harder and more uniformly-sampled than Im2GPS3K/YFCC4K (which are photographer-biased toward landmarks and North America/Europe).

## Relation to this thesis
- Identified in [[geoclip-rerank-line]] as the **winnable battleground**: on in-distribution YFCC4K the pipeline additions are net-negative, but the OOD uniform sampling of GWS15k is where a debiased pipeline could show real gains.
- Reported by [[g3]]; contrast with in-distribution [[yfcc4k]], [[im2gps3k]].
