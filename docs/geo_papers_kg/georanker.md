---
title: "GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization"
authors: (GeoRanker authors)
venue: arXiv
year: 2025
arxiv: 2505.13731
topics: [retrieval geolocation, learning-to-rank, distance-aware loss]
base_model: CLIP-family encoder + ranking head
benchmarks: [Im2GPS3K, YFCC4K]
role: retrieval-line competitor; rerank paradigm sibling
---

# GeoRanker

GeoRanker frames geolocation as **distance-aware ranking** over candidate locations, training a ranker whose loss respects geographic distance rather than treating candidates as unordered classes.

## Relation to this thesis
- Rerank paradigm sibling to the thesis's [[geoclip-rerank-line]] (rank candidates rather than regress coordinates), but GeoRanker LEARNS the ranker whereas the thesis reranks training-free with [[qwen25-vl]].
- Member of [[retrieval-geolocation-family]] alongside [[geoclip]], [[pigeon]], [[pinpoint]], [[img2loc]], [[geosearch]].
