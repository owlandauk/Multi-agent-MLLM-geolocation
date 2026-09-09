---
title: "PIGEON: Predicting Image Geolocations"
authors: Lukas Haas, Michal Skreta, Silas Alberti, Chelsea Finn
venue: CVPR
year: 2024
arxiv: 2307.05845
topics: [retrieval geolocation, geocell classification, CLIP, planet-scale]
base_model: CLIP (StreetCLIP-derived)
benchmarks: [Im2GPS3K, YFCC4K, GeoGuessr]
role: strong planet-scale geolocation baseline
---

# PIGEON / PIGEOTTO

PIGEON combines **semantic geocell classification**, contrastive pretraining, and retrieval-based refinement for planet-scale geolocation. PIGEOTTO is the generalization to arbitrary images. Descended from [[streetclip]].

## Relation to this thesis
- A high-accuracy retrieval-line competitor; sits alongside [[geoclip]], [[georanker]], [[pinpoint]] in the [[retrieval-geolocation-family]].
- Uses [[clip]]-family encoders; contrast with pure-MLLM [[geobayes]].
