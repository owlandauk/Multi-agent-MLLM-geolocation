---
title: "GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization"
authors: Vicente Vivanco Cepeda, Gaurav Kumar Nayak, Mubarak Shah
venue: NeurIPS
year: 2023
arxiv: 2309.16020
topics: [retrieval geolocation, CLIP alignment, GPS embedding, gallery retrieval]
base_model: CLIP (image encoder) + GPS location encoder
benchmarks: [Im2GPS3K, YFCC4K]
role: RETRIEVAL BASELINE / candidate provider for the GeoCLIP-rerank line
---

# GeoCLIP

GeoCLIP learns a **CLIP-style alignment between images and GPS coordinates**, encoding locations with a random-Fourier-feature GPS encoder into a shared embedding space. Geolocation becomes **retrieval**: rank a gallery of GPS embeddings by similarity to the image embedding.

## Reported results
- Im2GPS3K (1/25/200/750 km): 34.47 / 50.65 / 69.97 / 83.82.
- YFCC4K (1/25/200/750 km): 11.82 / 29.04 / 56.13 / 79.13.

## Relation to this thesis
- Supplies top-k GPS candidates in the [[geoclip-rerank-line]] (Qwen reranks WITHIN candidates, never invents a new location).
- On YFCC4K, GeoCLIP already beats [[geobayes]] — so the thesis set the GeoCLIP line aside to first optimize the [[thesis-pure-mllm-pipeline]].
- Uses base model [[clip]]. Contrast with pure-MLLM approaches like [[geobayes]] that use [[qwen25-vl]].
- Retrieval paradigm shared with [[georanker]], [[pinpoint]], [[img2loc]], [[geosearch]].
