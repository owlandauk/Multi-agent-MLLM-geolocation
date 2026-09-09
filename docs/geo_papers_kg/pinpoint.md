---
title: "PinPoint: Fine-Grained Image Geolocalization"
authors: (PinPoint authors)
venue: arXiv
year: 2026
arxiv: 2606.04133
topics: [retrieval geolocation, fine-grained localization]
base_model: CLIP-family encoder
benchmarks: [Im2GPS3K, YFCC4K]
role: recent retrieval-line competitor
---

# PinPoint

PinPoint targets **fine-grained (street/city-level)** worldwide geolocalization via retrieval, pushing accuracy at the tight distance thresholds.

## Relation to this thesis
- Fine-level focus contrasts with the thesis's diagnosed weakness at COARSE (country/continent) level (see [[coarse-subtraction-hypothesis]]).
- Member of [[retrieval-geolocation-family]]; shares the retrieval paradigm with [[geoclip]], [[georanker]].
