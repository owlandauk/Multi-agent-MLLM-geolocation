---
title: "Retrieval-Geolocation Family"
authors: (multiple)
venue: (family)
year: 2023
arxiv: n/a
topics: [retrieval, CLIP alignment, gallery ranking]
base_model: CLIP-family encoders
benchmarks: [Im2GPS3K, YFCC4K, GWS15k]
role: FAMILY node grouping retrieval-based geolocation methods
---

# Retrieval-Geolocation Family

Methods that treat geolocation as **retrieval/ranking over a gallery** of locations or geocells, built on [[clip]]-family encoders.

## Members
[[geoclip]], [[pigeon]], [[streetclip]], [[georanker]], [[pinpoint]], [[img2loc]], [[g3]], [[geosearch]].

## Relation
- Collectively the strong retrieval-line competitors; CONTRAST with the pure-MLLM generative paradigm ([[geobayes]], [[thesis-pure-mllm-pipeline]]).
- The thesis's own [[geoclip-rerank-line]] bridges the two (retrieve with GeoCLIP, rerank with an MLLM).
