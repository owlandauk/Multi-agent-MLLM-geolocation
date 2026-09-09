---
title: "GeoSearch: Web-Scale Retrieval-Augmented Image Geolocation"
authors: (GeoSearch authors)
venue: arXiv / alphaXiv
year: 2026
arxiv: 2604.25390
topics: [web-search augmented geolocation, agentic retrieval, evidence gating]
base_model: MLLM + web search tool
benchmarks: [Im2GPS3K, YFCC4K, OOD]
role: web-evidence retrieval reference; relevant to pipeline's web-verify branch
---

# GeoSearch

GeoSearch augments MLLM geolocation with **web-scale search**: it issues queries about visual entities (landmarks, signage text) and fuses retrieved web evidence into the location decision.

## Relation to this thesis
- Directly relevant to the thesis pipeline's **web-search verify branch** and the country-only web-evidence gate in the fusion design (see [[thesis-pure-mllm-pipeline]]).
- Agentic-retrieval sibling of [[geoagent]]; retrieval-family member alongside [[geoclip]], [[img2loc]].
