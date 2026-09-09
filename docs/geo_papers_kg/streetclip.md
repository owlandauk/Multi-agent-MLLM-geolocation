---
title: "StreetCLIP: Generalizable Zero-Shot Image Geolocalization"
authors: Lukas Haas, Silas Alberti, Michal Skreta
venue: arXiv
year: 2023
arxiv: 2302.00275
topics: [zero-shot geolocation, CLIP fine-tuning, geocell]
base_model: CLIP (domain-adapted for street scenes)
benchmarks: [Im2GPS, Im2GPS3K]
role: precursor encoder to PIGEON
---

# StreetCLIP

StreetCLIP domain-adapts [[clip]] for street-level scenes via synthetic caption pretraining, giving strong zero-shot geolocalization. Direct precursor to [[pigeon]].

## Relation to this thesis
- Part of the [[retrieval-geolocation-family]]; a CLIP-derived competitor line, contrasted with the pure-MLLM [[geobayes]] approach.
