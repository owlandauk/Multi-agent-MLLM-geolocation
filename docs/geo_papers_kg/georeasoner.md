---
title: "GeoReasoner: Reasoning-Enhanced Image Geolocalization with MLLMs"
authors: (GeoReasoner authors)
venue: arXiv
year: 2024
arxiv: 2406.18572
topics: [chain-of-thought geolocation, MLLM reasoning, cue grounding]
base_model: MLLM (instruction-tuned)
benchmarks: [Im2GPS3K, street-view]
role: CoT-reasoning geoloc reference; cue-grounded prompting
---

# GeoReasoner

GeoReasoner elicits **step-by-step reasoning** over visual geolocation cues (climate, vegetation, architecture, signage, driving-side) before committing to a location, rather than one-shot regression.

## Relation to this thesis
- Directly motivates the thesis's [[country-cue-prompting]] positive-cue experiments and the climate-first fixed-order idea in [[pomdp-module]].
- Reasoning paradigm shared with [[geocot]], [[geoagent]], [[gre-suite]].
- Contrast: reasoning-augmented but still generative — complements [[geobayes]]'s Bayesian scaffold.
