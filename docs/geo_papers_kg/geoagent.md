---
title: "GeoAgent: Agentic Tool-Use for Image Geolocation"
authors: (GeoAgent authors)
venue: arXiv
year: 2026
arxiv: 2602.12617
topics: [agentic geolocation, tool use, multi-step planning]
base_model: MLLM agent
benchmarks: [Im2GPS3K, YFCC4K]
role: agentic multi-tool geoloc reference
---

# GeoAgent

GeoAgent treats geolocation as an **agentic loop**: the MLLM plans, calls tools (maps, search, zoom/crop), observes, and iterates toward a location.

## Relation to this thesis
- The plan-act-observe loop is a maximal version of the thesis's Hypothesize-Verify-Update ([[geobayes]]) and [[pomdp-module]] action selection.
- Sibling of [[geosearch]] (tool/web use); reasoning-family with [[georeasoner]], [[geocot]].
