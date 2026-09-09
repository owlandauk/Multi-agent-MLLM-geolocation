---
title: "GeoCLIP-Rerank Line"
authors: (this thesis)
venue: Master's thesis
year: 2026
arxiv: n/a
topics: [thesis line, retrieval + rerank, GeoCLIP candidates, Qwen rerank]
base_model: GeoCLIP + Qwen2.5-VL-7B
benchmarks: [YFCC4K, GWS15k]
role: THESIS SECOND LINE (retrieval-rerank)
---

# GeoCLIP-Rerank Line

The thesis's retrieval-rerank line: [[geoclip]] supplies top-k GPS candidates; [[qwen25-vl]] **reranks WITHIN candidates** (never invents a new location); answer falls back to GeoCLIP rank-1 unless the reranked posterior clears an override margin. Goal: approach GeoCLIP oracle@5/@10.

## Status
- 2026-08-12 diagnosis: rerank additions are net-negative on in-distribution [[yfcc4k]] (GeoCLIP already beats [[geobayes]] there), so this line was set aside to first optimize [[thesis-pure-mllm-pipeline]].
- OOD [[gws15k]] is the identified winnable battleground.
- Rerank paradigm sibling of [[georanker]], [[img2loc]].
