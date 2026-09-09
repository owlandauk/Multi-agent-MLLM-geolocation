---
title: "Img2Loc: Revisiting Image Geolocalization using Multimodal Foundation Models and Retrieval-Augmented Generation"
authors: Zhongliang Zhou, et al.
venue: SIGIR (short)
year: 2024
arxiv: 2403.19584
topics: [RAG geolocation, MLLM, negative anchors, retrieval-augmented]
base_model: GPT-4V / multimodal foundation model + CLIP retrieval
benchmarks: [Im2GPS3K, YFCC4K]
role: RAG + negative-anchor prompting reference
---

# Img2Loc

Img2Loc combines **CLIP-based retrieval** with a multimodal foundation model, feeding both nearest (positive) and farthest (negative) candidate coordinates as anchors into the prompt so the model interpolates/excludes accordingly.

## Relation to this thesis
- The **negative-anchor prompt** idea is a candidate injection for the thesis fusion design (tell the model where it is NOT — see [[thesis-pure-mllm-pipeline]] fusion notes).
- RAG structure parallels the [[geoclip-rerank-line]]: retrieve candidates, let the MLLM decide within them.
- Member of [[retrieval-geolocation-family]].
