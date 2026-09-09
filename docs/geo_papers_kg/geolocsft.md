---
title: "GeoLocSFT: Supervised Fine-Tuning for Image Geolocation"
authors: (GeoLocSFT authors)
venue: arXiv
year: 2025
arxiv: 2506.01277
topics: [supervised fine-tuning, MLLM geolocation]
base_model: fine-tuned MLLM
benchmarks: [Im2GPS3K, YFCC4K]
role: SFT contrast to training-free line
---

# GeoLocSFT

GeoLocSFT fine-tunes an MLLM specifically for geolocation on curated data.

## Relation to this thesis
- Direct CONTRAST to the [[training-free-paradigm]] the thesis commits to (no fine-tuning) — GeoLocSFT buys accuracy with training data the thesis deliberately avoids.
- Sibling of [[reinforcing-image-geoloc]], [[geo-rl]] (both add training/RL).
