---
title: "Reinforcing Image Geolocalization with Reinforcement Learning"
authors: (authors)
venue: NeurIPS
year: 2025
arxiv: (NeurIPS 2025)
topics: [reinforcement learning, MLLM geolocation, reward shaping]
base_model: RL-tuned MLLM
benchmarks: [Im2GPS3K, YFCC4K]
role: RL-training contrast to training-free line
---

# Reinforcing Image Geolocation (RL)

Uses reinforcement learning (distance-based reward) to tune an MLLM for geolocation.

## Relation to this thesis
- CONTRAST to [[training-free-paradigm]]; sibling of [[geolocsft]], [[geo-rl]].
- The distance-based reward parallels the thesis's [[pomdp-module]] reward-equation experiments (but applied to weight updates, not action selection).
