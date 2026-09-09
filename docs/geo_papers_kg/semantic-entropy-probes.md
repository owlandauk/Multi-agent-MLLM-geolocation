---
title: "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs"
authors: Jannik Kossen, et al.
venue: arXiv
year: 2024
arxiv: 2406.15927
topics: [semantic entropy, hidden-state probes, efficient uncertainty]
base_model: LLMs
benchmarks: [hallucination detection]
role: efficient semantic-entropy estimation reference
---

# Semantic Entropy Probes (Kossen 2024)

Estimates semantic entropy from **hidden states via a linear probe** — capturing the uncertainty signal without sampling many generations.

## Relation to this thesis
- A cheaper alternative to [[sl-module]]'s N-sample approach: if a probe can read uncertainty from one forward pass, the sampling cost drops.
- Extends [[kuhn-semantic-uncertainty]] / [[semantic-entropy]]; sibling of [[kernel-language-entropy]], [[consensus-entropy]].
