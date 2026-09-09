---
title: "Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities"
authors: Alexander Nikitin, et al.
venue: NeurIPS
year: 2024
arxiv: 2405.20003
topics: [kernel language entropy, semantic similarity, fine-grained uncertainty]
base_model: LLMs
benchmarks: [uncertainty]
role: kernel-based semantic-entropy extension
---

# Kernel Language Entropy (Nikitin 2024)

Generalizes [[semantic-entropy]] using **semantic-similarity kernels** over generations, giving finer-grained uncertainty than hard meaning-clustering.

## Relation to this thesis
- A softer, continuous version of the semantic-entropy gate for [[sl-module]] — useful if hard clustering is too brittle at country/street granularity.
- Extends [[kuhn-semantic-uncertainty]]; sibling of [[semantic-entropy-probes]], [[consensus-entropy]], [[consistency-uncertainty]].
