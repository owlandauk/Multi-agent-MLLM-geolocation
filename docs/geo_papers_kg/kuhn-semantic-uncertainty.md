---
title: "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation"
authors: Lorenz Kuhn, Yarin Gal, Sebastian Farquhar
venue: ICLR
year: 2023
arxiv: 2302.09664
topics: [semantic entropy, uncertainty estimation, meaning-clustering]
base_model: LLMs
benchmarks: [QA uncertainty]
role: ORIGIN of semantic entropy (meaning-invariant uncertainty)
---

# Kuhn 2023 (Semantic Uncertainty)

The foundational paper for **semantic entropy**: cluster generations by meaning (bidirectional entailment) before computing entropy, so lexical paraphrase variance doesn't masquerade as uncertainty.

## Relation to this thesis
- The origin of the [[semantic-entropy]] idea the thesis wants to use as a HARD gate replacing [[sl-module]]'s soft variance-shrink.
- Precedes the Nature 2024 [[semantic-entropy]] follow-up; sibling of [[kernel-language-entropy]], [[semantic-entropy-probes]].
- Directly targets the [[sl-module]] failure: [[qwen25-vl]] can be lexically consistent but semantically wrong.
