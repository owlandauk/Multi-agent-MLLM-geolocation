---
title: "Language Models (Mostly) Know What They Know"
authors: Saurav Kadavath, et al. (Anthropic)
venue: arXiv
year: 2022
arxiv: 2207.05221
topics: [verbalized confidence, self-evaluation, calibration, P(True)]
base_model: LLMs
benchmarks: [calibration]
role: FOUNDATIONAL verbalized-confidence / self-knowledge reference
---

# Kadavath 2022 (Know What They Know)

Shows LLMs can be trained/prompted to **verbalize calibrated confidence** (P(True), self-evaluation) about their own answers.

## Relation to this thesis
- The basis for the "verbalized confidence" signal that [[geobayes]]'s support-weight c-score uses (c on a 1-5 scale, neutral pivot 3).
- CAVEAT the thesis exploits: verbalized confidence is often OVERconfident on wrong answers (see [[vlm-safety-audit]]), motivating [[sl-module]] and the semantic-entropy gate.
- Sibling of [[xiong-verbalized-confidence]], [[kuhn-semantic-uncertainty]].
