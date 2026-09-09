---
title: "Belief-Space Planning with Foundation Models as Uncertainty Estimators"
authors: (authors)
venue: arXiv
year: 2025
arxiv: 2504.03245
topics: [belief-space planning, foundation models, uncertainty estimation, POMDP]
base_model: foundation model as uncertainty estimator
benchmarks: n/a
role: modern bridge — LLM/foundation model inside belief-space planning
---

# Belief-Space Planning with Foundation Models (2504.03245)

Uses a **foundation model as the uncertainty estimator** inside a belief-space planner — the modern bridge between classical POMDP planning and LLM-driven perception.

## Relation to this thesis
- The closest analogue to the thesis architecture: [[qwen25-vl]] supplies the belief update, [[pomdp-module]] plans the next action on that belief.
- Bridges [[kaelbling-pomdp]] / [[platt-belief-space]] (classical) with the [[training-free-paradigm]] MLLM line.
