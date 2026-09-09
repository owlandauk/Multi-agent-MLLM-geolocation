---
title: "Planning and Acting in Partially Observable Stochastic Domains"
authors: Leslie Pack Kaelbling, Michael L. Littman, Anthony R. Cassandra
venue: Artificial Intelligence (journal)
year: 1998
arxiv: (AIJ classic)
topics: [POMDP, belief-state, value iteration, foundational]
base_model: n/a (framework)
benchmarks: n/a
role: FOUNDATIONAL POMDP reference (belief-state formulation)
---

# Kaelbling-Littman-Cassandra 1998

The canonical POMDP paper: formalizes planning under partial observability via **belief states** and the belief-MDP reduction, with exact value-iteration algorithms.

## Relation to this thesis
- The theoretical bedrock for [[pomdp-module]] — the thesis's geographic posterior IS a belief state, and choosing the next verification task IS a belief-MDP action.
- Foundational for [[active-perception-game]], [[integrated-control-pomdp]], [[perception-aware-pbvi]], [[belief-space-planning]], [[active-perception-pomdp-survey]].
