---
title: "Perception-Aware Point-Based Value Iteration for POMDPs"
authors: (Satsangi, Whiteson, et al. line)
venue: IJCAI
year: 2019
arxiv: (IJCAI 2019 proceedings)
topics: [active perception, point-based value iteration, information reward, POMDP]
base_model: n/a (framework)
benchmarks: n/a
role: active-perception POMDP solver reference
---

# Perception-Aware PBVI (IJCAI 2019)

An active-perception POMDP formulation where the reward is an **information-theoretic (belief-entropy-reduction) term**, solved by point-based value iteration — sensing actions chosen to reduce uncertainty.

## Relation to this thesis
- Directly models what [[pomdp-module]] does: pick the verification action that most reduces posterior uncertainty (expected information gain).
- Belief-entropy reward parallels the thesis's EIG action selection; sibling of [[active-perception-game]], [[active-perception-pomdp-survey]].
- KEY caveat for the thesis: myopic information reward regressed accuracy (v7-v9), motivating [[coarse-subtraction-hypothesis]] fixed-order at coarse levels.
