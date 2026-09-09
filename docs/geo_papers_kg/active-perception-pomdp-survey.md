---
title: "Exploiting Submodular Value Functions for Active Perception (POMDP survey)"
authors: Yash Satsangi, Shimon Whiteson, et al.
venue: Autonomous Robots (special issue on Active Perception)
year: 2018
arxiv: (Oxford / cs.ox.ac.uk)
topics: [active perception, submodularity, information gathering, POMDP]
base_model: n/a (framework)
benchmarks: n/a
role: active-perception theory / submodular info-reward reference
---

# Satsangi Active Perception (Submodular Value)

Frames active perception as a POMDP whose reward is **submodular information gain**, giving near-optimality guarantees for greedy sensing selection.

## Relation to this thesis
- Theoretical justification for GREEDY next-task selection in [[pomdp-module]] (submodularity → greedy is near-optimal), and a lens on WHY naive myopic EIG can still fail when submodularity breaks.
- Sibling of [[perception-aware-pbvi]], [[active-perception-game]]; extends [[kaelbling-pomdp]].
