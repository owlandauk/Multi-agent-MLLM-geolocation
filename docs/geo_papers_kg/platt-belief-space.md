---
title: "Belief Space Planning Assuming Maximum Likelihood Observations"
authors: Robert Platt, Russ Tedrake, Leslie Kaelbling, Tomas Lozano-Perez
venue: RSS (Robotics: Science and Systems)
year: 2010
arxiv: (MIT CSAIL)
topics: [belief-space planning, maximum-likelihood observation, POMDP approximation]
base_model: n/a (framework)
benchmarks: n/a
role: belief-space planning approximation reference
---

# Platt 2010 (Belief-Space Planning)

Casts partially observable control as **planning in belief space** under the maximum-likelihood-observation assumption, making otherwise-intractable POMDPs tractable.

## Relation to this thesis
- Approximation strategy relevant to [[pomdp-module]]: rather than full POMDP value iteration, plan the next verification greedily on the current belief (the pipeline's practical stance).
- Extends [[kaelbling-pomdp]]; sibling of [[belief-space-planning]], [[integrated-control-pomdp]].
