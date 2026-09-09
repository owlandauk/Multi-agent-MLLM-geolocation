---
title: "Integrated Perception and Control via POMDP"
authors: (authors)
venue: arXiv
year: 2025
arxiv: 2504.13288
topics: [POMDP, integrated control, belief-space planning]
base_model: n/a (framework)
benchmarks: n/a
role: POMDP formulation reference for the action-selection module
---

# Integrated Control POMDP (2504.13288)

POMDP formulation coupling perception and control in belief space.

## Relation to this thesis
- Formalism reference for [[pomdp-module]] (`POMDPModule`) — belief = geographic posterior, action = next verification task.
- KEY thesis finding: myopic EIG action selection REGRESSED accuracy in v7-v9, so the thesis DEGRADES POMDP to a **fixed climate-first order at coarse levels** (`POMDP_COARSE_FIXED_ORDER`) and keeps adaptive selection only at city/street.
- Sibling of [[active-perception-game]].
