---
title: "The Transferable Belief Model"
authors: Philippe Smets, Robert Kennes
venue: Artificial Intelligence (journal)
year: 1994
arxiv: (AIJ classic)
topics: [transferable belief model, open-world, pignistic transformation, unnormalized belief]
base_model: n/a (theory)
benchmarks: n/a
role: TBM — open-world, unnormalized belief functions
---

# Smets Transferable Belief Model (TBM)

The TBM keeps belief functions **unnormalized (open-world)** — conflict mass stays on the empty set rather than being renormalized away — and uses the **pignistic transform** to convert beliefs to decisions.

## Relation to this thesis
- An alternative to [[shafer-1976]]'s closed-world normalization that RETAINS conflict information — conceptually aligned with the thesis's goal of not flattening the North-America-vs-Europe contrast.
- Sibling combination-rule work to [[yager-1987]], [[dubois-prade-1992]]; motivates [[dst-module]]'s coarse-level design.
