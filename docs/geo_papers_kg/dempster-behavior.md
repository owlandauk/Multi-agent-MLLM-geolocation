---
title: "On the Behavior of Dempster's Rule of Combination"
authors: Jean Dezert, et al. (ONERA)
venue: HAL / technical report
year: (classic critique)
arxiv: (ONERA HAL)
topics: [Dempster's rule critique, conflict, counterintuitive results]
base_model: n/a
benchmarks: n/a
role: critique justifying DST-off at coarse level
---

# On the Behavior of Dempster's Rule (Dezert)

Demonstrates counterintuitive / overconfident results from Dempster's rule under high conflict, and argues for alternatives (e.g. Yager, DSmT) when disagreement must be retained.

## Relation to this thesis
- Justifies the thesis rule of thumb: **use Dempster only when conflict is low; avoid it when you need to RETAIN disagreement** — exactly the North-America-vs-Europe contrast at continent level.
- Direct support for `DST_DISABLE_LEVELS` at coarse in [[dst-module]]; extends [[dempster-shafer]].
