---
title: "Dempster-Shafer Theory of Evidence"
authors: Arthur Dempster, Glenn Shafer
venue: foundational (mathematical theory of evidence)
year: 1976
arxiv: (classic)
topics: [evidence fusion, belief functions, combination rule]
base_model: n/a (mathematical framework)
benchmarks: n/a
role: THEORETICAL basis for the DST fusion module
---

# Dempster-Shafer Theory (DST)

Belief-function framework for fusing evidence; Dempster's combination rule normalizes by 1/(1−K) where K is conflict mass. **Discards conflict / absolute evidence magnitude**, keeping only relative non-conflict support — counterintuitive and overconfident under high conflict.

## Relation to this thesis
- The theory behind [[dst-module]] (`DSTModule`), which uses Dempster's rule with a **Yager cautious-rule fallback** when conflict K > `DST_CONFLICT_THR`.
- KEY thesis insight: the 1/(1−K) normalization **flattens the strong North-America-vs-Europe contrast** that continent decisions need — so the thesis DISABLES DST at coarse levels (`DST_DISABLE_LEVELS`) and reverts to plain multiplicative Bayes ([[geobayes]]).
- See [[dempster-behavior]] for the critique.
- Origins and alternatives: [[dempster-1967]], [[shafer-1976]], [[yager-1987]], [[smets-tbm]], [[dubois-prade-1992]].
