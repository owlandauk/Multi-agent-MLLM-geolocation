---
title: "On the Dempster-Shafer Framework and New Combination Rules"
authors: Ronald R. Yager
venue: Information Sciences
year: 1987
arxiv: (Information Sciences, doi 0020-0255(87)90007-7)
topics: [Yager's rule, conflict management, combination rule]
base_model: n/a (theory)
benchmarks: n/a
role: SOURCE of Yager's cautious combination rule (used in DST module)
---

# Yager 1987

Introduces **Yager's rule of combination**: instead of Dempster's 1/(1−K) renormalization, assign conflict mass K to the universal set (ignorance) — a cautious rule that avoids counterintuitive results under high conflict.

## Relation to this thesis
- The EXACT rule [[dst-module]] falls back to when conflict K > `DST_CONFLICT_THR`. This is the module's "cautious" branch.
- Directly addresses the critique in [[dempster-behavior]]; alternative to [[shafer-1976]] normalization. Sibling rules: [[smets-tbm]], [[dubois-prade-1992]].
