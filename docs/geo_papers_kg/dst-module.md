---
title: "DST Module (Dempster-Shafer Fusion)"
authors: (this thesis)
venue: Master's thesis
year: 2026
arxiv: n/a
topics: [thesis module, evidence fusion, Yager fallback]
base_model: n/a (pipeline component)
benchmarks: n/a
role: THESIS MODULE — DST fusion with Yager fallback
---

# DST Module

`DSTModule` (geo_pipeline/modules/dst.py): fuses evidence with [[dempster-shafer]] combination; falls back to **Yager's cautious rule** when conflict mass K > `DST_CONFLICT_THR`. Also hosts the US-prior debias (`COUNTRY_PRIOR_DEBIAS_GAMMA`, `_debias_country_prior`).

## Diagnosis
- The 1/(1−K) normalization flattens the North-America-vs-Europe contrast needed for continent decisions (see [[dempster-behavior]]).
- Theory chain: [[dempster-1967]] → [[shafer-1976]] → combination-rule alternatives [[yager-1987]] (the Yager fallback this module uses), [[smets-tbm]], [[dubois-prade-1992]].
- FIX: disable DST at coarse (`DST_DISABLE_LEVELS`) → reduces to plain multiplicative Bayes ([[geobayes]], justified by [[dst-bayes-equivalence]]); keep Yager fallback at city/street for real conflict.
- Part of [[thesis-pure-mllm-pipeline]] alongside [[sl-module]], [[pomdp-module]].
