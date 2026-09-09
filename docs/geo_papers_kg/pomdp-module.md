---
title: "POMDP Module (Action Selection)"
authors: (this thesis)
venue: Master's thesis
year: 2026
arxiv: n/a
topics: [thesis module, action selection, EIG, LLM-as-policy]
base_model: n/a (pipeline component)
benchmarks: n/a
role: THESIS MODULE — next-verification-task selection
---

# POMDP Module

`POMDPModule` (geo_pipeline/modules/pomdp.py): selects the **next verification task** via LLM-as-policy or expected-information-gain (EIG). Belief = geographic posterior; action = which cue to verify next.

## Diagnosis
- Myopic EIG action selection REGRESSED accuracy in v7-v9.
- FIX: degrade to **fixed climate-first order** at coarse levels ([[geocot]] coarse-to-fine ordering, `POMDP_COARSE_FIXED_ORDER`); keep adaptive EIG/LLM selection at city/street.
- Formalism from [[kaelbling-pomdp]] (belief-state), [[integrated-control-pomdp]], [[active-perception-game]], [[platt-belief-space]], [[perception-aware-pbvi]], [[active-perception-pomdp-survey]], [[belief-space-planning]]. Reward-equation experiments were untested until the eig+combined vs llm A/B.
- Part of [[thesis-pure-mllm-pipeline]] alongside [[sl-module]], [[dst-module]].
