---
title: "Thesis Pure-MLLM Pipeline"
authors: (this thesis)
venue: Master's thesis
year: 2026
arxiv: n/a
topics: [thesis contribution, coarse-to-fine, Bayesian, MLLM]
base_model: Qwen2.5-VL-7B
benchmarks: [YFCC4K, Im2GPS3K]
role: THE THESIS ARTIFACT (non-GeoCLIP line)
---

# Thesis Pure-MLLM Pipeline

The thesis's own coarse-to-fine (Country → City → Street) Bayesian pipeline extending [[geobayes]] with three modules: [[sl-module]], [[dst-module]], [[pomdp-module]]. Training-free ([[training-free-paradigm]]), Qwen2.5-VL-7B verifier ([[qwen25-vl]]).

## Current diagnosis (2026-08-13)
- Same-code baseline (all knobs off): 5.0/14.73/24.0/42.07/64.2 on [[yfcc4k]].
- vs [[geobayes]] 4.9/16.1/30.9/55.8/75.4 — only Street wins; Country/Continent lag −13.7/−11.2.
- Root cause: [[pure-mllm-country-bias]] (US over-prediction). CORRECTED (2026-09-09): the old "bare Qwen 68.8@750, pipeline destroys ~27pt of country signal" was a threshold misread — 68.8 is the baseline @2500 (continent), not @750 (see [[geobayes]]). Bare Qwen @750 ≈ 50.7 (paper) / 42.07 same-code / ~48.6 hi-res; there is no 27pt country cliff. The real coarse story is the same-code lag on line 19, not a 68.8→ collapse.
- [[coarse-subtraction-hypothesis]]: adding machinery hurts coarse; fix by subtraction. But the combined P0 subtraction stack LOST (38.53/58.87) — bisection running to find which single knob (if any) helps.

## Fusion design (per-level)
- [[dst-module]]: pure multiplicative (Yager off) at coarse; Yager fallback at city/street.
- [[sl-module]]: hard agreement/[[semantic-entropy]] gate at coarse; soft variance-shrink at city/street.
- [[pomdp-module]]: fixed climate-first order ([[geocot]]) at coarse; adaptive EIG at city/street.
- Prompt: [[country-cue-prompting]] positive cues + [[img2loc]] negative anchors + [[geosearch]] web-evidence gate.
