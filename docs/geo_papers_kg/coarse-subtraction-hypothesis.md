---
title: "Coarse-Subtraction Hypothesis"
authors: (this thesis)
venue: Master's thesis
year: 2026
arxiv: n/a
topics: [thesis hypothesis, ablation, coarse-level accuracy]
base_model: n/a
benchmarks: [YFCC4K]
role: THESIS CENTRAL HYPOTHESIS (under test)
---

# Coarse-Subtraction Hypothesis

Central thesis hypothesis (AS ORIGINALLY POSED): the [[geobayes]] paradigm has a **systematic coarse-level (country/continent) defect** — added machinery ([[sl-module]], [[dst-module]], [[pomdp-module]]) destroys country signal the raw [[qwen25-vl]] already has. Fix by **SUBTRACTION** (disable/degrade modules at coarse), not addition.

> ⚠️ CORRECTED (2026-09-09): the founding comparison "bare Qwen 68.8@750 vs full GeoBayes 55.8" was a THRESHOLD MISREAD — 68.8 is the baseline @2500 (continent) cell, not @750 (see [[geobayes]]). At the SAME threshold, baseline @750 ≈ 50.7 → GeoBayes @750 = 55.8, so the paper's machinery (Google-Lens+Tavily retrieval) actually HELPS +5pt at country. The subtraction hypothesis's motivating "GeoBayes destroys 13pt of country signal" is thus an artifact. However, the thesis's OWN same-code ablations (below) independently show every ADDITION regresses coarse metrics on YFCC4K — so "coarse machinery is net-negative in *this* reimplementation" still holds empirically; it just isn't what the paper's Table 1 shows.

## Status (2026-08-13)
- Combined P0 subtraction stack LOST: 38.53/58.87 vs same-code baseline 42.07/64.2 — worse on every threshold.
- 4-way single-knob bisection RUNNING to test whether ANY single subtraction helps. If ALL hurt → hypothesis is WRONG for this pipeline; pivot to a raw-vs-pipeline country-leak diagnostic.
- Evidence: [[where-vlms-fail]], [[vlm-safety-audit]], [[dempster-behavior]].
- Geographic-bias evidence (external): [[vlms-geoguessr-masters]] (FairLocator), [[regional-bias-vlm]].
- Bias-aware benchmarks as second battleground: [[imageo-bench]], [[where-on-earth-bench]].
- Manifests as [[pure-mllm-country-bias]].
