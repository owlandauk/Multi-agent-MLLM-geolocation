---
title: "GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating"
authors: Qichuan Geng, Zhong Zhou, et al.
venue: AAAI (reported 2026)
year: 2026
arxiv: (paper PDF nave.vr3i.com)
topics: [training-free geolocation, Bayesian MAP, sequential update, MLLM geolocation]
base_model: Qwen2.5-VL-7B, Qwen2-VL-7B
benchmarks: [Im2GPS3K, YFCC4K]
role: THESIS BASELINE (the method this thesis extends)
---

# GeoBayes

GeoBayes formulates **image geolocation as MAP estimation** over geographic hypotheses, solved by **sequential Bayesian updating** in a **Hypothesize–Verify–Update** loop across geographic scales (continent → country → city → street). It is **training-free** and uses **Qwen2.5-VL-7B** as the verifier MLLM.

## Core method
- Multiplicative posterior update: P_t(l) ∝ P_0(l) · ∏ W(e_t | l), with support weight W = exp[α·β·(c−3)], β = ln2, centered at neutral pivot c=3.
- No Dempster-Shafer, no sampling-uncertainty, no POMDP action selection — plain multiplicative Bayes.

## Reported results
- +3.2% average localization accuracy over generation-based baselines without extra training.
- YFCC4K thresholds (Street/City/Region/Country/Continent km 1/25/200/750/2500): **4.9 / 16.1 / 30.9 / 55.8 / 75.4**.
- Im2GPS3K: 6.3 / 34.7 / 53.6 / 73.7 / 85.9.

## Relation to this thesis
- This thesis EXTENDS GeoBayes by adding three modules: [[sl-module]] sampling-likelihood, [[dst-module]] Dempster-Shafer fusion, [[pomdp-module]] action selection. See [[thesis-pure-mllm-pipeline]].
- CORRECTED (2026-09-09): earlier notes claimed "bare Qwen 68.8@750 → GeoBayes 55.8, machinery HURTS." This was a THRESHOLD MISREADING — 68.8 is the baseline @2500 (continent) cell, not @750. Same-threshold Table 1 story: baseline Qwen2.5-VL @750 ≈ 50.7 → GeoBayes @750 = 55.8, i.e. the paper's retrieval (Google Lens + Tavily) HELPS +5pt at country. Verified against the thesis's own same-code bare baseline (42.07@750 / 64.2@2500, ~48.6@750 hi-res): bare @750 is never 68.8; a bare model can't exceed GeoBayes-with-retrieval at the same threshold. See [[coarse-subtraction-hypothesis]] (the subtraction premise rests on this misread).
- Built on base model [[qwen25-vl]].
