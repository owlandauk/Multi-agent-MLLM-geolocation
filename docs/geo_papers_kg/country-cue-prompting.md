---
title: "Country-Cue Prompting (Positive Cues)"
authors: (this thesis)
venue: Master's thesis
year: 2026
arxiv: n/a
topics: [prompting, positive cues, driving-side, license-plate, text-script]
base_model: Qwen2.5-VL-7B
benchmarks: [YFCC4K]
role: THESIS TECHNIQUE — positive structured country cues
---

# Country-Cue Prompting

Give the MLLM **high-information positive cues** to narrow the country set: driving-side (L/R), license-plate style, text SCRIPT (Latin/Cyrillic/Han…), road markings, architecture, landscape. Implemented as `_COUNTRY_CUE_PROMPTS` + ensemble, gated by `COUNTRY_CUE_ENSEMBLE`.

## Relation
- Direct counter to [[pure-mllm-country-bias]] — positive cues, NOT US-suppression (v7's suppression over-corrected).
- Literature basis: [[georeasoner]], [[geocot]] (cue-grounded reasoning), [[llmgeo]] (prompt sensitivity).
- Part of the [[thesis-pure-mllm-pipeline]] fusion design.
