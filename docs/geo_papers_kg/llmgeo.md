---
title: "LLMGeo: Benchmarking Large Language Models on Image Geolocation In-the-wild"
authors: Zhiqiang Wang, Dejia Xu, Rana Muhammad Shahroz Khan, Yanbin Lin, Zhiwen Fan, Xingquan Zhu
venue: arXiv preprint
year: 2024
arxiv: 2405.20363
topics: [benchmark, training-free geolocation, prompt-based, MLLM evaluation]
base_model: multiple LLMs/VLMs (GPT-4V, etc.)
benchmarks: [in-the-wild geolocation]
role: benchmark / training-free evaluation reference
---

# LLMGeo

Benchmarks **training-free / prompt-based** LLM and VLM geolocation in the wild. Table 3 reports training-free evaluation across different prompt settings. Establishes that prompt design materially affects geolocation accuracy.

## Relation to this thesis
- Supports the [[training-free-paradigm]] the thesis operates in (no fine-tuning).
- Confirms prompt sensitivity — relevant to the thesis's [[country-cue-prompting]] positive-cue experiments (driving-side, license-plate, text script).
- Companion evaluation papers: [[eval-geoloc-mllm]], [[systematic-benchmark-geoloc]], [[geobench-vlm]], [[geo-privacy]].
