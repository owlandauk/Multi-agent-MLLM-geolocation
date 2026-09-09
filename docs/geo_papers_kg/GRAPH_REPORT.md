# Graph Report - geo_papers_kg  (2026-08-13)

## Corpus Check
- Corpus is ~7,534 words - fits in a single context window. You may not need a graph.

## Summary
- 63 nodes · 232 edges · 8 communities
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Training-Free Geolocation Paradigm
- SL Module (Sampling-Likelihood)
- DST Module (Dempster-Shafer Fusion)
- POMDP Module (Action Selection)
- Qwen2.5-VL Technical Report
- GeoBayes
- Coarse-Subtraction Hypothesis
- Thesis Pure-MLLM Pipeline

## God Nodes (most connected - your core abstractions)
1. `GeoBayes` - 21 edges
2. `Thesis Pure-MLLM Pipeline` - 17 edges
3. `Training-Free Geolocation Paradigm` - 17 edges
4. `POMDP Module (Action Selection)` - 16 edges
5. `SL Module (Sampling-Likelihood)` - 15 edges
6. `GeoCLIP` - 14 edges
7. `Qwen2.5-VL Technical Report` - 13 edges
8. `Retrieval-Geolocation Family` - 13 edges
9. `DST Module (Dempster-Shafer Fusion)` - 13 edges
10. `Coarse-Subtraction Hypothesis` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Where Do VLMs Fail? Regional Failure Analysis for Image Geolocation` --semantically_similar_to--> `Regional Biases in Large Language and Vision-Language Models`  [INFERRED] [semantically similar]
  where-vlms-fail.md → regional-bias-vlm.md
- `Perception-Aware PBVI (IJCAI 2019)` --semantically_similar_to--> `Satsangi Active Perception (Submodular Value)`  [INFERRED] [semantically similar]
  perception-aware-pbvi.md → active-perception-pomdp-survey.md
- `Kuhn 2023 (Semantic Uncertainty)` --semantically_similar_to--> `Kernel Language Entropy (Nikitin 2024)`  [INFERRED] [semantically similar]
  kuhn-semantic-uncertainty.md → kernel-language-entropy.md
- `Yager 1987 (New Combination Rules)` --semantically_similar_to--> `Smets Transferable Belief Model (TBM)`  [INFERRED] [semantically similar]
  yager-1987.md → smets-tbm.md
- `Active Perception as a Game` --conceptually_related_to--> `Integrated Perception and Control via POMDP`  [EXTRACTED]
  active-perception-game.md → integrated-control-pomdp.md

## Hyperedges (group relationships)
- **Multi-Sample Uncertainty / Self-Consistency Cluster** — consensus_entropy, consistency_uncertainty, sl_module [INFERRED 0.75]
- **Privacy & Bias in VLM Geolocation Cluster** — ai_sees_location, geo_privacy, eval_geoloc_mllm [INFERRED 0.75]
- **Uncertainty / Overconfidence Calibration Evidence** — semantic_entropy, vlm_safety_audit, sl_module, consistency_uncertainty [INFERRED 0.75]
- **Dempster-Shafer classic evidence theory** — dempster_1967, shafer_1976, yager_1987, smets_tbm, dubois_prade_1992 [INFERRED 0.85]
- **DST Evidence-Fusion Theory Cluster** — dempster_shafer, dempster_behavior, dst_bayes_equivalence, dst_module [INFERRED 0.85]
- **LLM uncertainty quantification theory** — kuhn_semantic_uncertainty, kadavath_know, xiong_verbalized_confidence, semantic_entropy_probes, kernel_language_entropy [INFERRED 0.85]
- **POMDP / active-perception belief-space theory** — kaelbling_pomdp, platt_belief_space, perception_aware_pbvi, active_perception_pomdp_survey, belief_space_planning [INFERRED 0.85]
- **Thesis Pure-MLLM Pipeline Modules** — sl_module, dst_module, pomdp_module, thesis_pure_mllm_pipeline [INFERRED 0.85]
- **US / North-America Over-Prediction Bias Evidence** — pure_mllm_country_bias, where_vlms_fail, regional_bias_vlm [INFERRED 0.85]

## Communities (8 total, 0 thin omitted)

### Community 0 - "Training-Free Geolocation Paradigm"
Cohesion: 0.40
Nodes (14): Country-Cue Prompting (Positive Cues), Evaluating MLLMs on Image Geolocation, Geo-Privacy: Image Geolocation Privacy Risks (PoPETs), Geo-R1 / Geo-RL: Reasoning-RL for Image Geolocation, GeoAgent: Agentic Tool-Use for Image Geolocation, GEOBench-VLM: A Benchmark for Geographic Understanding in VLMs, GeoCoT / GeoComp: Chain-of-Thought Geolocation, GeoLocSFT: Supervised Fine-Tuning for Image Geolocation (+6 more)

### Community 1 - "SL Module (Sampling-Likelihood)"
Cohesion: 0.64
Nodes (10): Consensus Entropy for Multi-Sample LLM Uncertainty, Consistency and Uncertainty in VLM Geolocation, Kadavath 2022 (Language Models Know What They Know), Kernel Language Entropy (Nikitin 2024), Kuhn 2023 (Semantic Uncertainty), Semantic Entropy (Farquhar Nature 2024), Semantic Entropy Probes (Kossen 2024), SL Module (Sampling-Likelihood) (+2 more)

### Community 2 - "DST Module (Dempster-Shafer Fusion)"
Cohesion: 0.75
Nodes (9): Dempster 1967 (Upper and Lower Probabilities), On the Behavior of Dempster's Rule (Dezert), Dempster-Shafer Theory of Evidence, On the Equivalence of Dempster-Shafer Fusion and Bayesian Updating, DST Module (Dempster-Shafer Fusion), Dubois-Prade 1992 (Combination of Evidence), Shafer 1976 (A Mathematical Theory of Evidence), Smets Transferable Belief Model (TBM) (+1 more)

### Community 3 - "POMDP Module (Action Selection)"
Cohesion: 0.68
Nodes (8): Active Perception as a Game, Satsangi Active Perception (Submodular Value), Belief-Space Planning with Foundation Models, Integrated Perception and Control via POMDP, Kaelbling-Littman-Cassandra 1998 (POMDP), Perception-Aware PBVI (IJCAI 2019), Platt 2010 (Belief-Space Planning, ML Observations), POMDP Module (Action Selection)

### Community 4 - "Qwen2.5-VL Technical Report"
Cohesion: 0.76
Nodes (7): CLIP, GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization, PIGEON: Predicting Image Geolocations, PinPoint: Fine-Grained Image Geolocalization, Qwen2.5-VL Technical Report, Retrieval-Geolocation Family, StreetCLIP: Generalizable Zero-Shot Image Geolocalization

### Community 5 - "GeoBayes"
Cohesion: 0.67
Nodes (6): G3: Multi-Stage Retrieval+Generation Geolocation, GeoBayes, GeoCLIP-Rerank Line, GWS15k: Global Worldwide Street-View Benchmark, Im2GPS3K Benchmark, YFCC4K Benchmark

### Community 6 - "Coarse-Subtraction Hypothesis"
Cohesion: 0.80
Nodes (5): AI Sees Your Location, Coarse-Subtraction Hypothesis, Pure-MLLM Country Bias (US Over-Prediction), Regional Biases in Large Language and Vision-Language Models, Where Do VLMs Fail? Regional Failure Analysis for Image Geolocation

### Community 7 - "Thesis Pure-MLLM Pipeline"
Cohesion: 1.00
Nodes (4): GeoCLIP, GeoSearch: Web-Scale Retrieval-Augmented Image Geolocation, Img2Loc: Revisiting Image Geolocalization using Multimodal Foundation Models and Retrieval-Augmented Generation, Thesis Pure-MLLM Pipeline

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GeoBayes` connect `GeoBayes` to `Training-Free Geolocation Paradigm`, `SL Module (Sampling-Likelihood)`, `DST Module (Dempster-Shafer Fusion)`, `POMDP Module (Action Selection)`, `Qwen2.5-VL Technical Report`, `Coarse-Subtraction Hypothesis`, `Thesis Pure-MLLM Pipeline`?**
  _High betweenness centrality (0.233) - this node is a cross-community bridge._
- **Why does `POMDP Module (Action Selection)` connect `POMDP Module (Action Selection)` to `Training-Free Geolocation Paradigm`, `SL Module (Sampling-Likelihood)`, `DST Module (Dempster-Shafer Fusion)`, `GeoBayes`, `Coarse-Subtraction Hypothesis`, `Thesis Pure-MLLM Pipeline`?**
  _High betweenness centrality (0.181) - this node is a cross-community bridge._
- **Why does `Training-Free Geolocation Paradigm` connect `Training-Free Geolocation Paradigm` to `POMDP Module (Action Selection)`, `Qwen2.5-VL Technical Report`, `GeoBayes`, `Coarse-Subtraction Hypothesis`, `Thesis Pure-MLLM Pipeline`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._