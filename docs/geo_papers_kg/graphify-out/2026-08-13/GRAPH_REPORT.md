# Graph Report - geo_papers_kg  (2026-08-13)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 67 nodes · 261 edges · 8 communities
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `63823076`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Training-Free Geolocation Paradigm
- SL Module (Sampling-Likelihood)
- DST Module (Dempster-Shafer Fusion)
- POMDP Module (Action Selection)
- Qwen2.5-VL Technical Report
- Coarse-Subtraction Hypothesis
- GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating
- Thesis Pure-MLLM Pipeline

## God Nodes (most connected - your core abstractions)
1. `GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating` - 21 edges
2. `Thesis Pure-MLLM Pipeline` - 18 edges
3. `Training-Free Geolocation Paradigm` - 17 edges
4. `Coarse-Subtraction Hypothesis` - 16 edges
5. `SL Module (Sampling-Likelihood)` - 16 edges
6. `POMDP Module (Action Selection)` - 16 edges
7. `Qwen2.5-VL Technical Report` - 14 edges
8. `GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization` - 14 edges
9. `Retrieval-Geolocation Family` - 13 edges
10. `DST Module (Dempster-Shafer Fusion)` - 13 edges

## Surprising Connections (you probably didn't know these)
- `VLMs as GeoGuessr Masters: Fairness / Geographic Bias in VLM Geolocation` --semantically_similar_to--> `Where Do VLMs Fail? Regional Failure Analysis for Image Geolocation`  [INFERRED] [semantically similar]
  vlms-geoguessr-masters.md → where-vlms-fail.md
- `Vision-Conditioned Semantic Entropy (UniVRSE / VASE)` --semantically_similar_to--> `Behavioral Safety Audit of Vision-Language Models: Confidence on Wrong Answers`  [INFERRED] [semantically similar]
  vision-conditioned-semantic-entropy.md → vlm-safety-audit.md
- `Kuhn 2023 (Semantic Uncertainty)` --semantically_similar_to--> `Kadavath 2022 (Language Models Know What They Know)`  [INFERRED] [semantically similar]
  kuhn-semantic-uncertainty.md → kadavath-know.md
- `Kuhn 2023 (Semantic Uncertainty)` --semantically_similar_to--> `Kernel Language Entropy (Nikitin 2024)`  [INFERRED] [semantically similar]
  kuhn-semantic-uncertainty.md → kernel-language-entropy.md
- `Yager 1987 (New Combination Rules)` --semantically_similar_to--> `Smets Transferable Belief Model (TBM)`  [INFERRED] [semantically similar]
  yager-1987.md → smets-tbm.md

## Hyperedges (group relationships)
- **Active perception / POMDP information-seeking frameworks** — active_perception_game, active_perception_pomdp_survey, belief_space_planning [INFERRED 0.75]
- **Dempster-Shafer Combination Rule Lineage** — shafer_1976, yager_1987, smets_tbm, dst_module [INFERRED 0.75]
- **Privacy & bias in VLM geolocation** — ai_sees_location, geo_privacy, eval_geoloc_mllm [INFERRED 0.75]
- **Semantic Uncertainty Line** — kuhn_semantic_uncertainty, kernel_language_entropy, kadavath_know [INFERRED 0.75]
- **US Over-Prediction Bias Evidence** — pure_mllm_country_bias, where_vlms_fail, vlms_geoguessr_masters, regional_bias_vlm [INFERRED 0.75]
- **Dempster-Shafer classic evidence theory** — dempster_1967, shafer_1976, yager_1987, smets_tbm, dubois_prade_1992 [INFERRED 0.85]
- **Dempster-Shafer classic theory papers** — dempster_1967, dempster_shafer, dubois_prade_1992 [INFERRED 0.85]
- **LLM uncertainty quantification theory** — kuhn_semantic_uncertainty, kadavath_know, xiong_verbalized_confidence, semantic_entropy_probes, kernel_language_entropy [INFERRED 0.85]
- **POMDP / active-perception belief-space theory** — kaelbling_pomdp, platt_belief_space, perception_aware_pbvi, active_perception_pomdp_survey, belief_space_planning [INFERRED 0.85]
- **POMDP Foundations** — kaelbling_pomdp, integrated_control_pomdp, perception_aware_pbvi, platt_belief_space [INFERRED 0.85]
- **Retrieval Geolocation Family** — geoclip, georanker, pigeon, pinpoint, img2loc [INFERRED 0.85]
- **Thesis Pure-MLLM Pipeline Modules** — thesis_pure_mllm_pipeline, sl_module, dst_module, pomdp_module [INFERRED 0.85]

## Communities (8 total, 0 thin omitted)

### Community 0 - "Training-Free Geolocation Paradigm"
Cohesion: 0.37
Nodes (16): Country-Cue Prompting (Positive Cues), Evaluating MLLMs on Image Geolocation, Geo-Privacy: Image Geolocation Privacy Risks (PoPETs), Geo-R1 / Geo-RL: Reasoning-RL for Image Geolocation, GeoAgent: Agentic Tool-Use for Image Geolocation, GEOBench-VLM: A Benchmark for Geographic Understanding in VLMs, GeoCoT / GeoComp: Chain-of-Thought Geolocation with a Human-Comparison Benchmark, GeoLocSFT: Supervised Fine-Tuning for Image Geolocation (+8 more)

### Community 1 - "SL Module (Sampling-Likelihood)"
Cohesion: 0.65
Nodes (11): Consensus Entropy for Multi-Sample LLM Uncertainty, Consistency and Uncertainty in Vision-Language Geolocation, Kadavath 2022 (Language Models Know What They Know), Kernel Language Entropy (Nikitin 2024), Kuhn 2023 (Semantic Uncertainty), Semantic Entropy (Farquhar Nature 2024), Semantic Entropy Probes (Kossen 2024), SL Module (Sampling-Likelihood) (+3 more)

### Community 2 - "DST Module (Dempster-Shafer Fusion)"
Cohesion: 0.75
Nodes (9): Dempster 1967 (Upper and Lower Probabilities), On the Behavior of Dempster's Rule of Combination, Dempster-Shafer Theory of Evidence, On the Equivalence of Dempster-Shafer Fusion and Bayesian Updating, DST Module (Dempster-Shafer Fusion), Dubois-Prade 1992 (Combination of Evidence), Shafer 1976 (A Mathematical Theory of Evidence), Smets Transferable Belief Model (TBM) (+1 more)

### Community 3 - "POMDP Module (Action Selection)"
Cohesion: 0.68
Nodes (8): Active Perception as a Game: Information-Seeking under Uncertainty, Satsangi Active Perception (Submodular Value), Belief-Space Planning with Foundation Models, Integrated Control POMDP (2504.13288), Kaelbling-Littman-Cassandra 1998 (POMDP), Perception-Aware PBVI (IJCAI 2019), Platt 2010 (Belief-Space Planning, ML Observations), POMDP Module (Action Selection)

### Community 4 - "Qwen2.5-VL Technical Report"
Cohesion: 0.76
Nodes (7): CLIP: Learning Transferable Visual Models From Natural Language Supervision, GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization, PIGEON: Predicting Image Geolocations, PinPoint: Fine-Grained Image Geolocalization, Qwen2.5-VL Technical Report, Retrieval-Geolocation Family, StreetCLIP: Generalizable Zero-Shot Image Geolocalization

### Community 5 - "Coarse-Subtraction Hypothesis"
Cohesion: 0.93
Nodes (6): AI Sees Your Location: Privacy and Bias in VLM Geolocation, Coarse-Subtraction Hypothesis, Pure-MLLM Country Bias (US Over-Prediction), Regional Biases in Large Language and Vision-Language Models, VLMs as GeoGuessr Masters: Fairness / Geographic Bias in VLM Geolocation, Where Do VLMs Fail? Regional Failure Analysis for Image Geolocation

### Community 6 - "GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating"
Cohesion: 0.67
Nodes (6): G3: Geolocation via Guidebook Grounding / Generative Geo-localization, GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating, GeoCLIP-Rerank Line, GWS15k: Global Worldwide Street-View Benchmark, Im2GPS3K Benchmark, YFCC4K Benchmark

### Community 7 - "Thesis Pure-MLLM Pipeline"
Cohesion: 1.00
Nodes (4): GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization, GeoSearch: Web-Scale Retrieval-Augmented Image Geolocation, Img2Loc: Revisiting Image Geolocalization using Multimodal Foundation Models and Retrieval-Augmented Generation, Thesis Pure-MLLM Pipeline

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating` connect `GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating` to `Training-Free Geolocation Paradigm`, `SL Module (Sampling-Likelihood)`, `DST Module (Dempster-Shafer Fusion)`, `POMDP Module (Action Selection)`, `Qwen2.5-VL Technical Report`, `Coarse-Subtraction Hypothesis`, `Thesis Pure-MLLM Pipeline`?**
  _High betweenness centrality (0.199) - this node is a cross-community bridge._
- **Why does `POMDP Module (Action Selection)` connect `POMDP Module (Action Selection)` to `Training-Free Geolocation Paradigm`, `SL Module (Sampling-Likelihood)`, `DST Module (Dempster-Shafer Fusion)`, `Coarse-Subtraction Hypothesis`, `GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating`, `Thesis Pure-MLLM Pipeline`?**
  _High betweenness centrality (0.167) - this node is a cross-community bridge._
- **Why does `Coarse-Subtraction Hypothesis` connect `Coarse-Subtraction Hypothesis` to `Training-Free Geolocation Paradigm`, `SL Module (Sampling-Likelihood)`, `DST Module (Dempster-Shafer Fusion)`, `POMDP Module (Action Selection)`, `Qwen2.5-VL Technical Report`, `GeoBayes: Probabilistic Image Geo-Localization Inference via Sequential Bayesian Updating`, `Thesis Pure-MLLM Pipeline`?**
  _High betweenness centrality (0.161) - this node is a cross-community bridge._