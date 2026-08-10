# Optimization Roadmap: GeoBayes-Style 7B Pipeline

This file is the durable next-step plan for continuing optimization. Use it at the start of each new session before launching experiments.

## Current Best Baseline

Use this as the main comparison point unless a later full run replaces it.

Run: `i14s48_geoclip_w015_country_fallback_relgate_full`

Settings:
- Model: `Qwen2.5-VL-7B-Instruct`
- Server: `i14s48`, Blackwell GPU, batch size `4`
- `WEB_SEARCH_ENABLED=0`
- `RETRIEVAL_PRIOR_ENABLED=1`
- `RETRIEVAL_PRIOR_WEIGHT=0.15`
- `--retrieval_country_fallback`
- `--retrieval_country_max_country_top 0.55`
- `--retrieval_country_same_continent_max_country_top 0.50`
- `--retrieval_country_cross_continent_max_country_top 0.55`

Full result on YFCC4K, 4536 images:

| Metric | Accuracy |
|---|---:|
| Street <1km | 5.22 |
| City <25km | 14.93 |
| Region <200km | 25.86 |
| Country <750km | 46.80 |
| Continent <2500km | 73.19 |

Previous continent-only reference: `i14s48_geoclip_w015_country_fallback_full_retry1` reached `5.09 / 14.70 / 25.73 / 46.25 / 73.26`. The relation-aware run is preferred because it improves street, city, region, and country while losing only `0.07` continent points.

## Decision Rules

Run `--limit 1500` before every full run. Do not launch full from 100 or 500 results.

Promote a 1500 experiment to full only if it satisfies at least one of these conditions:

1. Balanced improvement:
   - Country `>= 46.6`
   - Region `>= 26.8`
   - Continent `>= 72.3`
   - City `>= 15.2`

2. Country/region breakthrough:
   - Country improves by `+0.7` over the current 1500 reference, or
   - Region improves by `+0.8`,
   - and Continent does not fall below `71.8`.

3. Continent breakthrough:
   - Continent `>= 73.0` on 1500,
   - and Country `>= 45.8`,
   - and City `>= 14.8`.

Reject or revise an experiment when:
- City drops below `14.5` on 1500.
- Country drops below `45.5` on 1500.
- Continent drops below `71.5` unless region/city improves strongly and the run is explicitly a fine-grained ablation.
- `country_child_conflict_rate` rises above `28%` without a matching gain in country/continent.

## Step 1: Probability Thought Verification

Status: tested on 1500 records and not promoted to full.

Results:
- `i14s48_verify_support_relgate_limit1500`, `VERIFY_SUPPORT_FORMAT=1`, default `SL_SUPPORT_ALPHA=0.7`: `5.47 / 14.87 / 25.00 / 43.40 / 69.60`.
- `i14s48_verify_support_alpha035_relgate_limit1500`, `VERIFY_SUPPORT_FORMAT=1`, `SL_SUPPORT_ALPHA=0.35`: `5.27 / 14.73 / 26.13 / 45.67 / 71.40`.

Conclusion: the current support-line prompt over-confidently reinforces wrong country hypotheses on Qwen2.5-VL-7B. Keep `SL_SUPPORT_ALPHA` for future ablations, but do not use `VERIFY_SUPPORT_FORMAT=1` as the main path unless the prompt/parser is redesigned and diagnostics show better calibration.

Goal: make the verification stage closer to GeoBayes Fig. 1d by asking the model to rate the same evidence against every candidate as `S/C/N`, then using the centered likelihood parser already implemented in `SLModule`.

Hypothesis: this should improve country posterior calibration because contradictions are represented explicitly instead of being inferred through separate free-form per-hypothesis ratings.

Command changes from current best:

```bash
export VERIFY_SUPPORT_FORMAT=1
```

1500 target:

| Metric | Must beat / keep |
|---|---:|
| Street <1km | >= 5.0 |
| City <25km | >= 15.2 |
| Region <200km | >= 26.8 |
| Country <750km | >= 46.6 |
| Continent <2500km | >= 72.3 |

Full-worthy if it hits the balanced improvement rule or clearly raises country/region without hurting continent. If it fails, inspect whether `Support:` parsing is actually firing; if parsing is weak, improve prompt/parse before abandoning the idea.

## Step 2: Retrieval Neighbor Candidate Reranking

Goal: move retrieval from a final geocode fallback into the candidate-generation and posterior-update path.

Rationale from papers:
- GeoBayes gains from external clues only when they are converted into per-candidate likelihoods.
- RAG street-level papers get fine-grained gains by giving MLLMs similar and dissimilar retrieval examples, not just a country prior.

Implementation direction:
- Extend the GeoCLIP prior cache, if available, to expose top neighbor coordinates/countries/cities.
- Build candidate country/city sets from top retrieval neighbors.
- Ask the MLLM to verify these candidates with `S/C/N` evidence.
- Fuse through existing SL/DST/POMDP; do not directly replace the final answer.

1500 target:

| Metric | Must beat / keep |
|---|---:|
| Street <1km | >= 5.0 |
| City <25km | >= 15.5 |
| Region <200km | >= 27.0 |
| Country <750km | >= 47.0 |
| Continent <2500km | >= 72.0 |

Full-worthy if country reaches `>= 47.3` or region reaches `>= 27.5` while city remains `>= 15.0`.

## Step 3: Strict Object-Triggered Web/Image Search

Goal: reproduce the useful part of GeoBayes WebSearch without broad noisy search.

Strict conditions:
- Search only when there is a concrete visual/text entity: landmark, statue, road sign, shop name, phone number, route number, license plate clue, or explicit local language text.
- Search only when `country_stable=False`, `country_top < 0.65`, or city/street evidence is low-confidence.
- Use ImageSearch/Google Lens first when available; use Tavily only as fallback for a concrete entity query.
- Never replace the answer directly. Convert snippets into `S/C/N` support over existing candidates and fuse with DST.

1500 target:

| Metric | Must beat / keep |
|---|---:|
| Street <1km | >= 5.2 |
| City <25km | >= 15.5 |
| Region <200km | >= 26.8 |
| Country <750km | >= 46.8 |
| Continent <2500km | >= 72.5 |

Full-worthy if search enhancement rate stays small and high-quality, roughly `<10%` of records, while those enhanced records outperform non-enhanced low-confidence records.

## Step 4: True POMDP Reward Ablation

Goal: make the POMDP contribution real in the main experiment by using expected reward instead of the lightweight LLM policy.

Command changes:

```bash
export POMDP_POLICY=eig
export POMDP_REWARD_MODE=map_gain
export POMDP_EIG_LEVELS=country,city
```

Run a smaller sanity test first because this can be slower:
- First `--limit 300`
- Then `--limit 1500` only if runtime and metrics look sane.

300 sanity target:
- No runtime explosion beyond roughly `2x` per image.
- Country and continent should not collapse versus the same 300 baseline.

1500 target:

| Metric | Must beat / keep |
|---|---:|
| Street <1km | >= 5.0 |
| City <25km | >= 15.0 |
| Region <200km | >= 26.5 |
| Country <750km | >= 46.8 |
| Continent <2500km | >= 72.5 |

Full-worthy if the POMDP policy improves country/region or materially lowers country-child conflict without a large runtime penalty.

## Server Run Template

Use i14s48 when available.

```bash
cd /home/szuo/geo_i14s48_main_20260808_231711/geo_pipeline
export PATH=/cvhci/temp/szuo/vllm-blackwell-env/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export VLLM_TP=1
export MLLM_BACKEND=vllm
export MODEL_PATH=/cvhci/temp/szuo/models/qwen2.5-vl-7b
export WEB_SEARCH_ENABLED=0
export RETRIEVAL_PRIOR_ENABLED=1
export RETRIEVAL_PRIOR_PATH=/home/szuo/Multi-agent-MLLM-geolocation/geo_pipeline/results/geoclip_prior_full.json
export RETRIEVAL_PRIOR_WEIGHT=0.15

/cvhci/temp/szuo/vllm-blackwell-env/bin/python run_experiment.py \
  --name <experiment_name> \
  --limit 1500 \
  --batch_size 4 \
  --out results/<experiment_name>.json \
  --log results/<experiment_name>.log \
  --retrieval_country_fallback \
  --retrieval_country_max_country_top 0.55 \
  --retrieval_country_min_prior_top 0.15 \
  --retrieval_country_same_continent_max_country_top 0.50 \
  --retrieval_country_cross_continent_max_country_top 0.55 \
  --notes <short_notes>
```

After any run:

```bash
/cvhci/temp/szuo/vllm-blackwell-env/bin/python analyze_results.py --pred results/<experiment_name>.json \
  > results/<experiment_name>.analysis.txt
tail -n 5 results/experiment_runs.csv
sed -n '1,220p' results/<experiment_name>.analysis.txt
```

## Session Entry Checklist

At the start of the next session:

1. Check whether a run finished: `tail -n 8 results/experiment_runs.csv`.
2. Compare against the Current Best Baseline and the step-specific gate.
3. If the 1500 gate passes, launch full and stop after confirming it is running normally.
4. If the gate fails, inspect diagnostics before moving to the next roadmap step.
5. Record the result in this file or `docs/experiment-lessons-2026-08-08.md` before committing.
