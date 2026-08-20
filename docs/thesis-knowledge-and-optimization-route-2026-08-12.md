# Thesis Knowledge and Optimization Route

Date: 2026-08-12

## Core Position

The thesis should not claim that a Qwen-7B pipeline can beat GeoCLIP from
scratch. GeoCLIP is already a strong worldwide image-to-GPS retrieval model.
The thesis contribution should be framed as improving GeoCLIP-style retrieval
with uncertainty-aware MLLM reasoning:

```text
GeoCLIP top-k GPS candidates
  -> MLLM verification
  -> SL uncertainty-aware likelihood
  -> DST conflict-aware fusion
  -> POMDP adaptive evidence selection
  -> GeoKG consistency constraints
  -> reranked coordinate posterior
```

## Lessons From Papers

### GeoCLIP / GeoSURGE

- Strong geographic candidate generation is the load-bearing component.
- Do not collapse GeoCLIP top-k coordinates into country-only priors unless the
  goal is only coarse fallback.
- GeoSURGE shows that hierarchy-aware geographic embeddings and semantic fusion
  outperform plain GeoCLIP, which reinforces the need to keep multi-scale
  candidates instead of asking an MLLM to invent locations globally.

### GeoBayes

- Use a Hypothesize-Verify-Update loop.
- Keep centered Bayesian multiplicative update as the default evidence fusion.
- Per-candidate verification is important because free-form reasoning tends to
  confirm the current top guess.

### SL

- Single-source evidence can be uncertain or noisy.
- High-variance evidence should be shrunk toward neutral likelihood rather than
  allowed to dominate the posterior.

### DST

- Dempster-Shafer/Yager-style cautious fusion is useful only under high conflict.
- It should prevent overconfident collapse, not replace Bayesian multiplication
  everywhere.

### POMDP

- State is hidden true location; belief is the current posterior.
- Actions are verification choices over candidates, levels, visual cues, web
  clues, or GeoKG relations.
- Reward should combine entropy reduction, MAP gain, margin gain, and action
  cost. This is implemented in `modules/pomdp.py` with `POMDP_REWARD_MODE=combined`.

### GeoKG

- GeoKG is a constraint/validator, not a direct answer generator.
- Useful checks include city-country membership, landmark-country membership,
  script/language-country compatibility, road-side priors, and administrative
  hierarchy.

## Optimization Route

### Stage 1: Establish GeoCLIP Truth Baselines

Required before tuning:

```bash
python build_geoclip_candidate_cache.py --top_k 25 --out results/geoclip_candidates.json --resume
python evaluate_geoclip_candidates.py --candidate_cache results/geoclip_candidates.json --out results/geoclip_direct_oracle.json
```

Metrics to report:

- GeoCLIP direct top-1.
- GeoCLIP oracle@5 / oracle@10 / oracle@25.

If oracle@k is much higher than top-1, reranking has real headroom. If oracle@k
is weak, switch to a stronger candidate generator such as GeoSURGE/PIGEOTTO/RFM.

### Stage 2: Do-No-Harm Reranking

Use GeoCLIP top-1 as the default final prediction. SL-DST-POMDP may override it
only when the reranked posterior has a strong margin.

The first conservative rerank baseline is:

```bash
python evaluate_geoclip_qwen_rerank.py \
  --candidate_cache results/geoclip_candidates.json \
  --top_k 5 \
  --min_override_top 0.55 \
  --min_override_margin 0.08 \
  --out results/geoclip_qwen_rerank_limit1500.json \
  --limit 1500
```

Success condition:

```text
Hybrid >= GeoCLIP direct top-1 overall
```

or, at minimum, hybrid improves clear failure buckets without reducing aggregate
accuracy.

### Stage 3: Evidence Sources

Add evidence only if ablations prove it helps:

- MLLM visual verification over GeoCLIP candidates.
- GeoKG consistency checks.
- Google Lens / image search only for concrete entities.
- Tavily only for strict entity/text queries, not generic scene descriptions.

### Stage 4: Thesis Ablations

Required ablations:

- GeoCLIP direct top-1.
- GeoCLIP top-k oracle.
- GeoCLIP + Qwen rerank.
- GeoCLIP + SL.
- GeoCLIP + SL + DST.
- GeoCLIP + SL + DST + POMDP.
- GeoCLIP + SL + DST + POMDP + GeoKG.

## Default Answering Rule

When planning future work, start from these checks:

1. Does the change preserve GeoCLIP top-k coordinate information?
2. Does it have a do-no-harm fallback to GeoCLIP top-1?
3. Is the improvement measured against GeoCLIP direct and oracle@k?
4. Does the change fit SL, DST, POMDP, or GeoKG rather than adding unrelated complexity?
5. Can it be tested first on 1500 samples before full YFCC4K?
