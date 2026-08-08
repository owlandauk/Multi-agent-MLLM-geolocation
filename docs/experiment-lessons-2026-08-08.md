# Experiment Lessons: Retrieval-Backed GeoBayes

## Strong Prior Beats More Free Search

### Context
- The best gains came from adding a retrieval/GeoCLIP country prior and using it only when the visual country posterior was weak.
- Tavily/WebSearch and ImageSearch helped individual landmark cases, but broad web enhancement often injected noisy entities and did not improve aggregate YFCC4K accuracy.

### Root Cause / Core Insight
- GeoBayes-style updates are sensitive to the initial hypothesis set and geocoding target. A weak or wrong country prior sends city/street reasoning down the wrong branch, while external text snippets are too sparse and noisy to reliably repair that branch later.

### The Pattern
- Next time continent accuracy is the bottleneck, first strengthen the coarse prior and low-confidence geocode fallback before adding more search calls.
- Signal to recognize: country posterior top mass is low, North America false positives are high, and precise street geocodes are hurting continent distance.

## Evaluation Fallback Is Not the Same as Pipeline Belief

### Context
- `RETRIEVAL_PRIOR_WEIGHT=0.10` plus `retrieval_country_fallback` lifted full continent accuracy from the prior-only baseline, while preserving most street/city performance.
- On 1500 samples, `w010` reached about `45.20` country and `71.13` continent on i14s44, while historical `w015` reached about `45.87` country and `72.40` continent.

### Root Cause / Core Insight
- Replacing the pipeline posterior too early increases country-child conflict and can damage city/street descent. Using retrieval as an evaluation-side fallback for low-confidence country predictions gives most of the continent benefit without forcing every downstream child prediction to obey retrieval.

### The Pattern
- Next time a retrieval prior conflicts with visual evidence, prefer a soft prior plus low-confidence fallback over hard anchoring inside the hierarchy.
- Signal to recognize: `country_child_conflict_rate` jumps above roughly 25%, but `retrieval_country_fallback` rows have higher continent recall than the original low-confidence geocode.

## Gates Need Enough Samples and Same-Hardware Confirmation

### Context
- 100/500-sample runs were too noisy; 1500-sample gates better predicted full-run behavior.
- i14s44 4x2080 Ti can run 7B with tensor parallelism, but i14s48 Blackwell is the right machine for full runs because it avoids multi-GPU communication overhead.

### Root Cause / Core Insight
- Small samples overreact to a few continent-heavy or landmark-heavy images. Hardware also changes runtime enough that failed jobs and slow checks can distort the optimization loop.

### The Pattern
- Next time, use 1500 as the decision gate, then run full only when both country and continent pass the threshold on the same code path.
- Signal to recognize: a 500 run looks promising but full later regresses, or a slow multi-GPU run spends more time on communication than useful inference.

