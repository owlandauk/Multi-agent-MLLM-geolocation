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

## Fallback Thresholds Trade Continent for City

### Context
- A 1500-record offline sweep comparing `w015` with and without retrieval-country fallback showed that fallback rows improved continent more than country, but never improved city/street precision.
- On that sweep, `max_country_top=0.55` gave the best continent score, while `max_country_top=0.50` preserved more city/region accuracy and kept country accuracy similar.

### Root Cause / Core Insight
- Retrieval fallback is a coarse geocode repair, not a child-level localization repair. It helps when the original prediction is on the wrong continent, but it overwrites some otherwise-good city/street geocodes with country centers.

### The Pattern
- Next time the target metric is continent, use a wider fallback gate around `country_top < 0.55`; when city/region matter more, test `country_top < 0.50` before full runs.
- Signal to recognize: fallback improves many `2500km` misses but causes `City <25km` to drop, especially in the `country_top=0.50-0.55` bucket.

## Gates Need Enough Samples and Same-Hardware Confirmation

### Context
- 100/500-sample runs were too noisy; 1500-sample gates better predicted full-run behavior.
- i14s44 4x2080 Ti can run 7B with tensor parallelism, but i14s48 Blackwell is the right machine for full runs because it avoids multi-GPU communication overhead.

### Root Cause / Core Insight
- Small samples overreact to a few continent-heavy or landmark-heavy images. Hardware also changes runtime enough that failed jobs and slow checks can distort the optimization loop.

### The Pattern
- Next time, use 1500 as the decision gate, then run full only when both country and continent pass the threshold on the same code path.
- Signal to recognize: a 500 run looks promising but full later regresses, or a slow multi-GPU run spends more time on communication than useful inference.

## Relation-Aware Fallback Is the Current Balanced Choice

### Context
- A relation-aware fallback gate on i14s48 used `RETRIEVAL_PRIOR_WEIGHT=0.15`, `cross_continent country_top < 0.55`, and `same_continent country_top < 0.50`.
- The 1500 gate reached `5.40 / 15.73 / 26.80 / 46.33 / 72.33` for street/city/region/country/continent.
- The full run reached `5.22 / 14.93 / 25.86 / 46.80 / 73.19`, compared with the previous full `0.55` fallback at `5.09 / 14.70 / 25.73 / 46.25 / 73.26`.

### Root Cause / Core Insight
- Cross-continent retrieval disagreement is the high-value repair case: on full, applied cross-continent fallback had `51.71%` continent accuracy, while non-applied cross-continent rows had `31.97%`.
- Same-continent fallback is mostly a coarse repair and should stay conservative because it replaces potentially useful child geocodes with country centers.

### The Pattern
- Next time the target is balanced accuracy, prefer relation-aware fallback over a single global `0.55` gate.
- Signal to recognize: the relation-aware run raises country/city/region while losing only noise-level continent accuracy (`0.07` points on the full run).
