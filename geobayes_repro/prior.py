"""GeoBayes bare-prior layer (paper Eq.5), isolated for reproduction.

This module contains ONLY the paper's calibrated-prior math and the parsing of
its hierarchical-location-name response. No SL/DST/POMDP, no Bayesian update
loop, no WebSearch/Enhance. The point is to test whether the paper's *bare*
country@750 = 50.7 on YFCC4K is reproducible on our own split, given only:

  1. the paper's hierarchical-location-name prompt (multi-hypothesis + confidence)
  2. the paper's prior  P0(l_i) = softmax( min(s_i, tau_p) / T )   (Eq.5)
     with T = 1.5 (temperature) and tau_p = 0.6 (confidence cutoff)
  3. argmax of that prior -> single country prediction -> geocode

Pure logic here (no torch, no PIL) so it imports and unit-tests on any Python.
"""

from __future__ import annotations

import math

# Paper hyperparameters (GeoBayes, AAAI-26, Implementation details p.9001 + Eq.5).
PRIOR_TEMP = 1.5      # T
PRIOR_CUTOFF = 0.6    # tau_p


def eq5_prior(
    scores: dict[str, float],
    temp: float = PRIOR_TEMP,
    cutoff: float = PRIOR_CUTOFF,
) -> dict[str, float]:
    """Paper Eq.5: P0(l_i) = exp(min(s_i, tau_p)/T) / sum_j exp(min(s_j, tau_p)/T).

    `scores` maps a location name -> its raw confidence s_i in [0, 1].
    Truncation at tau_p smooths overconfident predictions before the softmax.
    Returns a normalized distribution over the same keys. Empty input -> {}.
    """
    if not scores:
        return {}
    logits = {name: min(s, cutoff) / temp for name, s in scores.items()}
    m = max(logits.values())
    exps = {name: math.exp(v - m) for name, v in logits.items()}
    z = sum(exps.values())
    return {name: v / z for name, v in exps.items()}


def argmax_prior(prior: dict[str, float]) -> str | None:
    """MAP pick over the prior. None if empty."""
    if not prior:
        return None
    return max(prior.items(), key=lambda kv: kv[1])[0]
