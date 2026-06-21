"""
DST — Multi-source Evidence Fusion via Dempster-Shafer Theory

GeoBayes fuses evidence with a naive product ∏W(e_t|l), which assumes
conditional independence and breaks down when two clues conflict strongly.

This module replaces that product with DST combination:
  - Each evidence item defines a Basic Belief Assignment (BBA) over the
    hypothesis set Θ = {l_1, ..., l_k, Θ} (the last element is the "ignorance" mass).
  - Multiple BBAs are combined with Dempster's rule (with conflict normalization).
  - When global conflict K > DST_CONFLICT_THR, we fall back to the cautious
    (Yager's) rule that routes conflict mass to ignorance rather than renormalizing.

Output: posterior probability distribution over hypotheses.
"""

from __future__ import annotations
import numpy as np
from config import DST_CONFLICT_THR


def _normalize_bba(bba: dict) -> dict:
    total = sum(bba.values())
    return {k: v / total for k, v in bba.items()} if total > 0 else bba


def likelihood_to_bba(
    w_scores: dict[str, float],
    theta_key: str = "__ignorance__",
    base_ignorance: float = 0.1,
) -> dict[str, float]:
    """
    Convert W_sl scores into a BBA.
    W > 1 → supporting evidence  → high focal mass on that hypothesis
    W < 1 → contradicting evidence → mass shifted away
    W = 1 → neutral              → mass goes to ignorance

    Strategy: softmax over W scores → hypothesis masses, then mix with a
    fixed ignorance mass to model MLLM uncertainty.
    """
    hyps = list(w_scores.keys())
    ws   = np.array([w_scores[h] for h in hyps], dtype=float)

    # W scores are already exp-scaled likelihoods — L1-normalise directly.
    # A second softmax would compress Country/Continent-level gaps where W
    # values are close (e.g. 1.3 vs 1.1), making the posterior nearly uniform.
    ws_clipped = np.clip(ws, 1e-9, None)
    probs = ws_clipped / ws_clipped.sum()

    bba = {h: float(p) * (1.0 - base_ignorance) for h, p in zip(hyps, probs)}
    bba[theta_key] = base_ignorance
    return bba


def dempster_combine(bba1: dict, bba2: dict, cautious_threshold: float = DST_CONFLICT_THR) -> dict:
    """
    Combine two BBAs using Dempster's rule.
    Falls back to Yager's cautious rule if conflict K > cautious_threshold.
    Both BBAs must share the same hypothesis keys + __ignorance__ key.
    """
    theta = "__ignorance__"
    hyps  = [k for k in bba1 if k != theta]

    combined: dict[str, float] = {h: 0.0 for h in hyps}
    combined[theta] = 0.0
    K = 0.0  # conflict mass

    all_keys = hyps + [theta]  # noqa: F841

    for k1, m1 in bba1.items():
        for k2, m2 in bba2.items():
            product = m1 * m2
            if k1 == k2:
                combined[k1] = combined.get(k1, 0.0) + product
            elif k1 == theta:
                combined[k2] = combined.get(k2, 0.0) + product
            elif k2 == theta:
                combined[k1] = combined.get(k1, 0.0) + product
            else:
                # conflict: two different singleton hypotheses
                K += product

    if K > cautious_threshold:
        # Yager: redirect conflict to ignorance
        combined[theta] = combined.get(theta, 0.0) + K
    elif K < 1.0:
        # standard Dempster normalization
        factor = 1.0 / (1.0 - K)
        combined = {k: v * factor for k, v in combined.items()}
    # K == 1.0 is total conflict — return uniform
    else:
        n = len(hyps)
        combined = {h: 1.0 / n for h in hyps}
        combined[theta] = 0.0

    return _normalize_bba(combined)


class DSTModule:
    """
    Fuses a list of per-evidence W_sl score dicts into a posterior distribution.
    Replaces GeoBayes's naive ∏W product.
    """

    def fuse(
        self,
        prior: dict[str, float],
        evidence_scores: list[dict[str, float]],
    ) -> dict[str, float]:
        """
        Args:
            prior:           {hypothesis: probability} from SL / GeoBayes prior step
            evidence_scores: list of {hypothesis: W_sl} dicts, one per evidence item

        Returns:
            posterior:       {hypothesis: probability} (ignorance key removed)
        """
        # initialise BBA from prior (prior already sums to 1, add small ignorance)
        theta = "__ignorance__"
        hyps  = list(prior.keys())
        bba   = {h: prior[h] * 0.9 for h in hyps}
        bba[theta] = 0.1

        for w_scores in evidence_scores:
            new_bba = likelihood_to_bba(w_scores, theta_key=theta)
            bba = dempster_combine(bba, new_bba)

        # strip ignorance mass → renormalize over hypotheses only
        posterior = {h: bba.get(h, 0.0) for h in hyps}
        total = sum(posterior.values())
        if total > 0:
            posterior = {h: v / total for h, v in posterior.items()}
        return posterior
