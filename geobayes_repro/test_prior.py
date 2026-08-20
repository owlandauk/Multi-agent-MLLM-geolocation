"""Unit tests for the GeoBayes Eq.5 bare-prior math (pure logic, no GPU/PIL)."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from prior import eq5_prior, argmax_prior, PRIOR_TEMP, PRIOR_CUTOFF


class Eq5PriorTests(unittest.TestCase):
    def test_matches_hand_computed_softmax(self):
        # Two candidates below the cutoff: straight softmax of s_i / T.
        scores = {"France": 0.4, "Spain": 0.2}
        prior = eq5_prior(scores)
        lf = 0.4 / PRIOR_TEMP
        ls = 0.2 / PRIOR_TEMP
        z = math.exp(lf) + math.exp(ls)
        self.assertAlmostEqual(prior["France"], math.exp(lf) / z, places=6)
        self.assertAlmostEqual(prior["Spain"], math.exp(ls) / z, places=6)
        self.assertAlmostEqual(sum(prior.values()), 1.0, places=6)

    def test_truncation_at_cutoff(self):
        # Confidences above tau_p=0.6 are clipped, so 0.9 and 0.6 become equal.
        scores = {"United States": 0.9, "Canada": 0.6}
        prior = eq5_prior(scores)
        self.assertAlmostEqual(prior["United States"], prior["Canada"], places=6)
        self.assertAlmostEqual(prior["United States"], 0.5, places=6)

    def test_truncation_flattens_overconfidence(self):
        # A 0.99 vs 0.5 pair: 0.99 clips to 0.6, 0.5 stays -> smaller gap than raw.
        prior = eq5_prior({"US": 0.99, "Mexico": 0.5})
        raw_gap = math.exp(0.99 / PRIOR_TEMP) / (
            math.exp(0.99 / PRIOR_TEMP) + math.exp(0.5 / PRIOR_TEMP)
        )
        self.assertLess(prior["US"], raw_gap)  # clipped mass is lower than raw

    def test_argmax_picks_highest(self):
        prior = eq5_prior({"Japan": 0.5, "China": 0.3, "Korea": 0.2})
        self.assertEqual(argmax_prior(prior), "Japan")

    def test_empty_yields_empty_and_none(self):
        self.assertEqual(eq5_prior({}), {})
        self.assertIsNone(argmax_prior({}))

    def test_single_candidate_is_certain(self):
        prior = eq5_prior({"Brazil": 0.3})
        self.assertEqual(prior, {"Brazil": 1.0})

    def test_all_equal_confidence_uniform(self):
        prior = eq5_prior({"A": 0.5, "B": 0.5, "C": 0.5})
        for v in prior.values():
            self.assertAlmostEqual(v, 1 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
