import math
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "geo_pipeline"))

from modules import sl


class SupportScoreTests(unittest.TestCase):
    def test_support_alpha_controls_s_c_n_likelihood_strength(self):
        text = "Observation: text clue.\nSupport: France=S; Canada=C; Germany=N"
        hypotheses = ["France", "Canada", "Germany"]

        strong = sl._parse_support_scores(text, hypotheses, support_alpha=0.7)
        mild = sl._parse_support_scores(text, hypotheses, support_alpha=0.35)

        self.assertGreater(strong["France"], mild["France"])
        self.assertLess(strong["Canada"], mild["Canada"])
        self.assertTrue(math.isclose(mild["Germany"], 1.0))


if __name__ == "__main__":
    unittest.main()
