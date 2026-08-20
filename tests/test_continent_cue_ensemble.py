import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "geo_pipeline"))

import pipeline


class MergeContinentScoresTests(unittest.TestCase):
    def _hyp(self, pairs):
        return {"hypotheses": [{"location": loc, "confidence": c} for loc, c in pairs]}

    def test_max_across_sources(self):
        # Same continent seen in two sources at different confidence -> keep the max.
        sources = [
            self._hyp([("Europe", 0.4)]),
            self._hyp([("Europe", 0.7)]),
        ]
        merged = pipeline._merge_continent_scores(sources)
        # max 0.7, plus a +0.05 multi-source bonus (2 sources -> +0.05).
        self.assertAlmostEqual(merged["Europe"], 0.75, places=6)

    def test_single_source_no_bonus(self):
        merged = pipeline._merge_continent_scores([self._hyp([("Asia", 0.6)])])
        self.assertAlmostEqual(merged["Asia"], 0.6, places=6)

    def test_multi_source_bonus_accumulates(self):
        sources = [
            self._hyp([("South America", 0.5)]),
            self._hyp([("South America", 0.5)]),
            self._hyp([("South America", 0.5)]),
        ]
        merged = pipeline._merge_continent_scores(sources)
        # 3 sources -> +0.05*(3-1) = +0.10.
        self.assertAlmostEqual(merged["South America"], 0.60, places=6)

    def test_non_continent_strings_dropped(self):
        # Country / region strings must be dropped by _collect_scores(level="continent").
        sources = [self._hyp([("France", 0.9), ("Southeast Asia", 0.8), ("Europe", 0.6)])]
        merged = pipeline._merge_continent_scores(sources)
        self.assertIn("Europe", merged)
        self.assertNotIn("France", merged)
        self.assertNotIn("Southeast Asia", merged)

    def test_continent_aliases_canonicalized(self):
        # "North America" variants collapse to one canonical key.
        sources = [
            self._hyp([("Northern America", 0.5)]),
            self._hyp([("North America", 0.6)]),
        ]
        merged = pipeline._merge_continent_scores(sources)
        self.assertEqual(set(merged.keys()), {"North America"})
        self.assertAlmostEqual(merged["North America"], 0.65, places=6)

    def test_empty_and_malformed_ignored(self):
        merged = pipeline._merge_continent_scores([None, {}, {"nope": 1}])
        self.assertEqual(merged, {})

    def test_top_k_cap(self):
        sources = [self._hyp([
            ("Europe", 0.9), ("Asia", 0.8), ("Africa", 0.7),
            ("North America", 0.6), ("South America", 0.5), ("Oceania", 0.4),
        ])]
        merged = pipeline._merge_continent_scores(sources, top_k=3)
        self.assertEqual(len(merged), 3)
        self.assertEqual(list(merged.keys())[0], "Europe")


if __name__ == "__main__":
    unittest.main()
