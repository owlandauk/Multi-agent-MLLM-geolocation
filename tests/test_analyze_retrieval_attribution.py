import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "geo_pipeline"))

import analyze_results


def _rec(before, after):
    return {"pre_fallback_dist_km": before, "dist_km": after}


class RetrievalAttributionTests(unittest.TestCase):
    def test_net_threshold_effect_and_classes(self):
        # thresholds = [1, 25, 200, 750, 2500]
        records = [
            # helped: 3000 -> 500 crosses under 750 and 2500 (gains at both, no loss)
            _rec(3000, 500),
            # hurt: 100 -> 1000 loses 200 and 750 (was under, now over)
            _rec(100, 1000),
            # unchanged: identical distance
            _rec(300, 300),
            # changed_neutral: moved 400 -> 300 but both sides same threshold band (200-750)
            _rec(400, 300),
            # no coverage: missing pre_fallback
            {"dist_km": 50},
        ]
        report = analyze_results.analyze(records)
        attr = report["retrieval_attribution"]

        self.assertEqual(attr["coverage"], 4)  # 4 of 5 have both fields
        self.assertEqual(attr["classes"]["helped"], 1)
        self.assertEqual(attr["classes"]["hurt"], 1)
        self.assertEqual(attr["classes"]["unchanged"], 1)
        self.assertEqual(attr["classes"]["changed_neutral"], 1)

        net = attr["net_threshold_effect"]
        # rec1 gains @750 (+1) and @2500 (+1); rec2 loses @200 (-1) and @750 (-1)
        self.assertEqual(net["1"], 0)
        self.assertEqual(net["25"], 0)
        self.assertEqual(net["200"], -1)
        self.assertEqual(net["750"], 0)  # +1 from rec1, -1 from rec2
        self.assertEqual(net["2500"], 1)

    def test_improvement_and_regression_percentiles(self):
        records = [_rec(3000, 500), _rec(100, 1000), _rec(300, 300)]
        attr = analyze_results.analyze(records)["retrieval_attribution"]
        # rec1 improved by 2500; rec2 regressed by 900; rec3 unchanged
        self.assertEqual(attr["improvement_km"]["n"], 1)
        self.assertEqual(attr["improvement_km"]["median"], 2500.0)
        self.assertEqual(attr["regression_km"]["n"], 1)
        self.assertEqual(attr["regression_km"]["median"], 900.0)

    def test_zero_coverage_is_safe(self):
        records = [{"dist_km": 50}, {"dist_km": 999}]
        attr = analyze_results.analyze(records)["retrieval_attribution"]
        self.assertEqual(attr["coverage"], 0)
        self.assertEqual(attr["coverage_rate"], 0.0)
        for thr in ("1", "25", "200", "750", "2500"):
            self.assertEqual(attr["net_threshold_effect"][thr], 0)

    def test_image_search_evidence_quality(self):
        records = [
            {
                "dist_km": 100,
                "country_image_search_enhanced": True,
                "country_image_search_evidence": "Eiffel Tower, Paris",
            },
            {
                "dist_km": 5000,
                "country_image_search_enhanced": True,
                "country_image_search_evidence": "",  # enhanced but no evidence text
            },
            {
                "dist_km": 10,
                "country_image_search_enhanced": False,
            },
        ]
        evq = analyze_results.analyze(records)["image_search_evidence_quality"]
        self.assertEqual(evq["enhanced_n"], 2)
        self.assertEqual(evq["enhanced_with_evidence_n"], 1)
        self.assertEqual(evq["vision_hit_rate"], 50.0)

    def test_distance_distribution(self):
        records = [{"dist_km": d} for d in (10, 20, 30, 40, 100)]
        dist = analyze_results.analyze(records)["distance_distribution"]
        self.assertEqual(dist["dist_km"]["n"], 5)
        self.assertEqual(dist["dist_km"]["median"], 30.0)
        self.assertEqual(dist["dist_km"]["max"], 100.0)


if __name__ == "__main__":
    unittest.main()
