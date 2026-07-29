import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "geo_pipeline"))

import analyze_results


class AnalyzeWebSearchReportingTests(unittest.TestCase):
    def test_reports_web_enhancement_for_each_level(self):
        records = [
            {
                "dist_km": 100,
                "gt_continent": "North America",
                "pred_country": "United States",
                "country_posterior": {"United States": 0.56, "Canada": 0.44},
                "continent_posterior": {"North America": 0.7},
                "country_web_enhanced": True,
                "country_web_delta": 0.03,
                "city_web_enhanced": False,
                "street_web_enhanced": True,
                "street_web_delta": 0.02,
            },
            {
                "dist_km": 3000,
                "gt_continent": "Europe",
                "pred_country": "Germany",
                "country_posterior": {"Germany": 0.52, "Austria": 0.48},
                "continent_posterior": {"Europe": 0.8},
                "country_web_enhanced": False,
                "city_web_enhanced": True,
                "city_web_delta": 0.04,
                "street_web_enhanced": False,
            },
        ]

        report = analyze_results.analyze(records)

        self.assertEqual(
            report["web_enhanced_rate"],
            {"country": 50.0, "city": 50.0, "street": 50.0},
        )
        self.assertEqual(report["city_web_enhanced_rate"], 50.0)
        self.assertEqual(report["street_web_enhanced_rate"], 50.0)
        self.assertEqual(report["city_web_delta"], {"mean": 0.04, "median": 0.04})
        self.assertEqual(report["street_web_delta"], {"mean": 0.02, "median": 0.02})
        self.assertIn("city_web_enhanced", report["diagnostic_buckets"])
        self.assertIn("street_web_enhanced", report["diagnostic_buckets"])


if __name__ == "__main__":
    unittest.main()
