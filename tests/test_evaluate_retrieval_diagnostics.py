import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "geo_pipeline"))

import evaluate


class FakeDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "photo_id": "img1",
            "image": object(),
            "gt_lat": 48.8566,
            "gt_lon": 2.3522,
        }


class FakePipeline:
    def __init__(self, mllm):
        pass

    def predict_batch(self, images):
        return [
            {
                "country": "France",
                "city": "Paris",
                "street": "Unknown",
                "country_posterior": {"france": 0.7, "germany": 0.3},
                "country_retrieval_enhanced": True,
                "country_retrieval_prior": {"germany": 1.0},
                "country_retrieval_weight": 0.15,
                "country_retrieval_effective_weight": 0.10,
                "country_retrieval_relation": "same_continent_conflict",
                "country_prior_before_retrieval": {"france": 0.8, "germany": 0.2},
            }
        ]


class EvaluateRetrievalDiagnosticsTests(unittest.TestCase):
    def test_evaluate_preserves_adaptive_retrieval_diagnostics(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        args = types.SimpleNamespace(
            img_dir="",
            gps_csv="",
            start=0,
            limit=1,
            batch_size=1,
            strict_child_geocode=False,
            allow_bare_city_geocode=True,
            retrieval_continent_fallback=False,
            retrieval_continent_max_country_top=0.50,
            retrieval_continent_min_prior_top=0.50,
            retrieval_country_fallback=False,
            retrieval_country_max_country_top=0.55,
            retrieval_country_min_prior_top=0.15,
            retrieval_country_same_continent_max_country_top=None,
            retrieval_country_cross_continent_max_country_top=None,
            out=out_path,
        )
        try:
            with patch.object(evaluate, "MLLMClient", return_value=object()), \
                 patch.object(evaluate, "GeoPipeline", FakePipeline), \
                 patch.object(evaluate, "YFCC4KDataset", return_value=FakeDataset()), \
                 patch.object(evaluate, "_geocode_level", return_value=((48.8566, 2.3522), "city_bare", "unchecked")):
                evaluate.evaluate(args)

            with open(out_path, encoding="utf-8") as f:
                record = json.load(f)["records"][0]
        finally:
            os.unlink(out_path)

        self.assertEqual(record["country_retrieval_effective_weight"], 0.10)
        self.assertEqual(record["country_retrieval_relation"], "same_continent_conflict")
        self.assertEqual(record["pre_fallback_pred_country"], "France")
        self.assertEqual(record["pre_fallback_geocode_source"], "city_bare")
        self.assertAlmostEqual(record["pre_fallback_dist_km"], 0.0, places=3)

    def test_retrieval_continent_fallback_uses_weak_country_belief(self):
        coords, diag = evaluate._retrieval_continent_fallback_coords(
            {
                "country_posterior": {"canada": 0.49, "france": 0.48},
                "country_retrieval_prior": {"france": 0.7, "germany": 0.2},
            },
            max_country_top=0.50,
            min_prior_top=0.50,
        )

        self.assertEqual(coords, evaluate._CONTINENT_CENTROIDS["Europe"])
        self.assertEqual(diag["continent"], "Europe")

    def test_retrieval_continent_fallback_skips_confident_country_belief(self):
        coords, _ = evaluate._retrieval_continent_fallback_coords(
            {
                "country_posterior": {"canada": 0.7, "france": 0.2},
                "country_retrieval_prior": {"france": 0.8},
            },
            max_country_top=0.50,
            min_prior_top=0.50,
        )

        self.assertIsNone(coords)

    def test_retrieval_country_fallback_uses_weak_country_belief(self):
        with patch.object(evaluate, "geocode", return_value=(46.2276, 2.2137)) as geocode:
            coords, diag = evaluate._retrieval_country_fallback_coords(
                {
                    "country_posterior": {"canada": 0.49, "france": 0.48},
                    "country_retrieval_prior": {"france": 0.7, "germany": 0.2},
                },
                max_country_top=0.50,
                min_prior_top=0.15,
            )

        self.assertEqual(coords, (46.2276, 2.2137))
        self.assertEqual(diag["country"], "france")
        geocode.assert_called_once_with("france")

    def test_retrieval_country_fallback_skips_confident_country_belief(self):
        with patch.object(evaluate, "geocode") as geocode:
            coords, diag = evaluate._retrieval_country_fallback_coords(
                {
                    "country_posterior": {"canada": 0.7, "france": 0.2},
                    "country_retrieval_prior": {"france": 0.8},
                },
                max_country_top=0.50,
                min_prior_top=0.15,
            )

        self.assertIsNone(coords)
        self.assertEqual(diag["country_top"], 0.7)
        geocode.assert_not_called()

    def test_retrieval_country_fallback_uses_cross_continent_gate(self):
        with patch.object(evaluate, "geocode", return_value=(46.2276, 2.2137)) as geocode:
            coords, diag = evaluate._retrieval_country_fallback_coords(
                {
                    "country_posterior": {"canada": 0.53, "france": 0.47},
                    "country_retrieval_prior": {"france": 0.7, "germany": 0.2},
                },
                max_country_top=0.50,
                min_prior_top=0.15,
                same_continent_max_country_top=0.45,
                cross_continent_max_country_top=0.55,
            )

        self.assertEqual(coords, (46.2276, 2.2137))
        self.assertEqual(diag["relation"], "cross_continent")
        self.assertEqual(diag["effective_max_country_top"], 0.55)
        geocode.assert_called_once_with("france")

    def test_retrieval_country_fallback_retries_child_in_retrieval_country(self):
        with patch.object(evaluate, "geocode", return_value=(48.8566, 2.3522)) as geocode:
            coords, diag = evaluate._retrieval_country_fallback_coords(
                {
                    "city": "Paris",
                    "street": "Unknown",
                    "country_posterior": {"canada": 0.53, "france": 0.47},
                    "country_retrieval_prior": {"france": 0.7, "germany": 0.2},
                },
                max_country_top=0.50,
                min_prior_top=0.15,
                same_continent_max_country_top=0.45,
                cross_continent_max_country_top=0.55,
                child_retry=True,
            )

        self.assertEqual(coords, (48.8566, 2.3522))
        self.assertEqual(diag["child_retry_level"], "city")
        self.assertEqual(diag["child_retry_query"], "Paris, france")
        geocode.assert_called_once_with("Paris, france")

    def test_retrieval_country_fallback_protects_same_continent_child_geocode(self):
        with patch.object(evaluate, "geocode") as geocode:
            coords, diag = evaluate._retrieval_country_fallback_coords(
                {
                    "country_posterior": {"france": 0.53, "germany": 0.47},
                    "country_retrieval_prior": {"germany": 0.7, "france": 0.2},
                },
                max_country_top=0.55,
                min_prior_top=0.15,
                same_continent_max_country_top=0.50,
                cross_continent_max_country_top=0.55,
            )

        self.assertIsNone(coords)
        self.assertEqual(diag["relation"], "same_continent")
        self.assertEqual(diag["effective_max_country_top"], 0.50)
        geocode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
