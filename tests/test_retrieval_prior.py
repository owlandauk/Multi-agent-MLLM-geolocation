import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "geo_pipeline"))

from retrieval_prior import (
    RetrievalPriorClient,
    adaptive_retrieval_weight,
    blend_country_priors,
)
from build_geoclip_prior_cache import gps_predictions_to_country_prior
import pipeline
from PIL import Image


class FakeMllm:
    def generate(self, messages, max_new_tokens=None):
        return (
            '{"hypotheses": ['
            '{"location": "France", "confidence": 0.8}, '
            '{"location": "Germany", "confidence": 0.2}'
            '], "verification_plan": []}'
        )


class RetrievalPriorTests(unittest.TestCase):
    def test_loads_json_mapping_and_canonicalizes_countries(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(
                '{"123": ['
                '{"country": "USA", "score": 0.8}, '
                '{"country": "Canada", "score": 0.2}, '
                '{"country": "Southeast Asia", "score": 0.9}'
                ']}'
            )
            path = f.name
        try:
            client = RetrievalPriorClient(enabled=True, path=path, weight=0.25)
            prior = client.country_prior_for_photo("123")
        finally:
            os.unlink(path)

        self.assertAlmostEqual(prior["united states"], 0.8)
        self.assertAlmostEqual(prior["canada"], 0.2)
        self.assertNotIn("Southeast Asia", prior)

    def test_blend_is_soft_and_preserves_existing_visual_prior(self):
        visual = {"france": 0.7, "germany": 0.3}
        retrieval = {"germany": 0.9, "france": 0.1}

        blended = blend_country_priors(visual, retrieval, weight=0.25)

        self.assertGreater(blended["france"], blended["germany"])
        self.assertAlmostEqual(sum(blended.values()), 1.0)
        self.assertAlmostEqual(blended["france"], 0.55)
        self.assertAlmostEqual(blended["germany"], 0.45)

    def test_adaptive_weight_keeps_full_weight_when_tops_agree(self):
        weight, relation = adaptive_retrieval_weight(
            {"france": 0.65, "germany": 0.35},
            {"france": 0.9, "spain": 0.1},
            base_weight=0.15,
        )

        self.assertEqual(relation, "agree")
        self.assertAlmostEqual(weight, 0.15)

    def test_adaptive_weight_reduces_cross_continent_conflict(self):
        weight, relation = adaptive_retrieval_weight(
            {"france": 0.55, "germany": 0.45},
            {"united states": 1.0},
            base_weight=0.15,
        )

        self.assertEqual(relation, "cross_continent_conflict")
        self.assertAlmostEqual(weight, 0.05)

    def test_pipeline_records_country_retrieval_prior_diagnostics(self):
        old_levels = pipeline.LEVELS
        pipeline.LEVELS = ["country"]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"img1": [{"country": "Germany", "score": 1.0}]}')
            path = f.name
        try:
            image = Image.new("RGB", (2, 2))
            image.info["photo_id"] = "img1"
            geo = pipeline.GeoPipeline(FakeMllm())
            geo.retrieval_prior = RetrievalPriorClient(enabled=True, path=path, weight=0.25)

            result = geo.predict(image)
        finally:
            pipeline.LEVELS = old_levels
            os.unlink(path)

        self.assertTrue(result["country_retrieval_enhanced"])
        self.assertEqual(result["country_retrieval_prior"], {"germany": 1.0})
        self.assertIn("germany", result["country_posterior"])

    def test_retrieval_verify_task_is_disabled_by_default(self):
        old_enabled = pipeline.RETRIEVAL_VERIFY_ACTION_ENABLED
        pipeline.RETRIEVAL_VERIFY_ACTION_ENABLED = False
        try:
            geo = pipeline.GeoPipeline(FakeMllm())
            task = geo._retrieval_verify_task(
                {
                    "applied": True,
                    "relation": "cross_continent_conflict",
                    "retrieval_prior": {"united states": 1.0},
                    "visual_prior": {"france": 0.7, "germany": 0.3},
                }
            )
        finally:
            pipeline.RETRIEVAL_VERIFY_ACTION_ENABLED = old_enabled

        self.assertIsNone(task)

    def test_retrieval_verify_task_uses_high_confidence_conflicts(self):
        old_enabled = pipeline.RETRIEVAL_VERIFY_ACTION_ENABLED
        old_min_prior = pipeline.RETRIEVAL_VERIFY_MIN_PRIOR_TOP
        old_relations = pipeline.RETRIEVAL_VERIFY_RELATIONS
        pipeline.RETRIEVAL_VERIFY_ACTION_ENABLED = True
        pipeline.RETRIEVAL_VERIFY_MIN_PRIOR_TOP = 0.55
        pipeline.RETRIEVAL_VERIFY_RELATIONS = ("cross_continent_conflict",)
        try:
            geo = pipeline.GeoPipeline(FakeMllm())
            task = geo._retrieval_verify_task(
                {
                    "applied": True,
                    "relation": "cross_continent_conflict",
                    "retrieval_prior": {"united states": 0.8, "canada": 0.2},
                    "visual_prior": {"france": 0.7, "germany": 0.3},
                }
            )
            skipped = geo._retrieval_verify_task(
                {
                    "applied": True,
                    "relation": "same_continent_conflict",
                    "retrieval_prior": {"germany": 0.8, "france": 0.2},
                    "visual_prior": {"france": 0.7, "spain": 0.3},
                }
            )
        finally:
            pipeline.RETRIEVAL_VERIFY_ACTION_ENABLED = old_enabled
            pipeline.RETRIEVAL_VERIFY_MIN_PRIOR_TOP = old_min_prior
            pipeline.RETRIEVAL_VERIFY_RELATIONS = old_relations

        self.assertIsNotNone(task)
        self.assertEqual(task["source"], "retrieval_verify")
        self.assertIn("united states (0.80)", task["desc"])
        self.assertIn("Visual prior candidates", task["desc"])
        self.assertIsNone(skipped)

    def test_client_records_adaptive_effective_weight(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"img1": [{"country": "United States", "score": 1.0}]}')
            path = f.name
        try:
            image = Image.new("RGB", (2, 2))
            image.info["photo_id"] = "img1"
            client = RetrievalPriorClient(
                enabled=True,
                path=path,
                weight=0.15,
                adaptive=True,
            )
            _, diag = client.blend_for_image(image, {"france": 0.7, "germany": 0.3})
        finally:
            os.unlink(path)

        self.assertEqual(diag["relation"], "cross_continent_conflict")
        self.assertAlmostEqual(diag["effective_weight"], 0.05)

    def test_retrieval_country_anchor_uses_weak_country_posterior(self):
        old_enabled = pipeline.RETRIEVAL_COUNTRY_ANCHOR_ENABLED
        old_max_top = pipeline.RETRIEVAL_COUNTRY_ANCHOR_MAX_COUNTRY_TOP
        old_min_prior = pipeline.RETRIEVAL_COUNTRY_ANCHOR_MIN_PRIOR_TOP
        old_weight = pipeline.RETRIEVAL_COUNTRY_ANCHOR_WEIGHT
        pipeline.RETRIEVAL_COUNTRY_ANCHOR_ENABLED = True
        pipeline.RETRIEVAL_COUNTRY_ANCHOR_MAX_COUNTRY_TOP = 0.55
        pipeline.RETRIEVAL_COUNTRY_ANCHOR_MIN_PRIOR_TOP = 0.15
        pipeline.RETRIEVAL_COUNTRY_ANCHOR_WEIGHT = 1.0
        try:
            posterior, diag = pipeline._retrieval_country_anchor_posterior(
                {"canada": 0.49, "france": 0.48, "germany": 0.03},
                {"france": 0.7, "germany": 0.2},
            )
        finally:
            pipeline.RETRIEVAL_COUNTRY_ANCHOR_ENABLED = old_enabled
            pipeline.RETRIEVAL_COUNTRY_ANCHOR_MAX_COUNTRY_TOP = old_max_top
            pipeline.RETRIEVAL_COUNTRY_ANCHOR_MIN_PRIOR_TOP = old_min_prior
            pipeline.RETRIEVAL_COUNTRY_ANCHOR_WEIGHT = old_weight

        self.assertTrue(diag["applied"])
        self.assertEqual(diag["country"], "france")
        self.assertEqual(max(posterior, key=posterior.get), "france")

    def test_retrieval_country_anchor_skips_strong_country_posterior(self):
        old_enabled = pipeline.RETRIEVAL_COUNTRY_ANCHOR_ENABLED
        pipeline.RETRIEVAL_COUNTRY_ANCHOR_ENABLED = True
        try:
            posterior, diag = pipeline._retrieval_country_anchor_posterior(
                {"canada": 0.7, "france": 0.2},
                {"france": 0.8},
            )
        finally:
            pipeline.RETRIEVAL_COUNTRY_ANCHOR_ENABLED = old_enabled

        self.assertFalse(diag["applied"])
        self.assertEqual(max(posterior, key=posterior.get), "canada")

    def test_geoclip_gps_predictions_aggregate_to_country_prior(self):
        prior = gps_predictions_to_country_prior(
            coords=[(48.8566, 2.3522), (52.52, 13.405)],
            probs=[0.75, 0.25],
        )

        scores = {item["country"]: item["score"] for item in prior}
        self.assertGreater(scores.get("france", 0), scores.get("germany", 0))
        self.assertAlmostEqual(sum(scores.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
