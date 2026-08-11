import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "geo_pipeline"))

import pipeline
from PIL import Image


class FakeMllm:
    def generate(self, messages, max_new_tokens=None):
        return (
            '{"hypotheses": ['
            '{"location": "Barcelona", "confidence": 0.8}, '
            '{"location": "Madrid", "confidence": 0.2}'
            '], "verification_plan": []}'
        )


class FakeSearch:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return {"answer": "La Deessa is associated with Barcelona."}


class WebSearchEnhancementTests(unittest.TestCase):
    def setUp(self):
        self._old_web_search_levels = pipeline.WEB_SEARCH_LEVELS
        pipeline.WEB_SEARCH_LEVELS = ("country", "city", "street")

    def tearDown(self):
        pipeline.WEB_SEARCH_LEVELS = self._old_web_search_levels

    def test_query_is_adapted_to_reasoning_level_and_parent(self):
        query = pipeline._build_web_search_query(
            level="city",
            posterior={"San Francisco": 0.51, "Los Angeles": 0.49},
            evidence=["red trolley bus with overhead tram wires"],
            parent_context="United States",
        )

        self.assertIn("which cities of United States", query)
        self.assertIn("red trolley bus", query)

    def test_latest_weak_evidence_can_trigger_search_without_key_evidence(self):
        self.assertTrue(
            pipeline._should_web_enhance_level(
                "country",
                {"United States": 0.52, "Canada": 0.48},
                visual_delta=0.01,
                search_evidence=["black-on-white license plate visible"],
            )
        )

    def test_stable_or_disabled_level_does_not_trigger_search(self):
        self.assertFalse(
            pipeline._should_web_enhance_level(
                "continent",
                {"Europe": 0.52, "Asia": 0.48},
                visual_delta=0.01,
                search_evidence=["sign text"],
            )
        )

    def test_default_search_update_mode_uses_candidate_verification(self):
        self.assertEqual(pipeline.WEB_SEARCH_UPDATE_MODE, "verify")

    def test_confirm_top_search_acceptance_rejects_country_hijack(self):
        old_mode = pipeline.WEB_SEARCH_ACCEPT_MODE
        pipeline.WEB_SEARCH_ACCEPT_MODE = "confirm_top"
        try:
            self.assertFalse(
                pipeline._accept_web_update(
                    "country",
                    {"Spain": 0.51, "France": 0.49},
                    {"Spain": 0.20, "France": 0.80},
                )
            )
            self.assertTrue(
                pipeline._accept_web_update(
                    "country",
                    {"Spain": 0.51, "France": 0.49},
                    {"Spain": 0.75, "France": 0.25},
                )
            )
        finally:
            pipeline.WEB_SEARCH_ACCEPT_MODE = old_mode

    def test_pipeline_reruns_level_from_web_snippets(self):
        old_mode = pipeline.WEB_SEARCH_UPDATE_MODE
        pipeline.WEB_SEARCH_UPDATE_MODE = "rehypothesize"
        try:
            geo = pipeline.GeoPipeline(FakeMllm())
            fake_search = FakeSearch()
            geo.web_search = fake_search

            enhanced = geo._web_enhance_level(
                Image.new("RGB", (2, 2)),
                "city",
                {"Madrid": 0.51, "Barcelona": 0.49},
                [],
                0.01,
                ["La Deessa statue"],
                "Spain",
            )
        finally:
            pipeline.WEB_SEARCH_UPDATE_MODE = old_mode

        self.assertIsNotNone(enhanced)
        posterior, evidence, raw_response, query, web_delta, observed = enhanced
        self.assertEqual(max(posterior, key=posterior.get), "Barcelona")
        self.assertIn("which cities of Spain", query)
        self.assertEqual(fake_search.queries, [query])
        self.assertIn("web search (city):", evidence[-1])
        self.assertIn("Barcelona", raw_response)
        self.assertEqual(web_delta, 0.0)
        self.assertEqual(observed, [])
        self.assertFalse(
            pipeline._should_web_enhance_level(
                "country",
                {"United States": 0.81, "Canada": 0.19},
                visual_delta=0.01,
                search_evidence=["sign text"],
            )
        )


if __name__ == "__main__":
    unittest.main()
