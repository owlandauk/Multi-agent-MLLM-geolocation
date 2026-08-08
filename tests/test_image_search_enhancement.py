import os
import sys
import unittest
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "geo_pipeline"))

import pipeline
from PIL import Image
from image_search import (
    ImageSearchClient,
    format_image_search_evidence,
    image_evidence_to_text_query,
    is_location_worthy_image_evidence,
    serpapi_lens_results_to_data,
)


class FakeMllm:
    def generate(self, messages, max_new_tokens=None):
        text = "\n".join(
            part.get("text", "")
            for msg in messages
            for part in msg.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if "External search evidence" in text:
            return "Observation: Search evidence identifies La Deessa in Barcelona.\nSupport: Madrid=C; Barcelona=S"
        return (
            '{"hypotheses": ['
            '{"location": "Barcelona", "confidence": 0.8}, '
            '{"location": "Madrid", "confidence": 0.2}'
            '], "verification_plan": []}'
        )


class FakeImageSearch:
    def __init__(self):
        self.enabled = True
        self.calls = 0

    def search_image(self, image):
        self.calls += 1
        return {
            "best_guess_labels": ["La Deessa Barcelona"],
            "entities": [{"description": "La Deessa", "score": 0.92}],
            "pages": [{"title": "La Deessa - Barcelona", "url": "https://example.com"}],
        }


class EmptyImageSearch:
    def __init__(self):
        self.enabled = True
        self.calls = 0

    def search_image(self, image):
        self.calls += 1
        return None


class FakeTextSearch:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return {"answer": "La Deessa is in Barcelona, Spain."}


class ImageSearchEnhancementTests(unittest.TestCase):
    def setUp(self):
        self._old_web_search_levels = pipeline.WEB_SEARCH_LEVELS
        pipeline.WEB_SEARCH_LEVELS = ("country", "city", "street")

    def tearDown(self):
        pipeline.WEB_SEARCH_LEVELS = self._old_web_search_levels

    def test_formats_google_vision_entities_as_searchable_evidence(self):
        evidence = format_image_search_evidence(
            {
                "best_guess_labels": ["La Deessa Barcelona"],
                "entities": [{"description": "La Deessa", "score": 0.92}],
                "pages": [{"title": "La Deessa - Barcelona", "url": "https://example.com"}],
            }
        )

        self.assertIn("Best guess: La Deessa Barcelona", evidence)
        self.assertIn("Entity 1: La Deessa", evidence)
        self.assertIn("La Deessa - Barcelona", evidence)

    def test_image_evidence_builds_strict_location_query(self):
        query = image_evidence_to_text_query(
            "city",
            (
                "Best guess: La Deessa Barcelona\n"
                "Entity 1: La Deessa (0.92)\n"
                "Page 1: La Deessa - Barcelona https://example.com"
            ),
            "Spain",
        )

        self.assertIn("location city in Spain", query)
        self.assertIn("La Deessa", query)
        self.assertNotIn("https://", query)
        self.assertNotIn("in which cities", query)

    def test_generic_vehicle_image_evidence_is_not_location_worthy(self):
        generic = "Best guess: street\nEntity 1: Audi A2 (0.72)\nEntity 2: Minivan (0.66)"
        generic_street = "Best guess: street"
        generic_media = "Best guess: Photograph\nEntity 1: Image sharing (0.60)\nEntity 2: Flickr (0.58)"
        generic_event = "Best guess: fireworks\nEntity 1: New Year (0.54)\nEntity 2: Fireworks (0.51)"
        generic_animal = "Best guess: cardigan welsh corgi\nEntity 1: Working dog (0.94)"
        generic_nature = "Best guess: moon\nEntity 1: Lunar eclipse (1.11)\nEntity 2: Earth (1.06)"
        generic_room = "Best guess: bedroom\nEntity 1: Nightstand (0.58)\nEntity 2: Bed Frame (0.51)"
        generic_bridge = "Best guess: concrete bridge\nEntity 1: Truss bridge (0.54)"
        generic_beverage = "Best guess: non-alcoholic beverage\nEntity 1: Caipirinha (0.52)"
        place = "Best guess: La Deessa Barcelona\nEntity 1: La Deessa (0.92)"

        self.assertFalse(is_location_worthy_image_evidence(generic))
        self.assertFalse(is_location_worthy_image_evidence(generic_street))
        self.assertFalse(is_location_worthy_image_evidence(generic_media))
        self.assertFalse(is_location_worthy_image_evidence(generic_event))
        self.assertFalse(is_location_worthy_image_evidence(generic_animal))
        self.assertFalse(is_location_worthy_image_evidence(generic_nature))
        self.assertFalse(is_location_worthy_image_evidence(generic_room))
        self.assertFalse(is_location_worthy_image_evidence(generic_bridge))
        self.assertFalse(is_location_worthy_image_evidence(generic_beverage))
        self.assertTrue(is_location_worthy_image_evidence(place))

    def test_query_drops_generic_image_search_terms(self):
        evidence = (
            "Best guess: La Deessa Barcelona\n"
            "Entity 1: Photograph (0.60)\n"
            "Entity 2: La Deessa (0.92)\n"
            "Entity 3: Flickr (0.58)"
        )

        query = image_evidence_to_text_query("city", evidence, "Spain")

        self.assertIn("La Deessa", query)
        self.assertNotIn("Photograph", query)
        self.assertNotIn("Flickr", query)

    def test_query_prioritizes_location_terms_over_serpapi_noise(self):
        evidence = (
            "Best guess: linux - Direct terminal output to Mysql table - Stack Overflow; "
            "Toronto weather takes nasty turn as rain forces GTA bus ...; "
            "93 Shoppers Drug Mart Corp Stock Photos, High-Res Pictures ...\n"
            "Entity 1: linux - Direct terminal output to Mysql table - Stack Overflow\n"
            "Entity 2: DSC09447 | Journey.ca | Flickr\n"
            "Entity 3: File:Corner of Queen Street and Parliament Street, Toronto -a ..."
        )

        query = image_evidence_to_text_query("city", evidence, "Canada")

        self.assertIn("Queen Street", query)
        self.assertNotIn("Stack Overflow", query)
        self.assertNotIn("Stock Photos", query)
        self.assertNotIn("DSC09447", query)

    def test_query_drops_serpapi_topic_noise_without_location_clue(self):
        evidence = (
            "Best guess: BRIDE GROOM Sailor FOUND WEDDING PHOTO Color Snapshot ...\n"
            "Entity 1: A List of What to Shoot At A Typical Wedding - Learning with ...\n"
            "Entity 2: What God Has Joined Together - by Deacon Marty McIndoe ..."
        )

        query = image_evidence_to_text_query("street", evidence, "United Kingdom")

        self.assertEqual("", query)

    def test_query_keeps_serpapi_place_terms_with_river_or_dori(self):
        evidence = (
            "Best guess: Digital Nomad in Kyoto, Day 2. Kiyamachi-dori (木屋町通 ...\n"
            "Entity 1: The Kamogawa River | Dave's Japan\n"
            "Entity 2: Small Irrigation Dam With Water Flowing Stock Photo ..."
        )

        query = image_evidence_to_text_query("city", evidence, "Japan")

        self.assertIn("Kiyamachi-dori", query)
        self.assertIn("Kamogawa River", query)
        self.assertNotIn("Stock Photo", query)

    def test_query_drops_plant_and_product_web_results(self):
        plant_evidence = (
            "Best guess: How Would U Rate This Hibiscus Picture : r/flower\n"
            "Entity 1: Hibiscus - Wikipedia\n"
            "Entity 2: What Time is it in Nature: Scarlet Hibiscus | NC Museum of ...\n"
            "Entity 3: Can someone tell me what this is? : r/NoLawns"
        )
        product_evidence = (
            "Best guess: Engraved Brass Plate on Sapele Mahogany Wood Plaque - Etsy\n"
            "Entity 1: Buy Cantebury Plaque w/ Stand-Out Ebony - Custom Plaques ...\n"
            "Entity 2: Amazon.com: Personalized Plaque - Customizable"
        )

        self.assertEqual("", image_evidence_to_text_query("country", plant_evidence))
        self.assertEqual("", image_evidence_to_text_query("city", product_evidence, "United States"))

    def test_query_drops_social_product_species_and_listing_noise(self):
        noisy_examples = [
            (
                "Best guess: Polka Dot White Comfort Shoes for Women for sale | eBay\n"
                "Entity 1: Lightning Themed Casual Star Sneakers For Kids ...\n"
                "Entity 2: Mary Jane Red Polka Dot Shoes Mary Jane Polka Dot Pumps ..."
            ),
            (
                "Best guess: Domestic Medium Hair Cats & Kittens for Adoption in Oak ...\n"
                "Entity 1: Fundraiser by Damian Perez : Support Damian and Bean to ..."
            ),
            (
                "Best guess: Nhn Sager Ln W, Deer Lodge, MT 59722 | MLS #30071899 | Zillow\n"
                "Entity 1: 3712 Baker St, San Diego, CA 92117 | Zillow"
            ),
            (
                "Best guess: Ronald Beitle Discography: Vinyl, CDs, & More | Discogs\n"
                "Entity 1: Happy Birthday, Tom Brechtlein!"
            ),
            (
                "Best guess: Henry Dever (@henrydevertattoo) · Napa, CA\n"
                "Entity 1: Today I have been a tattoo artist for 14 years"
            ),
            (
                "Best guess: Brown-breasted flycatcher (Muscicapa muttui)\n"
                "Entity 1: Asian brown flycatcher - Wikipedia"
            ),
        ]

        for evidence in noisy_examples:
            self.assertEqual("", image_evidence_to_text_query("city", evidence, "United States"))

    def test_query_keeps_named_landmark_lens_hits(self):
        evidence = (
            "Best guess: Sydney Harbour Bridge - Wikipedia\n"
            "Entity 1: The Quay to the City - Sydney's Circular Quay\n"
            "Entity 2: Statue of William III, Kensington Palace - Wikipedia\n"
            "Entity 3: Shinjuku Gyoen National Garden, Tokyo"
        )

        query = image_evidence_to_text_query("city", evidence, "Australia")

        self.assertIn("Sydney Harbour Bridge", query)
        self.assertIn("Kensington Palace", query)
        self.assertIn("Shinjuku Gyoen", query)

    def test_serpapi_lens_results_are_normalized_to_image_evidence(self):
        data = serpapi_lens_results_to_data(
            {
                "visual_matches": [
                    {
                        "title": "La Deessa - Barcelona",
                        "link": "https://example.com/la-deessa",
                        "source": "Example",
                    },
                    {
                        "title": "Plaça de Catalunya statue",
                        "link": "https://example.com/placa",
                    },
                ],
                "exact_matches": [
                    {
                        "title": "La Deessa sculpture",
                        "link": "https://example.com/exact",
                    }
                ],
            }
        )

        evidence = format_image_search_evidence(data)

        self.assertIn("Best guess: La Deessa - Barcelona", evidence)
        self.assertIn("Entity 1: La Deessa - Barcelona", evidence)
        self.assertIn("Page 1: La Deessa - Barcelona", evidence)
        self.assertTrue(is_location_worthy_image_evidence(evidence))

    def test_serpapi_client_reads_public_image_url_metadata(self):
        with patch.dict(
            os.environ,
            {
                "IMAGE_SEARCH_ENABLED": "1",
                "IMAGE_SEARCH_PROVIDER": "serpapi_lens",
                "SERPAPI_API_KEY": "test-key",
            },
        ):
            client = ImageSearchClient()

        image = Image.new("RGB", (2, 2))
        image.info["image_url"] = "https://example.com/photo.jpg"

        self.assertTrue(client.enabled)
        self.assertEqual(client._image_url(image, "/tmp/photo.jpg"), "https://example.com/photo.jpg")

    def test_serpapi_client_can_build_url_from_template(self):
        with patch.dict(
            os.environ,
            {
                "IMAGE_SEARCH_ENABLED": "1",
                "IMAGE_SEARCH_PROVIDER": "serpapi_lens",
                "SERPAPI_API_KEY": "test-key",
                "SERPAPI_IMAGE_URL_TEMPLATE": "https://cdn.example.com/{photo_id}.jpg",
            },
        ):
            client = ImageSearchClient()

        image = Image.new("RGB", (2, 2))
        image.info["photo_id"] = "12345"

        self.assertEqual(client._image_url(image, "/data/12345.jpg"), "https://cdn.example.com/12345.jpg")

    def test_serpapi_client_respects_uncached_call_cap(self):
        with patch.dict(
            os.environ,
            {
                "IMAGE_SEARCH_ENABLED": "1",
                "IMAGE_SEARCH_PROVIDER": "serpapi_lens",
                "SERPAPI_API_KEY": "test-key",
                "IMAGE_SEARCH_MAX_UNCACHED_CALLS": "1",
            },
        ):
            client = ImageSearchClient()

        first = Image.new("RGB", (2, 2))
        first.info["img_path"] = "/tmp/first.jpg"
        first.info["image_url"] = "https://example.com/first.jpg"
        second = Image.new("RGB", (2, 2))
        second.info["img_path"] = "/tmp/second.jpg"
        second.info["image_url"] = "https://example.com/second.jpg"

        with patch.object(
            client,
            "_serpapi_lens_search",
            return_value={"best_guess_labels": ["First Place"]},
        ) as search:
            self.assertIsNotNone(client.search_image(first))
            self.assertIsNone(client.search_image(second))

        self.assertEqual(1, search.call_count)

    def test_pipeline_uses_image_search_before_text_location_verification(self):
        geo = pipeline.GeoPipeline(FakeMllm())
        geo.image_search = FakeImageSearch()
        geo.web_search = FakeTextSearch()

        enhanced = geo._web_enhance_level(
            Image.new("RGB", (2, 2)),
            "city",
            {"Madrid": 0.51, "Barcelona": 0.49},
            [],
            0.01,
            ["white statue monument"],
            "Spain",
        )

        self.assertIsNotNone(enhanced)
        posterior, evidence, raw_response, query, web_delta, observed = enhanced
        self.assertEqual(max(posterior, key=posterior.get), "Barcelona")
        self.assertEqual(geo.image_search.calls, 1)
        self.assertEqual(len(geo.web_search.queries), 1)
        self.assertIn("location city in Spain", geo.web_search.queries[0])
        self.assertTrue(any("image search (city):" in item for item in evidence))
        self.assertIn("Barcelona", raw_response)

    def test_strict_image_search_can_probe_without_text_entity(self):
        geo = pipeline.GeoPipeline(FakeMllm())
        geo.image_search = FakeImageSearch()
        geo.web_search = FakeTextSearch()

        enhanced = geo._web_enhance_level(
            Image.new("RGB", (2, 2)),
            "city",
            {"Madrid": 0.51, "Barcelona": 0.49},
            [],
            0.01,
            ["ambiguous public space"],
            "Spain",
        )

        self.assertIsNotNone(enhanced)
        self.assertEqual(geo.image_search.calls, 1)
        self.assertIn("La Deessa", geo.web_search.queries[0])

    def test_verify_update_mode_applies_search_evidence_to_existing_candidates(self):
        old_mode = pipeline.WEB_SEARCH_UPDATE_MODE
        pipeline.WEB_SEARCH_UPDATE_MODE = "verify"
        try:
            geo = pipeline.GeoPipeline(FakeMllm())
            geo.image_search = FakeImageSearch()
            geo.web_search = FakeTextSearch()

            enhanced = geo._web_enhance_level(
                Image.new("RGB", (2, 2)),
                "city",
                {"Madrid": 0.51, "Barcelona": 0.49},
                [],
                0.01,
                ["white statue monument"],
                "Spain",
            )
        finally:
            pipeline.WEB_SEARCH_UPDATE_MODE = old_mode

        self.assertIsNotNone(enhanced)
        posterior, evidence, raw_response, query, web_delta, observed = enhanced
        self.assertEqual("Barcelona", max(posterior, key=posterior.get))
        self.assertIn("Support:", raw_response)
        self.assertTrue(any("web verify (city):" in item for item in evidence))
        self.assertTrue(any("Search evidence identifies" in item for item in observed))

    def test_image_search_failure_uses_text_search_fallback_for_concrete_visual_clue(self):
        old_mode = pipeline.WEB_SEARCH_UPDATE_MODE
        pipeline.WEB_SEARCH_UPDATE_MODE = "verify"
        try:
            geo = pipeline.GeoPipeline(FakeMllm())
            geo.image_search = EmptyImageSearch()
            geo.web_search = FakeTextSearch()

            enhanced = geo._web_enhance_level(
                Image.new("RGB", (2, 2)),
                "city",
                {"Madrid": 0.51, "Barcelona": 0.49},
                [],
                0.01,
                ["La Deessa statue sign"],
                "Spain",
            )
        finally:
            pipeline.WEB_SEARCH_UPDATE_MODE = old_mode

        self.assertIsNotNone(enhanced)
        self.assertEqual(geo.image_search.calls, 1)
        self.assertEqual(len(geo.web_search.queries), 1)
        self.assertIn("La Deessa", geo.web_search.queries[0])
        posterior, evidence, raw_response, query, web_delta, observed = enhanced
        self.assertIn("Support:", raw_response)
        self.assertTrue(any("web verify (city):" in item for item in evidence))

    def test_verify_prompt_does_not_treat_search_query_as_evidence(self):
        messages = pipeline._web_verify_prompt(
            "city",
            {"Mumbai, India": 0.51, "Bangalore, India": 0.49},
            "unrelated noisy clue location city in India",
            "Answer: The external result only mentions Zanzibar, Tanzania.",
            "India",
        )
        text = "\n".join(
            part.get("text", "")
            for msg in messages
            for part in msg.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        )

        self.assertIn("External search evidence", text)
        self.assertIn("Zanzibar, Tanzania", text)
        self.assertNotIn("unrelated noisy clue", text)
        self.assertNotIn("location city in India", text)


if __name__ == "__main__":
    unittest.main()
