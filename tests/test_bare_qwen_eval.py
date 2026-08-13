import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "geo_pipeline"))

import evaluate_bare_qwen as bare


class ParseCountryCityTests(unittest.TestCase):
    def test_parses_json_country_and_city(self):
        text = '{"country": "France", "city": "Paris", "reasons": "eiffel"}'
        country, city = bare._parse_country_city(text)
        self.assertEqual(country, "france")
        self.assertEqual(city, "Paris")

    def test_missing_fields_return_none(self):
        country, city = bare._parse_country_city("not json at all")
        self.assertIsNone(country)
        self.assertIsNone(city)

    def test_unknown_country_canonicalizes_to_none(self):
        country, _ = bare._parse_country_city('{"country": "Atlantis", "city": ""}')
        self.assertIsNone(country)


class GeocodeBareLadderTests(unittest.TestCase):
    def setUp(self):
        self._orig = bare.geocode
        self.calls = []

    def tearDown(self):
        bare.geocode = self._orig

    def _stub(self, hit_on):
        def _g(query):
            self.calls.append(query)
            return (1.0, 2.0) if query == hit_on else None
        return _g

    def test_city_country_qualified_first(self):
        bare.geocode = self._stub("Paris, France")
        coords, source = bare._geocode_bare("France", "Paris")
        self.assertEqual(coords, (1.0, 2.0))
        self.assertEqual(source, "city_country_qualified")
        self.assertEqual(self.calls[0], "Paris, France")

    def test_falls_back_to_country(self):
        bare.geocode = self._stub("France")
        coords, source = bare._geocode_bare("France", "Nowheresville")
        self.assertEqual(source, "country")

    def test_continent_centroid_last_resort(self):
        bare.geocode = self._stub("__never__")
        coords, source = bare._geocode_bare("France", "Nowheresville")
        self.assertEqual(source, "continent_fallback")
        self.assertEqual(coords, bare._CONTINENT_CENTROIDS["Europe"])

    def test_all_none_when_no_country(self):
        bare.geocode = self._stub("__never__")
        coords, source = bare._geocode_bare(None, None)
        self.assertIsNone(coords)
        self.assertEqual(source, "failed")


if __name__ == "__main__":
    unittest.main()
