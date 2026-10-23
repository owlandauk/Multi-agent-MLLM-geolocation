"""GeoCLIP top-k GPS candidate cache helpers.

This module preserves GeoCLIP's strongest signal: candidate coordinates. Older
retrieval-prior code collapsed these candidates to a country distribution, which
is useful for coarse fallback but loses the direct GPS baseline and top-k oracle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from country_aliases import canonicalize_country


def country_from_iso2(code: str) -> str | None:
    try:
        import pycountry
    except ImportError:
        return None
    country = pycountry.countries.get(alpha_2=(code or "").upper())
    return country.name if country else None


def normalize_scores(values: list[float]) -> list[float]:
    weights = [max(0.0, float(v)) for v in values]
    total = sum(weights)
    if total <= 0:
        n = max(len(weights), 1)
        return [1.0 / n for _ in weights]
    return [w / total for w in weights]


def candidates_to_country_prior(candidates: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores: dict[str, float] = {}
    for candidate in candidates:
        country = canonicalize_country(str(candidate.get("country") or ""))
        if not country:
            continue
        try:
            score = float(candidate.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        scores[country] = scores.get(country, 0.0) + max(0.0, score)

    total = sum(scores.values())
    if total <= 0:
        return []
    return [
        {"country": country, "score": round(score / total, 6)}
        for country, score in sorted(scores.items(), key=lambda kv: -kv[1])
    ]


def build_candidate_record(coords, probs) -> dict[str, Any]:
    """Convert GeoCLIP raw top-k tensors/lists into a JSON-safe record."""
    import reverse_geocoder as rg

    points = [(float(lat), float(lon)) for lat, lon in coords]
    raw_scores = [float(p) for p in probs]
    norm_scores = normalize_scores(raw_scores)
    geo_rows = rg.search(points, mode=1) if points else []

    candidates: list[dict[str, Any]] = []
    for idx, ((lat, lon), raw_score, score) in enumerate(
        zip(points, raw_scores, norm_scores), start=1
    ):
        row = geo_rows[idx - 1] if idx - 1 < len(geo_rows) else {}
        raw_country = country_from_iso2(str(row.get("cc", ""))) or row.get("cc", "")
        country = canonicalize_country(str(raw_country or ""))
        candidates.append(
            {
                "rank": idx,
                "lat": round(lat, 8),
                "lon": round(lon, 8),
                "score": round(score, 8),
                "raw_score": round(raw_score, 8),
                "country": country,
                "cc": row.get("cc"),
                "admin1": row.get("admin1"),
                "city": row.get("name"),
            }
        )

    return {
        "candidates": candidates,
        "country_prior": candidates_to_country_prior(candidates),
    }


def load_candidate_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"GeoCLIP candidate cache must be a JSON object: {path}")

    cache: dict[str, dict[str, Any]] = {}
    for photo_id, record in data.items():
        if isinstance(record, list):
            # Backward-compatible shape: photo_id -> candidates.
            candidates = record
            cache[str(photo_id)] = {
                "candidates": candidates,
                "country_prior": candidates_to_country_prior(candidates),
            }
        elif isinstance(record, dict):
            candidates = record.get("candidates") or []
            cache[str(photo_id)] = {
                **record,
                "candidates": candidates,
                "country_prior": record.get("country_prior")
                or candidates_to_country_prior(candidates),
            }
    return cache

