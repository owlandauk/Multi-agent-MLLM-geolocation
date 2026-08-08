"""Optional retrieval/GeoCLIP country prior support.

This module deliberately keeps retrieval as a soft prior, not a hard location
override. A retrieval backend can write top-k similar-image country evidence to
JSON/JSONL/CSV, then GeoBayes still performs the normal SL/DST/POMDP updates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from country_aliases import canonicalize_country, continent_of


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in scores.values())
    if total <= 0:
        return {}
    return {k: max(0.0, float(v)) / total for k, v in scores.items()}


def _canonical_country_scores(items: Any) -> dict[str, float]:
    """Convert retrieval rows into a normalized canonical country prior."""
    if isinstance(items, dict):
        iterable = [
            {"country": country, "score": score}
            for country, score in items.items()
        ]
    elif isinstance(items, list):
        iterable = items
    else:
        return {}

    scores: dict[str, float] = {}
    for item in iterable:
        if not isinstance(item, dict):
            continue
        country = canonicalize_country(str(item.get("country") or item.get("location") or ""))
        if country is None:
            continue
        raw_score = item.get("score", item.get("similarity", item.get("prob", 1.0)))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 1.0
        scores[country] = scores.get(country, 0.0) + max(0.0, score)
    return _normalize_scores(scores)


def blend_country_priors(
    visual_prior: dict[str, float],
    retrieval_prior: dict[str, float],
    weight: float,
) -> dict[str, float]:
    """Softly mix retrieval evidence into the MLLM country prior."""
    retrieval_prior = _normalize_scores(retrieval_prior)
    if not retrieval_prior or weight <= 0:
        return _normalize_scores(visual_prior) or dict(visual_prior or {})

    weight = min(max(float(weight), 0.0), 1.0)
    visual_prior = _normalize_scores(visual_prior)
    if not visual_prior or set(visual_prior) == {"Unknown"}:
        return retrieval_prior

    keys = set(visual_prior) | set(retrieval_prior)
    blended = {
        key: (1.0 - weight) * visual_prior.get(key, 0.0) + weight * retrieval_prior.get(key, 0.0)
        for key in keys
        if key != "Unknown"
    }
    return _normalize_scores(blended) or retrieval_prior


def _top_key(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores, key=scores.get)


def adaptive_retrieval_weight(
    visual_prior: dict[str, float],
    retrieval_prior: dict[str, float],
    base_weight: float,
    same_continent_weight: float = 0.10,
    cross_continent_weight: float = 0.05,
) -> tuple[float, str]:
    """Down-weight retrieval when its top country conflicts with visual prior."""
    base_weight = min(max(float(base_weight), 0.0), 1.0)
    visual_prior = _normalize_scores(visual_prior)
    retrieval_prior = _normalize_scores(retrieval_prior)
    visual_top = _top_key(visual_prior)
    retrieval_top = _top_key(retrieval_prior)
    if not visual_top or not retrieval_top:
        return base_weight, "missing_top"
    if visual_top == retrieval_top:
        return base_weight, "agree"

    visual_cont = continent_of(visual_top)
    retrieval_cont = continent_of(retrieval_top)
    if visual_cont and retrieval_cont and visual_cont == retrieval_cont:
        return min(base_weight, same_continent_weight), "same_continent_conflict"
    return min(base_weight, cross_continent_weight), "cross_continent_conflict"


class RetrievalPriorClient:
    """Loads precomputed retrieval/GeoCLIP country evidence when enabled."""

    def __init__(
        self,
        enabled: bool = False,
        path: str | None = None,
        weight: float = 0.25,
        adaptive: bool = False,
        same_continent_weight: float = 0.10,
        cross_continent_weight: float = 0.05,
    ):
        self.enabled = bool(enabled)
        self.path = Path(path).expanduser() if path else None
        self.weight = float(weight)
        self.adaptive = bool(adaptive)
        self.same_continent_weight = float(same_continent_weight)
        self.cross_continent_weight = float(cross_continent_weight)
        self._cache: dict[str, dict[str, float]] | None = None

    def _load(self) -> dict[str, dict[str, float]]:
        if self._cache is not None:
            return self._cache
        self._cache = {}
        if not self.enabled or self.path is None or not self.path.exists():
            return self._cache

        suffix = self.path.suffix.lower()
        if suffix == ".json":
            self._load_json()
        elif suffix == ".jsonl":
            self._load_jsonl()
        elif suffix == ".csv":
            self._load_csv()
        return self._cache

    def _load_json(self) -> None:
        assert self.path is not None and self._cache is not None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        for photo_id, items in data.items():
            prior = _canonical_country_scores(items)
            if prior:
                self._cache[str(photo_id)] = prior

    def _load_jsonl(self) -> None:
        assert self.path is not None and self._cache is not None
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                photo_id = row.get("photo_id") or row.get("id")
                if photo_id is None:
                    continue
                items = row.get("neighbors") or row.get("retrieval") or row.get("countries") or row
                prior = _canonical_country_scores(items)
                if prior:
                    self._cache[str(photo_id)] = prior

    def _load_csv(self) -> None:
        assert self.path is not None and self._cache is not None
        grouped: dict[str, list[dict[str, Any]]] = {}
        with self.path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                photo_id = row.get("photo_id") or row.get("id")
                if photo_id is None:
                    continue
                grouped.setdefault(str(photo_id), []).append(row)
        for photo_id, rows in grouped.items():
            prior = _canonical_country_scores(rows)
            if prior:
                self._cache[photo_id] = prior

    def country_prior_for_photo(self, photo_id: str | None) -> dict[str, float]:
        if not photo_id:
            return {}
        return dict(self._load().get(str(photo_id), {}))

    def country_prior_for_image(self, image) -> dict[str, float]:
        photo_id = getattr(image, "photo_id", None)
        if not photo_id and hasattr(image, "info"):
            photo_id = image.info.get("photo_id")
        if not photo_id and getattr(image, "filename", None):
            photo_id = Path(image.filename).stem
        return self.country_prior_for_photo(str(photo_id) if photo_id else None)

    def blend_for_image(self, image, visual_prior: dict[str, float]) -> tuple[dict[str, float], dict]:
        retrieval_prior = self.country_prior_for_image(image)
        if not retrieval_prior:
            return visual_prior, {"enabled": self.enabled, "applied": False}
        effective_weight = self.weight
        relation = "fixed"
        if self.adaptive:
            effective_weight, relation = adaptive_retrieval_weight(
                visual_prior,
                retrieval_prior,
                self.weight,
                self.same_continent_weight,
                self.cross_continent_weight,
            )
        blended = blend_country_priors(visual_prior, retrieval_prior, effective_weight)
        return blended, {
            "enabled": self.enabled,
            "applied": True,
            "weight": self.weight,
            "effective_weight": effective_weight,
            "relation": relation,
            "retrieval_prior": retrieval_prior,
            "visual_prior": _normalize_scores(visual_prior),
        }
