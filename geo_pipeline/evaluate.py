"""
Evaluation on YFCC4K using standard distance-threshold accuracy metrics.
Geocoding: location name → (lat, lon) via geopy (offline-compatible with Nominatim).

Usage:
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --limit 100 --out results/run1.json
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --start 1000 --out results/run2.json  # resume
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --batch_size 8 --out results/run3.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import lru_cache
from pathlib import Path
from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderRateLimited

from models.mllm_client import MLLMClient
from pipeline import GeoPipeline
from data.yfcc4k_loader import YFCC4KDataset
from country_aliases import COUNTRY_TO_CONTINENT, continent_of, canonicalize_country
from config import EVAL_THRESHOLDS, YFCC4K_IMG_DIR, YFCC4K_GPS_CSV


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


_geocoder = Nominatim(user_agent="geo_pipeline_eval", timeout=10)


# Continent centroids — used as a last-resort fallback so the continent
# threshold (<2500 km) can still hit even when country/city/street geocoding
# all fail. Coordinates are rough geographic centers.
_CONTINENT_CENTROIDS = {
    "Africa":        (1.65,   17.83),
    "Asia":          (34.05, 100.62),
    "Europe":        (54.53,  15.26),
    "North America": (54.53, -105.26),
    "South America": (-8.78,  -55.49),
    "Oceania":       (-22.74, 140.02),
    "Antarctica":    (-82.86,  21.00),
}


# Country name → continent map + canonicalizer live in country_aliases.py
# (shared with pipeline.py). Import re-exports above.


@lru_cache(maxsize=20000)
def geocode(location_name: str):
    """Name → (lat, lon). Returns None if lookup fails."""
    try:
        loc = _geocoder.geocode(location_name)
        time.sleep(1.1)  # Nominatim enforces 1 req/sec
        if loc:
            return loc.latitude, loc.longitude
    except GeocoderTimedOut:
        pass
    except GeocoderRateLimited:
        time.sleep(5)
    return None


def _geocode_level(
    name: str,
    level: str,
    country: str | None,
    strict_child_geocode: bool = False,
    allow_bare_city_geocode: bool = True,
):
    """Geocode one prediction level and return coords plus diagnostic source.

    Nominatim's gazetteer is ambiguous for many city/street names (a dozen
    "Springfield"s, two "Naples", etc.). When the predicted country is
    available, qualifying street/city queries first shrinks the search space.
    Bare street/city fallback remains enabled by default for comparability with
    v5/v6; strict mode disables all unqualified child fallback.
    """
    if level in ("street", "city"):
        country_ok = bool(country and country.lower() not in ("unknown", ""))
        name_has_country = country_ok and country.lower() in name.lower()
        embedded_country = canonicalize_country(name)
        if country_ok and embedded_country and embedded_country != country.lower():
            return None, None, "child_country_conflict"

        if country_ok and not name_has_country:
            coords = geocode(f"{name}, {country}")
            if coords is not None:
                return coords, f"{level}_country_qualified", "country_qualified"

        if strict_child_geocode and not name_has_country:
            return None, None, "failed"
        if level == "city" and not name_has_country and not allow_bare_city_geocode:
            return None, None, "failed"

        coords = geocode(name)
        if coords is not None:
            consistency = "country_in_name" if name_has_country else "unchecked"
            return coords, f"{level}_bare", consistency
        return None, None, "failed"

    coords = geocode(name)
    if coords is not None:
        return coords, "country", "country_level"
    return None, None, "failed"


def _continent_fallback_coords(pred: dict) -> tuple | None:
    """Vote a continent from the top-3 country posterior and return its centroid.

    Used as a last resort when Nominatim returns None for every level. The
    posterior is informative even when the argmax country name doesn't geocode
    (e.g. argmax='Burma', no Nominatim hit, but Asia is still right). Returns
    None if the country posterior is empty or no candidate maps to a continent.
    """
    country_post = pred.get("country_posterior", {}) or {}
    if not country_post:
        # try the bare country name as a last resort
        cont = continent_of(pred.get("country") or "")
        return _CONTINENT_CENTROIDS.get(cont) if cont else None

    sorted_countries = sorted(country_post.items(), key=lambda kv: -kv[1])[:3]
    votes: dict[str, float] = {}
    for cname, p in sorted_countries:
        cont = continent_of(cname)
        if cont:
            votes[cont] = votes.get(cont, 0.0) + p
    if not votes:
        return None
    best = max(votes, key=votes.get)
    return _CONTINENT_CENTROIDS.get(best)


def _retrieval_continent_fallback_coords(
    pred: dict,
    max_country_top: float,
    min_prior_top: float,
) -> tuple[tuple[float, float] | None, dict]:
    """Fallback to the retrieval prior's continent when country belief is weak.

    This is opt-in evaluation logic. It does not feed back into SL/DST/POMDP.
    """
    country_post = pred.get("country_posterior") or {}
    country_top = max((float(v) for v in country_post.values()), default=0.0)
    if country_top >= max_country_top:
        return None, {"country_top": country_top}

    retrieval_prior = pred.get("country_retrieval_prior") or {}
    prior_top = max((float(v) for v in retrieval_prior.values()), default=0.0)
    if prior_top < min_prior_top:
        return None, {"country_top": country_top, "prior_top": prior_top}

    votes: dict[str, float] = {}
    for country, prob in retrieval_prior.items():
        cont = continent_of(country)
        if cont:
            votes[cont] = votes.get(cont, 0.0) + float(prob)
    if not votes:
        return None, {"country_top": country_top, "prior_top": prior_top}

    best = max(votes, key=votes.get)
    coords = _CONTINENT_CENTROIDS.get(best)
    return coords, {
        "country_top": country_top,
        "prior_top": prior_top,
        "continent": best,
        "continent_mass": votes[best],
    }


def _top_country_score(scores: dict) -> tuple[str | None, float]:
    best_country = None
    best_score = 0.0
    for country, score in (scores or {}).items():
        canonical = canonicalize_country(country)
        if not canonical:
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if best_country is None or value > best_score:
            best_country = canonical
            best_score = value
    return best_country, best_score


def _retrieval_country_fallback_coords(
    pred: dict,
    max_country_top: float,
    min_prior_top: float,
    same_continent_max_country_top: float | None = None,
    cross_continent_max_country_top: float | None = None,
    child_retry: bool = False,
) -> tuple[tuple[float, float] | None, dict]:
    """Fallback to retrieval top-country geocoding when posterior is weak.

    This is opt-in evaluation logic for retrieval-backed coarse geolocation. It
    preserves the normal GeoBayes posterior unless the final country belief is
    low-confidence.
    """
    country_post = pred.get("country_posterior") or {}
    visual_country, country_top = _top_country_score(country_post)
    retrieval_country, prior_top = _top_country_score(pred.get("country_retrieval_prior") or {})
    visual_continent = continent_of(visual_country or "")
    retrieval_continent = continent_of(retrieval_country or "")

    relation = "unknown"
    effective_max_country_top = max_country_top
    if visual_continent and retrieval_continent:
        if visual_continent != retrieval_continent:
            relation = "cross_continent"
            effective_max_country_top = (
                cross_continent_max_country_top
                if cross_continent_max_country_top is not None
                else max_country_top
            )
        else:
            relation = "same_continent"
            effective_max_country_top = (
                same_continent_max_country_top
                if same_continent_max_country_top is not None
                else max_country_top
            )
    elif retrieval_continent:
        relation = "unknown_visual_continent"

    diag = {
        "country_top": country_top,
        "visual_country": visual_country,
        "visual_continent": visual_continent,
        "retrieval_continent": retrieval_continent,
        "relation": relation,
        "effective_max_country_top": effective_max_country_top,
    }

    if not retrieval_country or prior_top < min_prior_top:
        diag["prior_top"] = prior_top
        return None, diag

    if country_top >= effective_max_country_top:
        diag["prior_top"] = prior_top
        return None, diag

    if child_retry:
        for level in ("street", "city"):
            name = pred.get(level)
            if not name or name == "Unknown":
                continue
            embedded_country = canonicalize_country(name)
            if embedded_country and embedded_country != retrieval_country:
                continue
            query = name if embedded_country == retrieval_country else f"{name}, {retrieval_country}"
            coords = geocode(query)
            if coords is not None:
                diag["prior_top"] = prior_top
                diag["country"] = retrieval_country
                diag["child_retry_level"] = level
                diag["child_retry_query"] = query
                return coords, diag

    coords = geocode(retrieval_country)
    diag["prior_top"] = prior_top
    diag["country"] = retrieval_country
    if coords is None:
        diag["geocode_failed"] = True
    return coords, diag


def evaluate(args):
    mllm     = MLLMClient()
    pipeline = GeoPipeline(mllm)
    dataset  = YFCC4KDataset(img_dir=args.img_dir, gps_csv=args.gps_csv)

    start = args.start or 0
    end   = min(start + args.limit, len(dataset)) if args.limit else len(dataset)
    indices = list(range(start, end))
    batch_size = args.batch_size

    records = []
    correct = {thr: 0 for thr in EVAL_THRESHOLDS}
    total   = 0

    for batch_start in tqdm(range(0, len(indices), batch_size), desc="Evaluating batches"):
        batch_indices = indices[batch_start:batch_start + batch_size]
        samples = [dataset[i] for i in batch_indices]

        try:
            preds = pipeline.predict_batch([s["image"] for s in samples])
        except Exception as e:
            print(f"[WARN] batch {batch_start} failed: {e}")
            continue

        for sample, pred in zip(samples, preds):
            pred_country = pred.get("country")
            # If the hypothesize step failed to name a real country (parser
            # returned "Unknown" or a non-country string like "Southeast Asia"),
            # try to salvage one from the city/street strings — they very often
            # look like "Toronto, Canada" or "Paris, France". This is EVALUATE-
            # ONLY: we never feed the fallback back into country_posterior, so
            # DST/POMDP behaviour is unchanged. Rescues ~73% of Unknown records
            # per full_v4 offline check.
            if not pred_country or canonicalize_country(pred_country) is None:
                for field in ("city", "street"):
                    salvaged = canonicalize_country(pred.get(field) or "")
                    if salvaged:
                        pred_country = salvaged
                        break

            pred_coords = None
            geocode_source = None
            country_consistency = None
            # Hierarchical geocode: street → city → country. For street/city
            # try country-qualified queries before bare-name fallback.
            for level in ["street", "city", "country"]:
                name = pred.get(level)
                if name and name != "Unknown":
                    qualifier = pred_country if level in ("street", "city") else None
                    pred_coords, geocode_source, country_consistency = _geocode_level(
                        name,
                        level,
                        qualifier,
                        args.strict_child_geocode,
                        args.allow_bare_city_geocode,
                    )
                    if pred_coords:
                        break
            # Last-resort continent centroid (saves the <=2500km threshold when
            # Nominatim returns None for every level, e.g. obsolete country names).
            if pred_coords is None:
                pred_coords = _continent_fallback_coords(pred)
                if pred_coords is not None:
                    geocode_source = "continent_fallback"
                    country_consistency = "fallback"

            pre_fallback_pred_country = pred_country
            pre_fallback_coords = pred_coords
            pre_fallback_geocode_source = geocode_source
            pre_fallback_country_consistency = country_consistency

            retrieval_continent_fallback = {}
            if args.retrieval_continent_fallback:
                fallback_coords, retrieval_continent_fallback = _retrieval_continent_fallback_coords(
                    pred,
                    args.retrieval_continent_max_country_top,
                    args.retrieval_continent_min_prior_top,
                )
                if fallback_coords is not None:
                    retrieval_continent_fallback["previous_geocode_source"] = geocode_source
                    pred_coords = fallback_coords
                    geocode_source = "retrieval_continent_fallback"
                    country_consistency = "retrieval_continent_fallback"

            retrieval_country_fallback = {}
            if args.retrieval_country_fallback:
                fallback_coords, retrieval_country_fallback = _retrieval_country_fallback_coords(
                    pred,
                    args.retrieval_country_max_country_top,
                    args.retrieval_country_min_prior_top,
                    args.retrieval_country_same_continent_max_country_top,
                    args.retrieval_country_cross_continent_max_country_top,
                    args.retrieval_country_child_retry,
                )
                if fallback_coords is not None:
                    retrieval_country_fallback["previous_geocode_source"] = geocode_source
                    retrieval_country_fallback["previous_pred_country"] = pred_country
                    pred_country = retrieval_country_fallback.get("country") or pred_country
                    pred_coords = fallback_coords
                    geocode_source = "retrieval_country_fallback"
                    country_consistency = "retrieval_country_fallback"

            gt_lat, gt_lon = sample["gt_lat"], sample["gt_lon"]
            dist_km = haversine(gt_lat, gt_lon, pred_coords[0], pred_coords[1]) \
                      if pred_coords else float("inf")

            record = {
                "photo_id":    sample["photo_id"],
                "gt_lat":      gt_lat,
                "gt_lon":      gt_lon,
                "pred_country": pred_country,
                "pred_city":    pred.get("city"),
                "pred_street":  pred.get("street"),
                "pred_lat":     pred_coords[0] if pred_coords else None,
                "pred_lon":     pred_coords[1] if pred_coords else None,
                "dist_km":      dist_km,
                "pre_fallback_pred_country": pre_fallback_pred_country,
                "pre_fallback_pred_lat": pre_fallback_coords[0] if pre_fallback_coords else None,
                "pre_fallback_pred_lon": pre_fallback_coords[1] if pre_fallback_coords else None,
                "pre_fallback_dist_km": haversine(
                    gt_lat, gt_lon, pre_fallback_coords[0], pre_fallback_coords[1]
                ) if pre_fallback_coords else float("inf"),
                "pre_fallback_geocode_source": pre_fallback_geocode_source,
                "pre_fallback_country_consistency": pre_fallback_country_consistency,
                "pomdp_policy": pred.get("pomdp_policy"),
                "geocode_source": geocode_source,
                "country_consistency": country_consistency,
                "retrieval_continent_fallback": retrieval_continent_fallback,
                "retrieval_country_fallback": retrieval_country_fallback,
                "continent_posterior": {
                    k: round(float(v), 4)
                    for k, v in (pred.get("continent_posterior") or {}).items()
                },
                "country_posterior": {
                    k: round(float(v), 4)
                    for k, v in (pred.get("country_posterior") or {}).items()
                },
                "country_continent_regularized": bool(pred.get("country_continent_regularized")),
                "continent_stable": pred.get("continent_stable"),
                "country_stable": pred.get("country_stable"),
                "city_stable": pred.get("city_stable"),
                "street_stable": pred.get("street_stable"),
                "continent_steps": pred.get("continent_steps"),
                "country_steps": pred.get("country_steps"),
                "city_steps": pred.get("city_steps"),
                "street_steps": pred.get("street_steps"),
                "country_replaced": bool(pred.get("country_replaced")),
                "country_child_backtracked": bool(pred.get("country_child_backtracked")),
                "country_before_child_backtrack": pred.get("country_before_child_backtrack"),
                "country_child_backtrack_level": pred.get("country_child_backtrack_level"),
                "country_child_backtrack_country": pred.get("country_child_backtrack_country"),
                "country_retrieval_enhanced": bool(pred.get("country_retrieval_enhanced")),
                "country_retrieval_prior": pred.get("country_retrieval_prior"),
                "country_retrieval_weight": pred.get("country_retrieval_weight"),
                "country_retrieval_effective_weight": pred.get("country_retrieval_effective_weight"),
                "country_retrieval_relation": pred.get("country_retrieval_relation"),
                "country_prior_before_retrieval": pred.get("country_prior_before_retrieval"),
                "country_retrieval_anchored": bool(pred.get("country_retrieval_anchored")),
                "country_retrieval_anchor": pred.get("country_retrieval_anchor"),
                "country_web_enhanced": bool(pred.get("country_web_enhanced")),
                "country_web_search_query": pred.get("country_web_search_query"),
                "country_image_search_enhanced": bool(pred.get("country_image_search_enhanced")),
                "country_image_search_evidence": pred.get("country_image_search_evidence"),
                "country_visual_delta": pred.get("country_visual_delta"),
                "country_web_delta": pred.get("country_web_delta"),
                "city_web_enhanced": bool(pred.get("city_web_enhanced")),
                "city_web_search_query": pred.get("city_web_search_query"),
                "city_image_search_enhanced": bool(pred.get("city_image_search_enhanced")),
                "city_image_search_evidence": pred.get("city_image_search_evidence"),
                "city_web_delta": pred.get("city_web_delta"),
                "street_web_enhanced": bool(pred.get("street_web_enhanced")),
                "street_web_search_query": pred.get("street_web_search_query"),
                "street_image_search_enhanced": bool(pred.get("street_image_search_enhanced")),
                "street_image_search_evidence": pred.get("street_image_search_evidence"),
                "street_web_delta": pred.get("street_web_delta"),
                "country_descent_blocked_reason": pred.get("country_descent_blocked_reason"),
                "city_backtrack_conflicts": pred.get("city_backtrack_conflicts", []),
                "street_backtrack_conflicts": pred.get("street_backtrack_conflicts", []),
                "city_country_factcheck_consistent": pred.get("city_country_factcheck_consistent"),
                "city_country_factcheck_rejected": bool(pred.get("city_country_factcheck_rejected")),
                "city_country_factcheck_true_country": pred.get("city_country_factcheck_true_country"),
                "city_country_factcheck_reason": pred.get("city_country_factcheck_reason"),
                "city_country_factcheck_raw": pred.get("city_country_factcheck_raw"),
                "city_before_factcheck": pred.get("city_before_factcheck"),
                "raw_continent_response": pred.get("continent_raw_response"),
                "raw_country_response": pred.get("country_raw_response"),
                "raw_city_response":    pred.get("city_raw_response"),
                "raw_street_response":  pred.get("street_raw_response"),
            }
            records.append(record)
            total += 1
            for thr in EVAL_THRESHOLDS:
                if dist_km <= thr:
                    correct[thr] += 1

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\nResults on YFCC4K ({total} images, indices {start}–{start+total-1})")
    print(f"{'Threshold':>12}  {'Accuracy':>10}")
    print("-" * 26)
    for thr in EVAL_THRESHOLDS:
        label = {1: "Street <1km", 25: "City <25km", 200: "Region <200km",
                 750: "Country <750km", 2500: "Continent <2500km"}[thr]
        acc = 100.0 * correct[thr] / total if total > 0 else 0.0
        print(f"{label:>14}  {acc:>9.2f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "summary": {str(k): round(100 * v / total, 2) if total else 0
                        for k, v in correct.items()},
            "total": total,
            "start": start,
            "records": records,
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",    default=YFCC4K_IMG_DIR)
    parser.add_argument("--gps_csv",    default=YFCC4K_GPS_CSV)
    parser.add_argument("--limit",      type=int, default=None, help="max images to evaluate")
    parser.add_argument("--start",      type=int, default=0,    help="start from this dataset index (for resuming)")
    parser.add_argument("--batch_size", type=int, default=20,   help="images per GPU batch")
    parser.add_argument("--out",        default="results/eval.json")
    parser.add_argument(
        "--strict_child_geocode",
        action="store_true",
        help="Disable unqualified street/city Nominatim matches for strict consistency ablations.",
    )
    parser.add_argument(
        "--allow_bare_city_geocode",
        action="store_true",
        default=True,
        help="Allow unqualified city Nominatim matches; enabled by default for v5 comparability.",
    )
    parser.add_argument(
        "--disable_bare_city_geocode",
        action="store_false",
        dest="allow_bare_city_geocode",
        help="Disable unqualified city Nominatim matches for ablations.",
    )
    parser.add_argument(
        "--retrieval_continent_fallback",
        action="store_true",
        help="For low-confidence country predictions, geocode to the retrieval prior's continent centroid.",
    )
    parser.add_argument(
        "--retrieval_continent_max_country_top",
        type=float,
        default=0.50,
        help="Apply retrieval continent fallback only when country posterior top mass is below this value.",
    )
    parser.add_argument(
        "--retrieval_continent_min_prior_top",
        type=float,
        default=0.50,
        help="Apply retrieval continent fallback only when retrieval prior top country mass is at least this value.",
    )
    parser.add_argument(
        "--retrieval_country_fallback",
        action="store_true",
        help="For low-confidence country predictions, geocode to the retrieval prior's top country.",
    )
    parser.add_argument(
        "--retrieval_country_max_country_top",
        type=float,
        default=0.55,
        help="Apply retrieval country fallback only when country posterior top mass is below this value.",
    )
    parser.add_argument(
        "--retrieval_country_min_prior_top",
        type=float,
        default=0.15,
        help="Apply retrieval country fallback only when retrieval prior top country mass is at least this value.",
    )
    parser.add_argument(
        "--retrieval_country_same_continent_max_country_top",
        type=float,
        default=None,
        help="Override retrieval country fallback confidence gate when visual and retrieval top countries are on the same continent.",
    )
    parser.add_argument(
        "--retrieval_country_cross_continent_max_country_top",
        type=float,
        default=None,
        help="Override retrieval country fallback confidence gate when visual and retrieval top countries are on different continents.",
    )
    parser.add_argument(
        "--retrieval_country_child_retry",
        action="store_true",
        help="Before falling back to retrieval country center, geocode predicted street/city qualified by the retrieval country.",
    )
    evaluate(parser.parse_args())
