"""
Bare-Qwen geolocation baseline (Route A diagnostic).

Runs the MLLM directly with a single GeoReasoner-style JSON prompt (no
SL/DST/POMDP/web/retrieval), geocodes the predicted city/country, and reports
the same distance-threshold accuracy as evaluate.py. Purpose: confirm the
"bare Qwen2.5-VL-7B @750km = 68.8" ceiling from the GeoBayes paper on our own
YFCC4K split, to localize how much coarse signal the full pipeline loses.

Usage:
  MLLM_BACKEND=vllm CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path VLLM_TP=1 \
    python evaluate_bare_qwen.py --limit 1500 --batch_size 8 \
    --out results/bare_qwen_1500.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from models.mllm_client import MLLMClient
from data.yfcc4k_loader import YFCC4KDataset
from country_aliases import canonicalize_country, continent_of
from config import EVAL_THRESHOLDS, YFCC4K_IMG_DIR, YFCC4K_GPS_CSV

# Reuse the exact geocoder, haversine, centroids, and JSON parsing that the
# main pipeline / evaluate use, so numbers are apples-to-apples.
from evaluate import haversine, geocode, _CONTINENT_CENTROIDS
from pipeline import _georeasoner_country_prompt, _try_parse_json


def _parse_country_city(text: str) -> tuple[str | None, str | None]:
    """Parse {"country":..,"city":..} from a GeoReasoner JSON response."""
    parsed = _try_parse_json(text)
    raw_country = ""
    raw_city = ""
    if isinstance(parsed, dict):
        raw_country = str(parsed.get("country") or parsed.get("Country") or "")
        raw_city = str(parsed.get("city") or parsed.get("City") or "")
    country = canonicalize_country(raw_country) if raw_country else None
    city = raw_city.strip() or None
    return country, city


def _geocode_bare(country: str | None, city: str | None):
    """Hierarchical geocode: city-qualified-by-country -> country -> centroid.

    Returns (coords, source). Mirrors evaluate.py's fallback ladder but for the
    single bare prediction (no posterior).
    """
    if city and country:
        coords = geocode(f"{city}, {country}")
        if coords is not None:
            return coords, "city_country_qualified"
    if city:
        coords = geocode(city)
        if coords is not None:
            return coords, "city_bare"
    if country:
        coords = geocode(country)
        if coords is not None:
            return coords, "country"
    cont = continent_of(country or "")
    if cont and cont in _CONTINENT_CENTROIDS:
        return _CONTINENT_CENTROIDS[cont], "continent_fallback"
    return None, "failed"


def evaluate_bare(args):
    mllm = MLLMClient()
    dataset = YFCC4KDataset(img_dir=args.img_dir, gps_csv=args.gps_csv)

    start = args.start or 0
    end = min(start + args.limit, len(dataset)) if args.limit else len(dataset)
    indices = list(range(start, end))
    batch_size = args.batch_size

    records = []
    correct = {thr: 0 for thr in EVAL_THRESHOLDS}
    total = 0

    for batch_start in tqdm(range(0, len(indices), batch_size), desc="Bare batches"):
        batch_indices = indices[batch_start:batch_start + batch_size]
        samples = [dataset[i] for i in batch_indices]

        messages_list = [_georeasoner_country_prompt(s["image"]) for s in samples]
        try:
            responses = mllm.batch_generate(messages_list)
        except Exception as e:  # keep going on batch failure, like evaluate.py
            print(f"[WARN] batch {batch_start} failed: {e}")
            continue

        for sample, raw in zip(samples, responses):
            country, city = _parse_country_city(raw)
            coords, source = _geocode_bare(country, city)

            gt_lat, gt_lon = sample["gt_lat"], sample["gt_lon"]
            dist_km = (
                haversine(gt_lat, gt_lon, coords[0], coords[1])
                if coords else float("inf")
            )

            total += 1
            for thr in EVAL_THRESHOLDS:
                if dist_km <= thr:
                    correct[thr] += 1

            records.append({
                "photo_id": sample["photo_id"],
                "gt_lat": gt_lat,
                "gt_lon": gt_lon,
                "pred_country": country,
                "pred_city": city,
                "pred_lat": coords[0] if coords else None,
                "pred_lon": coords[1] if coords else None,
                "dist_km": dist_km,
                "geocode_source": source,
                "raw_response": raw,
            })

        out = {
            "summary": {
                str(thr): round(100.0 * correct[thr] / total, 2)
                for thr in EVAL_THRESHOLDS
            },
            "total": total,
            "start": start,
            "records": records,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False))

    print(f"[bare_qwen] total={total}")
    for thr in EVAL_THRESHOLDS:
        print(f"  @{thr}km: {100.0 * correct[thr] / total:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=YFCC4K_IMG_DIR)
    parser.add_argument("--gps_csv", default=YFCC4K_GPS_CSV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", default="results/bare_qwen.json")
    evaluate_bare(parser.parse_args())


if __name__ == "__main__":
    main()
