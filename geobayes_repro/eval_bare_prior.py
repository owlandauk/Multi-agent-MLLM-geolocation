"""GeoBayes paper bare-prior reproduction on YFCC4K.

Runs the MLLM ONCE per image with the paper's hierarchical-location-name prompt,
parses candidate countries with confidence scores, forms the Eq.5 calibrated
prior, takes the argmax country, geocodes it with the SAME harness as
evaluate.py, and reports the standard distance-threshold accuracy.

This is the minimal decisive experiment: does the paper's PROMPT + PRIOR alone
lift our bare country@750 from 45.2 toward the paper's 50.7? No SL/DST/POMDP, no
Bayesian update loop, no WebSearch. Only the coarse (country) prior layer.

Usage (server, GPU free only):
  MLLM_BACKEND=vllm CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path VLLM_TP=1 \
    python geobayes_repro/eval_bare_prior.py --limit 1500 --batch_size 8 \
    --out results/geobayes_repro_bareprior_1500.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make geo_pipeline importable when run from repo root.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "geo_pipeline"))

from tqdm import tqdm

from models.mllm_client import MLLMClient
from data.yfcc4k_loader import YFCC4KDataset
from country_aliases import canonicalize_country, continent_of
from config import EVAL_THRESHOLDS, YFCC4K_IMG_DIR, YFCC4K_GPS_CSV

# Reuse the EXACT geocoder / haversine / centroids / JSON parser the main
# pipeline uses, so the only variable vs evaluate_bare_qwen.py is prompt+prior.
from evaluate import haversine, geocode, _CONTINENT_CENTROIDS
from pipeline import _try_parse_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior import eq5_prior, argmax_prior


def _paper_prior_prompt(image) -> list:
    """Paper's hierarchical-location-name prompt: multiple candidate locations,
    each with a confidence score, so Eq.5 can form a calibrated prior.

    Wording follows the paper's description (p.8999: "instruct MLLM to generate
    candidate locations l_i with confidence scores s_i"). We ask for COUNTRY
    candidates specifically since country@750 is the level under test.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "You are an expert image geo-localization system. "
                    "Analyze the visual clues in this image (architecture, signage, "
                    "script/language, vegetation, vehicles, road markings, climate) "
                    "and list the most likely COUNTRIES where it was taken. "
                    "Give several candidates ranked by likelihood, each with a "
                    "confidence score between 0 and 1 reflecting how strongly the "
                    "visual evidence supports it. Do not commit to a single guess; "
                    "keep plausible alternatives.\n\n"
                    "Respond with valid JSON only, no markdown fences:\n"
                    '{"candidates": [{"country": "<name>", "confidence": <0-1>}, ...]}'
                )},
            ],
        }
    ]


def _parse_candidates(text: str) -> dict[str, float]:
    """Parse {"candidates":[{"country":..,"confidence":..}]} -> {country: score}.

    Canonicalizes country names and keeps the max score if a country repeats.
    Tolerates the model returning a bare list or a 'hypotheses'/'location' schema.
    """
    parsed = _try_parse_json(text)
    items = []
    if isinstance(parsed, dict):
        items = parsed.get("candidates") or parsed.get("hypotheses") or []
    elif isinstance(parsed, list):
        items = parsed
    scores: dict[str, float] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        raw = it.get("country") or it.get("location") or it.get("name") or ""
        canon = canonicalize_country(str(raw)) if raw else None
        if not canon:
            continue
        try:
            conf = float(it.get("confidence", it.get("score", 0.0)))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        scores[canon] = max(scores.get(canon, 0.0), conf)
    return scores


def _geocode_country(country: str | None):
    """Country name -> coords, with continent-centroid fallback. Mirrors the
    country branch of evaluate_bare_qwen._geocode_bare (no city here: the prior
    under test is country-level)."""
    if country:
        coords = geocode(country)
        if coords is not None:
            return coords, "country"
    cont = continent_of(country or "")
    if cont and cont in _CONTINENT_CENTROIDS:
        return _CONTINENT_CENTROIDS[cont], "continent_fallback"
    return None, "failed"


def evaluate(args):
    mllm = MLLMClient()
    dataset = YFCC4KDataset(img_dir=args.img_dir, gps_csv=args.gps_csv)

    start = args.start or 0
    end = min(start + args.limit, len(dataset)) if args.limit else len(dataset)
    indices = list(range(start, end))

    records = []
    correct = {thr: 0 for thr in EVAL_THRESHOLDS}
    total = 0

    for bstart in tqdm(range(0, len(indices), args.batch_size), desc="repro batches"):
        batch_indices = indices[bstart:bstart + args.batch_size]
        samples = [dataset[i] for i in batch_indices]
        messages_list = [_paper_prior_prompt(s["image"]) for s in samples]
        try:
            responses = mllm.batch_generate(messages_list)
        except Exception as e:
            print(f"[WARN] batch {bstart} failed: {e}")
            continue

        for sample, raw in zip(samples, responses):
            scores = _parse_candidates(raw)
            prior = eq5_prior(scores)
            country = argmax_prior(prior)
            coords, source = _geocode_country(country)

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
                "prior_top_mass": round(max(prior.values()), 4) if prior else None,
                "n_candidates": len(scores),
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
            "prior": {"temp": 1.5, "cutoff": 0.6, "eq": "Eq.5 GeoBayes AAAI-26"},
            "records": records,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False))

    print(f"[geobayes_repro] total={total}")
    for thr in EVAL_THRESHOLDS:
        print(f"  @{thr}km: {100.0 * correct[thr] / total:.2f}%")
    print("[geobayes_repro] target: paper YFCC4K country@750=50.7, ours-bare=45.2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=YFCC4K_IMG_DIR)
    parser.add_argument("--gps_csv", default=YFCC4K_GPS_CSV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", default="results/geobayes_repro_bareprior.json")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
