"""Faithful NAVIG-prompt reproduction on YFCC4K (Qwen2.5-VL-7B).

GeoBayes (AAAI-26) states it prompts the MLLM "to first generate a hierarchical
location name, which is then geocoded into GPS coordinates," FOLLOWING GeoReasoner
(Li et al. 2024) and NAVIG (Zhang et al. 2025, arxiv 2502.14638). The paper gives
no prompt text of its own; it defers to those two. NAVIG's prompt is the only
clean, standalone, un-finetuned "guess the location" prompt of the two
(GeoReasoner uses a LoRA-finetuned model + a terse prompt, not transferable).

This script reproduces NAVIG's `base_query` VERBATIM (from the NAVIG repo
SparrowZheyuan18/Navig, prompts.py) and — critically, matching NAVIG — takes the
model's OWN latitude/longitude as the prediction, NOT a country->centroid geocode.
That direct-coordinate output is NAVIG's design and the single biggest difference
from our earlier bare harness (which named a country then geocoded to a centroid,
losing precision for large countries).

NO SL/DST/POMDP, NO Bayesian loop, NO WebSearch. This is the bare-prompt layer
only, to answer: does the FAITHFUL upstream prompt lift our bare country@750 from
45.2 toward the paper's 50.7?

Usage (server, GPU free only):
  MLLM_BACKEND=vllm CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path VLLM_TP=1 \
    python geobayes_repro/eval_navig_prompt.py --limit 1500 --batch_size 8 \
    --out results/navig_prompt_1500.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make geo_pipeline importable when run from repo root.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "geo_pipeline"))

from tqdm import tqdm

from models.mllm_client import MLLMClient
from data.yfcc4k_loader import YFCC4KDataset
from config import EVAL_THRESHOLDS, YFCC4K_IMG_DIR, YFCC4K_GPS_CSV

# Reuse the EXACT haversine + JSON parser the main pipeline uses, so the only
# variables vs our earlier bare run are (1) the prompt and (2) direct-coord output.
from evaluate import haversine
from pipeline import _try_parse_json


# --- NAVIG base_query, VERBATIM from SparrowZheyuan18/Navig prompts.py ---------
# Do NOT edit this string. It is copied character-for-character from the NAVIG
# repository. The trailing newline is part of the original template.
NAVIG_BASE_QUERY = (
    "Suppose you are an expert in geo-localization. Given an image, guess the "
    "location of it, which contains the country, city, and its coordinates, with "
    'your answer formed as a json: {"country":"", "city":"", "latitude":, '
    '"longitude":}. Don\'t include any other information in your output. You '
    "should try your best to take a guess of each level in the json anyway, don't "
    "say you are unable to decide. Don't include unknown or empty value in the "
    "output dict, always take a guess anyway.\n"
)


def _navig_prompt(image) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": NAVIG_BASE_QUERY},
            ],
        }
    ]


def _parse_latlon(text: str):
    """Extract (lat, lon) from NAVIG's JSON output. Returns (lat, lon, country)
    or (None, None, country). country kept only for diagnostics."""
    parsed = _try_parse_json(text)
    if not isinstance(parsed, dict):
        return None, None, None
    country = parsed.get("country")
    lat = parsed.get("latitude")
    lon = parsed.get("longitude")
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None, None, country
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None, None, country
    return lat, lon, country


def evaluate(args):
    mllm = MLLMClient()
    dataset = YFCC4KDataset(img_dir=args.img_dir, gps_csv=args.gps_csv)

    start = args.start or 0
    end = min(start + args.limit, len(dataset)) if args.limit else len(dataset)
    indices = list(range(start, end))

    records = []
    correct = {thr: 0 for thr in EVAL_THRESHOLDS}
    total = 0
    n_parse_fail = 0

    for bstart in tqdm(range(0, len(indices), args.batch_size), desc="navig batches"):
        batch_indices = indices[bstart:bstart + args.batch_size]
        samples = [dataset[i] for i in batch_indices]
        messages_list = [_navig_prompt(s["image"]) for s in samples]
        try:
            responses = mllm.batch_generate(messages_list)
        except Exception as e:
            print(f"[WARN] batch {bstart} failed: {e}")
            continue

        for sample, raw in zip(samples, responses):
            lat, lon, country = _parse_latlon(raw)
            if lat is None:
                n_parse_fail += 1

            gt_lat, gt_lon = sample["gt_lat"], sample["gt_lon"]
            dist_km = (
                haversine(gt_lat, gt_lon, lat, lon)
                if lat is not None else float("inf")
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
                "pred_lat": lat,
                "pred_lon": lon,
                "dist_km": dist_km,
                "parsed": lat is not None,
                "raw_response": raw,
            })

        out = {
            "summary": {
                str(thr): round(100.0 * correct[thr] / total, 2)
                for thr in EVAL_THRESHOLDS
            },
            "total": total,
            "start": start,
            "parse_fail": n_parse_fail,
            "prompt": "NAVIG base_query verbatim; direct lat/lon (no geocode)",
            "records": records,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False))

    print(f"[navig_repro] total={total} parse_fail={n_parse_fail}")
    for thr in EVAL_THRESHOLDS:
        print(f"  @{thr}km: {100.0 * correct[thr] / total:.2f}%")
    print("[navig_repro] target: paper YFCC4K country@750=50.7, ours-bare=45.2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=YFCC4K_IMG_DIR)
    parser.add_argument("--gps_csv", default=YFCC4K_GPS_CSV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", default="results/navig_prompt.json")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
