"""Evaluate GeoCLIP direct top-1 and top-k oracle accuracy on YFCC4K.

This is the first required baseline for the do-no-harm hybrid pipeline. It tells
us whether SL-DST-POMDP reranking has room to improve GeoCLIP top-1, and it
prevents future pipeline changes from hiding a regression behind geocoding text.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import EVAL_THRESHOLDS, YFCC4K_GPS_CSV, YFCC4K_IMG_DIR
from geoclip_candidates import load_candidate_cache


_THRESHOLD_LABELS = {
    1: "Street <1km",
    25: "City <25km",
    200: "Region <200km",
    750: "Country <750km",
    2500: "Continent <2500km",
}


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _load_rows(img_dir: Path, gps_csv: Path, start: int, limit: int | None) -> list[dict]:
    meta = pd.read_csv(gps_csv)
    end = len(meta) if limit is None else min(len(meta), start + limit)
    rows = []
    for _, row in meta.iloc[start:end].iterrows():
        photo_id = str(row["photo_id"])
        if (img_dir / f"{photo_id}.jpg").exists():
            rows.append(
                {
                    "photo_id": photo_id,
                    "gt_lat": float(row["lat"]),
                    "gt_lon": float(row["lon"]),
                }
            )
    return rows


def _parse_oracle_ks(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(max(1, int(item)))
    return sorted(set(values)) or [1, 5, 10, 25]


def _accuracy(correct: dict[int, int], total: int) -> dict[str, float]:
    return {
        str(thr): round(100.0 * correct[thr] / total, 2) if total else 0.0
        for thr in EVAL_THRESHOLDS
    }


def _candidate_distance(gt_lat: float, gt_lon: float, candidate: dict) -> float:
    return haversine(gt_lat, gt_lon, float(candidate["lat"]), float(candidate["lon"]))


def evaluate(args) -> None:
    cache = load_candidate_cache(args.candidate_cache)
    rows = _load_rows(Path(args.img_dir), Path(args.gps_csv), args.start, args.limit)
    oracle_ks = _parse_oracle_ks(args.oracle_k)

    records = []
    top1_correct = {thr: 0 for thr in EVAL_THRESHOLDS}
    oracle_correct = {
        k: {thr: 0 for thr in EVAL_THRESHOLDS}
        for k in oracle_ks
    }
    missing = 0

    for row in tqdm(rows, desc="Evaluating GeoCLIP candidates"):
        photo_id = row["photo_id"]
        gt_lat, gt_lon = row["gt_lat"], row["gt_lon"]
        entry = cache.get(photo_id) or {}
        candidates = list(entry.get("candidates") or [])
        if not candidates:
            missing += 1
            dist_km = float("inf")
            top1 = {}
        else:
            top1 = candidates[0]
            dist_km = _candidate_distance(gt_lat, gt_lon, top1)

        oracle_distances = {}
        oracle_candidates = {}
        for k in oracle_ks:
            topk = candidates[:k]
            if not topk:
                oracle_dist = float("inf")
                best_candidate = None
            else:
                scored = [(_candidate_distance(gt_lat, gt_lon, cand), cand) for cand in topk]
                oracle_dist, best_candidate = min(scored, key=lambda item: item[0])
            oracle_distances[str(k)] = oracle_dist
            if best_candidate is not None:
                oracle_candidates[str(k)] = {
                    "rank": best_candidate.get("rank"),
                    "lat": best_candidate.get("lat"),
                    "lon": best_candidate.get("lon"),
                    "score": best_candidate.get("score"),
                    "country": best_candidate.get("country"),
                    "city": best_candidate.get("city"),
                }

        for thr in EVAL_THRESHOLDS:
            if dist_km <= thr:
                top1_correct[thr] += 1
            for k in oracle_ks:
                if oracle_distances[str(k)] <= thr:
                    oracle_correct[k][thr] += 1

        country_prior = {
            item["country"]: item["score"]
            for item in (entry.get("country_prior") or [])
            if item.get("country")
        }
        records.append(
            {
                "photo_id": photo_id,
                "gt_lat": gt_lat,
                "gt_lon": gt_lon,
                "pred_lat": top1.get("lat"),
                "pred_lon": top1.get("lon"),
                "pred_country": top1.get("country"),
                "pred_city": top1.get("city"),
                "pred_street": "GeoCLIP rank-1 GPS",
                "dist_km": dist_km,
                "geocode_source": "geoclip_top1_direct",
                "country_consistency": "geoclip_direct",
                "country_posterior": country_prior,
                "geoclip_top1_score": top1.get("score"),
                "geoclip_top1_raw_score": top1.get("raw_score"),
                "geoclip_oracle_dist_km": oracle_distances,
                "geoclip_oracle_candidate": oracle_candidates,
                "geoclip_candidates": candidates[: args.record_top_k],
            }
        )

    total = len(records)
    print(f"\nGeoCLIP candidate evaluation ({total} images, missing={missing})")
    print("Direct top-1")
    for thr in EVAL_THRESHOLDS:
        label = _THRESHOLD_LABELS[thr]
        acc = 100.0 * top1_correct[thr] / total if total else 0.0
        print(f"{label:>18}: {acc:6.2f}%")

    print("\nTop-k oracle")
    for k in oracle_ks:
        cells = []
        for thr in EVAL_THRESHOLDS:
            acc = 100.0 * oracle_correct[k][thr] / total if total else 0.0
            cells.append(f"{thr}km={acc:.2f}%")
        print(f"  oracle@{k}: " + ", ".join(cells))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "summary": _accuracy(top1_correct, total),
                "oracle_summary": {
                    str(k): _accuracy(oracle_correct[k], total)
                    for k in oracle_ks
                },
                "total": total,
                "start": args.start,
                "missing": missing,
                "candidate_cache": str(args.candidate_cache),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_cache", default="results/geoclip_candidates.json")
    parser.add_argument("--img_dir", default=YFCC4K_IMG_DIR)
    parser.add_argument("--gps_csv", default=YFCC4K_GPS_CSV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--oracle_k", default="1,5,10,25")
    parser.add_argument("--record_top_k", type=int, default=5)
    parser.add_argument("--out", default="results/geoclip_direct_oracle.json")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

