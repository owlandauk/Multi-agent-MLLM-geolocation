#!/usr/bin/env python3
"""Offline ceiling diagnostic for the premature-descent regression.

For each record in a v6b result JSON, compare:
  (A) the pipeline's descended fine coordinate (pred_lat/pred_lon as-is)
  (B) the COUNTRY-POINT coordinate (geocode of pred_country alone)

Question answered: if a PERFECT confidence gate kept whichever of {fine, country}
is closer to GT, what would country@750 become? That is the maximum recoverable
accuracy of any descent gate, and the target to beat.

Also reports: how often fine beats country, how often country beats fine, ties,
and the per-threshold accuracy under (A), (B-always-coarse), and (oracle-gate).

Reuses the repo's geocode() + haversine() so country-point matches eval exactly.
Run from inside geo_pipeline/ on a host with Nominatim access.
"""
import argparse
import json
import math
import sys
import time
from functools import lru_cache

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderRateLimited
from country_aliases import canonicalize_country

THRESHOLDS = [1, 25, 200, 750, 2500]

_geocoder = Nominatim(user_agent="geo_pipeline_eval", timeout=10)


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@lru_cache(maxsize=20000)
def geocode(location_name: str):
    """Name -> (lat, lon), matching evaluate.py exactly (1.1s Nominatim throttle)."""
    try:
        loc = _geocoder.geocode(location_name)
        time.sleep(1.1)
        if loc:
            return loc.latitude, loc.longitude
    except GeocoderTimedOut:
        pass
    except GeocoderRateLimited:
        time.sleep(5)
    return None


def country_point(pred_country, cache):
    if not pred_country or str(pred_country).lower() in ("", "unknown", "none"):
        return None
    key = canonicalize_country(pred_country) or str(pred_country).lower()
    if key in cache:
        return cache[key]
    coords = geocode(key)
    cache[key] = coords
    return coords


def acc(dists):
    n = len(dists)
    return {thr: round(100.0 * sum(1 for d in dists if d is not None and d <= thr) / n, 2)
            for thr in THRESHOLDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="v6b result JSON")
    ap.add_argument("--out", default=None, help="optional JSON summary out")
    args = ap.parse_args()

    data = json.load(open(args.result))
    recs = data["records"]
    cache = {}

    fine_d, coarse_d, oracle_d = [], [], []
    fine_wins = coarse_wins = ties = coarse_ungeocodable = 0

    for r in recs:
        gt = (r["gt_lat"], r["gt_lon"])
        # (A) fine = pipeline's final coordinate as shipped
        if r.get("pred_lat") is not None:
            df = haversine(gt[0], gt[1], r["pred_lat"], r["pred_lon"])
        else:
            df = float("inf")
        fine_d.append(df)

        # (B) country point
        cp = country_point(r.get("pred_country"), cache)
        if cp is None:
            dc = float("inf")
            coarse_ungeocodable += 1
        else:
            dc = haversine(gt[0], gt[1], cp[0], cp[1])
        coarse_d.append(dc)

        # oracle gate: keep whichever is closer
        best = min(df, dc)
        oracle_d.append(best)
        if abs(df - dc) < 1e-6:
            ties += 1
        elif df < dc:
            fine_wins += 1
        else:
            coarse_wins += 1

    out = {
        "result_file": args.result,
        "n": len(recs),
        "acc_fine_current": acc(fine_d),
        "acc_always_coarse": acc(coarse_d),
        "acc_oracle_gate": acc(oracle_d),
        "fine_closer": fine_wins,
        "coarse_closer": coarse_wins,
        "ties": ties,
        "country_ungeocodable": coarse_ungeocodable,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
