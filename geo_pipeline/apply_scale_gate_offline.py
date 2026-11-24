#!/usr/bin/env python3
"""Offline scale-asymmetric descent gate — reconstruct gated coords from a saved
pipeline result JSON, with NO GPU and NO pipeline rerun.

The v6b coarse-to-fine pipeline regresses country@750 vs bare because City/Street
descent geocodes to the WRONG country more often than it helps. This script tests
a STRUCTURAL fix: force the final coordinate to stay inside the coarse country.

Rule (per record):
  - If the pipeline's chosen geocode_source is already country-qualified or
    country-in-name, the fine point is in-country -> keep it unchanged.
  - Otherwise the fine point came from a BARE (unchecked) city/street hit that may
    sit in the wrong country -> replace it with the coarse-country geocode point.
  - If the country name itself won't geocode, keep the original (no-op).

Then recompute dist_km and compare baseline vs gated accuracy via
analyze_results.analyze (canonical denominator + thresholds), plus a break:fix
decomposition (records moved closer vs farther from GT).

Usage:
  python3 geo_pipeline/apply_scale_gate_offline.py --pred results/pipeline_qwen2vl2b_im2gps3k.json
  python3 geo_pipeline/apply_scale_gate_offline.py --pred results/a.json results/b.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderRateLimited

from analyze_results import analyze
from config import EVAL_THRESHOLDS

# Sources that guarantee the fine point is inside the coarse country. A
# country-qualified query ("Lyon, France") or a name that already embeds the
# country is in-country by construction; anything else (bare city/street hit) is
# the leak the gate closes.
_IN_COUNTRY_SOURCES = {
    "city_country_qualified",
    "street_country_qualified",
    "country_in_name",
    "country",          # country-level hit is trivially in-country
    "country_level",
}

_CACHE_PATH = Path(__file__).resolve().parent / "results" / ".scale_gate_geocode_cache.json"
_geocoder = Nominatim(user_agent="geo_pipeline_scale_gate", timeout=10)


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache))


def _geocode_country(name: str, cache: dict):
    """Country name -> (lat, lon) or None. Disk-cached; polite to Nominatim."""
    key = name.strip().lower()
    if key in cache:
        v = cache[key]
        return tuple(v) if v else None
    coords = None
    try:
        loc = _geocoder.geocode(name)
        time.sleep(1.1)  # Nominatim enforces 1 req/sec
        if loc:
            coords = (loc.latitude, loc.longitude)
    except GeocoderTimedOut:
        pass
    except GeocoderRateLimited:
        time.sleep(5)
    cache[key] = list(coords) if coords else None
    return coords


def haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _finite(d) -> float:
    try:
        f = float(d)
    except (TypeError, ValueError):
        return float("inf")
    return f if math.isfinite(f) else float("inf")


def gate_records(records: list[dict], cache: dict) -> tuple[list[dict], dict]:
    """Return (gated_records, stats). Does not mutate the input list."""
    gated = []
    n_kept = n_gated = n_country_ungeocodable = 0
    closer = farther = same = 0
    for r in records:
        g = copy.deepcopy(r)
        src = r.get("geocode_source")
        has_fine = r.get("pred_lat") is not None and r.get("pred_lon") is not None
        # Only levels below country can leak. If the final answer IS the country
        # level (or already in-country), or there's no coordinate at all, leave it.
        if has_fine and src not in _IN_COUNTRY_SOURCES:
            country = r.get("pred_country") or ""
            new_coords = _geocode_country(country, cache) if country else None
            if new_coords is not None:
                g["pred_lat"], g["pred_lon"] = new_coords
                g["geocode_source"] = "scale_gate_country"
                gt_lat, gt_lon = r.get("gt_lat"), r.get("gt_lon")
                if gt_lat is not None and gt_lon is not None:
                    g["dist_km"] = haversine(gt_lat, gt_lon, new_coords[0], new_coords[1])
                n_gated += 1
                before, after = _finite(r.get("dist_km")), _finite(g.get("dist_km"))
                if after < before:
                    closer += 1
                elif after > before:
                    farther += 1
                else:
                    same += 1
            else:
                n_country_ungeocodable += 1
                n_kept += 1
        else:
            n_kept += 1
        gated.append(g)
    stats = {
        "n_total": len(records),
        "n_kept": n_kept,
        "n_gated": n_gated,
        "n_country_ungeocodable": n_country_ungeocodable,
        "gated_closer": closer,
        "gated_farther": farther,
        "gated_same": same,
    }
    return gated, stats


def _acc(report: dict, thr: int) -> float:
    return float(report.get("accuracy", {}).get(str(thr), 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="pipeline result JSON(s)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    cache = _load_cache()
    out_rows = []
    try:
        for path in args.pred:
            data = json.loads(Path(path).read_text())
            # evaluate.py writes {"summary":..,"records":[..]}; older dumps may be a
            # bare list. Match analyze_results.main()'s data.get("records", []).
            records = data.get("records", []) if isinstance(data, dict) else data
            base_report = analyze(records)
            gated, gstats = gate_records(records, cache)
            _save_cache(cache)  # checkpoint after each file's geocoding
            gated_report = analyze(gated)
            row = {
                "file": Path(path).name,
                "total": base_report.get("total"),
                "gate_stats": gstats,
                "baseline": {str(t): _acc(base_report, t) for t in EVAL_THRESHOLDS},
                "gated": {str(t): _acc(gated_report, t) for t in EVAL_THRESHOLDS},
            }
            out_rows.append(row)
    finally:
        _save_cache(cache)

    if args.json:
        print(json.dumps(out_rows, indent=2))
        return

    labels = {1: "street@1", 25: "city@25", 200: "region@200",
              750: "country@750", 2500: "cont@2500"}
    for row in out_rows:
        print(f"\n=== {row['file']}  (n={row['total']}) ===")
        gs = row["gate_stats"]
        print(f"  gated {gs['n_gated']}/{gs['n_total']} fine->country "
              f"(closer={gs['gated_closer']} farther={gs['gated_farther']} "
              f"same={gs['gated_same']}; country-ungeocodable={gs['n_country_ungeocodable']})")
        print(f"  {'threshold':<14}{'baseline':>10}{'gated':>10}{'delta':>9}")
        for t in EVAL_THRESHOLDS:
            b, g = row["baseline"][str(t)], row["gated"][str(t)]
            print(f"  {labels[t]:<14}{b:>10.2f}{g:>10.2f}{g - b:>+9.2f}")


if __name__ == "__main__":
    main()
