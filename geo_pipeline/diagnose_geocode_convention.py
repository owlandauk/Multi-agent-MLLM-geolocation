#!/usr/bin/env python3
"""Offline geocode-CONVENTION sensitivity diagnostic.

Different geo-localization papers report country@750 under different, usually
undisclosed, geocode/scoring *conventions* (rulers). This script quantifies how
much the SAME bare-model predictions move under three legitimate conventions,
so the thesis can report its bare number with an explicit, reproducible ruler
instead of chasing an unpublished one.

Conventions (all operate on an existing bare result JSON, no GPU):
  (1) FULL-SET baseline        — pred_lat/pred_lon as-is, denominator = all records.
                                  This is the user's current ruler (name->geocode+distance).
  (2) CENTROID-SWAP (lenient)  — for records that fell back to a country CENTROID,
                                  replace the centroid with the country's CAPITAL point.
                                  Tests whether "better country anchor" is a real lever.
  (3) NON-LOCALIZABLE EXCLUSION (GaGA-style) — drop genuinely non-localizable images
                                  (model ABSTAINED: blank/"not specified"/"not
                                  determinable" city with a fallback country) from the
                                  DENOMINATOR, then score the rest with ruler (1).
                                  This is the convention GaGA notes (~35% of IM2GPS3K
                                  are non-localizable selfies/indoor shots).

Reports a per-threshold [1,25,200,750,2500] side-by-side table so the geocode
convention sensitivity is explicit and reproducible.

Reuses the repo's geocode() + haversine() so points match eval exactly.
Run from inside geo_pipeline/ on a host with Nominatim access.
"""
import argparse
import json
import math
import time
from functools import lru_cache

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderRateLimited
from country_aliases import canonicalize_country

THRESHOLDS = [1, 25, 200, 750, 2500]

_geocoder = Nominatim(user_agent="geo_pipeline_eval", timeout=10)

# city strings the model emits when it declines to localize
_ABSTAIN_MARKERS = ("not spec", "not determin", "unknown", "n/a", "none", "undetermin")


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


def is_abstention(rec) -> bool:
    """True if the model declined to localize: blank/placeholder city.

    A record that fell back to a country centroid because the model could only
    name a country (or named nothing) is a genuine non-localizable image, not a
    geocode failure. GaGA-style conventions exclude these from the denominator.
    """
    if rec.get("geocode_source") != "country":
        return False
    city = (rec.get("pred_city") or "").strip().lower()
    if city == "":
        return True
    return any(m in city for m in _ABSTAIN_MARKERS)


def capital_point(pred_country, cache):
    """Geocode the country's capital as a lenient anchor (vs centroid)."""
    if not pred_country:
        return None
    canon = canonicalize_country(pred_country)
    key = f"__capital__::{canon}"
    if key in cache:
        return tuple(cache[key]) if cache[key] else None
    pt = geocode(f"capital of {canon}") or geocode(canon)
    cache[key] = list(pt) if pt else None
    return pt


def accuracy_table(dists):
    n = len(dists)
    if n == 0:
        return {t: 0.0 for t in THRESHOLDS}, 0
    out = {}
    for t in THRESHOLDS:
        hits = sum(1 for d in dists if d is not None and d <= t)
        out[t] = 100.0 * hits / n
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="bare result JSON (list of records)")
    ap.add_argument("--cache", default="results/.convention_geocode_cache.json",
                    help="Nominatim lookup cache (capital points)")
    args = ap.parse_args()

    with open(args.pred) as f:
        data = json.load(f)
    # result JSONs are either a flat list of records or a {"records": [...]} wrapper
    recs = data["records"] if isinstance(data, dict) else data

    try:
        with open(args.cache) as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    # per-record GT + the three convention distances
    full_dists = []        # (1) full-set baseline
    swap_dists = []        # (2) centroid->capital swap
    excl_dists = []        # (3) non-localizable exclusion
    n_abstain = 0
    n_centroid = 0

    for rec in recs:
        gt_lat, gt_lon = rec.get("gt_lat"), rec.get("gt_lon")
        if gt_lat is None or gt_lon is None:
            full_dists.append(None); swap_dists.append(None)
            continue

        # (1) full-set, as saved
        d_full = rec.get("dist_km")
        if d_full is None and rec.get("pred_lat") is not None:
            d_full = haversine(gt_lat, gt_lon, rec["pred_lat"], rec["pred_lon"])
        full_dists.append(d_full)

        # (2) centroid-swap: only centroid fallbacks are affected
        if rec.get("geocode_source") == "country":
            n_centroid += 1
            cap = capital_point(rec.get("pred_country"), cache)
            swap_dists.append(haversine(gt_lat, gt_lon, cap[0], cap[1]) if cap else d_full)
        else:
            swap_dists.append(d_full)

        # (3) non-localizable exclusion: drop abstentions from denominator
        if is_abstention(rec):
            n_abstain += 1
        else:
            excl_dists.append(d_full)

    with open(args.cache, "w") as f:
        json.dump(cache, f)

    full_acc, n_full = accuracy_table(full_dists)
    swap_acc, n_swap = accuracy_table(swap_dists)
    excl_acc, n_excl = accuracy_table(excl_dists)

    print(f"\nrecords={len(recs)}  centroid-fallbacks={n_centroid}  "
          f"non-localizable(abstention)={n_abstain}\n")
    hdr = "threshold(km)".ljust(14) + "".join(
        f"{t:>10}" for t in THRESHOLDS)
    print(hdr)
    print("-" * len(hdr))

    def row(label, acc, n):
        line = f"{label}".ljust(14) + "".join(f"{acc[t]:>9.2f}%" for t in THRESHOLDS)
        print(f"{line}   (n={n})")

    row("(1) full", full_acc, n_full)
    row("(2) capital", swap_acc, n_swap)
    row("(3) excl-NL", excl_acc, n_excl)

    print("\ncountry@750 convention spread: "
          f"full={full_acc[750]:.2f}  capital={swap_acc[750]:.2f}  "
          f"excl-NL={excl_acc[750]:.2f}  "
          f"(Δ capital={swap_acc[750]-full_acc[750]:+.2f}, "
          f"Δ excl-NL={excl_acc[750]-full_acc[750]:+.2f})")


if __name__ == "__main__":
    main()
