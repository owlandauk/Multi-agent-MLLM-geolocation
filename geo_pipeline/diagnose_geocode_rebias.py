#!/usr/bin/env python3
"""Direction-A probe: can COUNTRY-BIASED geocoding rescue bare @750 misses?

Hypothesis under test: among bare misses@750 that already name a real city AND
whose predicted country equals the GT country (offline reverse-geocoded) AND sit
within 1500 km of GT, some are same-name-ambiguity errors -- Nominatim returned
the wrong same-named city. Re-geocoding "(city)" constrained to the GT country's
ISO code (country_codes=) should snap those to the correct city and flip
miss->hit, WITHOUT any model change (training-free, same name string).

This is NOT a legitimate test-time method (it peeks at GT country to set the
bias). It is a CEILING probe: it measures how many of the candidates are
geocode-disambiguation-fixable at all. If the ceiling is tiny, geocode
improvement is a dead lever for @750 and we move to direction B (GPU sampling).

Prints: candidate count, how many the country-biased re-geocode moves CLOSER,
and how many flip miss->hit (<=750). Zero GPU. Nominatim rate-limited (1 req/s).
Run from inside geo_pipeline/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reverse_geocoder as rg  # noqa: E402
import pycountry  # noqa: E402
from country_aliases import canonicalize_country as canon  # noqa: E402
from geopy.geocoders import Nominatim  # noqa: E402

_R_KM = 6371.0
_MK = ("not spec", "not determin", "unknown", "n/a", "none", "undetermin")


def hav(a, o, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = math.radians(c - a), math.radians(d - o)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _R_KM * math.asin(math.sqrt(x))


def blank(r):
    c = (r.get("pred_city") or "").strip().lower()
    return c == "" or any(m in c for m in _MK)


def iso2name(cc):
    try:
        return canon(pycountry.countries.get(alpha_2=cc).name)
    except Exception:
        return cc.lower()


def load(path):
    d = json.loads(Path(path).read_text())
    return d["records"] if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bare", required=True)
    ap.add_argument("--max_km", type=float, default=1500.0,
                    help="only probe same-country misses within this many km of GT")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (0=all)")
    args = ap.parse_args()

    recs = load(args.bare)
    miss = [r for r in recs
            if r.get("dist_km") is not None and r["dist_km"] > 750 and not blank(r)]

    # offline reverse-geocode GT -> country, keep same-country & within max_km
    res = rg.search([(r["gt_lat"], r["gt_lon"]) for r in miss])
    cands = []
    for r, g in zip(miss, res):
        gt_c = canon(iso2name(g["cc"]))
        pc = canon(r.get("pred_country") or "")
        if gt_c and pc and gt_c == pc and r["dist_km"] <= args.max_km:
            cands.append((r, g["cc"]))
    if args.limit:
        cands = cands[: args.limit]

    print(f"candidates (same-country miss, dist<= {args.max_km:.0f}km): {len(cands)}")
    geo = Nominatim(user_agent="geo-rebias-probe/1.0", timeout=10)

    closer = flipped = unchanged = failed = 0
    for i, (r, cc) in enumerate(cands):
        city = r["pred_city"]
        try:
            loc = geo.geocode(city, country_codes=cc, exactly_one=True)
        except Exception:
            loc = None
        time.sleep(1.05)  # Nominatim politeness
        if loc is None:
            failed += 1
            continue
        nd = hav(r["gt_lat"], r["gt_lon"], loc.latitude, loc.longitude)
        if nd < r["dist_km"] - 1:
            closer += 1
            if nd <= 750 and r["dist_km"] > 750:
                flipped += 1
        else:
            unchanged += 1
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(cands)}  closer={closer} flipped={flipped} "
                  f"unchanged={unchanged} failed={failed}", flush=True)

    n = len(recs)
    print(f"\n=== country-biased re-geocode ceiling probe ===")
    print(f"candidates:        {len(cands)}")
    print(f"moved closer:      {closer}")
    print(f"flipped miss->hit: {flipped}   (<=750km after re-geocode)")
    print(f"unchanged/worse:   {unchanged}")
    print(f"geocode failed:    {failed}")
    print(f"\nmax @750 lift if deployed: +{100.0*flipped/n:.2f} pt "
          f"(bare@750 would go to +{flipped} hits over n={n})")
    print("NOTE: uses GT country as bias => CEILING only, not a deployable method.")


if __name__ == "__main__":
    main()
