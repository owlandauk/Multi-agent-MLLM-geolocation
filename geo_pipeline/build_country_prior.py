"""Build an empirical country-frequency prior from YFCC4k GPS labels.

Reverse-geocodes every (lat, lon) in the GPS CSV to a country (offline via
`reverse_geocoder`), canonicalizes the name, and writes normalized frequencies
to JSON. This file is the divisor for the country-prior debias in DSTModule
(COUNTRY_PRIOR_FILE + COUNTRY_PRIOR_DEBIAS_GAMMA).

CAVEAT: frequencies computed over the full YFCC4k set include the eval-split
distribution. For a leakage-clean prior, point --gps_csv at a train/other split.
The debias is a population prior over countries, NOT a per-image label, but note
this in the thesis.

Run on the server (needs reverse_geocoder + pycountry):
    python build_country_prior.py --out results/country_prior.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from config import YFCC4K_GPS_CSV
from country_aliases import canonicalize_country


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gps_csv", default=YFCC4K_GPS_CSV)
    ap.add_argument("--out", default="results/country_prior.json")
    ap.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="add-k smoothing added to every observed country count before normalizing",
    )
    args = ap.parse_args()

    import reverse_geocoder as rg
    import pycountry

    meta = pd.read_csv(args.gps_csv)
    coords = [(float(r["lat"]), float(r["lon"])) for _, r in meta.iterrows()]
    print(f"[country_prior] reverse-geocoding {len(coords)} points ...")
    results = rg.search(coords)  # offline; returns dicts with 'cc' (ISO2)

    counts: Counter[str] = Counter()
    unresolved = 0
    for res in results:
        cc = res.get("cc")
        name = None
        if cc:
            country = pycountry.countries.get(alpha_2=cc)
            if country is not None:
                name = country.name
        canon = canonicalize_country(name) if name else None
        if canon is None:
            unresolved += 1
            continue
        counts[canon] += 1

    total = sum(counts.values())
    if total == 0:
        raise SystemExit("[country_prior] no countries resolved — check deps/CSV")

    k = args.smoothing
    denom = total + k * len(counts)
    freq = {c: (n + k) / denom for c, n in counts.items()}

    with open(args.out, "w") as fh:
        json.dump(freq, fh, indent=2, sort_keys=True, ensure_ascii=True)

    top = sorted(freq.items(), key=lambda kv: -kv[1])[:10]
    print(f"[country_prior] wrote {len(freq)} countries to {args.out} "
          f"({unresolved} points unresolved)")
    print("[country_prior] top-10:", ", ".join(f"{c}={p:.3f}" for c, p in top))


if __name__ == "__main__":
    main()
