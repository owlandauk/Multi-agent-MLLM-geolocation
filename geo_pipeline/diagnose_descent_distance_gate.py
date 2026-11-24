#!/usr/bin/env python3
"""Offline descent-DISTANCE gate validator (zero GPU, zero network).

Question: can a structural gate make the full pipeline's country@750 >= the
bare model's, WITHOUT losing the pipeline's only real gain (street@1)?

Mechanism under test (NOT a confidence gate):
  The bare country answer is already the model's best coarse guess (bare @750 is
  the ceiling). The pipeline's City/Street descent only helps when it refines
  the point INSIDE the correct country; it hurts when it drags the point far
  away (out of the correct country). So: if the descended pipeline point is more
  than tau km from the bare point, the descent "ran too far" -> reject it and
  fall back to the BARE DIRECT coordinate.

CRITICAL: fallback is the bare DIRECT point (pred_lat/pred_lon from the bare
run), NEVER a country centroid. The prior centroid-fallback gate regressed
-13.88 @750 (scale-asymmetric-gate-closed) precisely because bare fine hits are
in the RIGHT country but imprecise, and a centroid is worse. Using the bare
direct point means @750 can only improve or stay flat vs bare by construction.

Joins a pipeline results JSON and a bare results JSON by photo_id, both scored
with the same ruler (dist_km already present). Sweeps tau and reports the full
per-threshold table + @750 break/fix decomposition for each tau.

Decision gate: a tau is a WIN iff @750 >= bare@750 AND @1 > bare@1 (street gain
preserved). If no tau wins, the honest fallback is per-scale-best reporting.

Run from inside geo_pipeline/ (only needs the two JSONs; no geocoder).
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from country_aliases import canonicalize_country  # noqa: E402
except Exception:  # pragma: no cover - fallback if import path differs
    def canonicalize_country(name):
        return (name or "").strip().lower() or None

THRESHOLDS = [1, 25, 200, 750, 2500]
TAUS = [150, 250, 350, 500, 750, 1000, 2000, float("inf")]


def _cross_country(pr: dict, br: dict) -> bool:
    """True if the pipeline point and the bare point name DIFFERENT countries.

    This is the A2 signature: the descent crossed into another country's city
    while the bare answer stayed in (what is usually) the correct country.
    Canonicalized so "USA"/"United States" etc. are treated as the same.
    """
    pc = canonicalize_country(pr.get("pred_country") or "")
    bc = canonicalize_country(br.get("pred_country") or "")
    if pc is None or bc is None:
        return False
    return pc != bc


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d["records"] if isinstance(d, dict) else d


def acc_table(dists):
    n = len(dists)
    out = {}
    for t in THRESHOLDS:
        out[t] = 100.0 * sum(1 for d in dists if d is not None and d <= t) / n if n else 0.0
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", required=True, help="pipeline results JSON")
    ap.add_argument("--bare", required=True, help="bare results JSON (same dataset)")
    args = ap.parse_args()

    pipe = load(args.pipeline)
    bare = {r["photo_id"]: r for r in load(args.bare)}

    # bare baseline table (direct bare dist_km, over the joined set)
    bare_dists = []
    joined = []  # (pipe_rec, bare_rec, descent_dist_km_between_points)
    for pr in pipe:
        br = bare.get(pr["photo_id"])
        if br is None:
            continue
        bare_dists.append(br.get("dist_km"))
        # distance between pipeline point and bare point
        dd = None
        if (pr.get("pred_lat") is not None and br.get("pred_lat") is not None):
            dd = haversine(pr["pred_lat"], pr["pred_lon"],
                           br["pred_lat"], br["pred_lon"])
        joined.append((pr, br, dd))

    bare_acc, n = acc_table(bare_dists)
    bare750, bare1 = bare_acc[750], bare_acc[1]

    print(f"\njoined records = {n}   bare@750 = {bare750:.2f}   bare@1 = {bare1:.2f}\n")
    hdr = "tau(km)".ljust(10) + "".join(f"{t:>9}" for t in THRESHOLDS) \
        + "   gated  brk  fix  xcGate  xcFix   WIN?"
    print(hdr)
    print("-" * len(hdr))

    # tau=inf == never gate == raw pipeline (reference row).
    # xc_gated = gated cases where pipeline & bare name DIFFERENT countries (the A2
    #            signature: descent crossed into another country's city).
    # xc_fix   = those cross-country gates that flipped a @750 miss into a hit.
    for tau in TAUS:
        gated_dists = []
        n_gated = brk = fix = xc_gated = xc_fix = 0
        for pr, br, dd in joined:
            dp = pr.get("dist_km")      # pipeline (descended) distance to GT
            db = br.get("dist_km")      # bare distance to GT
            # gate: if descent moved the point > tau from bare point, fall back to bare
            gated_now = dd is not None and dd > tau
            if gated_now:
                chosen = db
                n_gated += 1
            else:
                chosen = dp
            gated_dists.append(chosen)
            # break/fix vs bare at 750 (does the GATED result lose/gain vs bare?)
            is_fix = False
            if db is not None and chosen is not None:
                bh, gh = db <= 750, chosen <= 750
                if bh and not gh:
                    brk += 1
                elif gh and not bh:
                    fix += 1
                    is_fix = True
            if gated_now and _cross_country(pr, br):
                xc_gated += 1
                if is_fix:
                    xc_fix += 1
        ga, _ = acc_table(gated_dists)
        win = "YES" if (ga[750] >= bare750 and ga[1] > bare1) else ""
        label = "inf(raw)" if tau == float("inf") else f"{int(tau)}"
        print(label.ljust(10)
              + "".join(f"{ga[t]:>8.2f}%" for t in THRESHOLDS)
              + f"  {n_gated:>5} {brk:>4} {fix:>4}  {xc_gated:>6} {xc_fix:>6}   {win}")

    print(f"\nWIN rule: @750 >= {bare750:.2f} AND @1 > {bare1:.2f}")
    print("xcGate = cross-country gated (A2 signature); xcFix = those that fixed @750.")


if __name__ == "__main__":
    main()
