#!/usr/bin/env python3
"""Oracle ceiling for {pipeline fine coord} vs {bare direct coord}.

The earlier ceiling diag showed falling back to the COUNTRY CENTROID cannot beat
bare (the centroid caps large countries). bare wins because its single direct
guess lands a CITY-level coordinate in roughly the right region. So the right
fallback candidate is the bare direct coordinate, not the country point.

This joins a pipeline result JSON and a bare result JSON by photo_id, then for
each image computes the oracle gate: keep whichever of {pipeline pred coord,
bare pred coord} is closer to GT. acc_oracle_gate is the MAX accuracy a perfect
"trust bare-direct when the descent is unreliable" gate could reach. If it is
>= bare at every threshold, the "keep bare coord as fallback + confidence gate"
direction can fully remove the regression.

No geocoding needed — both files already carry pred_lat/pred_lon. Pure offline.
"""
import argparse
import json
import math
import sys

THRESHOLDS = [1, 25, 200, 750, 2500]


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def dist_or_inf(r):
    if r is None or r.get("pred_lat") is None or r.get("pred_lon") is None:
        return float("inf")
    return haversine(r["gt_lat"], r["gt_lon"], r["pred_lat"], r["pred_lon"])


def acc(dists):
    n = len(dists)
    return {thr: round(100.0 * sum(1 for d in dists if d <= thr) / n, 2) for thr in THRESHOLDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--bare", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pipe = {str(r["photo_id"]): r for r in json.load(open(args.pipeline))["records"]}
    bare = {str(r["photo_id"]): r for r in json.load(open(args.bare))["records"]}
    common = sorted(set(pipe) & set(bare))

    fine_d, bare_d, oracle_d = [], [], []
    fine_wins = bare_wins = ties = 0
    for pid in common:
        df = dist_or_inf(pipe[pid])
        db = dist_or_inf(bare[pid])
        fine_d.append(df)
        bare_d.append(db)
        oracle_d.append(min(df, db))
        if abs(df - db) < 1e-6:
            ties += 1
        elif df < db:
            fine_wins += 1
        else:
            bare_wins += 1

    out = {
        "pipeline_file": args.pipeline,
        "bare_file": args.bare,
        "n_pipeline": len(pipe),
        "n_bare": len(bare),
        "n_joined": len(common),
        "acc_pipeline_fine": acc(fine_d),
        "acc_bare_direct": acc(bare_d),
        "acc_oracle_gate": acc(oracle_d),
        "pipeline_fine_closer": fine_wins,
        "bare_closer": bare_wins,
        "ties": ties,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
