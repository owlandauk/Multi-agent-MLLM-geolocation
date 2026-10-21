#!/usr/bin/env python3
"""Free-signal separability probe for the premature-descent regression.

The fine-vs-bare oracle (diagnose_fine_vs_bare_oracle.py) proved that keeping
whichever of {pipeline fine coord, bare direct coord} is closer to GT beats bare
at every threshold. That is the CEILING. This script asks the deployable
question: can any FREE recorded field (no GT, no GPU rerun, no geocoding) decide
"use bare instead of fine" well enough to approach that ceiling?

For each joined photo it evaluates 5 candidate gates. A gate FIRES = "distrust
the descent, fall back to bare's direct coordinate"; otherwise keep the pipeline
fine coordinate.

  S1  country_disagree   : pipeline.pred_country != bare.pred_country
  S2  not_country_stable : pipeline.country_stable is False
  S3  descended_fine     : geocode_source starts with "street" or "city"
  S4  S1 OR S3
  S5  S1 AND S3

Reported per gate: accuracy of the gated coordinate at all thresholds, how often
it fires (fires_pct), and how often its decision matches the oracle's choice
(decision_agree_oracle_pct). Compared against acc_pipeline_fine, acc_bare_direct,
and acc_oracle_gate. A gate is DEPLOYABLE-GOOD if its country@750 >= bare@750.

No geocoding, no model — both JSONs already carry pred_lat/pred_lon. Pure offline.
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


def norm_country(c):
    if c is None:
        return ""
    return str(c).strip().lower()


def descended_fine(r):
    src = str(r.get("geocode_source") or "").lower()
    return src.startswith("street") or src.startswith("city")


def signals_for(pr, br):
    """Return the boolean value of each candidate gate for one joined photo."""
    s1 = norm_country(pr.get("pred_country")) != norm_country(br.get("pred_country"))
    s2 = pr.get("country_stable") is False
    s3 = descended_fine(pr)
    return {
        "S1_country_disagree": s1,
        "S2_not_country_stable": s2,
        "S3_descended_fine": s3,
        "S4_S1_or_S3": s1 or s3,
        "S5_S1_and_S3": s1 and s3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--bare", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pipe = {str(r["photo_id"]): r for r in json.load(open(args.pipeline))["records"]}
    bare = {str(r["photo_id"]): r for r in json.load(open(args.bare))["records"]}
    common = sorted(set(pipe) & set(bare))

    gate_names = ["S1_country_disagree", "S2_not_country_stable",
                  "S3_descended_fine", "S4_S1_or_S3", "S5_S1_and_S3"]

    fine_d, bare_d, oracle_d = [], [], []
    gated_d = {g: [] for g in gate_names}
    fires = {g: 0 for g in gate_names}
    agree_oracle = {g: 0 for g in gate_names}

    for pid in common:
        pr, br = pipe[pid], bare[pid]
        df = dist_or_inf(pr)
        db = dist_or_inf(br)
        fine_d.append(df)
        bare_d.append(db)
        oracle_d.append(min(df, db))
        oracle_picks_bare = db < df  # oracle would switch to bare

        sig = signals_for(pr, br)
        for g in gate_names:
            fire = sig[g]
            if fire:
                fires[g] += 1
            # gated coord: bare if the gate fires, else pipeline fine
            gated_d[g].append(db if fire else df)
            if fire == oracle_picks_bare:
                agree_oracle[g] += 1

    n = len(common)
    bare750 = acc(bare_d)[750]

    gate_report = {}
    for g in gate_names:
        a = acc(gated_d[g])
        gate_report[g] = {
            "acc": a,
            "fires_pct": round(100.0 * fires[g] / n, 2),
            "decision_agree_oracle_pct": round(100.0 * agree_oracle[g] / n, 2),
            "beats_bare@750": a[750] >= bare750,
            "delta_vs_bare@750": round(a[750] - bare750, 2),
        }

    out = {
        "pipeline_file": args.pipeline,
        "bare_file": args.bare,
        "n_joined": n,
        "acc_pipeline_fine": acc(fine_d),
        "acc_bare_direct": acc(bare_d),
        "acc_oracle_gate": acc(oracle_d),
        "gates": gate_report,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
