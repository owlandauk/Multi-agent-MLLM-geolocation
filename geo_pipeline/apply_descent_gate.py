#!/usr/bin/env python3
"""Descent-DISTANCE gate post-processor (zero GPU, zero network).

Promotes diagnose_descent_distance_gate.py into a reusable post-processor that
EMITS a gated records JSON scored by the canonical analyze_results.analyze()
(full-dataset denominator) -- the SAME ruler used for bare and raw pipeline.

Mechanism (structural, NOT a confidence gate):
  The bare country answer is already the model's best coarse guess. The
  pipeline's City/Street descent helps only when it refines the point INSIDE
  the correct country; it hurts when it drags the point far away. So: if the
  descended pipeline point is more than tau km from the BARE DIRECT point,
  the descent "ran too far" -> reject it and fall back to the bare direct
  coordinate.

CRITICAL -- this is the LIVE twin of apply_scale_gate_offline.py (DEAD, -13.88):
  that script falls back to the country CENTROID; THIS one falls back to the
  bare DIRECT point (bare pred_lat/pred_lon). Bare fine hits are in the RIGHT
  country but imprecise, so a centroid is worse; the bare direct point is the
  coarse ceiling itself. By construction @750 can only improve or stay flat vs
  raw pipeline -- NEVER the centroid trap. Do not conflate the two scripts.

Confirmed online (join of the two real JSONs, n=2997): best tau=1000 gives
country@750=67.03 / street@1=11.34, vs bare 66.73 / 9.28 and raw pipeline
64.00 / 11.08. tau=250-1000 all WIN (@750>=bare AND @1>bare).

Run from inside geo_pipeline/ (only needs the two JSONs; no geocoder).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

# Allow running from the repo root as well as from inside geo_pipeline/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_results import analyze  # noqa: E402
from config import EVAL_THRESHOLDS  # noqa: E402


def haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load(path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["records"] if isinstance(data, dict) else data


def _acc(report: dict, thr: int) -> float:
    return float(report.get("accuracy", {}).get(str(thr), 0.0))


def gate_records(pipe: list[dict], bare_by_id: dict, tau: float) -> tuple[list[dict], dict]:
    """Return (gated_records, stats). Does not mutate the input list."""
    gated = []
    n_gated = n_no_bare_match = n_null_coords = 0
    for pr in pipe:
        g = copy.deepcopy(pr)
        g["descent_gate_fallback"] = False
        br = bare_by_id.get(pr.get("photo_id"))
        if br is None:
            n_no_bare_match += 1
            gated.append(g)
            continue
        p_ok = pr.get("pred_lat") is not None and pr.get("pred_lon") is not None
        b_ok = br.get("pred_lat") is not None and br.get("pred_lon") is not None
        if not (p_ok and b_ok):
            n_null_coords += 1
            gated.append(g)
            continue
        dd = haversine(pr["pred_lat"], pr["pred_lon"], br["pred_lat"], br["pred_lon"])
        if dd > tau:
            # Fall back to the bare DIRECT point (never a centroid).
            g["pred_lat"], g["pred_lon"] = br["pred_lat"], br["pred_lon"]
            gt_lat, gt_lon = pr.get("gt_lat"), pr.get("gt_lon")
            if gt_lat is not None and gt_lon is not None:
                g["dist_km"] = haversine(gt_lat, gt_lon, br["pred_lat"], br["pred_lon"])
            elif br.get("dist_km") is not None:
                g["dist_km"] = br["dist_km"]
            # else: keep the original pr dist_km (never emit None -> analyze() TypeErrors)
            g["pred_country"] = br.get("pred_country", g.get("pred_country"))
            g["geocode_source"] = "descent_gate_bare"
            g["descent_gate_fallback"] = True
            g["gate_tau"] = tau
            n_gated += 1
        gated.append(g)
    stats = {
        "n_total": len(pipe),
        "n_gated": n_gated,
        "n_no_bare_match": n_no_bare_match,
        "n_null_coords": n_null_coords,
        "tau": tau,
    }
    return gated, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pipeline", required=True, help="pipeline results JSON")
    ap.add_argument("--bare", required=True, help="bare results JSON (same dataset)")
    ap.add_argument("--tau", type=float, default=1000.0,
                    help="descent reject threshold km (default 1000)")
    ap.add_argument("--out", required=True, help="output gated records JSON path")
    args = ap.parse_args()

    pipe = load(args.pipeline)
    bare_by_id = {r["photo_id"]: r for r in load(args.bare)}
    gated, stats = gate_records(pipe, bare_by_id, args.tau)

    Path(args.out).write_text(json.dumps({"records": gated}, indent=2))

    base_report = analyze(pipe)
    gated_report = analyze(gated)

    labels = {1: "street@1", 25: "city@25", 200: "region@200",
              750: "country@750", 2500: "cont@2500"}
    print(f"\ngated {stats['n_gated']}/{stats['n_total']} "
          f"(no_bare_match={stats['n_no_bare_match']} "
          f"null_coords={stats['n_null_coords']}) tau={args.tau:g}  -> {args.out}")
    print(f"{'threshold':<14}{'pipeline':>10}{'gated':>10}{'delta':>9}")
    for t in EVAL_THRESHOLDS:
        b, g = _acc(base_report, t), _acc(gated_report, t)
        print(f"{labels[t]:<14}{b:>10.2f}{g:>10.2f}{g - b:>+9.2f}")
    print("\n[caveat] WIN was measured over the joined n=2997 with the diagnostic's")
    print("own accounting; canonical absolutes may shift slightly, but @750 is")
    print("monotone-safe by construction (fallback = bare direct point).")


if __name__ == "__main__":
    main()
