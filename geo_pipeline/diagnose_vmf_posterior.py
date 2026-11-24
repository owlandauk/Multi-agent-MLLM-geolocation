#!/usr/bin/env python3
"""Spherical von Mises-Fisher (vMF) posterior aggregator -- OFFLINE diagnostic.

NEW FRAMEWORK (not a gate, not a centroid, not a medoid):
  Model the location posterior for each image as a MIXTURE of vMF kernels on the
  unit sphere S^2. Each available coordinate estimate for the image (bare direct
  point, pipeline descended point, pipeline pre-fallback point) becomes one vMF
  kernel centered on that point; its concentration kappa encodes how trustworthy
  that estimate is (e.g. SL name-agreement). The posterior is the weighted vMF
  mixture (= spherical KDE). The answer is the posterior DENSITY MODE -- the
  continuous peak, found by numeric search over a geodesic grid seeded at the
  candidate points.

WHY this is mathematically distinct from every dead approach:
  * centroid (DEAD, -13.88): arithmetic mean of points -> dragged by outliers.
  * medoid: discrete, must pick an existing candidate point.
  * GeoBayes MAP: argmax over DISCRETE geographic hypotheses (country list).
  * vMF-mode (HERE): argmax of a CONTINUOUS spherical density. Near-duplicate
    candidates reinforce a shared peak; a lone outlier kernel cannot win unless
    it is the only mass. Reduces to the single point when only one kernel exists.

Literature license (training-free): "A Generative Approach to Global Visual
Geolocation" (arXiv 2412.06781) establishes modeling the geolocation posterior as
a continuous distribution on the sphere and taking its mode / entropy -- but with
a TRAINED Riemannian flow. We implement the SAME framework training-free via a
vMF kernel-density mixture over externally-provided candidate coordinates. No
model is trained; no trained localizer's coordinates are read.

IMPORTANT (dataset discipline): run ONLY on the SAME IM2GPS3K join used for the
descent gate (pipeline_qwen25vl7b_im2gps3k.json + bare_qwen25vl7b_im2gps3k_v2.json,
n=2997). Compare against bare 66.73/9.28 and gate best 67.03/11.34 on that SAME
ruler. NEVER compare to YFCC-4K numbers -- different dataset, not comparable.

Zero GPU, zero network. Run from inside geo_pipeline/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

THRESHOLDS = [1, 25, 200, 750, 2500]
# kappa swept as a bandwidth hyperparam (like the gate's tau). Larger kappa =
# tighter kernels = mode snaps to the densest single point; smaller = smoother.
KAPPAS = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]
_R_KM = 6371.0


def haversine(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _R_KM * math.asin(math.sqrt(a))


def to_unit(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return (math.cos(la) * math.cos(lo),
            math.cos(la) * math.sin(lo),
            math.sin(la))


def unit_to_latlon(x, y, z):
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    return math.degrees(math.asin(max(-1.0, min(1.0, z)))), math.degrees(math.atan2(y, x))


def slerp(a, b, t):
    """Spherical linear interpolation between two unit vectors a, b at param t."""
    dot = max(-1.0, min(1.0, sum(ai * bi for ai, bi in zip(a, b))))
    omega = math.acos(dot)
    if omega < 1e-9:
        return a
    s = math.sin(omega)
    c1, c2 = math.sin((1 - t) * omega) / s, math.sin(t * omega) / s
    v = tuple(c1 * ai + c2 * bi for ai, bi in zip(a, b))
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


def vmf_logdensity(x, kernels, kappa):
    """Unnormalized log mixture density at unit vector x.

    kernels: list of (mu_unit_vec, weight). Shared kappa (C_3 const drops out of
    the argmax). log sum_i w_i exp(kappa * mu_i . x).
    """
    terms = []
    for mu, w in kernels:
        dot = sum(mi * xi for mi, xi in zip(mu, x))
        terms.append(math.log(w + 1e-12) + kappa * dot)
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def mode_latlon(kernels, kappa):
    """Posterior density mode: evaluate on a geodesic grid seeded at the kernels.

    Candidate seeds = each kernel center + slerp midpoints between every pair.
    Then one local refinement pass of small geodesic steps around the best seed.
    Exact enough for the few-kernel case here; no optimizer dependency.
    """
    centers = [mu for mu, _ in kernels]
    seeds = list(centers)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            for t in (0.25, 0.5, 0.75):
                seeds.append(slerp(centers[i], centers[j], t))
    best, best_ld = None, -1e18
    for s in seeds:
        ld = vmf_logdensity(s, kernels, kappa)
        if ld > best_ld:
            best, best_ld = s, ld
    # local refinement: nudge toward each center in shrinking slerp steps
    for _ in range(3):
        improved = False
        for c in centers:
            for t in (0.1, 0.03, 0.01):
                cand = slerp(best, c, t)
                ld = vmf_logdensity(cand, kernels, kappa)
                if ld > best_ld:
                    best, best_ld, improved = cand, ld, True
        if not improved:
            break
    return unit_to_latlon(*best)


def posterior_entropy(kernels, kappa, n_samp=64):
    """Coarse entropy proxy of the mixture over a fixed sphere sample (localizability).

    Not used for scoring; reported as the paper's uncertainty signal, training-free.
    """
    import random
    rnd = random.Random(0)
    lds = []
    for _ in range(n_samp):
        z = rnd.uniform(-1, 1)
        th = rnd.uniform(0, 2 * math.pi)
        r = math.sqrt(1 - z * z)
        x = (r * math.cos(th), r * math.sin(th), z)
        lds.append(vmf_logdensity(x, kernels, kappa))
    m = max(lds)
    ps = [math.exp(l - m) for l in lds]
    s = sum(ps)
    ps = [p / s for p in ps]
    return -sum(p * math.log(p + 1e-12) for p in ps)


def load(path):
    d = json.loads(Path(path).read_text())
    return d["records"] if isinstance(d, dict) else d


def acc_table(dists):
    n = len(dists)
    out = {}
    for t in THRESHOLDS:
        out[t] = 100.0 * sum(1 for d in dists if d is not None and d <= t) / n if n else 0.0
    return out, n


def build_kernels(pr, br):
    """Up to 3 vMF kernels for one image: bare point, descended point, pre-fallback.

    Weight each by a confidence proxy:
      * bare point           -> fixed weight 1.0 (the coarse ceiling anchor)
      * descended point      -> street/city name-agreement (SL self-consistency)
      * pre-fallback point   -> same agreement, half weight (it is pre-correction)
    A kernel is skipped if its coords are missing. Returns [] if none usable.
    """
    kernels = []
    if br and br.get("pred_lat") is not None and br.get("pred_lon") is not None:
        kernels.append((to_unit(br["pred_lat"], br["pred_lon"]), 1.0))
    agr = pr.get("street_name_agreement")
    if agr is None:
        agr = pr.get("city_name_agreement")
    agr = 0.5 if agr is None else max(0.05, float(agr))
    if pr.get("pred_lat") is not None and pr.get("pred_lon") is not None:
        kernels.append((to_unit(pr["pred_lat"], pr["pred_lon"]), agr))
    pf_lat, pf_lon = pr.get("pre_fallback_pred_lat"), pr.get("pre_fallback_pred_lon")
    if pf_lat is not None and pf_lon is not None:
        # only add if materially different from the descended point
        if (pr.get("pred_lat") is None or
                haversine(pf_lat, pf_lon, pr["pred_lat"], pr["pred_lon"]) > 1.0):
            kernels.append((to_unit(pf_lat, pf_lon), 0.5 * agr))
    return kernels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", required=True, help="IM2GPS3K pipeline results JSON")
    ap.add_argument("--bare", required=True, help="IM2GPS3K bare results JSON (same dataset)")
    args = ap.parse_args()

    pipe = load(args.pipeline)
    bare = {r["photo_id"]: r for r in load(args.bare)}

    joined = []  # (pr, br, kernels)
    bare_dists, raw_dists = [], []
    n_multi = 0
    for pr in pipe:
        br = bare.get(pr["photo_id"])
        if br is None:
            continue
        bare_dists.append(br.get("dist_km"))
        raw_dists.append(pr.get("dist_km"))
        k = build_kernels(pr, br)
        if len(k) >= 2:
            n_multi += 1
        joined.append((pr, br, k))

    bare_acc, n = acc_table(bare_dists)
    raw_acc, _ = acc_table(raw_dists)
    b750, b1 = bare_acc[750], bare_acc[1]

    print(f"\njoined = {n}   multi-kernel images = {n_multi} "
          f"({100.0*n_multi/n:.1f}%)   (rest fall back to single point)")
    print(f"bare   @1={bare_acc[1]:.2f}  @750={bare_acc[750]:.2f}  "
          f"(ruler: IM2GPS3K; gate best was 67.03/11.34)")
    print(f"rawpipe@1={raw_acc[1]:.2f}  @750={raw_acc[750]:.2f}\n")

    hdr = "kappa".ljust(8) + "".join(f"{t:>9}" for t in THRESHOLDS) + "   WIN?"
    print(hdr)
    print("-" * len(hdr))
    for kappa in KAPPAS:
        dists = []
        for pr, br, kernels in joined:
            if not kernels:
                dists.append(pr.get("dist_km"))
                continue
            if len(kernels) == 1:
                lat, lon = unit_to_latlon(*kernels[0][0])
            else:
                lat, lon = mode_latlon(kernels, kappa)
            gt_lat, gt_lon = pr.get("gt_lat"), pr.get("gt_lon")
            if gt_lat is None or gt_lon is None:
                dists.append(pr.get("dist_km"))
            else:
                dists.append(haversine(gt_lat, gt_lon, lat, lon))
        acc, _ = acc_table(dists)
        win = "YES" if (acc[750] >= b750 and acc[1] > b1) else ""
        print(f"{kappa:<8.0f}" + "".join(f"{acc[t]:>8.2f}%" for t in THRESHOLDS) + f"   {win}")

    print(f"\nWIN rule: @750 >= bare {b750:.2f} AND @1 > bare {b1:.2f}")
    print("Single-kernel images reduce to their one point (no effect). vMF-mode only")
    print("differs from raw on the multi-kernel subset -- that subset size caps the gain.")


if __name__ == "__main__":
    main()
