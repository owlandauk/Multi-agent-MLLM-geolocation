#!/usr/bin/env python3
"""Apples-to-apples lenient-ruler comparison: bare vs descent-gate vs vMF-mode.

Problem this closes: the bare model's @750 jumps 66.73 -> 69.70 under the
"excl-NL" (GaGA-style non-localizable exclusion) convention ONLY because 235
genuine-abstention images leave the denominator (diagnose_geocode_convention.py).
The descent-gate (67.03) and vMF-mode (67.27) numbers were computed on the FULL
n=2997 denominator. Comparing gate/vMF (full) against bare 69.70 (reduced) is a
cross-ruler mistake -- the same error as comparing YFCC-4K to IM2GPS3K.

This script puts ALL THREE on the SAME denominator. It drops exactly the records
is_abstention(bare_rec) flags (geocode_source=="country" AND blank/placeholder
city), identical to convention (3) in diagnose_geocode_convention.py, then
recomputes bare / gate / vMF on the surviving set. Now the win-claim is honest:
does vMF still lead bare once abstentions are symmetrically removed from all three?

Reuses the gate rule from apply/diagnose_descent_distance_gate (fall back to bare
DIRECT point when descent > tau km from it; never a centroid) and the vMF-mixture
mode from diagnose_vmf_posterior (spherical KDE, continuous density peak). No
training, no network, zero GPU. Run from inside geo_pipeline/.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

THRESHOLDS = [1, 25, 200, 750, 2500]
_ABSTAIN_MARKERS = ("not spec", "not determin", "unknown", "n/a", "none", "undetermin")
GATE_TAU = 1000.0   # best @750 tau from the descent-distance sweep
VMF_KAPPA = 100.0   # coarse-best kappa from the vMF sweep
_R_KM = 6371.0


# --- geometry (copied verbatim from the two diagnostics) -------------------
def haversine(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _R_KM * math.asin(math.sqrt(a))


def to_unit(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))


def unit_to_latlon(x, y, z):
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    return math.degrees(math.asin(max(-1.0, min(1.0, z)))), math.degrees(math.atan2(y, x))


def slerp(a, b, t):
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
    terms = []
    for mu, w in kernels:
        dot = sum(mi * xi for mi, xi in zip(mu, x))
        terms.append(math.log(w + 1e-12) + kappa * dot)
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def mode_latlon(kernels, kappa):
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


def build_kernels(pr, br):
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
        if (pr.get("pred_lat") is None or
                haversine(pf_lat, pf_lon, pr["pred_lat"], pr["pred_lon"]) > 1.0):
            kernels.append((to_unit(pf_lat, pf_lon), 0.5 * agr))
    return kernels


# --- exclusion rules: two conventions for "non-localizable" --------------
def is_abstention(rec) -> bool:
    """TIGHT (committed diagnose_geocode_convention): blank city AND fell back
    to country centroid. Drops ~235 on bare 7B -> @750 69.70."""
    if rec.get("geocode_source") != "country":
        return False
    city = (rec.get("pred_city") or "").strip().lower()
    if city == "":
        return True
    return any(m in city for m in _ABSTAIN_MARKERS)


def is_abstention_loose(rec) -> bool:
    """LOOSE (GaGA-style, the throwaway /tmp script that printed ~71.7): blank or
    placeholder city REGARDLESS of geocode_source. Drops ~329 on bare 7B ->
    @750 71.70. A published convention (GaGA excludes non-localizable images)."""
    city = (rec.get("pred_city") or "").strip().lower()
    if city == "":
        return True
    return any(m in city for m in _ABSTAIN_MARKERS)


def load(path):
    d = json.loads(Path(path).read_text())
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
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--bare", required=True)
    args = ap.parse_args()

    pipe = load(args.pipeline)
    bare = {r["photo_id"]: r for r in load(args.bare)}

    # join; mask_tight = convention (3), mask_loose = GaGA-style (~71.7 script).
    # Both judged on the BARE record (gated images revert to the bare point).
    rows = []  # (pr, br, dd, keep_tight, keep_loose)
    for pr in pipe:
        br = bare.get(pr["photo_id"])
        if br is None:
            continue
        dd = None
        if pr.get("pred_lat") is not None and br.get("pred_lat") is not None:
            dd = haversine(pr["pred_lat"], pr["pred_lon"], br["pred_lat"], br["pred_lon"])
        rows.append((pr, br, dd, not is_abstention(br), not is_abstention_loose(br)))

    def run(mode):
        """mode: 'full' | 'tight' | 'loose'."""
        bare_d, gate_d, vmf_d = [], [], []
        for pr, br, dd, keep_tight, keep_loose in rows:
            if mode == "tight" and not keep_tight:
                continue
            if mode == "loose" and not keep_loose:
                continue
            gt_lat, gt_lon = pr.get("gt_lat"), pr.get("gt_lon")
            # bare
            bare_d.append(br.get("dist_km"))
            # gate: revert to bare DIRECT point if descent moved > tau
            chosen = br.get("dist_km") if (dd is not None and dd > GATE_TAU) else pr.get("dist_km")
            gate_d.append(chosen)
            # vMF mode
            kern = build_kernels(pr, br)
            if not kern:
                vmf_d.append(pr.get("dist_km"))
            elif len(kern) == 1:
                lat, lon = unit_to_latlon(*kern[0][0])
                vmf_d.append(haversine(gt_lat, gt_lon, lat, lon)
                             if gt_lat is not None else pr.get("dist_km"))
            else:
                lat, lon = mode_latlon(kern, VMF_KAPPA)
                vmf_d.append(haversine(gt_lat, gt_lon, lat, lon)
                             if gt_lat is not None else pr.get("dist_km"))
        return acc_table(bare_d), acc_table(gate_d), acc_table(vmf_d)

    for label, mode in (("FULL ruler (n=all)", "full"),
                        ("TIGHT excl-NL (blank city AND centroid; ~69.7)", "tight"),
                        ("LOOSE excl-NL GaGA-style (blank city anywhere; ~71.7)", "loose")):
        (ba, n), (ga, _), (va, _) = run(mode)
        print(f"\n=== {label}  n={n} ===")
        hdr = "method".ljust(10) + "".join(f"{t:>9}" for t in THRESHOLDS)
        print(hdr)
        print("-" * len(hdr))
        print("bare".ljust(10) + "".join(f"{ba[t]:>8.2f}%" for t in THRESHOLDS))
        print(f"gate t{int(GATE_TAU)}".ljust(10) + "".join(f"{ga[t]:>8.2f}%" for t in THRESHOLDS))
        print(f"vmf k{int(VMF_KAPPA)}".ljust(10) + "".join(f"{va[t]:>8.2f}%" for t in THRESHOLDS))
        print(f"@750: bare={ba[750]:.2f}  gate={ga[750]:.2f}  vmf={va[750]:.2f}   "
              f"(gate-bare={ga[750]-ba[750]:+.2f}, vmf-bare={va[750]-ba[750]:+.2f})")

    print("\nSAME exclusion as diagnose_geocode_convention (3): is_abstention(bare_rec).")
    print("If vmf-bare stays positive on the EXCL-NL ruler, the win survives the lenient denom.")


if __name__ == "__main__":
    main()
