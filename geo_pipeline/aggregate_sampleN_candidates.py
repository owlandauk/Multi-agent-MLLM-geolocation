#!/usr/bin/env python3
"""Direction-B offline aggregator over bare sample_n candidates (zero GPU/network).

Bare top-1 greedy misses @750 split into A1 = wrong COUNTRY (602/755 of the
city-named misses) which NO coordinate post-process can touch, because the
country name itself is wrong. The ONLY training-free lever for A1 is to sample
the MLLM N times and aggregate: if the correct country is present among the N
candidates even when top-1 is wrong, a majority/mode rule can flip miss->hit.

This script reads the sample_n run JSON (records each carry a 'candidates' list
of N {country,city,lat,lon}) and, on the SAME records, compares three readouts
on the SAME denominator:

  top1    : the first-sample coordinate (== the existing bare baseline)
  majvote : pick the candidate whose COUNTRY is the plurality among the N; among
            that country's candidates take the one closest to their own centroid
            (medoid-in-country) to avoid same-name outliers. Ties -> top1.
  vmfmode : spherical von Mises-Fisher mixture over the N candidate unit vectors
            (equal weight), continuous density MODE. Mirrors diagnose_vmf_posterior
            but the kernels are the N sibling samples, not pipeline+bare+prefallback.

With --greedy (join the temp=0 run by photo_id) two more readouts print:

  greedy  : the greedy (temp=0) direct coordinate -- the real deployable baseline
            (NOT the stochastic top1 slot, which is temp=1.0 degraded).
  override: causal selector -- default to the greedy point, switch to the vMF mode
            of {greedy + N samples} ONLY when a trigger fires. Two triggers:
            'g2med' (default) fires when greedy sits >tau km from the N-sample
            medoid -- a COORDINATE signal causal on the @750 error; 'name' fires
            when greedy's country disagrees with the N-sample plurality country.
            On IM2GPS3K g2med(tau=1000) Pareto-beats 'name' at both @1 and @750.
            Leaves untriggered images on the exact greedy point so street@1 is
            preserved. Also prints ORACLE (per-image min of greedy/mode) ceiling.

Prints @1/25/200/750/2500 for all three, the paired flip counts (fix vs break)
at @750, and the A1 rescue rate: of records whose top1 country != GT country,
how many each aggregator flips to a @750 hit.

GT country is read OFFLINE from gt_lat/gt_lon via reverse_geocoder (no network),
only to LABEL A1 -- it is never fed to the aggregators (they see candidates only).

Run from inside geo_pipeline/:
  python3 aggregate_sampleN_candidates.py --run results/bare_qwen25vl7b_im2gps3k_sampleN5.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

THRESHOLDS = [1, 25, 200, 750, 2500]
_R_KM = 6371.0
VMF_KAPPA = 100.0  # same bandwidth as diagnose_vmf_posterior coarse-best


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _R_KM * math.asin(math.sqrt(a))


def to_unit(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))


def unit_to_latlon(x, y, z):
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    return (math.degrees(math.asin(max(-1.0, min(1.0, z)))),
            math.degrees(math.atan2(y, x)))


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


def vmf_logdensity(x, mus, kappa):
    terms = [kappa * sum(mi * xi for mi, xi in zip(mu, x)) for mu in mus]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def vmf_mode(points_latlon, kappa=VMF_KAPPA):
    """Continuous density peak of an equal-weight vMF mixture on the sphere."""
    mus = [to_unit(la, lo) for la, lo in points_latlon]
    if len(mus) == 1:
        return points_latlon[0]
    seeds = list(mus)
    for i in range(len(mus)):
        for j in range(i + 1, len(mus)):
            for t in (0.25, 0.5, 0.75):
                seeds.append(slerp(mus[i], mus[j], t))
    best, best_ld = None, -1e18
    for s in seeds:
        ld = vmf_logdensity(s, mus, kappa)
        if ld > best_ld:
            best, best_ld = s, ld
    for _ in range(3):
        improved = False
        for mu in mus:
            for t in (0.1, 0.03, 0.01):
                cand = slerp(best, mu, t)
                ld = vmf_logdensity(cand, mus, kappa)
                if ld > best_ld:
                    best, best_ld, improved = cand, ld, True
        if not improved:
            break
    return unit_to_latlon(*best)


def cand_points(rec):
    """Valid (lat,lon,country) triples among the N candidates."""
    out = []
    for c in rec.get("candidates", []):
        if c.get("lat") is not None and c.get("lon") is not None:
            out.append((c["lat"], c["lon"], (c.get("country") or "").strip().lower()))
    return out


def majvote_coord(rec):
    """Plurality country among candidates; medoid-in-that-country coord. Fall
    back to top1 (pred_lat/lon) on tie or empty."""
    pts = cand_points(rec)
    if not pts:
        return rec.get("pred_lat"), rec.get("pred_lon")
    counts = {}
    for _, _, cc in pts:
        if cc:
            counts[cc] = counts.get(cc, 0) + 1
    if not counts:
        return pts[0][0], pts[0][1]
    top = max(counts.values())
    winners = [c for c, n in counts.items() if n == top]
    if len(winners) != 1:
        return rec.get("pred_lat"), rec.get("pred_lon")  # tie -> top1
    win = winners[0]
    group = [(la, lo) for la, lo, cc in pts if cc == win]
    # medoid: the group member minimizing summed haversine to the rest
    best, best_s = group[0], 1e18
    for la, lo in group:
        s = sum(haversine(la, lo, la2, lo2) for la2, lo2 in group)
        if s < best_s:
            best, best_s = (la, lo), s
    return best


def vmfmode_coord(rec):
    pts = cand_points(rec)
    if not pts:
        return rec.get("pred_lat"), rec.get("pred_lon")
    return vmf_mode([(la, lo) for la, lo, _ in pts])


def _canon(s):
    """Canonicalize a country string; fall back to a stripped lowercase form if
    country_aliases is unavailable so the override still runs offline."""
    try:
        from country_aliases import canonicalize_country
        return canonicalize_country(s or "") or ""
    except Exception:
        return (s or "").strip().lower()


OVERRIDE_G2MED_TAU = 1000.0  # km; greedy-to-medoid distance that fires the override


def _medoid(pts):
    """Point in pts minimizing summed haversine to the rest (same-name-robust)."""
    best, best_s = pts[0], 1e18
    for la, lo in pts:
        s = sum(haversine(la, lo, la2, lo2) for la2, lo2 in pts)
        if s < best_s:
            best, best_s = (la, lo), s
    return best


def override_coord(rec, greedy_rec, trigger="g2med", tau=OVERRIDE_G2MED_TAU):
    """Causal override. Default to the GREEDY point; switch to the vMF mode of
    {greedy + N samples} ONLY when a trigger fires. Leaves every untriggered image
    on the exact greedy point, so street@1 is preserved on the untouched set.

    trigger='g2med' (default): fire when the greedy point sits farther than `tau`
      km from the N-sample medoid -- a COORDINATE signal that is causal on the
      actual @750 error (distance), not on names. tau=1000 Pareto-beats the name
      trigger on IM2GPS3K at both @1 (9.18 vs 9.08) and @750 (67.58 vs 67.35).
    trigger='name': fire when the greedy country disagrees with the N-sample
      plurality country (the earlier A1 'greedy maybe wrong-country' rule).
    """
    gpt = (greedy_rec.get("pred_lat"), greedy_rec.get("pred_lon"))
    if gpt[0] is None:
        return vmfmode_coord(rec)  # no greedy anchor -> fall back to mode
    pts3 = cand_points(rec)
    if not pts3:
        return gpt
    pts = [(la, lo) for la, lo, _ in pts3]
    if trigger == "name":
        ccs = [cc for _, _, cc in pts3 if cc]
        if not ccs:
            return gpt
        plur = max(set(ccs), key=ccs.count)
        gcc = _canon(greedy_rec.get("pred_country"))
        if not gcc or gcc == _canon(plur):
            return gpt  # greedy agrees with the samples -> keep greedy (protect @1)
    else:  # 'g2med' coordinate trigger
        med = _medoid(pts)
        if haversine(gpt[0], gpt[1], med[0], med[1]) <= tau:
            return gpt  # greedy near the sample medoid -> keep greedy (protect @1)
    return vmf_mode([gpt] + pts)  # fire -> mode rescue over {greedy + N samples}


def dist_or_inf(rec, coord):
    if coord is None or coord[0] is None:
        return float("inf")
    return haversine(rec["gt_lat"], rec["gt_lon"], coord[0], coord[1])


def acc(dists):
    n = len(dists)
    return {t: 100.0 * sum(1 for d in dists if d <= t) / n for t in THRESHOLDS}, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="sample_n run JSON with candidates")
    ap.add_argument("--greedy", default=None,
                    help="greedy (temp=0) run JSON; enables the 'greedy' baseline "
                         "and the causal 'override' readout (join by photo_id)")
    ap.add_argument("--gt_country", action="store_true",
                    help="label A1 via offline reverse_geocoder (needs the lib)")
    ap.add_argument("--override_trigger", choices=("g2med", "name"), default="g2med",
                    help="override firing rule: 'g2med' greedy-far-from-medoid "
                         "(default) or 'name' greedy-country != sample-plurality")
    ap.add_argument("--override_tau", type=float, default=OVERRIDE_G2MED_TAU,
                    help="km threshold for the g2med trigger (default %(default)s)")
    args = ap.parse_args()

    d = json.loads(Path(args.run).read_text())
    recs = [r for r in d["records"] if r.get("candidates") and r.get("gt_lat") is not None]
    print(f"records with candidates: {len(recs)}  (N={len(recs[0]['candidates'])})")

    gmap = {}
    if args.greedy:
        gd = json.loads(Path(args.greedy).read_text())
        grecs = gd["records"] if isinstance(gd, dict) else gd
        gmap = {str(r["photo_id"]): r for r in grecs}
        recs = [r for r in recs if str(r["photo_id"]) in gmap]
        print(f"joined with greedy {args.greedy}: {len(recs)} common photo_ids")

    top1_d, maj_d, vmf_d = [], [], []
    greedy_d, over_d, oracle_d = [], [], []
    rows = []
    for r in recs:
        dt = dist_or_inf(r, (r.get("pred_lat"), r.get("pred_lon")))
        dm = dist_or_inf(r, majvote_coord(r))
        dv = dist_or_inf(r, vmfmode_coord(r))
        top1_d.append(dt); maj_d.append(dm); vmf_d.append(dv)
        rows.append((r, dt, dm, dv))
        if gmap:
            grec = gmap[str(r["photo_id"])]
            dg = dist_or_inf(r, (grec.get("pred_lat"), grec.get("pred_lon")))
            do = dist_or_inf(r, override_coord(r, grec, trigger=args.override_trigger,
                                               tau=args.override_tau))
            greedy_d.append(dg); over_d.append(do)
            oracle_d.append(min(dg, do))

    (ta, n), (ma, _), (va, _) = acc(top1_d), acc(maj_d), acc(vmf_d)
    hdr = "method".ljust(9) + "".join(f"{t:>9}" for t in THRESHOLDS)
    print("\n" + hdr); print("-" * len(hdr))
    if gmap:
        (ga, _), (oa, _), (ora, _) = acc(greedy_d), acc(over_d), acc(oracle_d)
        print("greedy".ljust(9) + "".join(f"{ga[t]:>8.2f}%" for t in THRESHOLDS))
    print("top1".ljust(9) + "".join(f"{ta[t]:>8.2f}%" for t in THRESHOLDS))
    print("majvote".ljust(9) + "".join(f"{ma[t]:>8.2f}%" for t in THRESHOLDS))
    print("vmfmode".ljust(9) + "".join(f"{va[t]:>8.2f}%" for t in THRESHOLDS))
    if gmap:
        print("override".ljust(9) + "".join(f"{oa[t]:>8.2f}%" for t in THRESHOLDS))
        print("ORACLE".ljust(9) + "".join(f"{ora[t]:>8.2f}%" for t in THRESHOLDS))
        fired = sum(1 for dg, do in zip(greedy_d, over_d) if dg != do)
        fix = sum(1 for dg, do in zip(greedy_d, over_d) if dg > 750 and do <= 750)
        brk = sum(1 for dg, do in zip(greedy_d, over_d) if dg <= 750 and do > 750)
    if gmap:
        trig = (f"g2med>{args.override_tau:g}km" if args.override_trigger == "g2med"
                else "country!=plurality")
        print(f"\n@750 override[{trig}] vs GREEDY: fired={fired}  "
              f"fix(miss->hit)={fix}  break(hit->miss)={brk}  net={fix - brk:+d}")

    for name, agg in (("majvote", maj_d), ("vmfmode", vmf_d)):
        fix = sum(1 for dt, da in zip(top1_d, agg) if dt > 750 and da <= 750)
        brk = sum(1 for dt, da in zip(top1_d, agg) if dt <= 750 and da > 750)
        print(f"\n@750 {name}: fix(miss->hit)={fix}  break(hit->miss)={brk}  "
              f"net={fix - brk:+d}")

    # A1 rescue: GT country via offline reverse geocode (label only)
    if args.gt_country:
        try:
            import reverse_geocoder as rg
            import pycountry
            from country_aliases import canonicalize_country as canon
        except Exception as e:
            print(f"\n[A1] reverse_geocoder/pycountry/country_aliases missing: {e}")
            return
        res = rg.search([(r["gt_lat"], r["gt_lon"]) for r in recs])

        def iso2canon(cc):
            try:
                return canon(pycountry.countries.get(alpha_2=cc).name)
            except Exception:
                return cc.lower()

        a1 = a1_maj = a1_vmf = 0
        for (r, dt, dm, dv), g in zip(rows, res):
            gt_c = iso2canon(g["cc"])
            pc = canon(r.get("pred_country") or "")
            if dt > 750 and (not pc or pc != gt_c):  # top1 wrong-country miss
                a1 += 1
                if dm <= 750:
                    a1_maj += 1
                if dv <= 750:
                    a1_vmf += 1
        print(f"\n=== A1 wrong-country miss@750 rescue ===")
        print(f"A1 cases (top1 country != GT country, miss@750): {a1}")
        print(f"  majvote rescues: {a1_maj}")
        print(f"  vmfmode rescues: {a1_vmf}")


if __name__ == "__main__":
    main()
