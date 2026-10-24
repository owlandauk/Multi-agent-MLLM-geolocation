#!/usr/bin/env python3
"""Does fine-level sample-N agreement separate 'descent helps' from 'descent hurts'?

Joins the gateN5 pipeline run (which records city/street name_agreement + the
pre_fallback fine coord) against the bare-direct run by photo_id. For each
image we compare the pipeline's PRE-FALLBACK fine distance (what you get if you
geocode the descended city/street) against bare's direct distance, bucketed by
the min(city,street) agreement. If agreement has separating power, low-agreement
buckets should be where bare wins (descent hurts) and high-agreement buckets
where the pipeline wins (descent helps).

Also runs an oracle+threshold sweep: for threshold T, pick bare coord when
min-agreement < T else pipeline fine coord, and report country@750. Compares
against always-pipeline, always-bare, and the per-image oracle ceiling.
"""
import json, sys
from math import radians, sin, cos, asin, sqrt

def hav(a, b, c, d):
    a, b, c, d = map(radians, (a, b, c, d))
    return 2 * 6371 * asin(sqrt(sin((c-a)/2)**2 + cos(a)*cos(c)*sin((d-b)/2)**2))

def load(p):
    x = json.load(open(p))
    return x.get("records", x) if isinstance(x, dict) else x

gate = {r["photo_id"]: r for r in load(sys.argv[1])}
bare = {r["photo_id"]: r for r in load(sys.argv[2])}
THR750 = 750

rows = []
for pid, g in gate.items():
    b = bare.get(pid)
    if not b:
        continue
    gt_lat, gt_lon = g.get("gt_lat"), g.get("gt_lon")
    if gt_lat is None:
        continue
    # pipeline fine coord = pre_fallback (the descended city/street geocode
    # BEFORE any gate/continent fallback kicked in)
    plat, plon = g.get("pre_fallback_pred_lat"), g.get("pre_fallback_pred_lon")
    blat, blon = b.get("pred_lat"), b.get("pred_lon")
    p_d = hav(gt_lat, gt_lon, plat, plon) if plat is not None else float("inf")
    b_d = hav(gt_lat, gt_lon, blat, blon) if blat is not None else float("inf")
    c_agr = g.get("city_name_agreement", 1.0)
    s_agr = g.get("street_name_agreement", 1.0)
    min_agr = min(c_agr, s_agr)
    rows.append((pid, min_agr, p_d, b_d))

n = len(rows)
print(f"joined n={n}")

# --- separating power: in each agreement bucket, who wins? ---
print("\n=== bucket by min(city,street) agreement: does bare beat pipeline-fine? ===")
print(f"{'bucket':>8} {'n':>5} {'pipe<750%':>9} {'bare<750%':>9} {'bare_wins%':>10}")
buckets = [(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]
for lo, hi in buckets:
    sub = [r for r in rows if lo <= r[1] < hi]
    if not sub:
        continue
    pin = sum(1 for _,_,pd,_ in sub if pd <= THR750) / len(sub) * 100
    bin_ = sum(1 for _,_,_,bd in sub if bd <= THR750) / len(sub) * 100
    bw = sum(1 for _,_,pd,bd in sub if bd < pd) / len(sub) * 100
    print(f"{lo:>4.1f}-{hi-0.01:<3.1f} {len(sub):>5} {pin:>8.1f}% {bin_:>8.1f}% {bw:>9.1f}%")

# --- threshold sweep: country@750 picking bare when min_agr<T else pipeline ---
print("\n=== threshold sweep (pick bare coord when min_agr < T, else pipeline-fine) ===")
always_pipe = sum(1 for _,_,pd,_ in rows if pd <= THR750) / n * 100
always_bare = sum(1 for _,_,_,bd in rows if bd <= THR750) / n * 100
oracle = sum(1 for _,_,pd,bd in rows if min(pd,bd) <= THR750) / n * 100
print(f"{'T':>5} {'country@750':>12}")
for T in [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]:
    acc = sum(1 for _,m,pd,bd in rows if (bd if m < T else pd) <= THR750) / n * 100
    print(f"{T:>5.2f} {acc:>11.2f}%")
print(f"\nalways-pipeline-fine : {always_pipe:.2f}%")
print(f"always-bare-direct   : {always_bare:.2f}%")
print(f"per-image oracle     : {oracle:.2f}%")
