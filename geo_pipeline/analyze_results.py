"""Offline diagnostics for geo_pipeline result JSON files.

Usage:
  python3 geo_pipeline/analyze_results.py --pred geo_pipeline/results/full_v5.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from statistics import mean, median

from country_aliases import canonicalize_country, continent_of
from config import EVAL_THRESHOLDS


_THRESHOLD_LABELS = {
    1: "Street <1km",
    25: "City <25km",
    200: "Region <200km",
    750: "Country <750km",
    2500: "Continent <2500km",
}


def _top_mass(record: dict) -> float | None:
    posterior = record.get("country_posterior") or {}
    if not posterior:
        return None
    return max(float(v) for v in posterior.values())


def _country_conflicts(record: dict) -> list[str]:
    pred_country = canonicalize_country(record.get("pred_country") or "")
    if not pred_country:
        return []

    conflicts = []
    for field in ("pred_city", "pred_street"):
        child_country = canonicalize_country(record.get(field) or "")
        if child_country and child_country != pred_country:
            conflicts.append(field)
    return conflicts


def analyze(records: list[dict]) -> dict:
    total = len(records)
    correct = {thr: 0 for thr in EVAL_THRESHOLDS}
    for record in records:
        dist = float(record.get("dist_km", float("inf")))
        for thr in EVAL_THRESHOLDS:
            if dist <= thr:
                correct[thr] += 1

    unknown = sum(1 for r in records if not canonicalize_country(r.get("pred_country") or ""))
    masses = [m for r in records if (m := _top_mass(r)) is not None]
    source_counts = Counter(r.get("geocode_source") or "missing" for r in records)
    consistency_counts = Counter(r.get("country_consistency") or "missing" for r in records)
    conflicts = [r for r in records if _country_conflicts(r)]

    pred_continent_counts = Counter()
    pred_continent_correct = defaultdict(int)
    gt_continent_counts = Counter()
    gt_continent_correct = defaultdict(int)

    for record in records:
        pred_cont = continent_of(canonicalize_country(record.get("pred_country") or "") or "")
        if pred_cont:
            pred_continent_counts[pred_cont] += 1
            if float(record.get("dist_km", float("inf"))) <= 2500:
                pred_continent_correct[pred_cont] += 1

        gt_cont = record.get("gt_continent")
        if gt_cont:
            gt_continent_counts[gt_cont] += 1
            if float(record.get("dist_km", float("inf"))) <= 2500:
                gt_continent_correct[gt_cont] += 1

    return {
        "total": total,
        "accuracy": {
            str(thr): round(100.0 * correct[thr] / total, 2) if total else 0.0
            for thr in EVAL_THRESHOLDS
        },
        "unknown_country_rate": round(100.0 * unknown / total, 2) if total else 0.0,
        "country_top_mass": {
            "mean": round(mean(masses), 4) if masses else None,
            "median": round(median(masses), 4) if masses else None,
        },
        "geocode_source": dict(source_counts),
        "country_consistency": dict(consistency_counts),
        "country_child_conflict_rate": round(100.0 * len(conflicts) / total, 2) if total else 0.0,
        "predicted_continent_breakdown": {
            cont: {
                "n": n,
                "continent_acc": round(100.0 * pred_continent_correct[cont] / n, 2),
            }
            for cont, n in sorted(pred_continent_counts.items())
        },
        "gt_continent_breakdown": {
            cont: {
                "n": n,
                "continent_acc": round(100.0 * gt_continent_correct[cont] / n, 2),
            }
            for cont, n in sorted(gt_continent_counts.items())
        },
    }


def _print_report(report: dict) -> None:
    print(f"Total records: {report['total']}")
    print("\nAccuracy")
    for thr in EVAL_THRESHOLDS:
        print(f"  {_THRESHOLD_LABELS[thr]:>18}: {report['accuracy'][str(thr)]:6.2f}%")

    print(f"\nUnknown country rate: {report['unknown_country_rate']:.2f}%")
    mass = report["country_top_mass"]
    print(f"Country posterior top mass: mean={mass['mean']} median={mass['median']}")
    print(f"Country-child conflict rate: {report['country_child_conflict_rate']:.2f}%")

    print("\nGeocode source")
    for key, value in sorted(report["geocode_source"].items()):
        print(f"  {key}: {value}")

    print("\nCountry consistency")
    for key, value in sorted(report["country_consistency"].items()):
        print(f"  {key}: {value}")

    print("\nPredicted continent breakdown")
    for cont, stats in report["predicted_continent_breakdown"].items():
        print(f"  {cont}: n={stats['n']} acc@2500={stats['continent_acc']:.2f}%")

    if report["gt_continent_breakdown"]:
        print("\nGT continent breakdown")
        for cont, stats in report["gt_continent_breakdown"].items():
            print(f"  {cont}: n={stats['n']} acc@2500={stats['continent_acc']:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to evaluate.py result JSON")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    with open(args.pred) as f:
        data = json.load(f)
    report = analyze(data.get("records", []))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
