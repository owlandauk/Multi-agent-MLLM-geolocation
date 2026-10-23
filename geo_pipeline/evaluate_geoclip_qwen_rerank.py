"""Conservative Qwen reranking over GeoCLIP top-k GPS candidates.

This script is the first do-no-harm hybrid baseline:

1. GeoCLIP supplies the candidate coordinates and prior scores.
2. Qwen verifies only those candidates; it does not invent a new city/street.
3. Evidence updates the candidate posterior.
4. The final answer falls back to GeoCLIP rank-1 unless the reranked posterior
   has a strong enough top margin.

It is intentionally narrower than the full SL-DST-POMDP pipeline so we can test
whether reranking GeoCLIP top-k is promising before adding more machinery.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from tqdm import tqdm

from config import BETA, EVAL_THRESHOLDS, YFCC4K_GPS_CSV, YFCC4K_IMG_DIR
from data.yfcc4k_loader import YFCC4KDataset
from geoclip_candidates import load_candidate_cache, normalize_scores


_SUPPORT_RE = re.compile(
    r"(?:#|candidate\s*)?(\d+)\s*[:=]\s*([SCN])\b",
    re.IGNORECASE,
)
_BEST_RE = re.compile(r"\bBest\s*:\s*(?:#\s*)?(\d+|none)\b", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"\bConfidence\s*:\s*(low|medium|high)\b", re.IGNORECASE)
_LABELS = {
    1: "Street <1km",
    25: "City <25km",
    200: "Region <200km",
    750: "Country <750km",
    2500: "Continent <2500km",
}


def haversine(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _candidate_name(candidate: dict) -> str:
    parts = [
        candidate.get("city"),
        candidate.get("admin1"),
        candidate.get("country"),
    ]
    text = ", ".join(str(p) for p in parts if p)
    return text or f"lat={candidate.get('lat')}, lon={candidate.get('lon')}"


def _make_prompt(image, candidates: list[dict]) -> list:
    candidate_lines = []
    for i, candidate in enumerate(candidates, start=1):
        candidate_lines.append(
            f"#{i}: {_candidate_name(candidate)} "
            f"(lat={candidate.get('lat')}, lon={candidate.get('lon')})"
        )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "GeoCLIP produced candidate locations for this image. "
                    "Your job is only to verify and rerank these candidates; "
                    "do not invent a new location.\n\n"
                    "Candidate locations (shown without GeoCLIP scores; use only visible evidence for verification):\n"
                    + "\n".join(candidate_lines)
                    + "\n\nStep 1 - Describe visible geographic evidence in 1-2 sentences, focusing on clues that can distinguish these candidates.\n"
                    "Step 2 - For each candidate, mark S only if a concrete visible clue supports that candidate over the others, "
                    "C only if a concrete visible clue contradicts it, or N if the evidence is generic, weak, or shared by multiple candidates. "
                    "Avoid marking all candidates S or all candidates C; use N for generic scenes and for candidates you cannot distinguish.\n"
                    "Step 3 - Pick one best candidate only when visible evidence differentiates it from the others; otherwise write none. "
                    "Confidence must be low, medium, or high.\n\n"
                    "Respond exactly in this format:\n"
                    "Observation: <visible evidence only>\n"
                    "Support: #1=S/C/N; #2=S/C/N; ...\n"
                    "Best: #k or none\n"
                    "Confidence: low/medium/high"
                )},
            ],
        }
    ]


def _parse_support(text: str, n: int) -> list[str]:
    ratings = ["N" for _ in range(n)]
    for match in _SUPPORT_RE.finditer(text or ""):
        idx = int(match.group(1)) - 1
        if 0 <= idx < n:
            ratings[idx] = match.group(2).upper()
    return ratings


def _parse_best_choice(text: str, n: int) -> tuple[int | None, str]:
    best_idx = None
    best_line = ""
    for line in (text or "").splitlines():
        if line.strip().lower().startswith("best"):
            best_line = line.lower()
            break
    if " or " in best_line or "/" in best_line or "," in best_line:
        return None, "low"
    match = _BEST_RE.search(text or "")
    if match and match.group(1).lower() != "none":
        idx = int(match.group(1)) - 1
        if 0 <= idx < n:
            best_idx = idx
    conf_match = _CONFIDENCE_RE.search(text or "")
    confidence = conf_match.group(1).lower() if conf_match else "low"
    return best_idx, confidence


def _confidence_alpha(confidence: str) -> float:
    return {"low": 0.25, "medium": 0.55, "high": 0.85}.get(confidence, 0.25)


def _support_weight(label: str, alpha: float) -> float:
    c = {"S": 5, "N": 3, "C": 1}.get(label.upper(), 3)
    return math.exp(alpha * BETA * (c - 3))


def _posterior(
    candidates: list[dict],
    ratings: list[str],
    alpha: float,
    best_idx: int | None,
    best_alpha: float,
) -> list[float]:
    priors = normalize_scores([float(c.get("score", 0.0)) for c in candidates])
    weighted = []
    for idx, (prior, label) in enumerate(zip(priors, ratings)):
        weight = prior * _support_weight(label, alpha)
        if best_idx is not None:
            c = 5 if idx == best_idx else 2.5
            weight *= math.exp(best_alpha * BETA * (c - 3))
        weighted.append(weight)
    return normalize_scores(weighted)


def _select_candidate(
    candidates: list[dict],
    posterior: list[float],
    min_override_top: float,
    min_override_margin: float,
) -> tuple[int, bool]:
    if not candidates:
        return 0, False
    best_idx = max(range(len(candidates)), key=lambda i: posterior[i])
    if best_idx == 0:
        return 0, False
    top1_prob = posterior[0]
    best_prob = posterior[best_idx]
    if best_prob >= min_override_top and (best_prob - top1_prob) >= min_override_margin:
        return best_idx, True
    return 0, False


def _accuracy(correct: dict[int, int], total: int) -> dict[str, float]:
    return {
        str(thr): round(100.0 * correct[thr] / total, 2) if total else 0.0
        for thr in EVAL_THRESHOLDS
    }


def evaluate(args) -> None:
    from models.mllm_client import MLLMClient

    cache = load_candidate_cache(args.candidate_cache)
    dataset = YFCC4KDataset(img_dir=args.img_dir, gps_csv=args.gps_csv)
    mllm = MLLMClient()

    start = args.start or 0
    end = min(start + args.limit, len(dataset)) if args.limit else len(dataset)
    indices = list(range(start, end))

    records = []
    correct = {thr: 0 for thr in EVAL_THRESHOLDS}
    top1_correct = {thr: 0 for thr in EVAL_THRESHOLDS}
    overrides = 0
    missing = 0

    for batch_start in tqdm(range(0, len(indices), args.batch_size), desc="GeoCLIP Qwen rerank"):
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        samples = [dataset[i] for i in batch_indices]
        batch_candidates = []
        prompts = []
        prompt_owners = []

        for local_idx, sample in enumerate(samples):
            entry = cache.get(str(sample["photo_id"])) or {}
            candidates = list(entry.get("candidates") or [])[: args.top_k]
            batch_candidates.append(candidates)
            if candidates:
                prompts.append(_make_prompt(sample["image"], candidates))
                prompt_owners.append(local_idx)

        responses = mllm.batch_generate(prompts, max_new_tokens=args.max_new_tokens) if prompts else []
        response_by_owner = {owner: response for owner, response in zip(prompt_owners, responses)}

        for local_idx, sample in enumerate(samples):
            candidates = batch_candidates[local_idx]
            gt_lat, gt_lon = sample["gt_lat"], sample["gt_lon"]
            if not candidates:
                missing += 1
                dist_km = float("inf")
                top1_dist = float("inf")
                selected = {}
                selected_idx = 0
                did_override = False
                ratings = []
                posterior = []
                raw_response = None
                best_idx = None
                best_confidence = "low"
            else:
                raw_response = response_by_owner.get(local_idx, "")
                ratings = _parse_support(raw_response, len(candidates))
                best_idx, best_confidence = _parse_best_choice(raw_response, len(candidates))
                posterior = _posterior(
                    candidates,
                    ratings,
                    args.evidence_alpha,
                    best_idx,
                    _confidence_alpha(best_confidence),
                )
                selected_idx, did_override = _select_candidate(
                    candidates,
                    posterior,
                    args.min_override_top,
                    args.min_override_margin,
                )
                selected = candidates[selected_idx]
                top1 = candidates[0]
                top1_dist = haversine(gt_lat, gt_lon, float(top1["lat"]), float(top1["lon"]))
                dist_km = haversine(gt_lat, gt_lon, float(selected["lat"]), float(selected["lon"]))
                overrides += int(did_override)

            for thr in EVAL_THRESHOLDS:
                if dist_km <= thr:
                    correct[thr] += 1
                if top1_dist <= thr:
                    top1_correct[thr] += 1

            records.append(
                {
                    "photo_id": sample["photo_id"],
                    "gt_lat": gt_lat,
                    "gt_lon": gt_lon,
                    "pred_lat": selected.get("lat"),
                    "pred_lon": selected.get("lon"),
                    "pred_country": selected.get("country"),
                    "pred_city": selected.get("city"),
                    "pred_street": "GeoCLIP Qwen rerank GPS",
                    "dist_km": dist_km,
                    "geocode_source": "geoclip_qwen_rerank",
                    "country_consistency": "geoclip_direct",
                    "geoclip_top1_dist_km": top1_dist,
                    "geoclip_selected_rank": selected.get("rank"),
                    "geoclip_rerank_override": did_override,
                    "geoclip_rerank_posterior": [round(v, 6) for v in posterior],
                    "geoclip_rerank_support": ratings,
                    "geoclip_rerank_best_idx": None if best_idx is None else best_idx + 1,
                    "geoclip_rerank_best_confidence": best_confidence,
                    "geoclip_rerank_raw": raw_response,
                    "geoclip_candidates": candidates,
                    "pomdp_policy": "geoclip_qwen_rerank_do_no_harm",
                }
            )

    total = len(records)
    print(f"\nGeoCLIP Qwen rerank ({total} images, missing={missing}, overrides={overrides})")
    print("Do-no-harm hybrid")
    for thr in EVAL_THRESHOLDS:
        acc = 100.0 * correct[thr] / total if total else 0.0
        print(f"{_LABELS[thr]:>18}: {acc:6.2f}%")
    print("\nGeoCLIP rank-1 within same rows")
    for thr in EVAL_THRESHOLDS:
        acc = 100.0 * top1_correct[thr] / total if total else 0.0
        print(f"{_LABELS[thr]:>18}: {acc:6.2f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "summary": _accuracy(correct, total),
                "geoclip_top1_summary": _accuracy(top1_correct, total),
                "total": total,
                "start": start,
                "missing": missing,
                "overrides": overrides,
                "candidate_cache": str(args.candidate_cache),
                "settings": {
                    "top_k": args.top_k,
                    "evidence_alpha": args.evidence_alpha,
                    "min_override_top": args.min_override_top,
                    "min_override_margin": args.min_override_margin,
                    "best_confidence_alpha": {"low": 0.25, "medium": 0.55, "high": 0.85},
                },
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_cache", default="results/geoclip_candidates.json")
    parser.add_argument("--img_dir", default=YFCC4K_IMG_DIR)
    parser.add_argument("--gps_csv", default=YFCC4K_GPS_CSV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--evidence_alpha", type=float, default=0.8)
    parser.add_argument("--min_override_top", type=float, default=0.22)
    parser.add_argument("--min_override_margin", type=float, default=0.00)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--out", default="results/geoclip_qwen_rerank.json")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
