"""Build a GeoCLIP country-prior cache for retrieval-assisted GeoBayes.

The output JSON maps photo_id -> [{country, score}], which can be consumed by
``RETRIEVAL_PRIOR_ENABLED=1 RETRIEVAL_PRIOR_PATH=... evaluate.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import YFCC4K_GPS_CSV, YFCC4K_IMG_DIR
from country_aliases import canonicalize_country


def _country_from_iso2(code: str) -> str | None:
    try:
        import pycountry
    except ImportError:
        return None
    country = pycountry.countries.get(alpha_2=(code or "").upper())
    return country.name if country else None


def gps_predictions_to_country_prior(coords, probs) -> list[dict[str, float]]:
    """Aggregate GeoCLIP top-k GPS predictions into country scores."""
    import reverse_geocoder as rg

    points = [(float(lat), float(lon)) for lat, lon in coords]
    weights = [float(p) for p in probs]
    if not points:
        return []

    geo_rows = rg.search(points, mode=1)
    scores: dict[str, float] = {}
    for row, weight in zip(geo_rows, weights):
        raw_country = _country_from_iso2(row.get("cc", "")) or row.get("cc", "")
        country = canonicalize_country(raw_country)
        if country is None:
            continue
        scores[country] = scores.get(country, 0.0) + max(0.0, weight)

    total = sum(scores.values())
    if total <= 0:
        return []
    return [
        {"country": country, "score": round(score / total, 6)}
        for country, score in sorted(scores.items(), key=lambda kv: -kv[1])
    ]


def _load_rows(img_dir: Path, gps_csv: Path, start: int, limit: int | None) -> list[tuple[str, Path]]:
    meta = pd.read_csv(gps_csv)
    rows: list[tuple[str, Path]] = []
    end = len(meta) if limit is None else min(len(meta), start + limit)
    for _, row in meta.iloc[start:end].iterrows():
        photo_id = str(row["photo_id"])
        img_path = img_dir / f"{photo_id}.jpg"
        if img_path.exists():
            rows.append((photo_id, img_path))
    return rows


def _patch_geoclip_transformers_compat(model) -> None:
    """Handle newer transformers returning model-output objects from CLIP."""
    import types
    import torch

    encoder = model.image_encoder

    def forward(self, x):
        features = self.CLIP.get_image_features(pixel_values=x)
        if not torch.is_tensor(features):
            candidate = getattr(features, "image_embeds", None)
            features = candidate if candidate is not None else getattr(features, "pooler_output", None)
        if features is None:
            vision_out = self.CLIP.vision_model(pixel_values=x)
            features = getattr(vision_out, "pooler_output", None)
        if features is None:
            raise TypeError("Could not extract CLIP image features for GeoCLIP")
        return self.mlp(features)

    encoder.forward = types.MethodType(forward, encoder)


def build_cache(args) -> None:
    import torch
    from geoclip import GeoCLIP

    img_dir = Path(args.img_dir)
    gps_csv = Path(args.gps_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {}
    if args.resume and out_path.exists():
        cache = json.loads(out_path.read_text(encoding="utf-8"))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GeoCLIP] loading model on {device}")
    model = GeoCLIP()
    _patch_geoclip_transformers_compat(model)
    model.to(device)
    model.device = device
    model.eval()

    rows = _load_rows(img_dir, gps_csv, args.start, args.limit)
    written = 0
    for photo_id, img_path in tqdm(rows, desc="GeoCLIP prior"):
        if args.resume and photo_id in cache:
            continue
        gps, probs = model.predict(str(img_path), top_k=args.top_k)
        cache[photo_id] = gps_predictions_to_country_prior(
            gps.detach().cpu().tolist(),
            probs.detach().cpu().tolist(),
        )
        written += 1
        if written % args.flush_every == 0:
            out_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    out_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[GeoCLIP] wrote {len(cache)} records to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=YFCC4K_IMG_DIR)
    parser.add_argument("--gps_csv", default=YFCC4K_GPS_CSV)
    parser.add_argument("--out", default="results/geoclip_prior_cache.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--flush_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
