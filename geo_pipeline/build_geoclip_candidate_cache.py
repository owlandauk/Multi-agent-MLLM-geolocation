"""Build a GeoCLIP top-k GPS candidate cache.

The output JSON maps photo_id to:

```
{
  "candidates": [{"rank", "lat", "lon", "score", "country", ...}],
  "country_prior": [{"country", "score"}]
}
```

This keeps GeoCLIP's coordinate-level signal for direct baselines, top-k oracle
analysis, and future SL-DST-POMDP reranking.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import YFCC4K_GPS_CSV, YFCC4K_IMG_DIR
from geoclip_candidates import build_candidate_record
from build_geoclip_prior_cache import _patch_geoclip_transformers_compat


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


def build_cache(args) -> None:
    import torch
    from geoclip import GeoCLIP

    if os.environ.get("GEOCLIP_DISABLE_CUDNN", "").lower() in {"1", "true", "yes"}:
        torch.backends.cudnn.enabled = False

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
    for photo_id, img_path in tqdm(rows, desc="GeoCLIP candidates"):
        if args.resume and photo_id in cache:
            continue
        gps, probs = model.predict(str(img_path), top_k=args.top_k)
        cache[photo_id] = build_candidate_record(
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
    parser.add_argument("--out", default="results/geoclip_candidates.json")
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

