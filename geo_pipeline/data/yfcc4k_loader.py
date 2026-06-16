import pandas as pd
from pathlib import Path
from PIL import Image
from config import YFCC4K_IMG_DIR, YFCC4K_GPS_CSV


class YFCC4KDataset:
    """
    YFCC4K evaluation dataset.
    CSV columns: photo_id, lat, lon, img_url
    Images are expected at: {img_dir}/{photo_id}.jpg
    """

    def __init__(self, img_dir: str = YFCC4K_IMG_DIR, gps_csv: str = YFCC4K_GPS_CSV):
        self.img_dir = Path(img_dir)
        self.meta = pd.read_csv(gps_csv)
        # keep only rows whose image file actually exists
        self.meta = self.meta[
            self.meta["photo_id"].apply(lambda pid: (self.img_dir / f"{pid}.jpg").exists())
        ].reset_index(drop=True)
        print(f"[YFCC4K] {len(self.meta)} images found in {img_dir}")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = self.img_dir / f"{row['photo_id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        return {
            "photo_id": str(row["photo_id"]),
            "image": image,
            "gt_lat": float(row["lat"]),
            "gt_lon": float(row["lon"]),
            "img_path": str(img_path),
        }
