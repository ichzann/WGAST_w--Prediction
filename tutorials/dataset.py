from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WGASTSurrogateDataset(Dataset):
    def __init__(self, manifest_path, tile_size=None, split=None):
        self.manifest = pd.read_parquet(manifest_path)
        if split is not None:
            if "split" not in self.manifest.columns:
                raise ValueError("manifest has no 'split' column -- rebuild with 03s_build_dataset.ipynb")
            self.manifest = self.manifest[self.manifest["split"] == split]
        self.manifest = self.manifest.reset_index(drop=True)
        self.tile_size = tile_size

        stats_path = Path(manifest_path).parent / "stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} not found. Run 03s_build_dataset.ipynb first (it writes stats.npz)."
            )
        s = np.load(stats_path)
        self.sp_mean = s["spatial_mean"][:, None, None].astype(np.float32)
        self.sp_std  = s["spatial_std"][:, None, None].astype(np.float32)
        self.sc_mean = s["scalars_mean"].astype(np.float32)
        self.sc_std  = s["scalars_std"].astype(np.float32)
        self.tg_mean = float(s["target_mean"])
        self.tg_std  = float(s["target_std"])

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        d = np.load(row["path"])
        # z-score valid pixels only; missing (zero-filled) pixels stay exactly 0
        # so the mask channels remain the sole "is this real?" signal (§10 H2).
        raw = d["spatial"]
        spatial = np.where(raw != 0, (raw - self.sp_mean) / self.sp_std, 0.0).astype(np.float32)
        scalars = (d["scalars"] - self.sc_mean) / self.sc_std
        target  = (d["target"]  - self.tg_mean) / self.tg_std

        if self.tile_size is not None:
            spatial, target = self._random_crop(spatial, target, self.tile_size)

        return {
            "spatial": torch.from_numpy(spatial.copy()),              # (C, H, W)
            "scalars": torch.from_numpy(scalars.copy()),              # (S,)
            "target":  torch.from_numpy(target.copy()).unsqueeze(0),  # (1, H, W)
            "city":    row["city"],
            "d0":      row["d0"],
        }

    @staticmethod
    def _random_crop(spatial, target, k):
        _, H, W = spatial.shape
        if H < k or W < k:
            raise ValueError(f"tile_size={k} > raster ({H},{W})")
        y = np.random.randint(0, H - k + 1)
        x = np.random.randint(0, W - k + 1)
        return spatial[:, y:y+k, x:x+k], target[y:y+k, x:x+k]
