import numpy as np 
import pandas as pd 
import torch 
from torch.utils.data import Dataset

class WGASTSurrogateDataset(Dataset):
    def __init__(self, manifest_path, tile_size=None):
        self.manifest = pd.read_parquet(manifest_path).reset_index(drop=True)
        self.tile_size = tile_size

    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row= self.manifest.iloc[idx]
        d= np.load(row["path"])
        spatial, scalars, target = d["spatial"], d["scalars"], d["target"]

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
        y = np.random.randint(0, H-k + 1)
        x = np.random.randint(0, W - k +1)
        return spatial[:, y:y+k, x:x+k], target[y:y+k, x:x+k] 
