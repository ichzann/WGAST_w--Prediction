#evaluate.py


import numpy as np 
import pandas as pd 
import torch 
import torch.nn.functional as F

from dataset import WGASTSurrogateDataset
from model_unet import ConditionalUNet
from losses import ssim

MANIFEST = "tutorials/data/secondary/samples_istanbul/manifest.parquet"
CKPT     = "tutorials/data/secondary/ckpt/smoke.pt"


def metrics(pred, target, data_range=1.0):
    diff = pred - target
    mse  = (diff ** 2).mean()
    return {
        "rmse": mse.sqrt().item(),
        "mae":  diff.abs().mean().item(),
        "bias": diff.mean().item(),
        "psnr": (10 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))).item(),
        "ssim": ssim(pred, target, data_range=data_range).item(),
    }


def pad16(x):
    """Pad spatial dims to next multiple of 16. Returns (padded, h, w)."""
    _, _, h, w = x.shape
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    return F.pad(x, (0, pw, 0, ph)), h, w


def main():
    device = ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_parquet(MANIFEST).sort_values("d0").reset_index(drop=True)

    model = ConditionalUNet().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)["model"])
    model.eval()

    rows = []
    prev_target = None
    with torch.no_grad():
        for _, row in df.iterrows():
            d = np.load(row["path"])
            spatial = torch.from_numpy(d["spatial"].copy()).unsqueeze(0).to(device)
            scalars = torch.from_numpy(d["scalars"].copy()).unsqueeze(0).to(device)
            target  = torch.from_numpy(d["target"].copy())[None, None].to(device)

            sp_pad, h, w = pad16(spatial)
            pred = model(sp_pad, scalars)[:, :, :h, :w]

            rows.append({"d0": row["d0"], "what": "model",
                        **metrics(pred, target)})

            if prev_target is not None:
                rows.append({"d0": row["d0"], "what": "persistence",
                            **metrics(prev_target.to(device), target)})

            prev_target = target.detach().cpu()

    out = pd.DataFrame(rows)
    print("Per-sample:")
    print(out.to_string(index=False))
    print("\nMeans by baseline:")
    print(out.groupby("what")[["rmse", "mae", "bias", "psnr", "ssim"]].mean())


if __name__ == "__main__":
    main()