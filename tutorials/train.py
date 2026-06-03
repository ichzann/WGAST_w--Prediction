# train.py
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset import WGASTSurrogateDataset
from model_unet import ConditionalUNet
from losses import L1SSIMLoss

MANIFEST = "tutorials/data/secondary/samples_istanbul/manifest.parquet"
CKPT_DIR = Path("tutorials/data/secondary/ckpt")

TILE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 200
LR = 1e-3
SSIM_DATA_RANGE = 6.0  # target is z-scored -> ~ ±3σ


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device", device)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ds = WGASTSurrogateDataset(MANIFEST, tile_size=TILE_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = ConditionalUNet().to(device)
    loss_fn = L1SSIMLoss(ssim_weight=0.1, data_range=SSIM_DATA_RANGE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss = tot_l1 = tot_ssim = 0.0
        n = 0
        for batch in loader:
            sp = batch["spatial"].to(device)
            sc = batch["scalars"].to(device)
            tg = batch["target"].to(device)

            pred = model(sp, sc)
            loss, parts = loss_fn(pred, tg)
            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = sp.size(0)
            tot_loss += loss.item() * bs
            tot_l1 += parts["l1"].item() * bs
            tot_ssim += parts["ssim"].item() * bs
            n += bs

        print(f"epoch {epoch:3d}  loss={tot_loss/n:.5f}  l1={tot_l1/n:.5f}  ssim={tot_ssim/n:.4f}")
        torch.save({"model": model.state_dict()}, CKPT_DIR / "smoke.pt")


if __name__ == "__main__":
    main()
