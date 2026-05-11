"""Compute per-scene and overall metrics for WGAST test predictions.

Background
----------
``experiment.test()`` writes a predicted 10 m Sentinel-grid LST TIF per scene
into ``test_dir`` but does not compute any quality metric. The weakly-supervised
target used during training is the **Landsat t1** LST at 30 m, so that is what
we score against here. The mask saved alongside the Landsat TIF tells us which
pixels are valid (cloud-free, in-bounds).

Procedure (per scene subdirectory under ``test_dir``):
  1. Locate the Landsat t1 ground truth + mask via ``get_pair_path_with_masks``.
  2. Locate the matching prediction TIF at ``test_dir/<...>_Sentinel_<date>.tif``.
  3. Downsample the prediction from 10 m to 30 m by 3x3 mean pooling.
  4. Apply the Landsat mask, compute RMSE / MAE / Bias / PSNR.

Units note: metrics are in the same units as the Landsat LST stored on disk
(Kelvin, Celsius, or a scaled value depending on ``data_preparation``). Check
your preprocessing before interpreting absolute numbers.

Usage
-----
    from pathlib import Path
    from runner.evaluate import evaluate_predictions

    df = evaluate_predictions(Path("data/Tdivision/test"))
    print(df.to_string(index=False))
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_loader.data import get_pair_path_with_masks  # noqa: E402

LANDSAT_KEY = "Landsat"
SENTINEL_KEY = "Sentinel"
DOWNSAMPLE = 3  # 10 m prediction -> 30 m Landsat reference


def _block_mean_2d(a: np.ndarray, k: int) -> np.ndarray:
    """k x k mean pooling on a 2-D array; trims edges to a multiple of k."""
    h, w = a.shape
    h2, w2 = (h // k) * k, (w // k) * k
    a = a[:h2, :w2]
    return a.reshape(h2 // k, k, w2 // k, k).mean(axis=(1, 3))


def _scene_metrics(pred30: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    valid = (mask > 0) & np.isfinite(pred30) & np.isfinite(gt)
    n = int(valid.sum())
    if n == 0:
        return {"n_pixels": 0, "RMSE": np.nan, "MAE": np.nan,
                "Bias": np.nan, "PSNR_dB": np.nan,
                "gt_min": np.nan, "gt_max": np.nan,
                "pred_min": np.nan, "pred_max": np.nan}

    p = pred30[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    err = p - g

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    data_range = float(g.max() - g.min())
    psnr = (20.0 * math.log10(data_range / rmse)
            if rmse > 0 and data_range > 0 else float("inf"))

    return {
        "n_pixels": n,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "PSNR_dB": float(psnr),
        "gt_min": float(g.min()),
        "gt_max": float(g.max()),
        "pred_min": float(p.min()),
        "pred_max": float(p.max()),
    }


def evaluate_predictions(test_dir: Path | str) -> pd.DataFrame:
    """Score every scene subdir under ``test_dir`` against its Landsat t1 target.

    Returns a DataFrame with one row per scene plus a final ``ALL`` row
    holding pixel-weighted overall RMSE / MAE / Bias.
    """
    test_dir = Path(test_dir)
    scene_dirs = sorted(p for p in test_dir.glob("*") if p.is_dir())
    if not scene_dirs:
        raise FileNotFoundError(f"No scene subdirectories under {test_dir}")

    rows: list[dict] = []
    for scene in scene_dirs:
        pairs = get_pair_path_with_masks(scene)
        if len(pairs) < 5:
            print(f"[skip] {scene.name}: incomplete pair set ({len(pairs)}/5)")
            continue
        gt_path, mask_path = pairs[-1]  # Landsat t1

        pred_name = gt_path.name.replace(LANDSAT_KEY, SENTINEL_KEY)
        pred_path = test_dir / pred_name
        if not pred_path.exists():
            print(f"[skip] {scene.name}: no prediction at {pred_path}")
            continue

        with rasterio.open(gt_path) as src:
            gt = src.read(1).astype(np.float32)
        with rasterio.open(pred_path) as src:
            pred10 = src.read(1).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        if mask.ndim == 3:
            mask = mask[0]

        pred30 = _block_mean_2d(pred10, DOWNSAMPLE)
        h = min(pred30.shape[0], gt.shape[0], mask.shape[0])
        w = min(pred30.shape[1], gt.shape[1], mask.shape[1])
        pred30, gt, mask = pred30[:h, :w], gt[:h, :w], mask[:h, :w]

        m = _scene_metrics(pred30, gt, mask)
        m["scene"] = scene.name
        m["date"] = gt_path.stem.split("_")[-1]
        rows.append(m)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    total_n = int(df["n_pixels"].sum())
    if total_n > 0:
        weighted_mse = float((df["n_pixels"] * df["RMSE"] ** 2).sum() / total_n)
        overall = {
            "scene": "ALL",
            "date": "",
            "n_pixels": total_n,
            "RMSE": math.sqrt(weighted_mse),
            "MAE": float((df["n_pixels"] * df["MAE"]).sum() / total_n),
            "Bias": float((df["n_pixels"] * df["Bias"]).sum() / total_n),
            "PSNR_dB": np.nan,
            "gt_min": float(df["gt_min"].min()),
            "gt_max": float(df["gt_max"].max()),
            "pred_min": float(df["pred_min"].min()),
            "pred_max": float(df["pred_max"].max()),
        }
        df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)

    cols = ["scene", "date", "n_pixels", "RMSE", "MAE", "Bias", "PSNR_dB",
            "gt_min", "gt_max", "pred_min", "pred_max"]
    return df[cols]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score WGAST test predictions.")
    ap.add_argument("test_dir", type=Path,
                    help="Path to the test directory (containing scene subdirs "
                         "and the saved prediction TIFs).")
    args = ap.parse_args()

    df = evaluate_predictions(args.test_dir)
    if df.empty:
        print("No scored scenes.")
    else:
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(df.to_string(index=False))
