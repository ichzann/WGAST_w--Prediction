# build_dataset.py
#
# For each clear-sky day d0 with a cached WGAST output, assemble the 10-day
# window d-10..d-1 of (S2 + LS + optical mask + past WGAST + WGAST mask) plus
# DEM and per-day daily weather, then save one .npz + a parquet manifest.
#
# After all samples are written, fit_stats() scans the manifest once and writes
# a stats.npz of per-channel mean/std used by dataset.py for normalisation.

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

WINDOW = 10
DATA_ROOT = Path(__file__).resolve().parent / "data" / "secondary"


# -- parsers ----------------------------------------------------------------
def _p_wgast(p): return datetime.strptime(p.stem.split("_")[1], "%Y%m%d").date()
def _p_s2(p):    return datetime.strptime(p.stem[:8], "%Y%m%d").date()
def _p_ls(p):    return datetime.strptime(p.stem.split("_")[-1], "%Y%m%d").date()

def _index(paths, parse):
    out = {}
    for p in paths:
        out.setdefault(parse(p), []).append(p)
    return out


# -- raster IO --------------------------------------------------------------
def _read_aligned(src_path, ref, bands):
    """Read `bands` from src_path, reproject onto the WGAST reference grid."""
    with rasterio.open(src_path) as src:
        out = np.zeros((len(bands), ref["height"], ref["width"]), dtype=np.float32)
        for i, b in enumerate(bands):
            reproject(
                source=rasterio.band(src, b), destination=out[i],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref["transform"], dst_crs=ref["crs"],
                resampling=Resampling.bilinear,
            )
    return out

def _stack_date(paths, ref, bands, n_bands):
    if not paths:
        return np.zeros((n_bands, ref["height"], ref["width"]), dtype=np.float32)
    arrs = [_read_aligned(p, ref, bands) for p in paths]
    return np.maximum.reduce(arrs) if len(arrs) > 1 else arrs[0]

def _opt_mask(stack):
    return (np.any(stack != 0, axis=0)).astype(np.float32)

def _ref_grid(wgast_path):
    with rasterio.open(wgast_path) as s:
        return dict(crs=s.crs, transform=s.transform, height=s.height, width=s.width)


# -- per-d0 sample ----------------------------------------------------------
def build_sample(d0, ref, wgast_idx, s2_idx, ls_idx, weather, dem):
    H, W = ref["height"], ref["width"]
    s2s, lss, wgs, opt_m, wg_m = [], [], [], [], []

    for k in range(1, WINDOW + 1):
        dk = d0 - timedelta(days=k)
        s2 = _stack_date(s2_idx.get(dk, []), ref, [1, 2, 3], 3)
        ls = _stack_date(ls_idx.get(dk, []), ref, [2, 3, 4], 3)   # skip LS band 1 (LST)
        opt = ((_opt_mask(s2) + _opt_mask(ls)) > 0).astype(np.float32)

        if dk in wgast_idx:
            wg = _read_aligned(wgast_idx[dk][0], ref, [1])[0]
            wm = np.ones((H, W), dtype=np.float32)
        else:
            wg = np.zeros((H, W), dtype=np.float32)
            wm = np.zeros((H, W), dtype=np.float32)

        s2s.append(s2); lss.append(ls); wgs.append(wg); opt_m.append(opt); wg_m.append(wm)

    # N=1 filters
    if not any(m.any() for m in opt_m): return None
    if not any(m.any() for m in wg_m):  return None

    # spatial stack: 10 * (3 S2 + 3 LS + 1 opt + 1 WGAST + 1 WGmask) + 1 DEM = 91
    slots = []
    for s2, ls, om, wg, wm in zip(s2s, lss, opt_m, wgs, wg_m):
        slots += [s2, ls, om[None], wg[None], wm[None]]
    spatial = np.concatenate(slots + [dem], axis=0)

    # scalars: 12 weather * 10 days + sin/cos(doy)
    wx = []
    for k in range(1, WINDOW + 1):
        dk = d0 - timedelta(days=k)
        row = weather.loc[weather.index.date == dk]
        wx.append(row.iloc[0].values if len(row) else np.zeros(weather.shape[1]))
    wx = np.concatenate(wx).astype(np.float32)
    doy = d0.timetuple().tm_yday
    season = np.array([np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)], dtype=np.float32)
    scalars = np.concatenate([wx, season])

    target = _read_aligned(wgast_idx[d0][0], ref, [1])[0]
    return spatial, scalars, target


# -- build all samples for one city -----------------------------------------
def build_city(city):
    cap = city.capitalize()
    wgast_dir = DATA_ROOT / "wgast_cache" / city.lower()
    s2_dir    = DATA_ROOT / "raw" / f"Sentinel2_window_{cap}"
    ls_dir    = DATA_ROOT / "raw" / f"Landsat8_window_{cap}"
    dem_path  = DATA_ROOT / "raw" / f"DEM_{cap}.tif"
    weather_p = DATA_ROOT / f"weather_{cap}_daily.parquet"
    out_dir   = DATA_ROOT / f"samples_{city.lower()}"
    out_dir.mkdir(exist_ok=True)

    wgast_idx = _index(sorted(wgast_dir.glob("wgast_*.tif")), _p_wgast)
    s2_idx    = _index(sorted(s2_dir.glob("*.tif")),          _p_s2)
    ls_idx    = _index(sorted(ls_dir.glob("LC08_*.tif")),     _p_ls)
    weather   = pd.read_parquet(weather_p)
    weather.index = pd.to_datetime(weather.index)

    rows = []
    for d0, paths in sorted(wgast_idx.items()):
        ref = _ref_grid(paths[0])
        dem = _read_aligned(dem_path, ref, [1])
        sample = build_sample(d0, ref, wgast_idx, s2_idx, ls_idx, weather, dem)
        if sample is None:
            print(f"  skip {d0} (filters)"); continue
        spatial, scalars, target = sample
        out = out_dir / f"sample_{d0:%Y%m%d}.npz"
        np.savez_compressed(out, spatial=spatial, scalars=scalars, target=target)
        rows.append(dict(city=city.lower(), d0=str(d0), path=str(out)))
        print(f"  kept {d0}  spatial={spatial.shape}  scalars={scalars.shape}")

    manifest = out_dir / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest)
    print(f"\n{len(rows)}/{len(wgast_idx)} samples written -> {out_dir}")
    return manifest


# -- per-channel z-score stats ---------------------------------------------
def fit_stats(manifest_path):
    """Single pass over the manifest. Per-channel mean/std for spatial (C,H,W),
    per-element mean/std for scalars (S,), scalar mean/std for target."""
    df = pd.read_parquet(manifest_path)

    sp_sum = sp_sq = sp_n = None
    sc_sum = sc_sq = sc_n = None
    tg_sum = tg_sq = tg_n = 0.0

    for path in df["path"]:
        d = np.load(path)
        sp, sc, tg = d["spatial"], d["scalars"], d["target"]

        # spatial: per-channel sums over H*W pixels
        if sp_sum is None:
            sp_sum = np.zeros(sp.shape[0], dtype=np.float64)
            sp_sq  = np.zeros(sp.shape[0], dtype=np.float64)
            sp_n   = 0
        sp_sum += sp.sum(axis=(1, 2))
        sp_sq  += (sp.astype(np.float64) ** 2).sum(axis=(1, 2))
        sp_n   += sp.shape[1] * sp.shape[2]

        # scalars: per-element across samples
        if sc_sum is None:
            sc_sum = np.zeros(sc.shape[0], dtype=np.float64)
            sc_sq  = np.zeros(sc.shape[0], dtype=np.float64)
            sc_n   = 0
        sc_sum += sc
        sc_sq  += sc.astype(np.float64) ** 2
        sc_n   += 1

        tg_sum += float(tg.sum())
        tg_sq  += float((tg.astype(np.float64) ** 2).sum())
        tg_n   += tg.size

    sp_mean = (sp_sum / sp_n).astype(np.float32)
    sp_std  = np.sqrt(np.maximum(sp_sq / sp_n - sp_mean ** 2, 1e-12)).astype(np.float32)
    sp_std  = np.where(sp_std < 1e-6, 1.0, sp_std).astype(np.float32)  # constant channels -> no scaling

    sc_mean = (sc_sum / sc_n).astype(np.float32)
    sc_std  = np.sqrt(np.maximum(sc_sq / sc_n - sc_mean ** 2, 1e-12)).astype(np.float32)
    sc_std  = np.where(sc_std < 1e-6, 1.0, sc_std).astype(np.float32)

    tg_mean = np.float32(tg_sum / tg_n)
    tg_std  = np.float32(max(np.sqrt(tg_sq / tg_n - float(tg_mean) ** 2), 1e-6))

    stats_path = Path(manifest_path).parent / "stats.npz"
    np.savez(stats_path,
             spatial_mean=sp_mean, spatial_std=sp_std,
             scalars_mean=sc_mean, scalars_std=sc_std,
             target_mean=tg_mean, target_std=tg_std)
    print(f"stats written -> {stats_path}")
    print(f"  spatial std range: {sp_std.min():.3g} .. {sp_std.max():.3g}")
    print(f"  scalars std range: {sc_std.min():.3g} .. {sc_std.max():.3g}")
    print(f"  target mean={tg_mean:.4g}  std={tg_std:.4g}")
    return stats_path


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "istanbul"
    manifest = build_city(city)
    fit_stats(manifest)
