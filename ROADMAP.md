# Roadmap — WGAST Day-Ahead LST Forecaster

> Living document. The goal of this file is two-fold:
> 1. Anchor what the project is about so any future Claude session can pick up cold.
> 2. Track decisions and open questions as we move from idea → implementation.

---

## 1. Project aim

WGAST produces a 10 m **Land Surface Temperature** (LST) map for a given day from
satellite imagery — but only for **day-0** (it cannot run on a day without cloud-free
satellite observations). This project adds a **secondary ML model** that answers:

> *"What will WGAST output tomorrow?"*

The secondary model is trained as a **WGAST surrogate over a past window**: given the
non-LST features WGAST sees (Sentinel-2 and Landsat indices) plus DEM and weather, all
collected over the **5 days preceding** a target day, it predicts the WGAST raster for
that target day. The raster itself is never a feature — only the past-window
non-LST signal.

```text
TRAINING     features over d-5..d-1  ──►  WGAST raster on day d0
INFERENCE    features over d-4..d0   ──►  predicted WGAST raster on day d+1
```

The **window-shift trick** is the core idea: at training time the last feature day
is `d-1` and the target is `d0`; at inference time the same model is fed features
through `d0` and asked for `d+1`. The temporal offset between "last feature day"
and "predicted day" is always +1, so train and inference distributions match.

Two consequences:

- **Data volume**: every clear-sky day where WGAST runs becomes a supervised sample
  (target = WGAST(d0)). Roughly 3–5× more rows than a `(d, d+1)`-pair scheme.
- **Distillation, not raster propagation**: the model learns the WGAST input→output
  mapping over a temporal window. It is not given a previous raster to start from.

The output is a 10 m LST map at the same resolution and shape as WGAST itself.
Crucially, at inference the model works **on cloudy d+1 too** — d+1 itself does
not need to be observed, since the features come from the past 5 days.

**Anchor use case:** *day-ahead urban-heat-island warning.* The 10 m raster lets you
flag specific neighbourhoods, parks, or city blocks that will be hottest tomorrow,
not just a citywide average. Air temperature is **not** a model target — if anyone
wants a scalar like T_max, they derive it post-hoc from the predicted raster.

**Generalisation philosophy:** the secondary model only sees intrinsic state. It is
never told *which* city it is in — no city_id, no climate zone, no lat/lon. The one
exception is **elevation**, which is included because it is a *physical scalar*
(it determines atmospheric pressure and the adiabatic temperature offset, ~6–10 °C/km),
not an identity label. Cross-city generalisation is by construction, not by labelling.

---

## 2. What WGAST is, technically

WGAST = *Weakly-Supervised Generative Network for Daily 10 m LST Estimation via
Spatio-Temporal Fusion* (Bouaziz et al., 2025, [arXiv:2508.06485](https://arxiv.org/abs/2508.06485)).

### 2.1 Inputs (per inference)

WGAST does **spatio-temporal fusion** of three satellites at two timestamps. Naming
follows the code (`model/WGAST.py::CombinFeatureGenerator.forward`,
`data_loader/data.py`):

| Input idx | Source              | Resolution | Bands fed into net      | Role                              |
|-----------|---------------------|-----------:|-------------------------|-----------------------------------|
| `inputs[0]` | Terra MODIS @ t₂  | 1 km       | 1 (LST)                 | Coarse LST at the **target** date |
| `inputs[1]` | Landsat 8 @ t₀    | 30 m       | 4 (LST + 3 indices)     | Reference high-spatial LST + spectral indices at a **past** date |
| `inputs[2]` | Sentinel-2 @ t₂   | 10 m       | 3 (indices)             | Fine-spatial spectral context at the **target** date |
| `inputs[3]` | Terra MODIS @ t₀  | 1 km       | 1 (LST)                 | Coarse LST at the reference date  |

`t₀` is the previous Landsat-available reference date; `t₂` is the inference target day.
By design the model never needs a *future* observation — making it real-time-friendly
for day-0.

### 2.2 Output

```text
output  : torch.Tensor of shape (B, NUM_BANDS=1, H, W)
units   : Land Surface Temperature (LST), in the same scale as the training targets
                       (Landsat-derived proxy LST; check data_preparation/ for the exact
                        scaling / normalisation that was applied).
spatial : 10 m per pixel — the Sentinel-2 grid (inputs[2]'s H, W).
temporal: one raster per target day t₂.
```

Internally the network is a **conditional GAN**:

- Generator (`CombinFeatureGenerator`): three parallel `FeatureExtract` encoders
  (MODIS, Landsat-LST, indices), per-level cosine-similarity refinement of Landsat
  LST features using Sentinel/Landsat indices, **AdaIN** to stylise high-res features
  with MODIS statistics at t₀, attention-based `SignificanceExtraction` that fuses
  the t₀ and t₂ MODIS features with the refined high-res features, then a 5-stage
  deconv decoder back to 10 m resolution.
- Discriminator (`NLayerDiscriminator`): standard PatchGAN.
- Weak supervision: 30 m Landsat LST is used as proxy ground-truth — there is no true
  10 m LST label.

### 2.3 Practical relationship to this project

For the secondary model we treat WGAST as a **black box that is correct**: on every
clear-sky day d0, WGAST(d0) is **the supervisory target**. The secondary model
never receives a previous WGAST raster as input — its job is to reconstruct WGAST(d0)
from the non-LST inputs WGAST itself sees, but pulled from the **5 days preceding
d0** instead of from d0. We deliberately do **not** question per-city WGAST quality
in v1 (see §4.2).

---

## 3. The secondary model

### 3.1 Inputs (intrinsic state only — no identity, no location, no LST)

Each training sample = `(region, date d0)` where d0 is a clear-sky day (so we have
a real WGAST(d0) as the supervisory target). All features come from the
**5-day window d-5..d-1** preceding d0 — d0 itself is never observed at training
time. At inference the window is shifted to d-4..d0 and the model produces the
prediction for d+1.

**Spatial inputs** (stacked into the U-Net encoder along the channel dimension):

- **Per-day optical indices**, for each of the 5 days in the window:
  - Sentinel-2 indices at 10 m (3 bands, as in WGAST).
  - Landsat indices at 30 m (3 bands, the non-LST ones from WGAST — Landsat is
    upsampled to the Sentinel-2 grid).
  - **Per-day cloud-mask channel** (1 = valid pixel, 0 = masked). Cloudy pixels
    in the band channels are set to 0; the mask channel tells the model which
    pixels are real.
- **Latest available WGAST raster strictly before d0** (1 channel, 10 m). This is
  the *most recent* WGAST output on any clear-sky day ≤ d-1 — could be d-1 itself,
  could be d-7 if the window had no further clear days. **Provides the LST signal
  to the model.** At inference this slot is filled by WGAST(d0), which is fresh
  by construction (we only run inference on clear d0). The model thus learns to
  use a possibly-stale LST snapshot at training time and gets the easy case at
  inference — robustness comes for free.
- **DEM** (digital elevation model, static per region, 1 channel at the Sentinel
  grid). Carries intra-city elevation variation — the very feature that produces
  spatial LST contrasts between hills, valleys, and built-up flats.

Channel count for the spatial branch: 5 days × (3 S2 + 3 LS + 1 mask) + 1 WGAST
LST + 1 DEM = **37 channels** at the encoder input (subject to revision when
bands are finalised in code).

**Scalar inputs** (enter the bottleneck via a small MLP):

- **Daily-aggregated weather** for each of d-5..d-1: air temperature (mean/max/min),
  dew point, relative humidity, wind speed, surface pressure, cloud cover,
  precipitation. ~7 vars × 5 days ≈ **35 scalars**. Source: Open-Meteo Archive.
- **Seasonal phase of the target day**: `sin(2π·doy_d0/365)`, `cos(2π·doy_d0/365)`.
- **Elevation scalar** (single scalar per city, at the representative point) —
  kept in addition to the DEM channel because it provides a region-level adiabatic
  offset to the bottleneck even if the encoder under-uses the DEM channel.
- **`age_days_lst`**: how many days old the latest-WGAST input channel is, relative
  to the target day. 0 if WGAST was available on d-1 itself (or on d0 at inference).
  Tells the model when to discount the LST channel.
- **Optional forecast for the target day** (forecast issued on d-1 for d0 at
  train time; forecast issued on d0 for d+1 at inference): forecast T_max, cloud
  cover, wind, humidity. ~4 scalars. Include if Open-Meteo coverage permits.

**N=1 cloud filter.** A sample is kept iff **at least one day in the window has a
usable optical observation** (any of Sentinel-2 / Landsat). In practice this filter
almost never bites — but it guarantees the spatial branch has *some* real signal
to work with rather than 5 all-cloudy days. Days within a kept sample that are
cloudy contribute zeros + the mask channel; the model learns to lean on the days
that are clear. Channel order encodes recency (channels for d-1 always sit in the
same slot), so no separate `age_days` scalar is needed.

**Explicitly excluded** (so generalisation is by design, not by label):

- ❌ city_id
- ❌ climate zone
- ❌ latitude / longitude
- ❌ **MODIS / Landsat raw LST**. The LST signal enters the model only through
  WGAST's own previous output (the latest-WGAST channel above). We deliberately do
  not give the model the raw coarse satellite LST that WGAST itself ingests —
  feeding WGAST(d-1) is cleaner, already at 10 m, and matches what's actually
  available in production. MODIS therefore drops out of the input set entirely.

All weather/forecast features come from **Open-Meteo** to keep observations, forecasts,
and any downstream sanity-checks on the same provider.

### 3.2 Target

The **real WGAST raster on the target day d0**:

```text
y = WGAST(MODIS_{d0}, Landsat_{t0}, Sentinel_{d0}, MODIS_{t0})  ∈  ℝ^{H × W}
```

Available only for clear-sky d0. Each such day in our pipeline becomes one training
row. At inference the same trained network is queried with the window shifted by
+1 day, producing the prediction for d+1.

### 3.3 v1 architecture — conditional U-Net (sequence-as-channels)

```text
                                                                         predicted
  spatial stack                                                          WGAST(d0)
  ─────────────                                                          ─────────
  • 5×(S2 indices + LS indices + cloud mask)                                 ▲
  • latest-WGAST raster (≤ d-1 at train, = d0 at infer)                      │
  • DEM                                                                      │
  ≈ 37 channels @ H×W                                                        │
        │                                                                    │
        ▼                                                                    │
  U-Net encoder ──┬──── bottleneck embedding ─────────────► U-Net decoder ──┘
  (with skips)    │                                            (uses skips)
                  └────── skip features ──────────────────────────► (skips)
                                                ▲
                                                │
                                          concat with conditioning
                                                ▲
                                                │
  scalar inputs                                 │
  ─────────────                                 │
  • 5×7 daily weather  ≈ 35 vals                │
  • sin/cos(doy_d0)                             │
  • elevation scalar                 small MLP ─┘
  • age_days_lst                  (~32-dim out)
  • forecast(d0) scalars (optional)
```

Design choices, all locked for v1:

| Decision | Choice |
|---|---|
| Encoder/decoder family | **U-Net** with skip connections — fine spatial detail matters for heat-island spotting |
| Temporal handling | **Days stacked as channels** (sequence position implicit). ConvLSTM / temporal attention deferred — data volume doesn't yet justify it |
| Cloud handling | **Per-day binary mask channel**; cloudy pixels zeroed in the band channels |
| Topography | **DEM as a static input channel** (was v2 in the previous plan; pulled forward into v1 because the new design has no raster-as-input) |
| How scalar features enter | **Concat at the bottleneck** (simplest; FiLM is a v2 option only if needed) |
| Loss | `L1 + λ·(1 − SSIM)`, **λ ≈ 0.1** (the WGAST paper's regime) |
| Optimiser | AdamW + cosine schedule, early stopping on val L1 |
| Capacity | ~0.5–2 M params (between a small U-Net and WGAST itself); trainable on a single GPU |
| Temporal / GRU branch on hourly weather | **Deferred to v3.** v1 = one image branch + one scalar branch only |

### 3.4 Tile-based training augmentation

At training time, crop random `K × K` tiles (e.g. `128 × 128` or `256 × 256`) from
the paired `(input_raster, target_raster)`. Same crop coordinates on both sides;
all scalar conditioning is replicated. This gives the encoder more spatial variation
per epoch, regularises a small U-Net, and side-steps any per-city size mismatch.

At inference / evaluation, run on the **full raster** (or tile + stitch with overlap,
which is exactly what `runner/experiment.py:test()` already does for WGAST).

### 3.5 Baselines that must always be reported

The model earns its place by beating these:

1. **Persistence-raster**: `predicted(d0) = WGAST(d_prev)`, where `d_prev` is the
   most recent clear-sky WGAST output strictly before d0 — i.e., exactly the
   latest-WGAST channel that the model itself receives as input. The "no model"
   floor. Trivially strong on stable weather. At inference for d+1 this becomes
   `predicted(d+1) = WGAST(d0)`.
2. **Weather-only U-Net**: identical architecture, but **all optical channels
   AND the latest-WGAST channel** are zeroed (DEM kept). Isolates exactly what
   the spatial inputs (optical stack + WGAST LST) contribute on top of weather
   + DEM + season alone.

The metric of interest is skill vs. baseline #1 (does the model do anything?) and
skill vs. baseline #2 (does the visual + DEM stack contribute, or is it all in the
weather?).

### 3.6 Evaluation metrics

Same metrics the WGAST paper reports against Landsat — applied here against the real
WGAST(d0) raster at train/val/test time (and, for the deployment use case, against
the real WGAST(d+1) when a clear-sky d+1 is available for evaluation):

- **RMSE** (in the LST unit on disk; the `gt_min/gt_max` columns from
  `runner/evaluate.py` tell us the scale)
- **MAE**
- **Bias** (mean of predicted − target)
- **PSNR**
- **SSIM**
- Optional sanity check: summary stats of predicted raster vs. observed T_max from
  Open-Meteo. Not a training signal — just a "does the predicted thermal field look
  physically plausible?" diagnostic.

### 3.7 Splits

**Time-based, never random.** Spatial holdout for an OOD test in v2.

| Split | Span | Purpose |
|---|---|---|
| Train | 2018-01 → 2023-12 | Fitting |
| Val | 2024-01 → 2024-06 | Early stopping, HP tuning |
| Test (IID, time-held-out) | 2024-07 → today | Honest in-distribution number |
| Test (OOD, city-held-out) | *v2* | One paper-ROI city never seen in train/val |

---

## 4. Data plan

### 4.1 Cities

Seven ROIs, all validated in the WGAST paper:

| City | Climate | Expected clear-sky days / yr |
|---|---|---|
| Orléans (current ROI, paper "Tours" band) | Cfb temperate | ~8 |
| Tours | Cfb temperate | ~8 |
| Montpellier | Csa Mediterranean | ~15 |
| Madrid | Csa hot-summer Mediterranean | ~18 |
| Rome | Csa hot-summer Mediterranean | ~15 |
| Istanbul | Csa / Cfa | ~12 |
| Cairo | BWh hot desert | ~25 |

Under the new design, **every clear-sky day where WGAST runs is one training row**
(target = WGAST on that day), provided the preceding 5-day window contains at least
one usable optical observation. Expected sample count over 2018-01 → 2024-12:
**~700–900 rows** across the seven cities (same WGAST clear-sky budget — the change
of target framing doesn't manufacture extra clear-sky days, it just removes the
"d+1 also clear" pairing constraint, which in the previous design dropped a large
fraction of usable d0 days). Effective supervised signal is much higher — each row
is a ~1 M-pixel raster, so the gradient information is comparable to ~10⁸ scalar
supervised examples, plenty for a U-Net at this capacity.

### 4.2 WGAST availability across cities — explicit assumption

For v1 we **assume WGAST output is trustworthy in all seven cities** (justified by the
paper's spatio-temporal generalisation section). Per-city quality is not
re-validated in this project. If the secondary model degrades in v2 OOD evaluation,
revisit:

1. Use the paper's released weights (if a multi-city checkpoint is published).
2. Retrain WGAST on the union of these seven ROIs.

### 4.3 Pipeline (reuses the existing repo)

```text
for region R in ROIs:
    dem = load_dem(R)                                        # static, cached once
    for d0 in clear_sky_days(R, 2018..2024):                 # supervised target dates
        raster_target = WGAST(at d0) [cached]                # supervisory raster

        # Latest-WGAST input — most recent WGAST output strictly before d0
        d_prev   = most_recent_wgast_date_before(R, d0)
        if d_prev is None: continue                          # no prior WGAST → drop
        raster_lst_in = WGAST(at d_prev) [cached]
        age_lst       = (d0 - d_prev).days                   # scalar conditioning

        # Past 5-day window of optical + weather
        window = []
        for k in 1..5:
            d_k = d0 - k days
            s2  = fetch_sentinel_indices(R, d_k)             # may be cloudy → mask
            ls  = fetch_landsat_indices (R, d_k)             # may be missing
            cm  = build_cloud_mask(s2, ls)                   # per-day mask
            wx  = open_meteo.archive(R.point, d_k)           # daily aggregates
            window.append({s2, ls, cm, wx})

        # N=1 filter
        if not any_day_has_usable_optical(window): continue

        fcst_d0 = open_meteo.forecast_archive(R.point, d0 - 1d, lead=1)  # optional
        write_sample(R, d0, raster_target, raster_lst_in, age_lst,
                     window, dem, fcst_d0)
```

WGAST runs **offline once** per `(R, date)` and writes a 10 m TIF to a cache
directory, reused across re-trains of the secondary model. Disk cost: ~4 MB per
raster × ~800 target rasters ≈ **~3 GB** for WGAST outputs, plus the per-day optical
stacks for the 5-day windows (cached at the Sentinel grid).

**Note on storing hourly weather:** even though v1 only consumes daily-aggregated
weather, the pipeline saves **hourly** observations alongside. This is so v2's
optional GRU branch can be added without re-fetching from Open-Meteo.

---

## 5. Decisions made + remaining open questions

### 5.1 Decided

- **Training framing**: target = WGAST raster on day d0; features = 5-day window
  d-5..d-1 of optical+weather **plus** the latest WGAST raster strictly before d0
  (could be d-1 or older, with an `age_days_lst` scalar telling the model how stale).
- **Inference framing**: shift the window to d-4..d0 and predict d+1; the latest-
  WGAST slot is filled by WGAST(d0). Same trained network; the +1 offset between
  "last feature day" and "predicted day" is identical to training.
- **Single horizon: h = +1 day.**
- **Use case: day-ahead urban-heat-island warning** at 10 m resolution.
- **Region scale = WGAST training scale** (city-sized polygon).
- **Cities = seven paper-validated ROIs** (§4.1).
- **WGAST is treated as ground truth across all cities for v1** (§4.2).
- **No identity / location labels in features** — city_id, climate_zone, lat, lon
  excluded by design. **Elevation IS included**, both as a static DEM channel in
  the spatial branch and as a region-level scalar in the conditioning MLP.
- **LST signal enters only via WGAST's own previous output** (the latest-WGAST
  channel). Raw MODIS LST and Landsat LST are not fed to the model. MODIS drops
  out of the input set entirely (it contributed only LST in WGAST).
- **Weather / forecast / observations source = Open-Meteo** (Archive + historical-
  forecast endpoints).
- **v1 architecture = conditional U-Net** (sequence-as-channels spatial branch,
  scalar conditioning concat at bottleneck), ~0.5–2 M params.
- **Cloud handling = strategy N=1 + per-day mask channel.** Sample is kept iff at
  least one day in the window has any usable optical observation. Cloudy pixels
  are zeroed in the band channels; a per-day binary mask channel tells the model
  which pixels are valid. Channel order encodes recency — no separate `age_days`
  scalar for the optical window (`age_days_lst` is a separate scalar for the
  latest-WGAST channel only).
- **Loss = L1 + λ·(1 − SSIM), λ ≈ 0.1.**
- **GRU / hourly-weather temporal branch deferred to v3.**
- **Tile-based training augmentation** (random crops with the same coordinates
  across all temporal channels), full raster at inference.
- **Baselines** = persistence-raster (most recent WGAST output strictly before the
  target day — same thing the model receives in its latest-WGAST channel) +
  weather-only U-Net (optical channels AND latest-WGAST channel zeroed, DEM kept).
- **Metrics** = RMSE / MAE / Bias / PSNR / SSIM on the raster vs. real WGAST(d0).
  Optional summary-stat sanity check vs. observed T_max.
- **Splits = time-based** (IID test on last months); OOD city in v2.

### 5.2 Still open

1. **Window length** — fixed at 5 days for v1. Revisit if val performance is
   bottlenecked by stale optical inputs (shorten to 3) or by data scarcity
   (lengthen to 7).
1b. **`age_days_lst` distribution** — once `build_dataset.py` runs, plot the
   histogram. If most samples have very-old LST inputs (e.g. weeks), consider
   capping (drop samples where age > N) so the model isn't dominated by a stale-
   input regime that will never occur at inference. At inference age_lst is
   almost always 0.
2. **Tile size for training augmentation** — `128²` vs. `256²`. Pick after the
   first cache is built; depends on actual per-city raster dimensions.
3. **U-Net depth & channel widths** — start small (4 down/up blocks, channels
   `[16, 32, 64, 128, 256]` like WGAST). Tune if val L1 plateaus high.
4. **Forecast-for-target-day scalar** — include if Open-Meteo's historical-forecast
   coverage is reliable across all seven cities; otherwise drop in v1.

---

## 6. Architecture roadmap (v1 → v3)

A documented upgrade lane so the multi-head idea isn't lost:

| Version | Data scale | Model | What's new |
|---|---|---|---|
| **v1** (now) | ~700–900 supervised rasters, 7 cities | Conditional U-Net, sequence-as-channels spatial branch (5-day window of S2/LS indices + per-day cloud mask + latest-WGAST LST channel + DEM) + scalar conditioning | distillation framing, window-shift trick, DEM channel folded in, LST signal via WGAST's own previous output |
| **v2** | same data + held-out city | same model | OOD city evaluation; possibly FiLM conditioning if scalar fusion is bottlenecking |
| **v3** | ~5 k+ pairs (more years, denser S2 era, possibly more ROIs) | Adds a **ConvLSTM / temporal-attention branch** over the per-day optical stack, and a **GRU branch over hourly weather** feeding the bottleneck | temporal-aware architecture only pays off once the dataset supports it |

v1 stores hourly weather even though it isn't consumed — v3 can drop in without
re-fetching.

---

## 7. Repo layout (proposed)

```
WGAST/
├── data_download/        # (existing) satellites
├── data_preparation/     # (existing) build triples → WGAST inputs
├── model/                # (existing) WGAST.py
├── runner/               # (existing) WGAST training/eval, evaluate.py (metrics)
├── tutorials/            # (existing)
│
├── prediction/           # NEW — secondary model
│   ├── weather_fetcher.py    # Open-Meteo client (archive + historical-forecast)
│   ├── build_dataset.py      # loop in §4.3, writes paired-raster index + scalars parquet
│   ├── dataset.py            # PyTorch Dataset reading parquet + cached rasters
│   ├── model_unet.py         # conditional U-Net (§3.3)
│   ├── losses.py             # L1 + SSIM
│   ├── train.py              # full training loop
│   └── evaluate.py           # raster RMSE / MAE / Bias / PSNR / SSIM vs. baselines
└── ROADMAP.md            # this file
```

---

## 8. Status

- [x] WGAST repo understood and roadmap drafted (this file).
- [x] Region scale, ROIs, weather provider, role of forecast — decided.
- [x] Target reframed: WGAST(d0) raster (distillation framing + window-shift at inference).
- [x] Feature design (intrinsic only, no labels; LST signal via latest WGAST output) — decided.
- [x] v1 architecture (conditional U-Net + sequence-as-channels + latest-WGAST LST channel + DEM channel + scalar conditioning) — decided.
- [x] Cloud-handling strategy (N=1 filter + per-day mask channel) — decided.
- [x] Loss (L1 + 0.1·(1−SSIM)), tile augmentation, baselines — decided.
- [x] `runner/evaluate.py` — WGAST test-prediction metrics ready.
- [ ] `prediction/weather_fetcher.py` — Open-Meteo client (next code task).
- [ ] `prediction/build_dataset.py` — windowed pipeline, writes per-target-day sample with 5-day optical+weather stack.
- [ ] First end-to-end sample built for one (city, d0).
- [ ] v1 U-Net trained and reported alongside persistence-raster + weather-only baselines.
- [ ] v2: OOD city test added.

---

*Source of truth for project intent. Edit it as the plan changes — don't let it drift.*

To-do list (v1)                                                                                                                                                                                         
                                                                                                                                                                                                          
  Setup & data acquisition                  
                                                                                                                                                                
  1. Pick ≥5 cities from the seven in §4.1 (Orléans is already done; Madrid + Cairo are the highest-yield additions).                                                                                     
  2. Smoke-test the existing WGAST checkpoint on ONE new city before committing to all of them. Your model was trained on Orléans alone — run it on, say, one Madrid date and visually check the output   
  looks like a plausible LST map. If it's clearly broken, you have to either retrain WGAST multi-city or grab the paper's released weights, and that decision changes the schedule. Don't skip this —
  discovering it after 800 inferences hurts.
  3. For each city, run the existing pipeline end-to-end: data_download (01_…ipynb) → data_preparation (02_…ipynb) → data_structuring (03_…ipynb), covering 2018-01 → 2024-12. This produces the
  MODIS/Landsat/Sentinel triples per city.
  4. Run WGAST inference on every clear-sky date in every city; save each output raster to a per-city cache directory (~1600 rasters total, ~6 GB).

  Pipeline code (the new prediction/ package)
  5. Write prediction/weather_fetcher.py — Open-Meteo client (archive + historical-forecast endpoints). Smallest, most testable piece; build it first.
  6. Write prediction/build_dataset.py — for each clear-sky day d0 in the WGAST cache, assemble the 5-day window d-5..d-1 (per-day Sentinel-2 + Landsat indices with a per-day cloud mask, daily-aggregated Open-Meteo weather), look up the most recent WGAST output strictly before d0 (record its date and age_days_lst), and (optionally) the forecast for d0 issued on d-1. Apply the N=1 filter and drop samples with no prior WGAST raster at all. Write one parquet row per kept d0 pointing at the target raster, the latest-WGAST raster, and the windowed optical stack on disk.
  7. Write prediction/dataset.py — PyTorch Dataset that yields (spatial_stack, scalar_conditioning, target_raster) triples. spatial_stack = (5 × (S2+LS+mask) + latest-WGAST + DEM) channels at H×W. Random tile cropping at train-time (same coords across all spatial channels), full-raster at val/test-time.

  Model & training
  8. Write prediction/model_unet.py (conditional U-Net) and prediction/losses.py (L1 + 0.1·(1−SSIM)).
  9. Write prediction/train.py with the time-based splits from §3.7 (train 2018–2023, val 2024-H1, test 2024-H2).
  10. Train three models on identical splits:
      - v1 conditional U-Net (the real model)
      - Weather-only U-Net (same arch, blank raster input) — baseline #2
      - No training needed for persistence-raster baseline — it's a constant function.

  Evaluation & reporting
  11. Write prediction/evaluate.py — RMSE / MAE / Bias / PSNR / SSIM against the real WGAST(d0), plus skill scores vs. both baselines. For the deployment use case, also evaluate the window-shifted inference (predict d+1 from window d-4..d0) whenever a clear-sky d+1 is available for ground truth.
  12. Run on the held-out test split (Orléans 2024-H2 + the other cities' 2024-H2).

  Things easy to forget

  - LST units consistency. Check whether the cached WGAST rasters are Kelvin, Celsius, or normalised, and write training/eval in the same unit. runner/evaluate.py will tell you the range from
  gt_min/gt_max.
  - Coordinate point per city for Open-Meteo. Open-Meteo returns a time series at a point, not a polygon — fix one representative point per city (centroid of the WGAST polygon) and store it next to the
  city config.
  - Cache invalidation. If you retrain WGAST mid-project, you have to regenerate the raster cache. Worth a single-line wgast_checkpoint_hash column in the parquet so you don't accidentally mix outputs
  from two different checkpoints.
  - Cloud-mask sanity check. Once build_dataset.py runs, plot per-window statistics: how many of the 5 days have a usable optical observation? If the distribution is dominated by 1-out-of-5, the model is effectively learning from a single-day snapshot for most samples, and the multi-day window is mostly cosmetic — that's a signal to shorten the window or revisit feature engineering.
  - Tile-crop consistency. When random-cropping for training augmentation, the same (y, x) window must be applied across all temporal channels AND the DEM AND the target raster. A mismatched crop here is a silent bug that destroys training.

  Out of scope for v1 (don't do these yet)

  - OOD city test (v2)
  - GRU branch on hourly weather (v3) — but do save the hourly data now so v3 isn't blocked
  - Retraining WGAST itself, unless step 2 fails