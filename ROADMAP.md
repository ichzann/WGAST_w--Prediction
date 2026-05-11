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

Concretely, the secondary model takes today's WGAST raster (plus weather state and
the official forecast) and predicts the **WGAST raster for day d+1**:

```text
WGAST(d), weather(d), forecast(d+1)  ──►  predicted WGAST raster for day d+1
```

The output is a 10 m LST map at the same resolution and shape as WGAST itself.
Crucially, the model works **on cloudy d+1 too** — it does not require satellite
imagery for tomorrow. That's what makes the system useful: WGAST stops at day-0, the
secondary model extends one day forward.

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

For the secondary model we treat WGAST as a **black box that is correct**: WGAST(d)
is both an **input feature** (today's raster) and, on supervised pairs, **the ground
truth** for what WGAST(d+1) would have been. We deliberately do **not** question
per-city WGAST quality in v1 (see §4.2).

---

## 3. The secondary model

### 3.1 Inputs (intrinsic state only — no identity, no location)

Each training sample = `(region, date d)` where day d+1 happens to be a clear-sky
day (so we have a real WGAST(d+1) as the supervisory target).

- **WGAST raster on day d** (the previous available WGAST output if d itself is cloudy
  — in v1 we use the most recent clear-sky raster ≤ d, plus an `age_days` scalar).
  Shape: `1 × H × W`.
- **Observed weather on day d** (morning of d, latest fully-observed window): air
  temperature, dew point, relative humidity, wind speed, surface pressure, cloud
  cover, precipitation in the past 24 h. Source: Open-Meteo Archive.
- **Forecast for day d+1** (issued at 00 UTC of day d): forecast T_max, forecast
  cloud cover, forecast wind, forecast humidity. Source: Open-Meteo historical-
  forecast endpoint.
- **Seasonal phase**: `sin(2π·doy/365)`, `cos(2π·doy/365)`.
- **`age_days`**: how many days old the input WGAST raster is (0 if d itself was clear).
- **`elevation`** (single scalar per city, in metres at the representative point):
  included because the dry/moist adiabatic lapse rate (~6–10 °C/km) imposes a real
  physical offset on LST. Without it, the model would have to learn weird
  compensations from humidity/cloud features to fake the mean-temperature offset
  between low- and high-elevation cities. **This is the only "static city-level"
  feature in v1 — it earns its place by being mechanism-bearing, not identity-bearing.**

**Explicitly excluded** (so generalisation is by design, not by label):

- ❌ city_id
- ❌ climate zone
- ❌ latitude / longitude

All weather/forecast features come from **Open-Meteo** to keep observations, forecasts,
and any downstream sanity-checks on the same provider.

### 3.2 Target

The **real WGAST raster on day d+1**:

```text
y = WGAST(MODIS_{d+1}, Landsat_{t0}, Sentinel_{d+1}, MODIS_{t0})  ∈  ℝ^{H × W}
```

Available only for clear-sky d+1. That's the supervision constraint that bounds
sample count — each clear-sky day in our pipeline becomes one training row.

### 3.3 v1 architecture — conditional U-Net

```text
  WGAST(d) raster  ──►  CNN encoder ──┬──── bottleneck embedding ──┐
   (1 × H × W)         (with skips     │                            │
                        to decoder)    └─── skip features ──┐       │
                                                            │       ▼
                                                            │   concat with conditioning
  weather + forecast + sin/cos(doy)                         │       │
        + age_days  (~12 scalars)                           │       ▼
                ▼                                           │   CNN decoder
            small MLP  ──►  conditioning embedding ─────────┴────► (uses skips)
                            (~16-dim)                               │
                                                                    ▼
                                                              predicted WGAST(d+1)
                                                                (1 × H × W)
```

Design choices, all locked for v1:

| Decision | Choice |
|---|---|
| Encoder/decoder family | **U-Net** with skip connections — fine spatial detail matters for heat-island spotting |
| How scalar features enter | **Concat at the bottleneck** (simplest; FiLM is a v2 option only if needed) |
| Loss | `L1 + λ·(1 − SSIM)`, **λ ≈ 0.1** (the WGAST paper's regime) |
| Optimiser | AdamW + cosine schedule, early stopping on val L1 |
| Capacity | ~0.5–2 M params (between a small U-Net and WGAST itself); trainable on a single GPU |
| Temporal / GRU branch on hourly weather | **Deferred to v2.** v1 = one image branch + one scalar branch only |

### 3.4 Tile-based training augmentation

At training time, crop random `K × K` tiles (e.g. `128 × 128` or `256 × 256`) from
the paired `(input_raster, target_raster)`. Same crop coordinates on both sides;
all scalar conditioning is replicated. This gives the encoder more spatial variation
per epoch, regularises a small U-Net, and side-steps any per-city size mismatch.

At inference / evaluation, run on the **full raster** (or tile + stitch with overlap,
which is exactly what `runner/experiment.py:test()` already does for WGAST).

### 3.5 Baselines that must always be reported

The model earns its place by beating these:

1. **Persistence-raster**: `predicted(d+1) = WGAST(d)`. The "no model" floor. Trivially
   strong on stable weather.
2. **Weather-only U-Net**: identical architecture, but the raster input is replaced
   with a blank/constant tensor. Isolates exactly what the past WGAST raster contributes
   on top of weather + forecast + season alone.

The metric of interest is skill vs. baseline #1 (does the model do anything?) and
skill vs. baseline #2 (does the past raster contribute, or is it all in the weather?).

### 3.6 Evaluation metrics

Same metrics the WGAST paper reports against Landsat — applied here against the real
WGAST(d+1) raster:

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

Total expected supervised pairs over 2018-01 → 2024-12: **~700–900 rows**.
Effective supervised signal is much higher — each row is a ~1 M-pixel raster, so the
gradient information is comparable to ~10⁸ scalar supervised examples, which is plenty
for a U-Net at this capacity.

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
    for year y in 2018..2024:
        triples = data_download.find_common_dates(R, y)      # existing cloud filtering
        for d_plus_1 in triples:                             # supervised target dates
            d         = most_recent_available_date_before(d_plus_1)
            raster_in = WGAST(at d) [cached]                 # input raster
            raster_y  = WGAST(at d_plus_1) [cached]          # supervisory raster
            x_obs     = open_meteo.archive(R.point, d)
            x_fcst    = open_meteo.forecast_archive(R.point, d, lead=1)
            age       = (d_plus_1 - d).days
            write_pair(R, d, d_plus_1, raster_in, raster_y, x_obs, x_fcst, age)
```

WGAST runs **offline once** per `(R, date)` and writes a 10 m TIF to a cache
directory, reused across re-trains of the secondary model. Disk cost: ~4 MB per
raster × ~1 600 rasters (input + target) ≈ **~6 GB total**.

**Note on storing hourly weather:** even though v1 only consumes daily-aggregated
weather, the pipeline saves **hourly** observations alongside. This is so v2's
optional GRU branch can be added without re-fetching from Open-Meteo.

---

## 5. Decisions made + remaining open questions

### 5.1 Decided

- **Target = WGAST raster on day d+1** (not a scalar; not air temperature).
- **Single horizon: h = +1 day.**
- **Use case: day-ahead urban-heat-island warning** at 10 m resolution.
- **Region scale = WGAST training scale** (city-sized polygon).
- **Cities = seven paper-validated ROIs** (§4.1).
- **WGAST is treated as ground truth across all cities for v1** (§4.2).
- **No identity / location labels in features** — city_id, climate_zone, lat, lon
  excluded by design. **Elevation IS included** as a physical scalar (it carries
  adiabatic lapse-rate information that the model would otherwise have to fake
  from indirect features).
- **Weather / forecast / observations source = Open-Meteo** (Archive + historical-
  forecast endpoints).
- **v1 architecture = conditional U-Net** (CNN encoder with skips, scalar conditioning
  concat at bottleneck, CNN decoder with skips), ~0.5–2 M params.
- **Loss = L1 + λ·(1 − SSIM), λ ≈ 0.1.**
- **GRU / temporal branch deferred to v2.**
- **Tile-based training augmentation** (random crops of paired rasters), full raster
  at inference.
- **Baselines** = persistence-raster + weather-only U-Net.
- **Metrics** = RMSE / MAE / Bias / PSNR / SSIM on the raster, computed against the
  real WGAST(d+1). Optional summary-stat sanity check vs. observed T_max.
- **Splits = time-based** (IID test on last months); OOD city in v2.

### 5.2 Still open

1. **Tile size for training augmentation** — `128²` vs. `256²`. Pick after the first
   raster cache is built; depends on actual per-city raster dimensions.
2. **Handling of input-raster age (`age_days`)** — encode as a scalar feature
   (current plan) vs. drop samples where `age_days > N` (e.g. > 14). Decide
   empirically after seeing the age distribution in the assembled dataset.
3. **U-Net depth & channel widths** — start small (4 down/up blocks, channels
   `[16, 32, 64, 128, 256]` like WGAST). Tune if val L1 plateaus high.

---

## 6. Architecture roadmap (v1 → v3)

A documented upgrade lane so the multi-head idea isn't lost:

| Version | Data scale | Model | What's new |
|---|---|---|---|
| **v1** (now) | ~700–900 paired rasters, 7 cities | Conditional U-Net (one image branch + scalar conditioning) | learned representation, no identity labels |
| **v2** | same data + held-out city | same model, plus an optional per-pixel **DEM channel** fed to the CNN encoder | OOD city evaluation; DEM channel captures *intra-city* elevation effects on top of v1's city-level elevation scalar (Istanbul hills, Rome's seven hills, etc.) |
| **v3** | ~5 k+ pairs (more years, denser S2 era, possibly more ROIs) | Adds a **GRU branch over hourly weather** (the temporal head from earlier discussion) feeding the bottleneck, plus optionally FiLM conditioning throughout the decoder | the temporal head only pays off once the dataset supports it |

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
- [x] Target reframed: WGAST(d+1) raster, not scalar T_max.
- [x] Feature design (intrinsic only, no labels) — decided.
- [x] v1 architecture (conditional U-Net + concat-at-bottleneck) — decided.
- [x] Loss (L1 + 0.1·(1−SSIM)), tile augmentation, baselines — decided.
- [x] `runner/evaluate.py` — WGAST test-prediction metrics ready.
- [ ] `prediction/weather_fetcher.py` — Open-Meteo client (next code task).
- [ ] `prediction/build_dataset.py` — full pipeline, writes paired-raster dataset.
- [ ] First end-to-end pair built for one (city, day).
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
  6. Write prediction/build_dataset.py — for each clear-sky day d+1 in the WGAST cache, find the most recent prior raster (age_days), fetch weather on d, fetch forecast for d+1, write one row to a
  parquet index that points to the two raster files on disk.
  7. Write prediction/dataset.py — PyTorch Dataset that yields (input_raster, scalar_conditioning, target_raster) triples, with random tile cropping at train-time and full-raster at val/test-time.

  Model & training
  8. Write prediction/model_unet.py (conditional U-Net) and prediction/losses.py (L1 + 0.1·(1−SSIM)).
  9. Write prediction/train.py with the time-based splits from §3.7 (train 2018–2023, val 2024-H1, test 2024-H2).
  10. Train three models on identical splits:
      - v1 conditional U-Net (the real model)
      - Weather-only U-Net (same arch, blank raster input) — baseline #2
      - No training needed for persistence-raster baseline — it's a constant function.

  Evaluation & reporting
  11. Write prediction/evaluate.py — RMSE / MAE / Bias / PSNR / SSIM against the real WGAST(d+1), plus skill scores vs. both baselines.
  12. Run on the held-out test split (Orléans 2024-H2 + the other cities' 2024-H2).

  Things easy to forget

  - LST units consistency. Check whether the cached WGAST rasters are Kelvin, Celsius, or normalised, and write training/eval in the same unit. runner/evaluate.py will tell you the range from
  gt_min/gt_max.
  - Coordinate point per city for Open-Meteo. Open-Meteo returns a time series at a point, not a polygon — fix one representative point per city (centroid of the WGAST polygon) and store it next to the
  city config.
  - Cache invalidation. If you retrain WGAST mid-project, you have to regenerate the raster cache. Worth a single-line wgast_checkpoint_hash column in the parquet so you don't accidentally mix outputs
  from two different checkpoints.
  - age_days distribution. Once build_dataset.py runs, plot the histogram of age_days before training. If it's bimodal (e.g. lots of pairs with age = 1 from dry climates, lots with age = 21 from wet
  ones), you may want to cap it (drop samples with age > N) so the model doesn't learn that "very old input = signal lost".

  Out of scope for v1 (don't do these yet)

  - OOD city test (v2)
  - GRU branch on hourly weather (v3) — but do save the hourly data now so v3 isn't blocked
  - Retraining WGAST itself, unless step 2 fails