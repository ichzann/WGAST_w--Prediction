# cities.py
#
# Single source of truth for the per-city AOIs (ROADMAP §4.1) and for the
# final-run experiment design (ROADMAP §3.7): which city belongs to which
# split, and which date ranges to download/build. Every notebook/script does:
#
#     from cities import CITIES, EXPERIMENT, DATA_ROOT
#     city = "Istanbul"            # the ONE switch
#     roi  = CITIES[city]["roi"]   # [west, south, east, north]
#
# All boxes share the same degree spans as the original Orléans AOI
# (~0.158° lon × ~0.106° lat). Keys are capitalised; use city.lower()
# for cache/sample directory names.

import os
from pathlib import Path

# Override with WGAST_DATA_ROOT to point at an external disk / mounted drive
# (e.g. on a training PC). Default: tutorials/data/secondary next to this file.
DATA_ROOT = Path(os.environ.get(
    "WGAST_DATA_ROOT",
    Path(__file__).resolve().parent / "data" / "secondary",
))

# Final-run design: splits are CITY-based (cross-city generalisation is the
# point — and Orléans is WGAST's own training city, so it must not be used
# for val/test). Date ranges are inclusive; a list means disjoint chunks
# (run the 01s/02s download loop once per chunk).
#   train: Istanbul + Orléans, full year 2022
#   val:   Rome,  Apr + Sep 2022 (mixed season, drives checkpoint selection)
#   test:  Cairo, Apr + Sep 2022 (held out until the very end)
EXPERIMENT = {
    "Istanbul": {"split": "train", "ranges": [("2022-01-01", "2022-12-31")]},
    "Orleans":  {"split": "train", "ranges": [("2022-01-01", "2022-12-31")]},
    # Val/test chunks are 2-month windows: a t0 needs Landsat AND Sentinel-2
    # clear on the SAME day (16-day vs 5-day cycles -> rare), and t0 may precede
    # a d0 by up to max_gap_days=50 — single months found no t0 at all.
    "Rome":     {"split": "val",   "ranges": [("2022-03-01", "2022-04-30"),
                                              ("2022-08-01", "2022-09-30")]},
    "Cairo":    {"split": "test",  "ranges": [("2022-03-01", "2022-04-30"),
                                              ("2022-08-01", "2022-09-30")]},
}


def split_of(city):
    """Split for a city (capitalised or lowercase key). KeyError if the city
    is not part of the experiment — better loud than silently mis-split."""
    return EXPERIMENT[city.capitalize()]["split"]

CITIES = {
    "Orleans": {
        "roi": [1.8352731328559473, 47.844911277451844, 1.99343366332974, 47.95130583226066],
    },
    "Tours": {  # derived from city centre with the standard box — verify AOI before first download
        "roi": [0.60572, 47.34090, 0.76388, 47.44730],
    },
    "Montpellier": {  # derived from city centre with the standard box — verify AOI before first download
        "roi": [3.79812, 43.55770, 3.95628, 43.66410],
    },
    "Madrid": {
        "roi": [-3.7828802652368965, 40.36360272259559, -3.6247197347631035, 40.46999727740441],
    },
    "Rome": {
        "roi": [12.413119734763104, 41.83700272259559, 12.571280265236896, 41.94339727740441],
    },
    "Istanbul": {
        "roi": [28.963878734763104, 40.99138522259559, 29.122039265236896, 41.09777977740441],
    },
    "Cairo": {
        "roi": [31.156619734763103, 29.991202722595592, 31.314780265236896, 30.09759727740441],
    },
}


def centroid(roi):
    """(lat, lon) representative point for Open-Meteo."""
    return (roi[1] + roi[3]) / 2, (roi[0] + roi[2]) / 2
