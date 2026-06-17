import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry



# ======================================================================
# ======================================================================
#Claude
# ======================================================================
# ======================================================================


# Default variables — picked for LST relevance, see ROADMAP §3.1.
DEFAULT_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "dew_point_2m_mean",
    "relative_humidity_2m_mean",
    "shortwave_radiation_sum",     # primary LST driver
    "cloud_cover_mean",
    "cloud_cover_max",             # diurnal cloud structure: overcast vs. partly cloudy
    "cloud_cover_min",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "et0_fao_evapotranspiration",  # latent-heat-flux proxy
    "surface_pressure_mean",
    "sunshine_duration",
    "soil_moisture_0_to_7cm_mean",     # Bowen-ratio control: integrates dry-down, unlike precip
    "soil_temperature_0_to_7cm_mean",  # ~9 km LST prior + ground-heat-flux memory
    "snowfall_sum",                    # snow albedo decouples LST from radiation
]

# Target-day (d0 at train / d+1 at inference) forecast block. Same list minus the
# soil vars: forecast models use different soil layering than ERA5, and soil state
# is slow enough that the d-1 archive value covers it.
TARGET_DAY_VARS = [v for v in DEFAULT_DAILY_VARS
                   if not v.startswith("soil_")]

DEFAULT_HOURLY_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
]


def _block_to_df(block, var_names, utc_offset_seconds=0):
    """Turn an Open-Meteo Hourly/Daily block into a DataFrame indexed by *local*
    (timezone-naive) time. block.Time() is UTC epoch seconds; with timezone="auto"
    the daily rows describe local days, so we shift by the location's UTC offset —
    otherwise every daily row lands on the previous UTC day for positive-offset
    cities and per-day lookups are off by one (ROADMAP §10 C2)."""
    index = pd.date_range(
        start=pd.to_datetime(block.Time() + utc_offset_seconds, unit="s"),
        end=pd.to_datetime(block.TimeEnd() + utc_offset_seconds, unit="s"),
        freq=pd.Timedelta(seconds=block.Interval()),
        inclusive="left",
    )
    data = {name: block.Variables(i).ValuesAsNumpy() for i, name in enumerate(var_names)}
    return pd.DataFrame(data, index=index).rename_axis("date")


def _client(cache_dir):
    cache_session = requests_cache.CachedSession(cache_dir, expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _params(latitude, longitude, start_date, end_date):
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }


def fetch_weather(
    latitude,
    longitude,
    start_date,
    end_date,
    daily_vars=None,
    hourly_vars=None,
    cache_dir=".cache/openmeteo",
):
    """Fetch Open-Meteo Archive (ERA5) data for one point.

    Returns
    -------
    (daily_df, hourly_df) : tuple of pd.DataFrame
        Both indexed by local (timezone-naive) datetime, one column per variable.
        Daily rows land on local midnight of the day they describe.
    """
    daily_vars = daily_vars if daily_vars is not None else DEFAULT_DAILY_VARS
    hourly_vars = hourly_vars if hourly_vars is not None else DEFAULT_HOURLY_VARS

    params = _params(latitude, longitude, start_date, end_date)
    params["daily"] = daily_vars
    params["hourly"] = hourly_vars

    url = "https://archive-api.open-meteo.com/v1/archive"
    response = _client(cache_dir).weather_api(url, params=params)[0]
    offset = response.UtcOffsetSeconds()
    daily_df = _block_to_df(response.Daily(), daily_vars, offset)
    hourly_df = _block_to_df(response.Hourly(), hourly_vars, offset)
    return daily_df, hourly_df


def fetch_target_weather(
    latitude,
    longitude,
    start_date,
    end_date,
    daily_vars=None,
    live=False,
    cache_dir=".cache/openmeteo",
):
    """Fetch the target-day forecast block (ROADMAP §3.1) for one point.

    Training uses the Historical Forecast API (archived NWP output, from 2021),
    NOT the ERA5 archive, so the model sees the same kind of input it will get
    at inference, where `live=True` hits the live Forecast API for d+1.

    Returns a daily DataFrame indexed by local (timezone-naive) midnight.
    """
    daily_vars = daily_vars if daily_vars is not None else TARGET_DAY_VARS

    params = _params(latitude, longitude, start_date, end_date)
    params["daily"] = daily_vars

    url = ("https://api.open-meteo.com/v1/forecast" if live
           else "https://historical-forecast-api.open-meteo.com/v1/forecast")
    response = _client(cache_dir).weather_api(url, params=params)[0]
    offset = response.UtcOffsetSeconds()
    return _block_to_df(response.Daily(), daily_vars, offset)
