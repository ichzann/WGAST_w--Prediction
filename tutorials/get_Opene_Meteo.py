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
    "wind_speed_10m_mean",
    "precipitation_sum",
    "et0_fao_evapotranspiration",  # latent-heat-flux proxy
    "surface_pressure_mean",
    "sunshine_duration",
]

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


def _block_to_df(block, var_names):
    """Turn an Open-Meteo Hourly/Daily block into a DataFrame indexed by UTC time."""
    index = pd.date_range(
        start=pd.to_datetime(block.Time(), unit="s", utc=True),
        end=pd.to_datetime(block.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=block.Interval()),
        inclusive="left",
    )
    data = {name: block.Variables(i).ValuesAsNumpy() for i, name in enumerate(var_names)}
    return pd.DataFrame(data, index=index).rename_axis("date")


def fetch_weather(
    latitude,
    longitude,
    start_date,
    end_date,
    daily_vars=None,
    hourly_vars=None,
    cache_dir=".cache/openmeteo",
):
    """Fetch Open-Meteo Archive data for one point.

    Returns
    -------
    (daily_df, hourly_df) : tuple of pd.DataFrame
        Both indexed by UTC datetime, one column per variable.
    """
    daily_vars = daily_vars if daily_vars is not None else DEFAULT_DAILY_VARS
    hourly_vars = hourly_vars if hourly_vars is not None else DEFAULT_HOURLY_VARS

    cache_session = requests_cache.CachedSession(cache_dir, expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": daily_vars,
        "hourly": hourly_vars,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }

    response = openmeteo.weather_api(url, params=params)[0]
    daily_df = _block_to_df(response.Daily(), daily_vars)
    hourly_df = _block_to_df(response.Hourly(), hourly_vars)
    return daily_df, hourly_df
