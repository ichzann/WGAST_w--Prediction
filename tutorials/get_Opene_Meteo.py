import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 52.52,
	"longitude": 13.41,
	"start_date": "2026-04-27",
	"end_date": "2026-05-11",
	"daily": ["apparent_temperature_min", "sunrise", "daylight_duration", "sunset", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "apparent_temperature_max", "apparent_temperature_mean", "temperature_2m_min", "temperature_2m_max", "temperature_2m_mean", "weather_code", "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min", "dew_point_2m_mean", "dew_point_2m_max", "dew_point_2m_min", "et0_fao_evapotranspiration_sum", "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min", "pressure_msl_mean", "snowfall_water_equivalent_sum", "pressure_msl_max", "pressure_msl_min", "surface_pressure_max", "surface_pressure_mean", "surface_pressure_min", "winddirection_10m_dominant", "wind_gusts_10m_mean", "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min", "wet_bulb_temperature_2m_max", "wet_bulb_temperature_2m_mean", "wet_bulb_temperature_2m_min", "vapour_pressure_deficit_max", "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean", "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean", "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean", "soil_temperature_7_to_28cm_mean", "soil_temperature_28_to_100cm_mean"],
	"hourly": ["temperature_2m", "shortwave_radiation", "relative_humidity_2m", "apparent_temperature", "dew_point_2m", "precipitation", "rain", "snowfall", "snow_depth", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", "soil_temperature_28_to_100cm", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", "soil_temperature_100_to_255cm", "soil_moisture_100_to_255cm", "terrestrial_radiation", "global_tilted_irradiance", "direct_normal_irradiance", "diffuse_radiation", "terrestrial_radiation_instant", "global_tilted_irradiance_instant", "direct_normal_irradiance_instant", "diffuse_radiation_instant", "direct_radiation_instant", "shortwave_radiation_instant", "direct_radiation", "boundary_layer_height", "wet_bulb_temperature_2m", "total_column_integrated_water_vapour", "sunshine_duration", "albedo", "snow_depth_water_equivalent"],
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_shortwave_radiation = hourly.Variables(1).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(4).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(5).ValuesAsNumpy()
hourly_rain = hourly.Variables(6).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(7).ValuesAsNumpy()
hourly_snow_depth = hourly.Variables(8).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(9).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(10).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(11).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(12).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(13).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(14).ValuesAsNumpy()
hourly_et0_fao_evapotranspiration = hourly.Variables(15).ValuesAsNumpy()
hourly_vapour_pressure_deficit = hourly.Variables(16).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(17).ValuesAsNumpy()
hourly_wind_speed_100m = hourly.Variables(18).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(19).ValuesAsNumpy()
hourly_wind_direction_100m = hourly.Variables(20).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(21).ValuesAsNumpy()
hourly_soil_temperature_0_to_7cm = hourly.Variables(22).ValuesAsNumpy()
hourly_soil_temperature_7_to_28cm = hourly.Variables(23).ValuesAsNumpy()
hourly_soil_temperature_28_to_100cm = hourly.Variables(24).ValuesAsNumpy()
hourly_soil_moisture_0_to_7cm = hourly.Variables(25).ValuesAsNumpy()
hourly_soil_moisture_7_to_28cm = hourly.Variables(26).ValuesAsNumpy()
hourly_soil_moisture_28_to_100cm = hourly.Variables(27).ValuesAsNumpy()
hourly_soil_temperature_100_to_255cm = hourly.Variables(28).ValuesAsNumpy()
hourly_soil_moisture_100_to_255cm = hourly.Variables(29).ValuesAsNumpy()
hourly_terrestrial_radiation = hourly.Variables(30).ValuesAsNumpy()
hourly_global_tilted_irradiance = hourly.Variables(31).ValuesAsNumpy()
hourly_direct_normal_irradiance = hourly.Variables(32).ValuesAsNumpy()
hourly_diffuse_radiation = hourly.Variables(33).ValuesAsNumpy()
hourly_terrestrial_radiation_instant = hourly.Variables(34).ValuesAsNumpy()
hourly_global_tilted_irradiance_instant = hourly.Variables(35).ValuesAsNumpy()
hourly_direct_normal_irradiance_instant = hourly.Variables(36).ValuesAsNumpy()
hourly_diffuse_radiation_instant = hourly.Variables(37).ValuesAsNumpy()
hourly_direct_radiation_instant = hourly.Variables(38).ValuesAsNumpy()
hourly_shortwave_radiation_instant = hourly.Variables(39).ValuesAsNumpy()
hourly_direct_radiation = hourly.Variables(40).ValuesAsNumpy()
hourly_boundary_layer_height = hourly.Variables(41).ValuesAsNumpy()
hourly_wet_bulb_temperature_2m = hourly.Variables(42).ValuesAsNumpy()
hourly_total_column_integrated_water_vapour = hourly.Variables(43).ValuesAsNumpy()
hourly_sunshine_duration = hourly.Variables(44).ValuesAsNumpy()
hourly_albedo = hourly.Variables(45).ValuesAsNumpy()
hourly_snow_depth_water_equivalent = hourly.Variables(46).ValuesAsNumpy()

hourly_data = {
	"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)
}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["apparent_temperature"] = hourly_apparent_temperature
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["snow_depth"] = hourly_snow_depth
hourly_data["pressure_msl"] = hourly_pressure_msl
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["et0_fao_evapotranspiration"] = hourly_et0_fao_evapotranspiration
hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
hourly_data["soil_temperature_0_to_7cm"] = hourly_soil_temperature_0_to_7cm
hourly_data["soil_temperature_7_to_28cm"] = hourly_soil_temperature_7_to_28cm
hourly_data["soil_temperature_28_to_100cm"] = hourly_soil_temperature_28_to_100cm
hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm
hourly_data["soil_moisture_7_to_28cm"] = hourly_soil_moisture_7_to_28cm
hourly_data["soil_moisture_28_to_100cm"] = hourly_soil_moisture_28_to_100cm
hourly_data["soil_temperature_100_to_255cm"] = hourly_soil_temperature_100_to_255cm
hourly_data["soil_moisture_100_to_255cm"] = hourly_soil_moisture_100_to_255cm
hourly_data["terrestrial_radiation"] = hourly_terrestrial_radiation
hourly_data["global_tilted_irradiance"] = hourly_global_tilted_irradiance
hourly_data["direct_normal_irradiance"] = hourly_direct_normal_irradiance
hourly_data["diffuse_radiation"] = hourly_diffuse_radiation
hourly_data["terrestrial_radiation_instant"] = hourly_terrestrial_radiation_instant
hourly_data["global_tilted_irradiance_instant"] = hourly_global_tilted_irradiance_instant
hourly_data["direct_normal_irradiance_instant"] = hourly_direct_normal_irradiance_instant
hourly_data["diffuse_radiation_instant"] = hourly_diffuse_radiation_instant
hourly_data["direct_radiation_instant"] = hourly_direct_radiation_instant
hourly_data["shortwave_radiation_instant"] = hourly_shortwave_radiation_instant
hourly_data["direct_radiation"] = hourly_direct_radiation
hourly_data["boundary_layer_height"] = hourly_boundary_layer_height
hourly_data["wet_bulb_temperature_2m"] = hourly_wet_bulb_temperature_2m
hourly_data["total_column_integrated_water_vapour"] = hourly_total_column_integrated_water_vapour
hourly_data["sunshine_duration"] = hourly_sunshine_duration
hourly_data["albedo"] = hourly_albedo
hourly_data["snow_depth_water_equivalent"] = hourly_snow_depth_water_equivalent

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_apparent_temperature_min = daily.Variables(0).ValuesAsNumpy()
daily_sunrise = daily.Variables(1).ValuesInt64AsNumpy()
daily_daylight_duration = daily.Variables(2).ValuesAsNumpy()
daily_sunset = daily.Variables(3).ValuesInt64AsNumpy()
daily_sunshine_duration = daily.Variables(4).ValuesAsNumpy()
daily_precipitation_sum = daily.Variables(5).ValuesAsNumpy()
daily_rain_sum = daily.Variables(6).ValuesAsNumpy()
daily_snowfall_sum = daily.Variables(7).ValuesAsNumpy()
daily_precipitation_hours = daily.Variables(8).ValuesAsNumpy()
daily_wind_speed_10m_max = daily.Variables(9).ValuesAsNumpy()
daily_wind_gusts_10m_max = daily.Variables(10).ValuesAsNumpy()
daily_wind_direction_10m_dominant = daily.Variables(11).ValuesAsNumpy()
daily_shortwave_radiation_sum = daily.Variables(12).ValuesAsNumpy()
daily_et0_fao_evapotranspiration = daily.Variables(13).ValuesAsNumpy()
daily_apparent_temperature_max = daily.Variables(14).ValuesAsNumpy()
daily_apparent_temperature_mean = daily.Variables(15).ValuesAsNumpy()
daily_temperature_2m_min = daily.Variables(16).ValuesAsNumpy()
daily_temperature_2m_max = daily.Variables(17).ValuesAsNumpy()
daily_temperature_2m_mean = daily.Variables(18).ValuesAsNumpy()
daily_weather_code = daily.Variables(19).ValuesAsNumpy()
daily_cloud_cover_mean = daily.Variables(20).ValuesAsNumpy()
daily_cloud_cover_max = daily.Variables(21).ValuesAsNumpy()
daily_cloud_cover_min = daily.Variables(22).ValuesAsNumpy()
daily_dew_point_2m_mean = daily.Variables(23).ValuesAsNumpy()
daily_dew_point_2m_max = daily.Variables(24).ValuesAsNumpy()
daily_dew_point_2m_min = daily.Variables(25).ValuesAsNumpy()
daily_et0_fao_evapotranspiration_sum = daily.Variables(26).ValuesAsNumpy()
daily_relative_humidity_2m_mean = daily.Variables(27).ValuesAsNumpy()
daily_relative_humidity_2m_max = daily.Variables(28).ValuesAsNumpy()
daily_relative_humidity_2m_min = daily.Variables(29).ValuesAsNumpy()
daily_pressure_msl_mean = daily.Variables(30).ValuesAsNumpy()
daily_snowfall_water_equivalent_sum = daily.Variables(31).ValuesAsNumpy()
daily_pressure_msl_max = daily.Variables(32).ValuesAsNumpy()
daily_pressure_msl_min = daily.Variables(33).ValuesAsNumpy()
daily_surface_pressure_max = daily.Variables(34).ValuesAsNumpy()
daily_surface_pressure_mean = daily.Variables(35).ValuesAsNumpy()
daily_surface_pressure_min = daily.Variables(36).ValuesAsNumpy()
daily_winddirection_10m_dominant = daily.Variables(37).ValuesAsNumpy()
daily_wind_gusts_10m_mean = daily.Variables(38).ValuesAsNumpy()
daily_wind_speed_10m_mean = daily.Variables(39).ValuesAsNumpy()
daily_wind_gusts_10m_min = daily.Variables(40).ValuesAsNumpy()
daily_wind_speed_10m_min = daily.Variables(41).ValuesAsNumpy()
daily_wet_bulb_temperature_2m_max = daily.Variables(42).ValuesAsNumpy()
daily_wet_bulb_temperature_2m_mean = daily.Variables(43).ValuesAsNumpy()
daily_wet_bulb_temperature_2m_min = daily.Variables(44).ValuesAsNumpy()
daily_vapour_pressure_deficit_max = daily.Variables(45).ValuesAsNumpy()
daily_soil_moisture_0_to_100cm_mean = daily.Variables(46).ValuesAsNumpy()
daily_soil_moisture_0_to_7cm_mean = daily.Variables(47).ValuesAsNumpy()
daily_soil_moisture_28_to_100cm_mean = daily.Variables(48).ValuesAsNumpy()
daily_soil_moisture_7_to_28cm_mean = daily.Variables(49).ValuesAsNumpy()
daily_soil_temperature_0_to_100cm_mean = daily.Variables(50).ValuesAsNumpy()
daily_soil_temperature_0_to_7cm_mean = daily.Variables(51).ValuesAsNumpy()
daily_soil_temperature_7_to_28cm_mean = daily.Variables(52).ValuesAsNumpy()
daily_soil_temperature_28_to_100cm_mean = daily.Variables(53).ValuesAsNumpy()

daily_data = {
	"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)
}

daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
daily_data["sunrise"] = daily_sunrise
daily_data["daylight_duration"] = daily_daylight_duration
daily_data["sunset"] = daily_sunset
daily_data["sunshine_duration"] = daily_sunshine_duration
daily_data["precipitation_sum"] = daily_precipitation_sum
daily_data["rain_sum"] = daily_rain_sum
daily_data["snowfall_sum"] = daily_snowfall_sum
daily_data["precipitation_hours"] = daily_precipitation_hours
daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration
daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
daily_data["temperature_2m_min"] = daily_temperature_2m_min
daily_data["temperature_2m_max"] = daily_temperature_2m_max
daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
daily_data["weather_code"] = daily_weather_code
daily_data["cloud_cover_mean"] = daily_cloud_cover_mean
daily_data["cloud_cover_max"] = daily_cloud_cover_max
daily_data["cloud_cover_min"] = daily_cloud_cover_min
daily_data["dew_point_2m_mean"] = daily_dew_point_2m_mean
daily_data["dew_point_2m_max"] = daily_dew_point_2m_max
daily_data["dew_point_2m_min"] = daily_dew_point_2m_min
daily_data["et0_fao_evapotranspiration_sum"] = daily_et0_fao_evapotranspiration_sum
daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean
daily_data["relative_humidity_2m_max"] = daily_relative_humidity_2m_max
daily_data["relative_humidity_2m_min"] = daily_relative_humidity_2m_min
daily_data["pressure_msl_mean"] = daily_pressure_msl_mean
daily_data["snowfall_water_equivalent_sum"] = daily_snowfall_water_equivalent_sum
daily_data["pressure_msl_max"] = daily_pressure_msl_max
daily_data["pressure_msl_min"] = daily_pressure_msl_min
daily_data["surface_pressure_max"] = daily_surface_pressure_max
daily_data["surface_pressure_mean"] = daily_surface_pressure_mean
daily_data["surface_pressure_min"] = daily_surface_pressure_min
daily_data["winddirection_10m_dominant"] = daily_winddirection_10m_dominant
daily_data["wind_gusts_10m_mean"] = daily_wind_gusts_10m_mean
daily_data["wind_speed_10m_mean"] = daily_wind_speed_10m_mean
daily_data["wind_gusts_10m_min"] = daily_wind_gusts_10m_min
daily_data["wind_speed_10m_min"] = daily_wind_speed_10m_min
daily_data["wet_bulb_temperature_2m_max"] = daily_wet_bulb_temperature_2m_max
daily_data["wet_bulb_temperature_2m_mean"] = daily_wet_bulb_temperature_2m_mean
daily_data["wet_bulb_temperature_2m_min"] = daily_wet_bulb_temperature_2m_min
daily_data["vapour_pressure_deficit_max"] = daily_vapour_pressure_deficit_max
daily_data["soil_moisture_0_to_100cm_mean"] = daily_soil_moisture_0_to_100cm_mean
daily_data["soil_moisture_0_to_7cm_mean"] = daily_soil_moisture_0_to_7cm_mean
daily_data["soil_moisture_28_to_100cm_mean"] = daily_soil_moisture_28_to_100cm_mean
daily_data["soil_moisture_7_to_28cm_mean"] = daily_soil_moisture_7_to_28cm_mean
daily_data["soil_temperature_0_to_100cm_mean"] = daily_soil_temperature_0_to_100cm_mean
daily_data["soil_temperature_0_to_7cm_mean"] = daily_soil_temperature_0_to_7cm_mean
daily_data["soil_temperature_7_to_28cm_mean"] = daily_soil_temperature_7_to_28cm_mean
daily_data["soil_temperature_28_to_100cm_mean"] = daily_soil_temperature_28_to_100cm_mean

daily_dataframe = pd.DataFrame(data = daily_data)
print("\nDaily data\n", daily_dataframe)
