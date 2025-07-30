"""
Default dictionaries for various configurations and parameters.
"""

from meshflow import __version__

mesh_forcing_units_default = {
    "precipitation": "millimeter / second",
    "air_pressure": "pascal",
    "air_temperature": "kelvin",
    "wind_speed": "meter / second",
    "shortwave_radiation": "watt / meter ** 2",
    "longwave_radiation": "watt / meter ** 2",
    "specific_humidity": "dimensionless",
}

mesh_drainage_database_units_default = {
    "river_slope": "dimensionless",
    "river_length": "meter",
    "rank": "dimensionless",
    "next": "dimensionless",
    "gru": "dimensionless",
    "landclass": "dimensionless",
}

# Minimum values for drainage database parameters
mesh_drainage_database_minimums_default = {
    "river_slope": 1e-3,
    "river_length": 1e-3,
    "subbasin_area": 1e-3,
}

# Default settings for mesh setups
mesh_settings_default = {
     "forcing_files": "multiple",
}

# Default names for drainage database elements
mesh_drainage_database_names_default = {
    "river_slope": "ChnlSlope",
    "river_length": "ChnlLength",
    "river_class": "IAK",
    "rank": "Rank",
    "next": "Next",
    "subbasin_area": "GridArea",
    "landclass": "GRU",
    'lat': 'lat',
    'lon': 'lon',
}
