from typing import Any, Dict

parameters_local_attrs: Dict[str, Dict[str, str | float]] = {
    "lon": {
        "units": "degrees_east",
        "long_name": "longitude",
        "standard_name": "longitude",
        "axis": "X"
    },
    "lat": {
        "units": "degrees_north",
        "long_name": "latitude",
        "standard_name": "latitude",
        "axis": "Y"
    },
    "clay": {
        "units": "%",
        "long_name": "Clay Content of Soil Layer",
        "grid_mapping": "crs",
        "standard_name": "clay",
        "coordinates": "lat lon"
    },
    "sand": {
        "units": "%",
        "long_name": "Sand Content of Soil Layer",
        "grid_mapping": "crs",
        "standard_name": "sand",
        "coordinates": "lat lon"
    },
    "orgm": {
        "units": "%",
        "long_name": "Organic Matter Content of Soil Layer",
        "grid_mapping": "crs",
        "standard_name": "orgm",
        "coordinates": "lat lon"
    },
    "elevation": {
        "units": "m",
        "long_name": "Average Elevation of GRU",
        "grid_mapping": "crs",
        "standard_name": "elevation",
        "coordinates": "lat lon"
    },
    "slope": {
        "units": "degree",
        "long_name": "Average slope of GRU",
        "grid_mapping": "crs",
        "standard_name": "slope",
        "coordinates": "lat lon"
    },
    "aspect": {
        "units": "degree",
        "long_name": "Average Aspect of GRU",
        "grid_mapping": "crs",
        "standard_name": "aspect",
        "coordinates": "lat lon"
    },
    "delta": {
        "units": "m",
        "long_name": "Change in elevation between basin and of GRU",
        "grid_mapping": "crs",
        "standard_name": "delta",
        "coordinates": "lat lon"
    },
    "curvature": {
        "units": "-",
        "long_name": "Average curvature parameter of GRU",
        "grid_mapping": "crs",
        "standard_name": "curvature",
        "coordinates": "lat lon"
    },
    "skyviewfactor": {
        "units": "-",
        "long_name": "Average skyviewfactor parameter of GRU",
        "grid_mapping": "crs",
        "standard_name": "skyviewfactor",
        "coordinates": "lat lon"
    },
    "dd": {
        "units": "m m-2",
        "long_name": "Subbasin average drainage density; also dden",
        "grid_mapping": "crs",
        "standard_name": "dd",
        "coordinates": "lat lon"
    },
    "crs": {
        "grid_mapping_name": "latitude_longitude",
        "longitude_of_prime_meridian": 0,
        "semi_major_axis": 6378137,
        "inverse_flattening": 298.257223563
    },
    "pwr": {
        "units": "-",
        "long_name": "Exponent on the lower zone storage in the lower zone function; values may range from 2.0 to 3.0 when using an hourly time-step; typically calibrated",
        "standard_name": "pwr",
        "coordinates": "subbasin"
    },
    "flz": {
        "units": "-",
        "long_name": "Lower zone function; values may range from 1.0e-06 to 1.0e-04; typically calibrated",
        "standard_name": "flz",
        "coordinates": "subbasin"
    }
}
