"""
Default attribute dictionaries
"""

from meshflow import __version__


ddb_global_attrs_default = {
    'author': 'University of Calgary',
    'license': 'GNU General Public License v3 (or any later version)',
    'purpose': 'Create a drainage database .nc file for MESH',
    'featureType': 'point'
}


ddb_local_attrs_default = {
    'ChnlSlope': {
        '_FillValue': 'NaN',
        'long_name': 'Segment slope',
        'grid_mapping': 'crs',
        'coordinates': 'lon lat time'
    },
    'ChnlLength': {
        '_FillValue': 'NaN',
        'long_name': 'Segment length',
        'grid_mapping': 'crs',
        'coordinates': 'lon lat time'
    },
    'Rank': {
        '_FillValue': -1.,
        'standard_name': 'Rank',
        'long_name': 'Element ID',
        'grid_mapping': 'crs',
        'coordinates': 'lon lat time'
    },
    'Next': {
        '_FillValue': -1.,
        'standard_name': 'Next',
        'long_name': 'Receiving ID',
        'grid_mapping': 'crs',
        'coordinates': 'lon lat time'
    },
    'GridArea': {
        '_FillValue': 'NaN',
        'long_name': 'HRU Area',
        'grid_mapping': 'crs',
        'coordinates': 'lon lat time'
    },
    'GRU': {
        '_FillValue': -1.,
        'standard_name': 'GRU',
        'long_name': 'Group Response Unit',
        'coordinates': 'lon lat time'
    },
    'LandUse': {
        'standard_name': 'Landuse classification name',
    },
}


forcing_local_attrs_default = {
}


forcing_global_attrs_default = {
    'author': 'University of Calgary',
    'license': 'GNU General Public License v3 (or any later version)',
    'purpose': 'Create forcing .nc file for MESH',
    'Conventions': 'CF-1.6',
    'history': f'Created using MESHFlow version {__version__}'
}

default_attrs = {
    'lat': {
        'standard_name': 'latitude',
        'units': 'degrees_north',
        'axis': 'X',
    },
    'lon': {
        'standard_name': 'longitude',
        'units': 'degrees_east',
        'axis': 'Y',
    },
    'crs': {
        'grid_mapping_name': 'latitude_longitude',
        'longitude_of_prime_meridian': 0.,
        'semi_major_axis': 6378137.0,
        'inverse_flattening': 298.257223563,
    },
}
