"""
Default dictionaries
"""

ddb_global_attrs_default = {
    'author':'University of Calgary',
    'license':'GNU General Public License v3 (or any later version)',
    'purpose':'Create a drainage database .nc file for MESH',
    'featureType':'point'
}


ddb_local_attrs_default = {
    'ChnlSlope': {
        '_FillValue':'NaN',
        'unit':1.,
        'long_name':'Segment slope',
        'grid_mapping':'crs',
        'coordinates':'lon lat time'
    },
    'ChnlLength': {
        '_FillValue':'NaN',
        'unit':'m',
        'long_name':'Segment length',
        'grid_mapping':'crs',
        'coordinates':'lon lat time'
    },
    'Rank': {
        '_FillValue': -1.,
        'unit':1.,
        'standard_name':'Rank',
        'long_name':'Element ID',
        'grid_mapping':'crs',
        'coordinates':'lon lat time'
    },
    'Next': {
        '_FillValue': -1.,
        'unit':1.,
        'standard_name':'Next',
        'long_name':'Receiving ID',
        'grid_mapping':'crs',
        'coordinates':'lon lat time'
    },
    'GridArea': {
        '_FillValue':'NaN',
        'unit':'m^2',
        'long_name':'HRU Area',
        'grid_mapping':'crs',
        'coordinates':'lon lat time'
    },
    'GRU': {
        '_FillValue': -1.,
        'standard_name':'GRU',
        'long_name':'Group Response Unit',
        'unit':1.,
        'coordinates':'lon lat time'
    },
    'crs': {
        'grid_mapping_name': 'latitude_longitude',
        'longitude_of_prime_meridian': 0.,
        'semi_major_axis': 6378137.,
        'inverse_flattening': 298.257223563,
    },
    'lat': {
        '_FillValue':'NaN',
        'standard_name':'latitude',
        'units':'degrees_north',
        'axis': 'Y',
    },
    'lon': {
        '_FillValue':'NaN',
        'standard_name':'longitude',
        'units':'degrees_east',
        'axis': 'X',
    },
}
            