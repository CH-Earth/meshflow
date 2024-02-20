#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary libraries
import meshflow # version v0.1.0-dev1

import os # python 3.10.2


# In[ ]:


# main work path - modify
work_path = '/path/to/your/experiments/'

# using meshflow==v0.1.0-dev1
# modify the following to match your settings
config = {
    'riv': os.path.join(work_path, 'your_rivers_file.shp'),
    'cat': os.path.join(work_path, 'your_catchments_file.shp'),
    'landcover': os.path.join(work_path, 'your_landcover_fractions_file.csv'),
    'forcing_files': os.path.join(work_path, 'easymore-outputs'),
    'forcing_vars': [ # MESH usuall needs 7 variables, list them below
        "var1",
        "var2",
        "var3",
        "var4",
        "var5",
        "var6",
        "var7",
    ],
    'forcing_units': { # Enter original units for each variable listed under 'forcing_vars'
        "var1": 'millibar',
        "var2": 'kg/kg',
        "var3": 'celsius',
        "var4": 'knot',
        "var5": 'm/hr',
        "var6": 'W/m^2',
        "var7": 'W/m^2',
    },
    'forcing_to_units': { # And here, the units that MESH needs to read
         "var1": 'm/s',
         "var2": 'W/m^2',
         "var3": 'W/m^2',
         "var4": 'mm/s',
         "var5": 'pascal',
         "var6": 'kelvin',
         "var7": 'kg/kg',
    },
    'main_id': 'main_riv_id', # what is the main ID of each river segment? Column name in the `cat` object
    'ds_main_id': 'ds_main_id', # what is the downstream segment ID for each river segment? ditto.
    'landcover_classes': { # these are the classes defined for NALCMS-Landsat 2015 dataset. Is this accurate?
        0: 'Class A', # "key" values are foudn in `landcover` object
        1: 'Class B',
        2: 'Class C',
        3: 'Class D',
        4: 'Class E',
        5: 'Class F',
    },
    'ddb_vars': { # drainage database variables that MESH needs
        # FIXME: in later versions, the positions of keys and values below will be switched
        'seg_slope': 'ChnlSlope',
        'Shape_Leng': 'ChnlLength',
        'Rank': 'Rank', # Rank variable - for WATROUTE routing
        'Next': 'Next', # Next variable - for WATROUTE routing
        'landcover': 'GRU', # GRU fractions variable
        'Shape_Area': 'GridArea', # Sub-basin area variable
        'landcover_names': 'LandUse', # LandUse names
    },
    'ddb_units': { # units of variables in the drainage database
        'ChnlSlope': 'm/m',
        'ChnlLength': 'm',
        'Rank': 'dimensionless',
        'Next': 'dimensionless',
        'GRU': 'dimensionless',
        'GridArea': 'm^2',
        'LandUse': 'dimensionless',
    },
    'ddb_to_units': { # units of variables in the drainage database the MESH needs
        'ChnlSlope': 'm/m',
        'ChnlLength': 'm',
        'Rank': 'dimensionless',
        'Next': 'dimensionless',
        'GRU': 'dimensionless',
        'GridArea': 'm^2',
        'LandUse': 'dimensionless',
    },
    'ddb_min_values': { # minimum values in the drainage database
        'ChnlSlope': 1e-10, # in case there are 0s in the `rivers` Shapefile, we need minimums for certain variables
        'ChnlLength': 1e-3,
        'GridArea': 1e-3,
    },
    'gru_dim': 'NGRU', # change to `NGRU` for 'MESH>=r1860', keep for 'MESH<=1860', for example for r1813.
    'hru_dim': 'subbasin', # consistent in various versions, no need to change
    'outlet_value': 0, # modify depending on the outlet values specific in `ds_main_id` object
}


# In[ ]:


exp1 = meshflow.MESHWorkflow(**config)


# In[ ]:


exp1.run()


# In[ ]:


exp1.forcing


# In[ ]:


exp1.ddb


# In[ ]:


# create a directory for MESH setup
os.makedirs('/path/to/your/experiments/MESH-model')

# saving drainage database and forcing files
exp1.save('/path/to/your/experiments/MESH-model')


# ____

# If there is any issue, open a ticket on [MESHFlow's GitHub Webpage](https://github.com/kasra-keshavarz/meshflow/issues). Once this Notebook runs, then have the setting files along with a `MESH` executable all in the `MESH-smm` directory. Please be mindful of the settings, and assure the settings are all accurate to run `MESH` properly.
