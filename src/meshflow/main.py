"""
This package automates MESH model setup through a flexible workflow that can
be set up using a JSON configuration file, a Command Line Interface (CLI) or
directly inside a Python script/environment.

Much of the work has been adopted from workflows developed by Dr. Ala Bahrami
and Cooper Albano at the University of Saskatchewan applying MESH modelling
framework to the North American domain.
"""

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

import hydrant.topology as htp # pre-release version available only

from typing import (
    Iterable,
    Tuple,
    Dict,
    Sequence,
)
import re

import pint # pint >0.22
import glob


from .meshflow import __version__
from ._default_dicts import (
    ddb_global_attrs_default,
    ddb_local_attrs_default,
)


class MESHWorkflow(object):
    """
    Main Workflow class of MESH
    
    
    Attributes
    ----------
    
    
    Parameters
    ----------
    
    
    """
    # main constructor
    def __init__(
        self,
        riv : str,
        cat : str,
        landcover : str,
        coords : str,
        forcing_files : str,
        output_dir : str,
        forcing_vars : Sequence[str],
        forcing_units : Dict[str, str], # pint standards
        forcing_to_units : Dict[str, str] = None,
        main_id : str = None,
        ds_main_id: str = None,
        outlet_val: int = -9999,
        riv_cols : Dict[str, Union[str, int]] = None,
        cat_cols : Dict[str, Union[str, int]] = None,
        ddb_vars : Dict[str, str] = None,
        ddb_local_attrs : Dict[str, str] = None,
        ddb_global_attrs : Dict[str, str] = None,
        ddb_min_vals : Dict[str, float] = None,
    ) -> None:
        """Main constructor of MESHWorkflow
        """
        # `main_id` and `ds_main_id` are necessary
        if any(main_id, ds_main_id):
            raise ValueError("`main_id` and `ds_main_id` must be entered")
        
        # forcing variables
        if not forcing_vars:
            raise ValueError("`forcing_vars` cannot be empty")
            
        # units
        if not forcing_units:
            raise ValueError("`forcing_units` cannot be empty")
            
        # if ddb_local_attrs not entered
        if not ddb_global_attrs:
            self.ddb_global_attrs = ddb_global_attrs_default
        
        # if ddb_local_attrs not entred
        if not ddb_local_attrs:
            self.ddb_local_attrs = ddb_local_attrs_default
        
        # assign input values
        self.ddb_global_attrs = ddb_global_attrs
        self.ddb_local_attrs = ddb_local_attrs
        
        # read `riv` and `cat` using GeoPandas
        self.riv = gpd.read_file(riv)
        self.cat = gpd.read_file(cat)
        
        # read `landcover` using pandas
        # [FIXME]: this is a bit over simplified as a temporary
        #          solution
        self.landcover = pd.read_csv(landcover)
        
        # read `coords` using pandas
        # [FIXME]: this is a bit over simplified as a temporary
        #          solution
        self.coords = pd.read_csv(coords)
        
        # assign inputs
        self.main_id = main_id
        self.ds_main_id = ds_main_id
        self.forcing_files = forcing_files
        self.output_dir = output_dir
        self.forcing_vars = forcing_vars
        self.forcing_files = forcing_files
        self.forcing_units = forcing_units
        self.forcing_to_units = forcing_to_units
        self.outlet_val = outlet_val
        self.riv_cols = riv_cols
        self.cat_cols = cat_cols
        self.ddb_vars = ddb_vars
        self.ddb_min_vals = ddb_min_vals
        
        
    @classmethod
    def from_dict(
        cls: 'MESHWorkflow',
        init_dict: Dict = {},
    ) -> 'MESHWorkflow':
        """
        Constructor to use a dictionary to instantiate
        """
        if len(init_dict) == 0:
            raise KeyError("`init_dict` cannot be empty")
        assert isinstance(init_dict, dict), "`init_dict` must be a `dict`"

        return cls(**init_dict)

    
    @classmethod
    def from_json(
        cls: 'Easymore',
        json_str: str,
    ) -> 'Easymore':
        """
        Constructor to use a loaded JSON string
        """
        # building customized Easymore's JSON string decoder object
        decoder = json.JSONDecoder(object_hook=Easymore._easymore_decoder)
        json_dict = decoder.decode(json_str)

        return cls.from_dict(json_dict)

    
    @classmethod
    def from_json_file(
        cls: 'Easymore',
        json_file: 'str',
    ) -> 'Easymore':
        """
        Constructor to use a JSON file path
        """
        with open(json_file) as f:
            json_dict = json.load(f,
                                  object_hook=Easymore._easymore_decoder)

        return cls.from_dict(json_dict)

    
    @staticmethod
    def _env_var_decoder(s):
        """
        OS environmental variable decoder
        """
        # RE patterns
        env_pat = r'\$(.*?)/'
        bef_pat = r'(.*?)\$.*?/?'
        aft_pat = r'\$.*?(/.*)'
        # strings after re matches
        e = re.search(env_pat, s).group(1)
        b = re.search(bef_pat, s).group(1)
        a = re.search(aft_pat, s).group(1)
        # extract environmental variable
        v = os.getenv(e)
        # return full: before+env_var+after
        if v:
            return b+v+a
        return s

    
    @staticmethod
    def _easymore_decoder(obj):
        """
        Decoding typical JSON strings returned into valid Python objects
        """
        if obj in ["true", "True", "TRUE"]:
            return True
        elif obj in ["false", "False", "FALSE"]:
            return False
        elif isinstance(obj, str):
            if '$' in obj:
                return Easymore._env_var_decoder(obj)
        elif isinstance(obj, dict):
            return {k: Easymore._easymore_decoder(v) for k, v in obj.items()}
        return obj

    
    # main section
    def run(self):
        """
        Run the workflow and prepare a MESH setup
        """
        # some local variables
        ddb_file = "MESH_drainage_database.nc"
        forcing_file = "MESH_forcing.nc"
        
        # select rows and columns for the landcover
        sel_rows = landcover[main_id].isin(cat[main_id])
        sel_cols = landcover.columns[~landcover.columns.isin(['lat', 'lon'])]
        
        # preparing landcover
        landcover_c = landcover_df.loc[sel_rows, sel_cols].copy()

        # preparing `Rank` and `Next`
        rank_var, next_var, seg_id, to_seg = extract_rank_next(riv[main_id], riv[ds_main_id])

        # necessary values
        cols = ['Rank', 'Next', cat_dim]
        data = [rank_var, next_var, seg_id]

        # new riv_c
        riv_c = gpd.GeoDataFrame(pd.merge(left=riv_c.set_index(cat_dim).loc[seg_id].reset_index(), 
                                          right=pd.DataFrame({col:datum for col,datum in zip(cols, data)}),
                                          on=cat_dim))
        
        # generate drainage database
        # preparations
        y = prepare_mesh_ddb(riv=riv_c,
                             cat=self.cat,
                             landcover=landcover_c,
                             coords=self.coords,
                             cat_dim=main_id,
                             gru_dim='gru',
                             include_vars=self.forcing_vars,
                             attr_local=self.ddb_local_attrs,
                             attr_global=self.ddb_global_attrs,
                             min_values=self.min_values,
                             fill_na=None,
                             ordered_dims={cat_dim:seg_id},
                            )
        
        # printing drainage database netcdf file
        y.to_netcdf(os.path.join(self.output_dir, forcing_file))
        
        # forcing unit registry
        ureg = pint.UnitRegistry(force_ndarray_like=True)
        ureg.define('millibar = 1e-3 * bar')
        
        # printing forcing netcdf file
        ds = mesh_forcing(path = self.forcing_files,
                          variables=self.forcing_vars,
                          output_path=self.output_dir,
                          units=self.units,
                          to_units=self.to_units,
                          unit_registry=ureg)
        
        # printin forcing netcdf file
        ds.to_netcdf(os.path.join(self.output_dir, forcing_file))
        
        # copy settings file
        # [FIXME]: check and copy template files as an immediate solution
        os.copy()
        
        return
    
    def _progress_bar():
        pass
    
    def _submit_job():
        pass
