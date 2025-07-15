"""
This package automates the setup of MESH models through a flexible workflow.
Configuration can be provided via a JSON file, Command Line Interface (CLI),
or directly within a Python script or environment.

Much of the workflow is based on approaches developed by Dr. Ala Bahrami and
Cooper Albano at the University of Saskatchewan for applying the MESH
modelling framework to North American domains.
"""

# third-party libraries
import dateutil.parser
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import pint
import dateutil
import timezonefinder

# built-in libraries
from typing import (
    Dict,
    Any,
    Union,
    Optional,
)

import re
import json
import sys
import glob
import os
import shutil
import warnings
import datetime

# local imports
from ._default_attrs import (
    ddb_global_attrs_default,
    ddb_local_attrs_default,
    forcing_local_attrs_default,
    forcing_global_attrs_default,
    default_attrs,
)
from ._default_dicts import (
    mesh_forcing_units_default,
    mesh_drainage_database_units_default,
    mesh_drainage_database_minimums_default,
    mesh_drainage_database_names_default,
)
from meshflow import utility

# custom type hints
try:
    from os import PathLike
except ImportError:  # <Python3.8
    from typing import Union
    PathLike = Union[str, bytes]


class MESHWorkflow(object):
    """
    A comprehensive workflow automation class for MESH (Modélisation Environmentale 
    Communautaire - Surface and Hydrology) model setup.

    This class provides a flexible and automated approach to setting up MESH model 
    configurations through various input methods including JSON configuration files, 
    Command Line Interface (CLI), or direct Python scripting. It handles the 
    preparation of geospatial data, forcing files, drainage databases, and model 
    configuration files required for MESH model execution.

    The workflow encompasses:
    - Processing geofabric data (river networks and catchments)
    - Preparing landcover classification data
    - Processing meteorological forcing data
    - Generating drainage database files
    - Creating model configuration files (CLASS, hydrology, run options)
    - Coordinate system transformations and unit conversions

    This implementation is based on workflows developed by Dr. Ala Bahrami and 
    Cooper Albano at the University of Saskatchewan for applying the MESH modelling 
    framework to North American domains.

    Attributes
    ----------
    riv : geopandas.GeoDataFrame
        River network geometry and attributes
    cat : geopandas.GeoDataFrame  
        Catchment/subbasin geometry and attributes
    landcover : pandas.DataFrame
        Landcover fractions for each catchment and land class
    forcing : xarray.Dataset
        Processed meteorological forcing data (when single file mode)
    ddb : xarray.Dataset
        MESH drainage database containing catchment and routing parameters
    coords : pandas.DataFrame
        Centroid coordinates for each catchment
    main_id : str
        Primary identifier field name for catchments
    ds_main_id : str
        Downstream catchment identifier field name
    forcing_vars : dict
        Mapping of input forcing variable names to standard names
    forcing_units : dict
        Units of input forcing variables
    ddb_vars : dict
        Mapping of drainage database variable names
    landcover_classes : dict
        Mapping of landcover class codes to descriptive names
    settings : dict
        Model configuration settings and parameters

    Examples
    --------
    Basic usage with file paths:

    >>> workflow = MESHWorkflow(
    ...     riv='path/to/rivers.shp',
    ...     cat='path/to/catchments.shp', 
    ...     landcover='path/to/landcover.csv',
    ...     landcover_classes={1: 'forest', 2: 'grassland'},
    ...     forcing_files='path/to/forcing/*.nc',
    ...     forcing_vars={'PRES': 'air_pressure', 'TEMP': 'air_temperature'},
    ...     settings={'core': {'forcing_files': 'single'}}
    ... )
    >>> workflow.run(save_path='output_directory')

    Using JSON configuration:

    >>> workflow = MESHWorkflow.from_json_file('config.json')
    >>> workflow.run(save_path='output_directory')

    See Also
    --------
    from_dict : Create instance from dictionary
    from_json : Create instance from JSON string  
    from_json_file : Create instance from JSON file
    """
    # main constructor
    def __init__(
        self,
        riv: PathLike,
        cat: PathLike,
        landcover: PathLike,
        landcover_classes: Dict[str, str],
        forcing_files: PathLike = None,
        forcing_vars: Dict[str, str] = None,
        forcing_units: Dict[str, str] = None,
        ddb_vars: Dict[str, str] = None,
        ddb_units: Dict[str, str] = None,
        main_id: str = None,
        ds_main_id: str = None,
        forcing_to_units: Dict[str, str] = mesh_forcing_units_default,
        forcing_local_attrs: Dict[str, str] = forcing_local_attrs_default,
        forcing_global_attrs: Dict[str, str] = forcing_global_attrs_default,
        ddb_local_attrs: Dict[str, str] = ddb_local_attrs_default,
        ddb_global_attrs: Dict[str, str] = ddb_global_attrs_default,
        ddb_min_values: Dict[str, float] = mesh_drainage_database_minimums_default,
        ddb_to_units: Dict[str, str] = mesh_drainage_database_units_default,
        settings: Dict[str, str] = None,
        gru_dim: str = 'NGRU',
        hru_dim: str = 'subbasin',
        outlet_value: int = -9999,
    ) -> None:
        """Initialize a new MESHWorkflow instance for automated model setup.

        Parameters
        ----------
        riv : str or PathLike
            Path to the river network shapefile or geospatial data file containing
            river segments with routing connectivity information.
        cat : str or PathLike  
            Path to the catchment/subbasin shapefile or geospatial data file containing
            watershed boundaries and attributes.
        landcover : str or PathLike
            Path to the landcover CSV file containing fractional coverage of different
            land classes for each catchment. Expected format has catchment IDs as 
            index and columns prefixed with 'frac_' for each land class.
        landcover_classes : dict of {str: str}
            Mapping of landcover class codes/numbers to descriptive class names.
            Keys should match the numeric suffixes in landcover file column names.
            Example: {1: 'needleleaf forest', 2: 'grassland', 3: 'urban'}
        forcing_files : str or PathLike, optional
            Path to meteorological forcing files. Can be a single NetCDF file,
            a glob pattern (e.g., '*.nc'), or a directory containing forcing files.
            Required for full workflow execution.
        forcing_vars : dict of {str: str}, optional
            Mapping of input forcing variable names to MESH standard variable names.
            Example: {'PRES': 'air_pressure', 'TEMP': 'air_temperature', 'PRECIP': 'precipitation'}
        forcing_units : dict of {str: str}, optional
            Units of the input forcing variables. Keys should match those in forcing_vars.
            Example: {'air_pressure': 'Pa', 'air_temperature': 'K', 'precipitation': 'mm/h'}
        ddb_vars : dict of {str: str}, optional
            Mapping of drainage database variable names from input data to MESH standards.
            Used for customizing which catchment attributes to include in the drainage database.
        ddb_units : dict of {str: str}, optional
            Units of drainage database variables. Keys should match those in ddb_vars.
        main_id : str, optional
            Name of the primary identifier field in the catchment and river data.
            This field links river segments to their corresponding catchments.
            Default behavior attempts to auto-detect common field names.
        ds_main_id : str, optional  
            Name of the downstream segment identifier field in river data.
            Used to establish routing connectivity between river segments.
            Default behavior attempts to auto-detect common field names.
        forcing_to_units : dict of {str: str}, optional
            Target units for forcing variables after conversion. Uses MESH defaults
            if not specified. Example: {'air_pressure': 'Pa', 'air_temperature': 'K'}
        forcing_local_attrs : dict of {str: str}, optional
            Local (variable-level) attributes to apply to forcing variables in output files.
            Uses package defaults if not specified.
        forcing_global_attrs : dict of {str: str}, optional
            Global (file-level) attributes to apply to forcing NetCDF files.
            Uses package defaults if not specified.
        ddb_local_attrs : dict of {str: str}, optional
            Local (variable-level) attributes to apply to drainage database variables.
            Uses package defaults if not specified.
        ddb_global_attrs : dict of {str: str}, optional
            Global (file-level) attributes to apply to drainage database NetCDF file.
            Uses package defaults if not specified.
        ddb_min_values : dict of {str: float}, optional
            Minimum allowable values for drainage database variables.
            Used for quality control and replacing unrealistic values.
            Uses package defaults if not specified.
        ddb_to_units : dict of {str: str}, optional
            Target units for drainage database variables after conversion.
            Uses MESH defaults if not specified.
        settings : dict, optional
            Comprehensive model configuration dictionary containing parameters for:
            - Core settings (dates, file handling, time zones)
            - CLASS parameters (measurement heights, land surface physics)
            - Hydrology parameters (routing, subsurface processes)
            Required for full workflow execution. Can be loaded from JSON files.
        gru_dim : str, optional, default='NGRU'
            Dimension name for Group Response Units (land classes) in output files.
            Typically does not need to be changed unless using custom MESH builds.
        hru_dim : str, optional, default='subbasin'
            Dimension name for Hydrologic Response Units (catchments) in output files.
            Should match the primary catchment identifier dimension.
        outlet_value : int, optional, default=-9999
            Sentinel value used to identify outlet/terminal segments in the river network.
            Segments with this value in the downstream ID field are treated as outlets.

        Raises
        ------
        TypeError
            If dictionary parameters are not of type dict when provided.
        FileNotFoundError
            If required input files (riv, cat, landcover) cannot be found.
        KeyError
            If required keys are missing from landcover_classes or settings dictionaries.
        ValueError
            If data validation fails (e.g., mismatched IDs between files).

        Notes
        -----
        The constructor performs lazy loading of geospatial files - data is read when
        the instance is created but heavy processing is deferred until run() is called.
        
        All file paths support environment variable expansion using the pattern $VAR/
        where VAR is an environment variable name.

        The workflow expects specific data formats:
        - River and catchment files: Any format readable by GeoPandas (Shapefile, GeoPackage, etc.)
        - Landcover file: CSV with catchment IDs as index, columns named 'frac_<class_number>'
        - Forcing files: NetCDF format with time dimension and spatial coverage

        Examples
        --------
        Minimal setup for testing:
        
        >>> workflow = MESHWorkflow(
        ...     riv='rivers.shp',
        ...     cat='catchments.shp', 
        ...     landcover='landcover.csv',
        ...     landcover_classes={1: 'forest', 2: 'grassland'},
        ...     settings={'core': {'forcing_files': 'single'}}
        ... )

        Complete setup with forcing data:
        
        >>> workflow = MESHWorkflow(
        ...     riv='rivers.shp',
        ...     cat='catchments.shp',
        ...     landcover='landcover.csv', 
        ...     landcover_classes={1: 'needleleaf forest', 2: 'grassland', 3: 'urban'},
        ...     forcing_files='/data/forcing/*.nc',
        ...     forcing_vars={'PRES': 'air_pressure', 'TEMP': 'air_temperature'},
        ...     forcing_units={'air_pressure': 'hPa', 'air_temperature': 'C'},
        ...     main_id='COMID',
        ...     ds_main_id='NextDownID',
        ...     settings=my_settings_dict
        ... )
        """
        # dictionary to check types
        _dict_items = [forcing_units,
                       forcing_to_units,
                       ddb_units,
                       ddb_to_units]

        # check dictionary dtypes
        for item in _dict_items:
            if not isinstance(item, dict) and (item is not None):
                raise TypeError(f"`{item}` must be of type dict")

        # if `ddb_local_attrs` not assigned
        if not ddb_global_attrs:
            self.ddb_global_attrs = ddb_global_attrs_default
        else:
            self.ddb_global_attrs = ddb_global_attrs

        # if `ddb_local_attrs` not assigned
        if not ddb_local_attrs:
            self.ddb_local_attrs = ddb_local_attrs_default
        else:
            self.ddb_local_attrs = ddb_local_attrs

        # if `forcing_local_attrs` not assigned
        if not forcing_local_attrs:
            self.forcing_local_attrs = forcing_local_attrs_default
        else:
            self.forcing_local_attrs = forcing_local_attrs

        # if `forcing_global_attrs` not assigned
        if not forcing_global_attrs:
            self.forcing_global_attrs = forcing_global_attrs_default
        else:
            self.forcing_global_attrs = forcing_global_attrs

        # assigning geofabric files to be "lazy loaded"
        self._riv_path = riv
        self._cat_path = cat

        # assgining landcover files to be "lazy loaded"
        self._landcover_path = landcover

        # assgining forcing files path to be "lazy loaded"
        self._forcing_path = forcing_files

        # assign inputs
        # geofabric specs
        self.main_id = main_id
        self.ds_main_id = ds_main_id
        self.outlet_value = outlet_value

        # forcing specs
        self.forcing_vars = forcing_vars
        self.forcing_units = forcing_units
        self.forcing_to_units = forcing_to_units

        # drainage database specs
        self.ddb_vars = ddb_vars
        self.ddb_units = ddb_units
        self.ddb_to_units = ddb_to_units
        self.ddb_min_values = ddb_min_values

        # landcover specs
        self.landcover_classes = landcover_classes

        # assing inputs read from files
        self._read_input_files()

        # core variable and dimension names
        self.gru_dim = gru_dim
        self.gru_var = self.gru_dim + '_var'
        self.hru_dim = hru_dim
        self.hru_var = self.hru_dim + '_var'

        # MESH-specific variables
        self.rank_str = 'Rank'
        self.next_str = 'Next'

        # If settings are provided as a dictionary
        assert isinstance(settings, dict), "`settings` must be a `dict`"
        self.settings = settings

    def _read_input_files(self):
        """Read necessary input files
        [FIXME]: lazy loading may be necessary in the future to save time
        when istantiating from MESHWorkflow class
        [FIXME]: eventually, this needs to turn into its own object,
        rather than reading a simple file in pandas DataFrame format
        """
        self.riv = gpd.read_file(self._riv_path)
        self.cat = gpd.read_file(self._cat_path)
        self.landcover = self._read_landcover()

    def _read_landcover(self):
        """
        Return landcover object for elements given in the input domain for
        MESH model setup
        """

        # [FIXME]: This needs to be flexible in future versions
        _lc_prefix = 'frac_'

        # local _seg_ids and _ds_seg_ids
        _seg_ids = self.cat.loc[:, self.main_id]

        # read the landcover data as a pandas.DataFrame
        _lc_df = pd.read_csv(self._landcover_path,
                             index_col=0,
                             header=0)

        # select rows and columns to be included for landcover object
        _rows = [row for row in _lc_df.index if _seg_ids.isin([row]).any()]
        _cols = [col for col in _lc_df.columns if
                 col.startswith(_lc_prefix)]

        # return a copy of the dataframe including hrus available in the
        # input domain and only fractions

        return _lc_df.loc[_rows, _cols].copy()

    @property
    def coords(self):
        # calculate centroid latitude and longitude values
        return utility.extract_centroid(gdf=self.cat,
                                        obj_id=self.main_id)

    @property
    def forcing_files(self):
        pattern = r"\.nc\*?|\.nc"
        if re.search(pattern, self._forcing_path):
            if glob.glob(self._forcing_path):
                return self._forcing_path
        else:
            _path = os.path.join(self._forcing_path, '*.nc*')
            if glob.glob(_path):
                return _path

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
        cls: 'MESHWorkflow',
        json_str: str,
    ) -> 'MESHWorkflow':
        """
        Constructor to use a loaded JSON string
        """
        # building customized MESHWorkflow's JSON string decoder object
        decoder = json.JSONDecoder(object_hook=MESHWorkflow._json_decoder)
        json_dict = decoder.decode(json_str)
        # return class instance
        return cls.from_dict(json_dict)

    @classmethod
    def from_json_file(
        cls: 'MESHWorkflow',
        json_file: 'str',
    ) -> 'MESHWorkflow':
        """
        Constructor to use a JSON file path
        """
        with open(json_file) as f:
            json_dict = json.load(f, object_hook=MESHWorkflow._json_decoder)

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
    def _json_decoder(obj):
        """
        Decoding typical JSON strings returned into valid Python objects
        """
        if obj in ["true", "True", "TRUE"]:
            return True
        elif obj in ["false", "False", "FALSE"]:
            return False
        elif isinstance(obj, str):
            if '$' in obj:
                return MESHWorkflow._env_var_decoder(obj)
            if MESHWorkflow._is_valid_integer(obj):
                return int(obj)
        elif isinstance(obj, dict):
            return {MESHWorkflow._json_decoder(k): MESHWorkflow._json_decoder(v) for k, v in obj.items()}
        return obj
    
    @staticmethod
    def _is_valid_integer(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def run(
        self,
        save_path: Optional[PathLike] = None,
    ) -> None:
        """
        Run the workflow and prepare a MESH setup
        """
        # Initilize drainage database and forcing objects
        self.init()

        # Generate drainage database
        self.init_ddb() # creates self.ddb automatically

        # Generate forcing data
        if self.settings['core']['forcing_files'] == 'multiple':
            warnings.warn(
                "Since multiple forcing files are needed, "
                "each file will be processed and saved during "
                "initialization.",
                UserWarning,
            )
            if save_path is None:
                raise ValueError("`save_path` cannot be None when processing multiple forcing files.")

            # hard-coded 'forcings' path in the `save_path` path
            self.init_forcing(save=True, save_path=os.path.join(save_path, 'forcings'))

        else:
            self.init_forcing(save=False) # creates self.forcing automatically

        # 3 generate land-cover dependant setting files for MESH
        self.class_text = self.init_class(return_text=True)

        # 4 generate other setting files for MESH
        self.hydrology_text = self.init_hydrology(return_text=True)

        # 5 generate run options for MESH
        self.run_options_text = self.init_options(return_text=True)

        return

    def init(
        self
    ) -> None:
        """
        Initialize the MESH workflow by setting up drainage database and
        forcing objects
        """
        # initialize MESH-specific variables including `rank` and `next`
        # and the re-ordered `main_seg` and `ds_main_seg`;
        # This will assign: 1) self.rank, 2) self.next, 3) self.main_seg,
        # and 4) self.ds_main_seg
        self._init_idx_vars()

        # reorder river segments and add rank and next to the
        # geopandas.DataFrame object
        self.reordered_riv = self._reorder_riv(rank_name=self.rank_str,
                                               next_name=self.next_str)

        # defined ordered_dims variables
        self.ordered_dims = {self.main_id: self.main_seg}

        # assigning coordinates to both `forcing` and `ddb` of an instance
        self._coords_ds = utility.prepare_mesh_coords(self.coords,
                                                      cat_dim=self.main_id,
                                                      hru_dim=self.hru_dim)

    def init_ddb(
        self,
        return_ds = False,
        save_path: Optional[PathLike] = None,
    ) -> None:
        """
        Initialize the drainage database object
        """
        # ddb variables
        if not self.ddb_vars:
            raise ValueError("`ddb_vars` cannot be empty")

        # Creating local dictionaries for drainage database variables
        ddb_vars_renamed = {}
        ddb_units_renamed = {}
        ddb_to_units_renamed = {}
        ddb_min_values_renamed = {}

        # Based on the input `ddb_vars`, adjust the names with MESH standard
        # values
        for k, v in self.ddb_vars.items():
            if k in mesh_drainage_database_names_default:
                ddb_vars_renamed[v] = mesh_drainage_database_names_default[k]
        # Assuring default variables are all added
        for k in ('rank', 'next'):
                v = mesh_drainage_database_names_default[k]
                ddb_vars_renamed[v] = v
        # Similarly for the `landclass` variable, while its naming scheme is
        # an outlier
        ddb_vars_renamed['landclass'] = mesh_drainage_database_names_default['landclass']
        ddb_vars_renamed['area'] = 'GridArea' # FIXME: this needs to be automated

        # similarly for `ddb_units`
        for k, v in self.ddb_units.items():
            if k in mesh_drainage_database_names_default:
                new_k = mesh_drainage_database_names_default[k]
                ddb_units_renamed[new_k] = v

        # and for `ddb_to_units`
        for k, v in self.ddb_to_units.items():
            if k in mesh_drainage_database_names_default:
                new_k = mesh_drainage_database_names_default[k]
                ddb_to_units_renamed[new_k] = v

        # finally, for the minimum values of the drainage database
        for k, v in self.ddb_min_values.items():
            if k in mesh_drainage_database_names_default:
                new_k = mesh_drainage_database_names_default[k]
                ddb_min_values_renamed[new_k] = v

        # generate mesh drainage database
        self.ddb = utility.prepare_mesh_ddb(
            riv=self.reordered_riv,
            cat=self.cat,
            landcover=self.landcover,
            cat_dim=self.main_id,
            gru_dim=self.gru_dim,
            hru_dim=self.hru_dim,
            gru_names=self.landcover_classes,
            include_vars=ddb_vars_renamed,
            attr_local=self.ddb_local_attrs,
            attr_global=self.ddb_global_attrs,
            min_values=ddb_min_values_renamed,
            fill_na=None,
            ordered_dims=self.ordered_dims,
            ddb_units=ddb_units_renamed,
            ddb_to_units=ddb_to_units_renamed)

        # ad-hoc manipulations on the drainage database
        self.ddb = self._adhoc_mesh_vars(self.ddb)

        # assign the `GridArea` units attributes
        # FIXME: this needs to be changed to pint's
        #        `pint.Quantity` object
        self.ddb['GridArea'].attrs['units'] = 'm ** 2'

        if return_ds:
            return self.ddb
        else:
            return

    def init_forcing(
        self,
        save: bool = False,
        save_path: Optional[PathLike] = None,
    ) -> Optional[xr.Dataset]:
        """
        Initialize the forcing object
        """
        # Type checks
        assert isinstance(save, bool), "`save` must be a boolean value"
        if save_path is not None:
            assert isinstance(save_path, (str, PathLike)), \
                "`save_path` must be a string or PathLike object"

        # Check if forcing files are provided
        if not self.forcing_files:
            raise ValueError("`forcing_files` cannot be empty")

        # Check if forcing variables are provided
        if not self.forcing_vars:
            raise ValueError("`forcing_vars` cannot be empty")

        # Check if forcing units are provided
        if not self.forcing_units:
            raise ValueError("`forcing_units` cannot be empty")

        # Check if forcing to_units are provided
        if not self.forcing_to_units:
            raise ValueError("`forcing_to_units` cannot be empty")

        # Check if save is enabled and save_path is provided
        if save and save_path is None:
            raise ValueError("`save_path` cannot be None if `save` is True")
        
        # if save_path is not None, make sure the directory exists
        if save_path is not None:
            # get the absolute path
            save_path = os.path.abspath(save_path)

            # make sure the directory exists
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            else:
                warnings.warn(
                    f"Directory {save_path} already exists. "
                    "Forcing files will be saved there.",
                    UserWarning,
                )

        # Forcing unit registry
        _ureg = pint.UnitRegistry(force_ndarray_like=True)
        _ureg.define('millibar = 1e-3 * bar')
        _ureg.define('degrees_north = 1 * degree')
        _ureg.define('degrees_east = 1 * degree')

                    
        # forcing units need internal names for keys
        forcing_units_renamed = {}
        # renaming forcing units based on the provided forcing_vars
        for k, v in self.forcing_units.items():
            if k in self.forcing_vars:
                new_k = self.forcing_vars[k]
                forcing_units_renamed[new_k] = v

        # same for forcing_to_units
        forcing_to_units_renamed = {}
        for k, v in self.forcing_to_units.items():
            if k in self.forcing_vars:
                new_k = self.forcing_vars[k]
                forcing_to_units_renamed[new_k] = v

        # Generate forcing data
        # making a list of variables
        if self.settings['core']['forcing_files'] == 'multiple':
            # Make a list of forcing files
            files = sorted(glob.glob(self.forcing_files))

            if not files:
                raise ValueError("No forcing files found matching the pattern")

            for forcing_file in files:
                ds = utility.prepare_mesh_forcing(
                    path=forcing_file,
                    variables=[v for k, v in self.forcing_vars.items()],
                    hru_dim=self.hru_dim,
                    units=forcing_units_renamed,
                    to_units=forcing_to_units_renamed,
                    unit_registry=_ureg,
                    aggregate=self.settings['core']['forcing_files'],
                    local_attrs=self.forcing_local_attrs,
                    global_attrs=self.forcing_global_attrs
                )

                # modify adhoc mesh variables
                ds = self._adhoc_mesh_vars(ds)

                # modify time encoding of the forcing object, though MESH
                # does not care about the time encoding anymore >r1860
                ds = self._modify_forcing_encodings(ds)

                # save the forcing object to a file
                if save:
                    file_name = os.path.basename(forcing_file)
                    # save the forcing object to a file
                    ds.to_netcdf(os.path.join(save_path, file_name),
                                 format='NETCDF4_CLASSIC',
                                 unlimited_dims=['time'])
                else:
                    warnings.warn(
                        "Forcing object is not saved, but returned as "
                        "xarray.Dataset object(s).",
                        UserWarning,
                    )

                # closing the file
                ds.close()

        else:
            # if `self.settings['core']['forcing_files']` is not 'multiple', we assume
            # that the forcing files should be merged
            ds = utility.prepare_mesh_forcing(
                    path=self.forcing_files,
                    variables=[v for v in self.forcing_vars.values()],
                    hru_dim=self.hru_dim,
                    units=forcing_units_renamed,
                    to_units=forcing_to_units_renamed,
                    unit_registry=_ureg,
                    aggregate=self.settings['core']['forcing_files'],
                    local_attrs=self.forcing_local_attrs,
                    global_attrs=self.forcing_global_attrs
                )
            # modify adhoc mesh variables
            ds = self._adhoc_mesh_vars(ds)

            # modify time encoding of the forcing object, though MESH
            # does not care about the time encoding anymore >r1860
            ds = self._modify_forcing_encodings(ds)

            # save the forcing object to a file
            if save:
                # save the forcing object to a file
                ds.to_netcdf(os.path.join(save_path, "MESH_forcing.nc"),
                                format='NETCDF4_CLASSIC',
                                unlimited_dims=['time'])

            self.forcing = ds

        return

    def init_class(
        self,
        return_text: bool = False,
    ) -> Optional[str]:
        """
        Initialize the class text file for MESH
        """
        # routine checks
        # check if 'measurement_height' is provided in the 'class_params'
        if 'measurement_heights' not in self.settings['class_params']:
            raise KeyError("`measurement_heights` must be provided in `class_params`")
        # check if necessary parameters are provided in `measurement_height`
        if not all(key in self.settings['class_params']['measurement_heights']
                   for key in ['wind_speed', 'specific_humidity', 'air_temperature', 'roughness_length']):
            raise KeyError("`measurement_heights` must contain 'wndspd', 'spechum',"
                           " 'airtemp', and 'roughness_length' keys")
        # if the values of `specific_humidity`, `air_temperature` are not equal
        # raise an error
        if self.settings['class_params']['measurement_heights']['specific_humidity'] != \
           self.settings['class_params']['measurement_heights']['air_temperature']:
            raise ValueError("`measurement_heights['specific_humidity']` and "
                             "`measurement_heights['air_temperature']` must be equal")

        # calculate area's centroid coordinates
        area_centroids = utility.extract_centroid(
            self.cat.dissolve(),
            obj_id=self.main_id)

        area_centroid_x = area_centroids['lon'][0]
        area_centroid_y = area_centroids['lat'][0]

        # extract subbasin ID counts
        subbasin_counts = len(self.cat[self.main_id].index)

        # extract landcover class counts
        landcover_counts = len(self.landcover.columns.str.startswith('frac_'))

        # build the CLASS "case"s dictionary
        class_case = {
            "centroid_lat": area_centroid_y,
            "centroid_lon": area_centroid_x,
            "reference_height_wndspd": self.settings['class_params']['measurement_heights']['wind_speed'],
            "reference_height_spechum_airtemp": self.settings['class_params']['measurement_heights']['specific_humidity'],
            "reference_height_surface_roughness": self.settings['class_params']['measurement_heights']['roughness_length'],
            "NL": subbasin_counts,
            "NM": landcover_counts,
        }

        # build the CLASS "info" dictionary
        if 'copyright' in self.settings['class_params']:
            if 'author' in self.settings['class_params']['copyright'] and \
               'email' in self.settings['class_params']['copyright']:
                class_info = {
                    'author': self.settings['class_params']['copyright']['author'],
                    'location': self.settings['class_params']['copyright']['email'],
                }

        # if `class_info` is not provided, use the default values
        if 'class_info' not in locals():
            class_info = {
                'author': 'MESHFlow',
                'location': 'University of Calgary, Canada',
            }

        # build the automated CLASS "gru" dictionary to be updated later by
        # user's manual inputs
        class_gru = {}
        for gru in self.landcover.columns:
            # extract the GRU name, typically an integer
            gru_number = int(gru.replace('frac_', ''))

            # extract class names from the landcover_classes dictionary
            if gru_number not in self.landcover_classes:
                raise KeyError(f"GRU `{gru_number}` not found in landcover_classes")
            
            mid = self._return_short_gru_name(self.landcover_classes[gru_number])
            # split the class name into a list of class names
            # e.g., 'needleleaf deciduous' -> ['needleleaf', 'deciduous']
            # build the gru dictionary for the gru block
            class_gru[gru_number] = {
                'class': 'needleleaf', # default value for everything
                'mid': mid,
            }

        # update the class_gru dictionary with user inputs
        if 'class_params' in self.settings and 'grus' in self.settings['class_params']:
            for gru, params in self.settings['class_params']['grus'].items():
                # check if the gru is in the class_gru dictionary
                if gru in class_gru:
                    # update the class_gru dictionary with user inputs
                    class_gru[gru].update(params)
                else:
                    warnings.warn(f"GRU {gru} not found in landcover classes. Skipping...")

        # generate the CLASS text file
        class_text = utility.render_class_template(
            class_info=class_info,
            class_case=class_case,
            class_grus=class_gru,
        )

        # if return is requested, return the class text
        if return_text:
            return class_text

    def init_hydrology(
        self,
        return_text: bool = False,
    ) -> Optional[str]:
        
        # build the routing dictionary
        # if order is provided, use it
        if hasattr(self, 'ddb_vars'):
            if 'river_class' in self.ddb_vars:
                river_classes = len(set(self.ddb_vars['river_class']))
                if river_classes > 5:
                    raise ValueError("`river_class` cannot have more than 5 "
                                     "classes. Adjust the `ddb_vars`'s "
                                     "`river_class` value.")

        # if river_classes variable exist
        if 'river_classes' in locals():
            routing_dict = [dict() for i in range(river_classes)]
        else:
            routing_dict = [{}]

        # build the GRU-dependent hydrology part
        hydrology_dict = {int(k.replace('frac_', '')): {} for k in self.landcover.columns}

        # If user provided more infomration, update these dictionaries
        if 'hydrology_params' in self.settings:
            # update the routing dictionary, assuming user has taken
            # care of the order of classes
            if 'routing' in self.settings['hydrology_params'] and \
                len(self.settings['hydrology_params']['routing']) > 0:
                    routing_dict = self.settings['hydrology_params']['routing']

            # update the hydrology dictionary
            if 'hydrology' in self.settings['hydrology_params']:
                hydrology_dict = self.settings['hydrology_params']['hydrology']
        
        # build the hydrology text file
        hydrology_text = utility.render_hydrology_template(
            routing_params=routing_dict,
            hydrology_params=hydrology_dict,
        )

        # if return is requested, return the hydrology text
        if return_text:
            return hydrology_text

        return

    def init_options(
        self,
        return_text: bool = False,
    ) -> Optional[str]:
        """
        Initialize the options text file for MESH
        """

        # build the options dictionary
        # identify the forcing variables
        # forcing start date
        forcing_start_date = self.settings['core']['forcing_start_date']
        forcing_start_date = self.format_date(forcing_start_date, '%Y%m%d')

        # extract simulation dates
        if 'simulation_start_date' in self.settings['core']:
            start_date = self.settings['core']['simulation_start_date']
            start_date = dateutil.parser.parse(start_date)
        else:
            start_date = dateutil.parser.parse(forcing_start_date)

        # extract end date
        if 'simulation_end_date' in self.settings['core']:
            end_date = self.settings['core']['simulation_end_date']
            end_date = dateutil.parser.parse(end_date)
        else:
            end_date = '2100-12-31 23:00:00'

        # if multiple forcing files are used, then providing a list of forcing
        # files is necessary, otherwise, just the default "MESH_forcing" for the
        # fname in the forcing dictionary is enough
        if self.settings['core']['forcing_files'] == 'multiple':
            # if multiple forcing files are used, then providing a list of forcing
            # files is necessary, otherwise, just the default "MESH_forcing" for the
            # fname in the forcing dictionary is enough
            forcing_name = ''
            forcing_list = 'forcing_files_list'
        else:
            # if single forcing file is used, then just the default "MESH_forcing" for the
            # fname in the forcing dictionary is enough
            forcing_name = 'MESH_forcing'
            forcing_list = ''

        # check the time-zone of the forcing file and the target time-zone to 
        # calculate the time-difference in hours
        if 'forcing_time_zone' in self.settings['core']:
            forcing_time_zone = self.settings['core']['forcing_time_zone']
        else:
            forcing_time_zone = 'UTC'
            warnings.warn(
                "No `forcing_time_zone` provided in the settings. "
                "Assuming UTC time zone.",
                UserWarning,
            )

        # check the model time-zone
        if 'model_time_zone' in self.settings['core']:
            model_time_zone = self.settings['core']['model_time_zone']
        else:
            # try extracting the time zone from the provided `cat` file
            # calculate area's centroid coordinates---basically what is done
            # in the `init_class` method
            warnings.warn(
                "No `model_time_zone` provided in the settings. "
                "Autodetecting the time zone using `timezonefinder` "
                "based on the centroid of the catchment area.",
                UserWarning,
            )
            area_centroids = utility.extract_centroid(
                self.cat.dissolve(),
                obj_id=self.main_id
            )
            # assigning the latitude and longitude values
            area_centroid_x = area_centroids['lon'][0]
            area_centroid_y = area_centroids['lat'][0]
            # extracing the model time zone from the coordinates
            model_time_zone = timezonefinder.TimezoneFinder().timezone_at(
                lat=area_centroid_y,
                lng=area_centroid_x
            )
            # Print the model time zone
            if model_time_zone:
                warnings.warn(
                    f"Autodetected model time zone: {model_time_zone}",
                    UserWarning,
                )
            # if the model time zone is None, then assume UTC
            # and warn the user
            else:
                model_time_zone = 'UTC'
                warnings.warn(
                    "No `model_time_zone` provided in the settings and"
                    " autodetection using `timezonefinder` failed."
                    " Assuming UTC time zone.",
                    UserWarning,
                )

        # calculate the time difference in hours
        time_diff = utility.calculate_time_difference(
            initial_time_zone=forcing_time_zone,
            target_time_zone=model_time_zone
        )

        # if integer, turn into an integer
        time_diff = self.maybe_int(time_diff)

        options_dict = {
            "flags": {
                "forcing": {
                    "BASINSHORTWAVEFLAG": f'name_var={self.forcing_vars.get("shortwave_radiation")}',
                    "BASINHUMIDITYFLAG": f'name_var={self.forcing_vars.get("specific_humidity")}',
                    "BASINRAINFLAG": f'name_var={self.forcing_vars.get("precipitation")}',
                    "BASINPRESFLAG": f'name_var={self.forcing_vars.get("air_pressure")}',
                    "BASINLONGWAVEFLAG": f'name_var={self.forcing_vars.get("longwave_radiation")}',
                    "BASINWINDFLAG": f'name_var={self.forcing_vars.get("wind_speed")}',
                    "BASINTEMPERATUREFLAG": f'name_var={self.forcing_vars.get("air_temperature")}',
                    "BASINFORCINGFLAG": {
                        "start_date": forcing_start_date,
                        "hf": 60, # FIXME: hardcoded value for now
                        "time_shift": time_diff,
                        "fname": forcing_name,
                    },
                    "FORCINGLIST": forcing_list,
                },
                "etc": {
                    "PBSMFLAG": "off",
                    "TIMESTEPFLAG": 60, # FIXME: hardcoded value for now
                },
            },
            "outputs": {
                "result": "results",
            },
            "dates": {
                "start_year": start_date.timetuple().tm_year,
                "start_day": start_date.timetuple().tm_yday,
                "start_hour": start_date.timetuple().tm_hour,
                "end_year": end_date.timetuple().tm_year,
                "end_day": end_date.timetuple().tm_yday,
                "end_hour": end_date.timetuple().tm_hour,
            },
        }

        self.options_text = utility.render_run_options_template(options_dict)

        if return_text:
            return self.options_text
        
        return

    def save(self, output_dir):
        """
        Save the drainage database, forcing files, and configuration files to
        the specified output directory.

        This method exports the model's drainage database and forcing files in
        NetCDF or text format, copies default setting files, and writes
        configuration files required for the model setup.

        Parameters
        ----------
        output_dir : str or PathLike
            Path to the directory where the model setup and associated files will be saved.

        Returns
        -------
        None
            This method does not return any value. Files are written to the specified output directory.

        Notes
        -----
        - If a single forcing file is used, it is saved as 'MESH_forcing.nc' in NetCDF format.
        - If multiple forcing files are used, a list of their paths is saved as 'forcing_files_list.txt'.
        - Default setting files from the package are copied to the output directory.
        - Additional configuration files such as CLASS, hydrology, and run options are saved as .ini files.
        """
        # Make the final output directory absolute
        output_dir = os.path.abspath(output_dir)
        # Check if the output directory exists, if not, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # create the final results directory
        os.makedirs(os.path.join(output_dir, 'results'),
            exist_ok=True)

        # saving drainage database
        ddb_file = 'MESH_drainage_database.nc'
        # printing drainage database netcdf file
        self.ddb.to_netcdf(os.path.join(output_dir, ddb_file))

        # saving forcing files
        if self.settings['core']['forcing_files'] == 'single':
            forcing_file = 'MESH_forcing.nc'
            # necessary operations on the xarray.Dataset self.focring object
            self._modify_forcing_encodings()
            # printing forcing netcdf file
            self.forcing.to_netcdf(
                os.path.join(output_dir, forcing_file),
                format='NETCDF4_CLASSIC',
                unlimited_dims=['time'])

        else:
            # meaning multiple forcing files are used, and they have already been saved
            # during the initialization of the forcing object, so just creating a list
            # files and saving it under save_path/forcing_files_list.txt
            forcing_file = 'forcing_files_list.txt'
            # creating a list of forcing files
            forcing_files = sorted(glob.glob(os.path.join(output_dir, 'forcings', '*.nc*')))
            # writing the forcing files to a text file
            with open(os.path.join(output_dir, forcing_file), 'w') as f:
                for file in forcing_files:
                    f.write(f"{file}\n")

        # copy crude setting files that are not automated YET
        pkgdir = sys.modules['meshflow'].__path__[0]
        setting_path = os.path.join(pkgdir, 'default_settings')
        for f in glob.glob(os.path.join(setting_path, '*')):
            if os.path.isfile(f):
                shutil.copy2(f, output_dir)

        # save the class text file
        class_file = 'MESH_parameters_CLASS.ini'
        with open(os.path.join(output_dir, class_file), 'w') as f:
            f.write(self.class_text)

        # save the hydrology text file
        hydrology_file = 'MESH_parameters_hydrology.ini'
        with open(os.path.join(output_dir, hydrology_file), 'w') as f:
            f.write(self.hydrology_text)

        # save the run options text file
        run_options_file = 'MESH_input_run_options.ini'
        with open(os.path.join(output_dir, run_options_file), 'w') as f:
            f.write(self.run_options_text)

        return

    def _init_idx_vars(self):
        """
        Initiating MESH variables such as `rank`, `next`, `main_seg`, and
        `ds_main_seg`
        """
        # extracting pandas.Series for river segments and their downstream
        # segments from the `self.riv` geopandas.geoDataFrame
        _seg = self.riv.loc[:, self.main_id]
        _ds_seg = self.riv.loc[:, self.ds_main_id]

        # assigning modified "index" variables compatible with MESH
        # requirements

        # building a dictionary as an input to `extract_rank_next`
        # function
        _kwargs = {
            'seg': _seg,
            'ds_seg': _ds_seg,
            'outlet_value': self.outlet_value,
        }

        # calling `extract_rank_next` function
        (self.rank,
         self.next,
         self.main_seg,
         self.ds_main_seg) = utility.extract_rank_next(**_kwargs)

        return

    def _reorder_riv(
        self,
        rank_name='rank',
        next_name='next'
    ) -> gpd.GeoDataFrame:

        # necessary values
        _cols = [rank_name, next_name, self.main_id]
        _data = [self.rank, self.next, self.main_seg]

        # pandas.DataFrame of `_cols` and `_data`
        _idx_df = pd.DataFrame({col: datum
                                for col, datum in
                                zip(_cols, _data)})

        # re-ordered view of `self.riv` based on the `self.main_seg`
        # that are ordered based on `self.rank`
        _riv_reindexed = self.riv.set_index(self.main_id).copy()

        # getting a copy of `self.riv` with the new order of
        # `self.main_seg`
        _riv_reordered = _riv_reindexed.loc[self.main_seg].reset_index().copy()

        # reordered elements
        _reordered_riv = gpd.GeoDataFrame(pd.merge(left=_riv_reordered,
                                                   right=_idx_df,
                                                   on=self.main_id)
                                          )

        return _reordered_riv

    def _modify_forcing_encodings(
        self,
        ds: [xr.Dataset] = None,
    ) -> xr.Dataset:
        """Necessary adhoc modifications on the forcing object's
        encoding
        """
        if ds is None:
            ds = self.forcing

        # check if the `time` variable is present
        if 'time' not in ds:
            raise ValueError("`time` variable not found in the forcing object")

        # empty encoding dictionary of the `time` variable
        ds.time.encoding = {}
        # estimate the frequency offset value
        _freq = pd.infer_freq(ds.time)
        # get the full name
        _freq_long = utility.forcing_prep.freq_long_name(_freq)

        _encoding = {
            'time': {
                'units': f'{_freq_long} since 1900-01-01 12:00:00'
            }
        }
        ds.time.encoding = _encoding

        return ds

    def _adhoc_mesh_vars(
        self,
        ds,
    ) -> xr.Dataset:
        """
        Ad-hoc manipulations on drainage database and forcing objects
        including adding `crs` variables, adding coordinates, and adding
        necessary default variable attributes
        """
        # fix coordinates of the forcing and ddb objects
        ds = xr.combine_by_coords([self._coords_ds, ds])

        # adding `crs` variable to both main Dataset groups
        # i.e., forcing and drainage database
        ds['crs'] = 1

        # adjust local attributes of common variables
        for k in default_attrs:
            for attr, desc in default_attrs[k].items():
                ds[k].attrs[attr] = desc

        # downcast datatype of coordinate variables
        for c in ds.coords:
            if np.issubdtype(ds[c], float):
                try:
                    ds[c] = pd.to_numeric(ds[c].to_numpy(), downcast='integer')
                except Exception:
                    pass

        # assure the order of `self._hru_dim` is correct
        ds = ds.loc[{self.hru_dim: self.main_seg}].copy()

        return ds

    def _return_short_gru_name(
        self,
        string: str,
    ) -> str:
        """
        Return a short name for the GRU based on the class name
        """        
        class_name_list = string.split(' ')
        # drop the non-informative words
        class_name_list = [word for word in class_name_list if len(word) > 3]
        # create an short_name with a maximum length of 34 characters
        short_name = '_'.join([word[:4] for word in class_name_list])[:34]

        return short_name

    def format_date(
        self,
        date_str,
        date_format='%Y-%m-%d',
    ) -> str:
        """
        Convert a date string to 'yyyymmdd' format, handling various formats"""
        dt = dateutil.parser.parse(date_str)

        return dt.strftime(date_format)

    def maybe_int(self, x: Any):
        assert isinstance(x, (int, float)), "Input must be an int or float"
        if isinstance(x, float) and x.is_integer():
            return int(x)
        return x

    def _progress_bar():
        return
