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
    Tuple,
    List,
)

from importlib import resources

import re
import json
import sys
import glob
import os
import shutil
import warnings

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
from . import utility

# custom type hints
try:
    from os import PathLike
except ImportError:  # <Python3.8
    from typing import Union
    PathLike = Union[str, bytes]

# constants
with open(
    resources.files("meshflow.templates").joinpath("default_process_parameters.json"),
    'r'
) as f:
    DEFAULT_PROCESS_PARAMETERS = json.load(f)
DEFAULT_RUN_OPTIONS = resources.files("meshflow.templates").joinpath("default_input_run_options.json")

class MESHWorkflow(object):
    """
    Automates the setup of MESH models through a flexible workflow.

    This class provides methods to initialize, configure, and save all
    necessary files for running a MESH hydrological model. It supports
    configuration via direct Python objects, dictionaries, or JSON files,
    and handles geospatial, landcover, and meteorological forcing data.

    Parameters
    ----------
    riv : str or PathLike
        Path to the river network shapefile or geospatial data file
        containing river segments.
    cat : str or PathLike
        Path to the catchment/subbasin shapefile or geospatial data file
        containing watershed boundaries.
    landcover : str or PathLike
        Path to the landcover CSV file containing fractional coverage of
        land classes for each catchment.
    landcover_classes : dict of {str: str}
        Mapping of landcover class codes/numbers to descriptive class names.
    forcing_files : str or PathLike, optional
        Path to meteorological forcing files (NetCDF, glob pattern, or
        directory).
    forcing_vars : dict of {str: str}, optional
        Mapping of input forcing variable names to MESH standard variable
        names.
    forcing_units : dict of {str: str}, optional
        Units of the input forcing variables.
    ddb_vars : dict of {str: str}, optional
        Mapping of drainage database variable names from input data to MESH
        standards.
    ddb_units : dict of {str: str}, optional
        Units of drainage database variables.
    main_id : str, optional
        Name of the primary identifier field in the catchment and river data.
    ds_main_id : str, optional
        Name of the downstream segment identifier field in river data.
    forcing_to_units : dict of {str: str}, optional
        Target units for forcing variables after conversion.
    forcing_local_attrs : dict of {str: str}, optional
        Local attributes to apply to forcing variables in output files.
    forcing_global_attrs : dict of {str: str}, optional
        Global attributes to apply to forcing NetCDF files.
    ddb_local_attrs : dict of {str: str}, optional
        Local attributes to apply to drainage database variables.
    ddb_global_attrs : dict of {str: str}, optional
        Global attributes to apply to drainage database NetCDF file.
    ddb_min_values : dict of {str: float}, optional
        Minimum allowable values for drainage database variables.
    ddb_to_units : dict of {str: str}, optional
        Target units for drainage database variables after conversion.
    settings : dict, optional
        Comprehensive model configuration dictionary.
    gru_dim : str, optional
        Dimension name for Group Response Units (land classes) in output
        files.
    hru_dim : str, optional
        Dimension name for Hydrologic Response Units (catchments) in
        output files.
    outlet_value : int, optional
        Sentinel value used to identify outlet/terminal segments in the
        river network.

    Attributes
    ----------
    riv : geopandas.GeoDataFrame
        River network data.
    cat : geopandas.GeoDataFrame
        Catchment/subbasin data.
    landcover : pandas.DataFrame
        Landcover fractions for each catchment.
    main_id : str
        Primary identifier field name.
    ds_main_id : str
        Downstream segment identifier field name.
    outlet_value : int
        Value used to identify outlet segments.
    forcing_vars : dict
        Mapping of input forcing variable names to MESH standard names.
    forcing_units : dict
        Units of input forcing variables.
    forcing_to_units : dict
        Target units for forcing variables.
    ddb_vars : dict
        Mapping of drainage database variable names.
    ddb_units : dict
        Units of drainage database variables.
    ddb_to_units : dict
        Target units for drainage database variables.
    ddb_min_values : dict
        Minimum allowable values for drainage database variables.
    landcover_classes : dict
        Mapping of landcover class codes/numbers to descriptive class names.
    settings : dict
        Model configuration dictionary.
    gru_dim : str
        Dimension name for Group Response Units. For MESH versions beyond
        r1860, the recommended value is 'NGRU'.
    hru_dim : str
        Dimension name for subbasins (Hydrologic Response Units). The default
        value is 'subbasin', but it can be customized.
    class_text : str
        Generated CLASS configuration text by the `init_class` method.
    hydrology_text : str
        Generated hydrology configuration text by the `init_hydrology` method.
    run_options_text : str
        Generated run options configuration text by the `init_options` method.
    ddb : xarray.Dataset
        Drainage database dataset with catchment attributes and routing
        information. It is generated by the `init_ddb` method.
    forcing : xarray.Dataset
        Forcing dataset with meteorological variables for each catchment.
        It is generated by the `init_forcing` method.

    Methods
    -------
    run(save_path=None)
        Run the workflow and prepare a MESH setup. In case of multiple
        forcing files, each file will be processed and saved during
        the workflow execution. Therefore, `save_path` cannot be None.
    init()
        Initialize the workflow by necessary variables to start the
        rest of the setup process.
    init_ddb(return_ds=False, save_path=None)
        Initialize the drainage database object.
    init_forcing(save=False, save_path=None)
        Initialize the forcing object. If `save` is True, the forcing
        files will be processed and saved to the specified `save_path`.
        For `multiple` forcing files, each file must be processed and
        saved individually, therefore, `save_path` cannot be None.
    init_class(return_text=False)
        Initialize the CLASS configuration text for MESH.
    init_hydrology(return_text=False)
        Initialize the hydrology configuration text for MESH.
    init_options(return_text=False)
        Initialize the run options configuration text for MESH.
    save(output_dir)
        Save the drainage database, forcing files, and configuration files
        to the specified output directory.
    from_dict(init_dict)
        Instantiate a MESHWorkflow object from a dictionary.
    from_json(json_str)
        Instantiate a MESHWorkflow object from a JSON string.
    from_json_file(json_file)
        Instantiate a MESHWorkflow object from a JSON file.

    See Also
    --------
    meshflow.utility : Utility functions for geospatial and data
        processing.
    """
    # main constructor
    def __init__(
        self,
        riv: PathLike, # type: ignore
        cat: PathLike, # type: ignore
        landcover: PathLike, # type: ignore
        landcover_classes: Dict[str, str],
        forcing_files: PathLike = None, # type: ignore
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
        """
        Initialize a MESHWorkflow instance for automating MESH model setup.

        Parameters
        ----------
        riv : PathLike
            Path to the river network shapefile or geospatial data file
            containing river segments.
        cat : PathLike
            Path to the catchment/subbasin shapefile or geospatial data file
            containing watershed boundaries.
        landcover : PathLike
            Path to the landcover CSV file containing fractional coverage of
            land classes for each catchment. Note that currently, only MAF
            compliant files are supported.
        landcover_classes : Dict[str, str]
            Mapping of landcover class codes/numbers to descriptive class
            names.
        forcing_files : PathLike, optional
            Path to meteorological forcing files (NetCDF, glob pattern, or
            directory).
        forcing_vars : Dict[str, str], optional
            Mapping of input forcing variable names to MESH standard variable
            names.
        forcing_units : Dict[str, str], optional
            Units of the input forcing variables.
        ddb_vars : Dict[str, str], optional
            Mapping of drainage database variable names from input data to
            MESH standards.
        ddb_units : Dict[str, str], optional
            Units of drainage database variables.
        main_id : str, optional
            Name of the primary identifier field in the catchment and river
            data.
        ds_main_id : str, optional
            Name of the downstream segment identifier field in river data.
        forcing_to_units : Dict[str, str], optional
            Target units for forcing variables after conversion.
        forcing_local_attrs : Dict[str, str], optional
            Local attributes to apply to forcing variables in output files.
        forcing_global_attrs : Dict[str, str], optional
            Global attributes to apply to forcing NetCDF files.
        ddb_local_attrs : Dict[str, str], optional
            Local attributes to apply to drainage database variables.
        ddb_global_attrs : Dict[str, str], optional
            Global attributes to apply to drainage database NetCDF file.
        ddb_min_values : Dict[str, float], optional
            Minimum allowable values for drainage database variables.
        ddb_to_units : Dict[str, str], optional
            Target units for drainage database variables after conversion.
        settings : Dict[str, str], optional
            Comprehensive model configuration dictionary. Refer to the
            documentation for more details.
        gru_dim : str, optional
            Dimension name for Group Response Units (land classes) in output
            files. Default is 'NGRU'.
        hru_dim : str, optional
            Dimension name for Hydrologic Response Units (catchments) in
            output files. Default is 'subbasin'.
        outlet_value : int, optional
            Sentinel value used to identify outlet/terminal segments in the
            river network. Default is -9999.

        Raises
        ------
        TypeError
            If any of the dictionary parameters are not of type dict.
        AssertionError
            If settings is not a dictionary.
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
        """
        Read and load necessary input files for the workflow.

        Note:
            - Currently, files are loaded eagerly during instantiation.
            - Future versions may implement lazy loading for efficiency.
            - Landcover loading is currently limited to MAF-compliant CSV files.
        """
        # reading objects
        self.riv = gpd.read_file(self._riv_path)
        self.cat = gpd.read_file(self._cat_path)
        self.landcover = self._read_landcover()

        # check if at least one outlet segment exists in `.riv` object
        if not np.any(self.riv[self.ds_main_id] == self.outlet_value):
            raise ValueError("System requires at least one outlet river"
                    "segments.")

        # limit the `river_class` values to 5 only
        river_class_name = self.ddb_vars['river_class']
        # if `river_class` numbers are more than 5, set anything more than the
        # fifth largest number, to the fifth largest number seen in 
        # the IAK variable
        if river_class_name in self.riv.columns:
            # set the minimum river class value, if provided
            if 'river_class' in self.ddb_min_values:
                self.riv[river_class_name] = self.riv[river_class_name].clip(lower=self.ddb_min_values['river_class'])
            # set the maximum river class types to 5 distinct values only
            u = np.unique(self.riv[river_class_name].to_numpy())
            if u.size > 5:
                fifth_largest_distinct = u[4]
                self.riv[river_class_name] =  self.riv[river_class_name].clip(upper=fifth_largest_distinct)

    def _read_landcover(self):
        """
        Reads and returns the landcover DataFrame for catchments present in the input domain.

        Returns
        -------
        pd.DataFrame
            Landcover fractions for each catchment, filtered to include only those
            present in the catchment file and only columns starting with 'frac_'.
        """

        # [FIXME]: This needs to be flexible in future versions
        _lc_prefix = 'frac_'

        # local _seg_ids and _ds_seg_ids
        _seg_ids = self.cat.loc[:, self.main_id]

        # read the landcover data as a pandas.DataFrame
        _lc_df = pd.read_csv(self._landcover_path,
                             index_col=0,
                             header=0)

        # if a landcover class with `9999` value is present, change it to 
        if 'frac_9999' in _lc_df.columns:
            # rename the value to frac_255 meaning no value
            _lc_df = _lc_df.rename(columns={'frac_9999': 'frac_255'})

        # downcast landcover index values to integer if possible
        # FIXME: this will need to be flexible
        if pd.api.types.is_float_dtype(_lc_df.index):
            try:
                _lc_df.index = _lc_df.index.astype('int64')  # succeeds only if all floats are whole numbers
            except ValueError:
                # some index values are not whole numbers; leave as-is or handle as needed
                warnings.warn("Landcover index values must be integer values."
                              " This will be fixed in the upcoming versions.")

        # select rows and columns to be included for landcover object
        _rows = [row for row in _lc_df.index if _seg_ids.isin([row]).any()]
        _cols = [col for col in _lc_df.columns if
                 col.startswith(_lc_prefix)]
 
        # return a copy of the dataframe including hrus available in the
        # input domain and only fractions
        return _lc_df.loc[_rows, _cols].copy()

    @property
    def coords(self):
        """
        Calculate and return centroid coordinates (latitude and longitude) for the main object.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            Tuple containing centroid latitude and longitude values.

        Notes
        -----
        Utilizes `utility.extract_centroid` to compute the centroid based on
        the object's geodataframe (`self.cat`) and its identifier (`self.main_id`).
        """
        # calculate centroid latitude and longitude values
        return utility.extract_centroid(gdf=self.cat,
                                        obj_id=self.main_id)

    @property
    def forcing_files(self):
        """
        Returns the path or glob pattern to the forcing files.

        Determines the correct path or glob pattern for NetCDF forcing files
        based on the provided forcing path. If the path matches a NetCDF file
        pattern ('.nc' or '.nc*') and files exist, returns the path directly.
        Otherwise, constructs a glob pattern for NetCDF files in the directory
        and returns the pattern if matching files are found.

        Returns
        -------
        str or None
            Path or glob pattern to the forcing files if found, otherwise None.
        """
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
        Instantiate a MESHWorkflow object from a dictionary.

        Parameters
        ----------
        init_dict : dict
            Dictionary containing initialization parameters for the
            MESHWorkflow class.

        Returns
        -------
        MESHWorkflow
            An instance of the MESHWorkflow class.

        Raises
        ------
        KeyError
            If `init_dict` is empty.
        AssertionError
            If `init_dict` is not a dictionary.

        Notes
        -----
        The keys in `init_dict` must match the arguments required by the
        MESHWorkflow constructor.
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
        Instantiate a MESHWorkflow object from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON string containing initialization parameters for the
            MESHWorkflow class.

        Returns
        -------
        MESHWorkflow
            An instance of the MESHWorkflow class.

        Notes
        -----
        The keys in the JSON string must match the arguments required by
        the MESHWorkflow constructor. Decoding uses a custom object hook
        to handle typical JSON-to-Python conversions.
        """
        decoder = json.JSONDecoder(object_hook=MESHWorkflow._json_decoder)
        json_dict = decoder.decode(json_str)
        return cls.from_dict(json_dict)

    @classmethod
    def from_json_file(
        cls: 'MESHWorkflow',
        json_file: str,
    ) -> 'MESHWorkflow':
        """
        Instantiate a MESHWorkflow object from a JSON file.

        Parameters
        ----------
        json_file : str
            Path to the JSON file containing initialization parameters
            for the MESHWorkflow class.

        Returns
        -------
        MESHWorkflow
            An instance of the MESHWorkflow class.

        Notes
        -----
        The keys in the JSON file must match the arguments required by
        the MESHWorkflow constructor. Decoding uses a custom object hook
        to handle typical JSON-to-Python conversions.
        """
        with open(json_file) as f:
            json_dict = json.load(f, object_hook=MESHWorkflow._json_decoder)

        return cls.from_dict(json_dict)

    @staticmethod
    def _env_var_decoder(s):
        """
        Decode OS environmental variables embedded in a string.

        Parameters
        ----------
        s : str
            Input string containing an environmental variable in the form
            `$VARNAME/`.

        Returns
        -------
        str
            String with the environmental variable replaced by its value,
            if found. Otherwise, returns the original string.

        Notes
        -----
        The method uses regular expressions to extract the environmental
        variable name and replace it with its value from the OS environment.
        If the variable is not found, the original string is returned.
        """
        # Regular expression patterns
        env_pat = r'\$(.*?)/'
        bef_pat = r'(.*?)\$.*?/?'
        aft_pat = r'\$.*?(/.*)'

        # Extract parts of the string
        e = re.search(env_pat, s).group(1)
        b = re.search(bef_pat, s).group(1)
        a = re.search(aft_pat, s).group(1)

        # Get environmental variable value
        v = os.getenv(e)

        # Return reconstructed string if variable found
        if v:
            return b + v + a
        return s

    @staticmethod
    def _json_decoder(obj):
        """
        Decode typical JSON strings into valid Python objects.

        Parameters
        ----------
        obj : Any
            The object to decode, typically a string or dictionary
            from a JSON structure.

        Returns
        -------
        Any
            The decoded Python object. Converts JSON booleans to Python
            bool, environment variables to their values, and integer-like
            strings to int. Recursively decodes dictionaries.

        Notes
        -----
        - Recognizes boolean strings ("true", "false") and converts them.
        - Replaces environment variable references (e.g., "$VARNAME/").
        - Converts integer-like strings to int.
        - Recursively decodes dictionaries.
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
            return {MESHWorkflow._json_decoder(k): MESHWorkflow._json_decoder(v)
                    for k, v in obj.items()}
        return obj

    @staticmethod
    def _is_valid_integer(s):
        """
        Check if a string represents a valid integer.

        Parameters
        ----------
        s : str
            Input string to check.

        Returns
        -------
        bool
            True if the string can be converted to an integer, False otherwise.

        Notes
        -----
        This method attempts to convert the input string to an integer.
        If successful, returns True. Otherwise, returns False.
        """
        try:
            int(s)
            return True
        except ValueError:
            return False

    def run(
        self,
        save_path: Optional[PathLike] = None, # type: ignore
    ) -> None:
        """
        Run the workflow and prepare a MESH model setup.

        This method initializes the drainage database and forcing objects,
        generates configuration files, and prepares all necessary files for
        running a MESH hydrological model.

        Parameters
        ----------
        save_path : str or PathLike, optional
            Path to the directory where output files will be saved. Required
            if multiple forcing files are processed.

        Raises
        ------
        ValueError
            If `save_path` is None when processing multiple forcing files.

        Notes
        -----
        - For multiple forcing files, each file is processed and saved
          during initialization.
        - For single forcing files, the forcing object is created but not
          saved automatically.
        - Generates CLASS, hydrology, and run options configuration files.
        """
        # Initialize drainage database and forcing objects
        self.init()

        # Generate drainage database
        self.init_ddb()  # creates self.ddb automatically

        # Generate forcing data
        if self.settings['core']['forcing_files'] == 'multiple':
            warnings.warn(
                "Since multiple forcing files are needed, each file will be "
                "processed and saved during initialization.",
                UserWarning,
            )
            if save_path is None:
                raise ValueError(
                    "`save_path` cannot be None when processing multiple "
                    "forcing files."
                )
            # Hard-coded 'forcings' path in the `save_path` directory
            self.init_forcing(
                save=True,
                save_path=os.path.join(save_path, 'forcings')
            )
        else:
            self.init_forcing(save=False)  # creates self.forcing automatically

        # Generate land-cover dependent setting files for MESH
        self.class_dict = self.init_class(return_dict=True)

        # Generate hydrology setting files for MESH
        self.hydrology_dict = self.init_hydrology(return_dict=True)

        # Generate run options for MESH
        self.run_options_dict = self.init_options(return_dict=True)

        # Check for included processes to customize parameters
        included_processes = self.check_process_parameters(
            options_dict=self.options_dict,
            return_processes=True)

        # Render configuration texts for the MESH instance
        self.class_text, self.hydrology_text, self.run_options_text = self.render_configs(
            class_dicts=self.class_dict,
            hydrology_dicts=self.hydrology_dict,
            options_dict=self.run_options_dict,
            process_details=included_processes,
            return_texts=True,
        )

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
    ) -> None:
        """
        Initialize the drainage database object for the workflow.

        Parameters
        ----------
        return_ds : bool, optional
            If True, return the drainage database as an xarray.Dataset.
            If False (default), assign to self.ddb and return None.

        Returns
        -------
        None or xarray.Dataset
            If `return_ds` is True, returns the drainage database as an
            xarray.Dataset. Otherwise, assigns to self.ddb and returns None.

        Raises
        ------
        ValueError
            If `ddb_vars` is empty.

        Notes
        -----
        - Renames input variables to MESH standard names and ensures required
          variables are present.
        - Generates the drainage database using geospatial and landcover data.
        - Performs ad-hoc manipulations and assigns units to GridArea.
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
            ddb_to_units=ddb_to_units_renamed
            )

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
        save_path: Optional[PathLike] = None, # type: ignore
    ) -> Optional[xr.Dataset]:
        """
        Initialize the forcing object for the MESH workflow.

        Parameters
        ----------
        save : bool, optional
            If True, save the processed forcing files to disk. Default is False.
        save_path : str or PathLike, optional
            Directory path where forcing files will be saved if `save` is True.
            If not provided, files are not saved.

        Returns
        -------
        xarray.Dataset or None
            Returns the processed forcing dataset if not saving to disk.
            Otherwise, returns None after saving files. In case of `multiple`
            files, each is processed and saved individually, and nothing is
            returned.

        Raises
        ------
        AssertionError
            If `save` is not a boolean or `save_path` is not a string or PathLike.
        ValueError
            If required inputs (`forcing_files`, `forcing_vars`, `forcing_units`,
            `forcing_to_units`) are missing, or if `save` is True and `save_path`
            is None.

        Notes
        -----
        - Handles both single and multiple forcing files based on settings.
        - Performs unit conversion and applies attributes using Pint registry.
        - For multiple files, each is processed and saved individually.
        - For single file, returns the xarray.Dataset unless saving is requested.
        - Artificial outlet segments are handled for forcing variables if present.
        - Time encoding is modified for compatibility with MESH (>r1860).
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
                    aggregate=False,
                    local_attrs=self.forcing_local_attrs,
                    global_attrs=self.forcing_global_attrs,
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
                    # try saving and setting the `time` dimension as
                    # unlimited
                    try:
                        ds.to_netcdf(os.path.join(save_path, file_name),
                                format='NETCDF4_CLASSIC',
                                unlimited_dims=['time'])
                    except RuntimeError:
                        # if a RuntimeError occured during this step,
                        # it is probably related to the setting unlimited
                        # size for the time dimension, so ignoring it.
                        ds.to_netcdf(os.path.join(save_path, file_name),
                                format='NETCDF4_CLASSIC')
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
                    aggregate=True,
                    local_attrs=self.forcing_local_attrs,
                    global_attrs=self.forcing_global_attrs,
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
        return_dict: bool = False,
    ) -> dict | None:
        """
        Generate the CLASS configuration text for the MESH model.

        Parameters
        ----------
        return_dict : bool, optional
            If True, returns the generated CLASS configuration dictionary.
            If False (default), assigns the dictionary to `self.class_dict` and returns
            None.

        Returns
        -------
        dict or None
            The CLASS configuration dictionary if `return_dict` is True, otherwise None.

        Raises
        ------
        KeyError
            If `measurement_heights` is missing in `class_params` or required
            keys are missing in `measurement_heights`. This is an important
            requirement for the CLASS configuration.
        ValueError
            If `specific_humidity` and `air_temperature` measurement heights are
            not equal. MESH currently requires these heights to be the same.

        Notes
        -----
        - Calculates centroid coordinates for the model domain.
        - Extracts subbasin and landcover class counts.
        - Builds CLASS configuration dictionaries for case, info, and GRUs.
        - Updates GRU dictionary with user-provided class assignments if present.
        - Renders the CLASS configuration text using a template utility.
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
            for gru, _class_dict in self.settings['class_params']['grus'].items():

                # check if the gru is in the class_gru dictionary
                if gru in class_gru:
                    # if only a string is provided—must be the class name only
                    if isinstance(_class_dict, str):
                        class_gru[gru]['class'] = _class_dict
                    elif isinstance(_class_dict, dict) and 'class' in _class_dict:
                        # update the class_gru dictionary with user inputs
                        class_gru[gru]['class'] = _class_dict['class']
                        # adding whatever is in `_class_dict` to the class_gru
                        # dictionary, except for the 'class' key
                        for key, value in _class_dict.items():
                            if key.lower() != 'class':
                                class_gru[gru][key.lower()] = value
                else:
                    warnings.warn(f"GRU {gru} not found in landcover classes. Skipping...")

        # if return is requested, return the class dictionary
        if return_dict:
            return {
                'class_case': class_case,
                'class_info': class_info,
                'class_grus': class_gru,
            }

        return

    def init_hydrology(
        self,
        return_dict: bool = False,
    ) -> dict | None:
        """
        Generate the hydrology configuration text for the MESH model.

        Parameters
        ----------
        return_dict : bool, optional
            If True, returns the generated hydrology configuration dictionary.
            If False (default), assigns the dictionary to `self.hydrology_dict`
            and returns None.

        Returns
        -------
        dict or None
            The hydrology configuration dictionary if `return_dict` is True,
            otherwise None.

        Raises
        ------
        ValueError
            If the number of river classes exceeds 5.

        Notes
        -----
        - Builds routing and GRU-dependent hydrology dictionaries.
        - Updates dictionaries with user-provided parameters if available.
        - Renders hydrology configuration text using a template utility.
        """
        # build the routing dictionary
        # if order is provided, use it
        if hasattr(self, 'ddb_vars'):
            if 'river_class' in self.ddb_vars:
                river_class_name = self.ddb_vars['river_class']
                # extract the number of river classes
                river_classes = len(set(self.riv[river_class_name].values))
                if river_classes > 5:
                    raise ValueError(
                        "`river_class` cannot have more than 5 classes. "
                        "Adjust the `ddb_vars`'s `river_class` value."
                    )

        # if river_classes variable exist
        if 'river_classes' in locals():
            routing_dict = [dict() for i in range(river_classes)]
        else:
            routing_dict = [{}]

        # build the GRU-dependent hydrology part
        hydrology_dict = {
            int(k.replace('frac_', '')): {} for k in self.landcover.columns
        }

        # If user provided more information, update these dictionaries
        if 'hydrology_params' in self.settings:
            # update the routing dictionary, assuming user has taken
            # care of the order of classes
            if (
                'routing' in self.settings['hydrology_params'] and
                len(self.settings['hydrology_params']['routing']) > 0
            ):
                for iak_class, r_dict in enumerate(self.settings['hydrology_params']['routing']):
                    # check if the class is in the routing_dict
                    if iak_class < len(routing_dict):
                        # assure that the params keys are lower-cased
                        r_dict = {k.lower(): v for k, v in r_dict.items()}
                        # update the routing dictionary with user inputs
                        routing_dict[iak_class].update(r_dict)
                    else:
                        warnings.warn(
                            f"Routing class {iak_class} not found in routing_dict. Skipping...",
                            UserWarning,
                        )

            # update the hydrology dictionary
            if 'hydrology' in self.settings['hydrology_params']:
                for gru, params in self.settings['hydrology_params']['hydrology'].items():
                    # check if gru is in the hydrology_dict
                    if gru in hydrology_dict:
                        # assure that the params keys are lower-cased
                        params = {k.lower(): v for k, v in params.items()}
                        # update the hydrology dictionary with user inputs
                        hydrology_dict[gru].update(params)
                    else:
                        warnings.warn(
                            f"GRU {gru} not found in landcover classes. Skipping...",
                            UserWarning,
                        )

        # if return is requested, return the hydrology dictionary
        if return_dict:
            return {
                'routing': routing_dict,
                'hydrology': hydrology_dict,
            }

        return

    def init_options(
        self,
        default_options: PathLike = DEFAULT_RUN_OPTIONS, # type: ignore
        return_dict: bool = False,
    ) -> dict | None:
        """
        Generate the MESH run options configuration text.

        Parameters
        ----------
        default_options : PathLike, optional
            Path to the default run options JSON file. If not provided,
            uses the built-in default.
        return_dict : bool, optional
            If True, returns the generated run options configuration text as a
            dictionary. If False (default), assigns the dictionary to `self.options_dict`
            and returns None.

        Returns
        -------
        dict or None
            The run options configuration text if `return_text` is True,
            otherwise None.

        Notes
        -----
        - Builds the options dictionary for MESH run configuration, including
          flags, output settings, and simulation dates.
        - Handles both single and multiple forcing file scenarios.
        - Automatically detects and sets time zones if not provided in
          settings, using the centroid of the catchment area.
        - Calculates time difference between forcing and model time zones.
        - Renders the options text using a template utility.
        - The user's custom settings in `self.settings['run_options']`
          will override the automatically generated options.
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
                "No `model_time_zone` provided in the settings.core. "
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
                        "hf": 60, # FIXME: hardcoded value, need to be changed
                        "time_shift": time_diff,
                        "fname": forcing_name,
                    },
                    "FORCINGLIST": forcing_list,
                },
                "etc": {
                    "PBSMFLAG": "off",
                    "TIMESTEPFLAG": 60, # FIXME: hardcoded value, need to be changed
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

        # also try to look at the default options and update them
        # to provide the full picture to the rendering engine
        with open(default_options, 'r') as file:
            default_options = json.load(file)
        default_options['settings'].update(utility.templating.deep_merge(default_options['settings'], options_dict))

        # update the built options with that of user input
        if 'run_options' in self.settings and len(self.settings['run_options']) > 0:
            # deep update the options_dict with user inputs
            # this updates `options_dict` in place
            self._recursive_update(default_options['settings'], self.settings['run_options'])

        # making the object available to the instance
        self.options_dict = default_options

        if return_dict:
            return self.options_dict

        return

    def check_process_parameters(
        self,
        options_dict: Dict[str, Any],
        process_details: Dict[str, Any] = DEFAULT_PROCESS_PARAMETERS,
        return_processes: bool = False,
    ) -> Dict[str, List] | None:
        """
        Check and process run options parameters for consistency and correctness.

        Parameters
        ----------
        options_dict : dict
            Dictionary containing run options configuration parameters.
        process_details : dict, optional
            Dictionary defining process details, including necessary hydrology
            and routing parameters for each process. Default is `DEFAULT_PROCESS_PARAMETERS`.
        return_processes : bool, optional
            If True, returns the extracted process parameters as a dictionary.
            If False (default), assigns the parameters to instance attributes
            and returns None.

        Raises
        ------
        ValueError
            If any required keys are missing in the `options_dict`.

        Notes
        -----
        - Validates the presence of essential keys in the run options dictionary.
        - Ensures that the configuration is complete before rendering.
        """
        required_keys = ['flags', 'outputs', 'dates']
        for key in required_keys:
            if key not in options_dict['settings']:
                raise ValueError(f"Missing required key '{key}' in options_dict")

        # FIXME: this is an experimental step to check the involved processes
        #        and include necessary parameters for each process. This will
        #        be further developed to be a coherent mechanism.

        # present processes to be put into a dictionary and then passed to the
        # hydrology file; important processes to check are:
        # flags:
        #   1. RUNMODE
        #   2. BASEFLOWFLAG
        #   3. to be continued...

        # extract process definition values from the options_dict
        runmode = options_dict['settings']['flags']['etc'].get('RUNMODE').lower()
        baseflowflag = options_dict['settings']['flags']['etc'].get('BASEFLOWFLAG').lower()

        processes = {
            'runmode': runmode,
            'baseflowflag': baseflowflag
        }

        # extract the relevant `hydrology` and `routing` dictionaries
        # and combine them to be further passed to the text rendering engines
        # first define necessary empty sequences
        self.hydrology_prcess_params = []
        self.routing_process_params = []

        # now iterate over available processes and extract what parameters
        # need to be included in the hydrology and routing dictionaries
        for process_name, process_type in processes.items():
            if process_name in process_details:
                # extract the process parameters
                self.hydrology_prcess_params += process_details[process_name][process_type]['hydrology']
                self.routing_process_params += process_details[process_name][process_type]['routing']

        # we also need, certain minimal parameters that are necessary
        # these parameters are in the "_necessary" key of the process_details
        self.hydrology_prcess_params += process_details['_necessary']['hydrology']
        self.routing_process_params += process_details['_necessary']['routing']

        # make sure everything in both lists are in lower case
        self.hydrology_prcess_params = [param.lower() for param in self.hydrology_prcess_params]
        self.routing_process_params = [param.lower() for param in self.routing_process_params]

        if return_processes:
            return {
                'hydrology': self.hydrology_prcess_params,
                'routing': self.routing_process_params
            }

        return

    def render_configs(
        self,
        class_dicts: Dict[str, Dict[str, Any]],
        hydrology_dicts: Dict[str, Dict[str, Any]],
        options_dict: Dict[str, Any],
        process_details: Optional[Dict[str, List]] = None,
        return_texts: Optional[bool] = False,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Render configuration texts for CLASS, hydrology, and run options.

        Parameters
        ----------
        class_dicts : dict of dict
            Dictionary containing CLASS configuration parameters
            that are stored in three distinct dictionaries, including
            `class_info`, `class_case`, and `class_grus` (key names).
        hydrology_dicts : dict of dict
            Dictionary containing hydrology configuration parameters
            stored in two distinct dictionaries, including 
            `hydrology_info` and `hydrology_case` (key names).
        options_dict : dict
            Dictionary containing run options configuration parameters.
        process_details : dict, optional
            Dictionary defining process details, including necessary hydrology
            and routing parameters for each process. If provided, it will be
            used to check and extract necessary parameters for hydrology and
            routing configurations.
        return_texts : bool, optional
            If True, return the rendered configuration texts as strings
            instead of writing to files.

        Returns
        -------
        Tuple[str, str, str]
            A tuple containing the rendered configuration texts for CLASS,
            hydrology, and run options.

        Notes
        -----
        - Utilizes utility functions to render configuration texts based on
          provided dictionaries created via `init_class`, `init_hydrology`,
          and `init_options` methods.
        - Users can directly use this function if the configuration dictionaries
          are created externally.
        """
        # check whether the data dictionaries are provided
        if not isinstance(class_dicts, dict) or \
            not isinstance(hydrology_dicts, dict) or \
                not isinstance(options_dict, dict):
            raise ValueError("`class_dicts`, `hydrology_dicts`, and `options_dict` "
                             "must be dictionaries")

        # render CLASS configuration text
        self.class_text = utility.render_class_template(
            class_info=class_dicts.get('class_info'),
            class_case=class_dicts.get('class_case'),
            class_grus=class_dicts.get('class_grus')
        )

        # render run options configuration text
        self.options_text = utility.render_run_options_template(options_dict)

        # render hydrology configuration text
        # check if the process_details is provided
        self.hydrology_text = utility.render_hydrology_template(
            routing_params=hydrology_dicts.get('routing'),
            hydrology_params=hydrology_dicts.get('hydrology'),
            process_details=process_details
        )

        if return_texts:
            return self.class_text, self.hydrology_text, self.options_text

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
            f.write(self.options_text)

        return

    def _init_idx_vars(self):
        """
        Initialize MESH index variables: `rank`, `next`, `main_seg`, and
        `ds_main_seg`.

        Notes
        -----
        Extracts river segment and downstream segment indices from the river
        GeoDataFrame. Calls `utility.extract_rank_next` to compute and assign
        the workflow's routing variables for MESH compatibility.
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
        ds: xr.Dataset = None,
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

    def _recursive_update(
        self,
        base_dict: dict,
        update_with: dict
    ) -> dict:
        """
        Recursively update a nested dictionary with another dictionary.

        Parameters
        ----------
        base_dict : dict
            The dictionary to be updated in-place.
        update_with : dict
            The dictionary whose keys and values are used to update ``base_dict``.
            Nested dictionaries trigger recursive updates; non-dictionary values
            overwrite existing entries.

        Returns
        -------
        dict
            The updated ``base_dict`` dictionary after applying all changes from
            ``update_with``.

        Examples
        --------
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> patch = {"b": {"c": 20}, "e": 5}
        >>> _recursive_update(base, patch)
        {'a': 1, 'b': {'c': 20, 'd': 3}, 'e': 5}
        """
        for key, value in update_with.items():
            # Check if the key exists in the base dictionary and if both
            # the base value and the new value are dictionaries.
            if (key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):

                # If both are dicts, recurse deeper
                self._recursive_update(base_dict[key], value)
            else:
                # Otherwise, overwrite the value (or add it if it didn't exist)
                if key in base_dict:
                    base_dict[key] = value
                else:
                    warnings.warn(
                        f"Key '{key}' not found in the target dictionary; adding it.",
                        UserWarning,
                    )
                    base_dict[key] = value

        return base_dict
