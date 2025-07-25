"""
containing modules to set up MESH models
"""

# third-party libraries
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

from hydrant.topology import river_graph

# built-in libraries
from typing import (
    Sequence,
    Iterable,
    Tuple,
    Dict,
)
import re

# internal imports
from .geom import _calculate_polygon_areas


def extract_rank_next(
    seg: Iterable,
    ds_seg: Iterable,
    outlet_value: int = -9999,
) -> Tuple[np.ndarray, ...]:
    """
    Generate MESH-compatible rank and next variables for river segments.

    Parameters
    ----------
    seg : array-like or list-like
        Ordered segment IDs for river reaches in the area of interest.
    ds_seg : array-like or list-like
        Ordered downstream segment IDs corresponding to `seg` elements.
    outlet_value : int, optional
        Value assigned to `to_segment` indicating outlet/sink from the
        system. Default is -9999.

    Returns
    -------
    rank_var : numpy.ndarray
        Rank of each segment ID, following MESH modelling standards.
    next_var : numpy.ndarray
        Downstream segment index for each river reach, matching `rank_var`.
    seg_id : numpy.ndarray
        Segment IDs reordered to match `rank_var` and `next_var`.
    to_segment : numpy.ndarray
        Downstream segment IDs reordered to match `rank_var` and `next_var`.

    Notes
    -----
    Developed by Dr. Ala Bahrami and Cooper Albano for North American MESH
    model workflows. Minor changes by Kasra Keshavarz.

    Original workflow:
        https://github.com/MESH-Model/MESH-Scripts
    """

    # extracting numpy array out of input iterables
    seg_arr = np.array(seg)
    ds_seg_arr = np.array(ds_seg)

    # re-order ids to match MESH's requirements
    seg_id, to_segment = _adjust_ids(seg_arr, ds_seg_arr)

    # Count the number of outlets
    outlets = np.where(to_segment == outlet_value)[0]

    # Search over to extract the subbasins drain into each outlet
    rank_var_id_domain = np.array([]).astype(int)
    outlet_number = np.array([]).astype(int)

    for k in range(len(outlets)):
        # initial step
        seg_id_target = seg_id[outlets[k]]
        # set the rank_var of the outlet
        rank_var_id = outlets[k]

        # find upstream seg_ids draining into the chosen outlet [indexed `k`]
        while (np.size(seg_id_target) >= 1):
            if (np.size(seg_id_target) == 1):
                r = np.where(to_segment == seg_id_target)[0]
            else:
                r = np.where(to_segment == seg_id_target[0])[0]
            # updated the target seg_id
            seg_id_target = np.append(seg_id_target, seg_id[r])
            # remove the first searched target
            seg_id_target = np.delete(seg_id_target, 0, 0)
            if (len(seg_id_target) == 0):
                break
            # update the rank_var_id
            rank_var_id = np.append(rank_var_id, r)
        rank_var_id = np.flip(rank_var_id)
        if (np.size(rank_var_id) > 1):
            outlet_number = np.append(outlet_number,
                                      (k)*np.ones((len(rank_var_id), 1)).astype(int))
        else:
            outlet_number = np.append(outlet_number, (k))
        rank_var_id_domain = np.append(rank_var_id_domain, rank_var_id)
        rank_var_id = []

    # reorder seg_id and to_segment
    seg_id = seg_id[rank_var_id_domain]
    to_segment = to_segment[rank_var_id_domain]

    # rearrange outlets to be consistent with MESH outlet structure
    # In MESH outlets should be placed at the end of the `NEXT` variable
    na = len(rank_var_id_domain)
    fid1 = np.where(to_segment != outlet_value)[0]
    fid2 = np.where(to_segment == outlet_value)[0]
    fid = np.append(fid1, fid2)

    rank_var_id_domain = rank_var_id_domain[fid]
    seg_id = seg_id[fid]
    to_segment = to_segment[fid]
    outlet_number = outlet_number[fid]

    # construct rank_var and next_var variables
    next_var = np.zeros(na).astype(np.int32)

    for k in range(na):
        if (to_segment[k] != outlet_value):
            r = np.where(to_segment[k] == seg_id)[0] + 1
            next_var[k] = r
        else:
            next_var[k] = 0

    # Construct rank_var from 1:na
    rank_var = np.arange(1, na+1).astype(np.int32)

    return rank_var, next_var, seg_id, to_segment

def _check_geo_obj_dtype(obj):
    # check `obj`'s dtype
    # if it is a path string
    if isinstance(obj, str):
        obj = gpd.read_file(obj)
    # if it is a geopandas.GeoDataFrame
    elif isinstance(obj, gpd.GeoDataFrame):
        pass
    # if it is a list of str paths
    elif isinstance(obj[0], str):
        obj = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in obj]))
    else:
        raise TypeError("The type of the `riv` could be a string path, "
                        "a list of string paths, or a geopandas.GeoDataframe "
                        "object")

    # return a copy of the GeoDataFrame object
    return obj.copy()

def _adjust_ids(
    seg_id: np.ndarray,
    ds_seg_id: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [Temporary solution]: readjusts segments (and therefore downstream
    segments) to allow all segments to be in between the nodes of the
    longest branch found in a given river network
    """
    # function limited names for the segments and downstream
    # river segments
    main_id_str = 'main_id'
    ds_main_id_str = 'ds_main_id'

    # creating a pandas DataFrame of `seg_id` and `ds_seg_id`
    riv_df = pd.concat([pd.Series(arr) for arr in [seg_id, ds_seg_id]],
                       names=[main_id_str, ds_main_id_str],
                       axis=1)
    # naming columns, in case needed
    riv_df.columns = [main_id_str, ds_main_id_str]

    # extracting the longest branch out of "hydrant"
    longest_branch = river_graph.longest_branch(
                                    riv=riv_df,
                                    main_id=main_id_str,
                                    ds_main_id=ds_main_id_str)

    # selecting first and last (while not counting the outlet value, so -2)
    first_node = longest_branch[0]
    last_node = longest_branch[-2]

    # extracting the index of first and last nodes
    first_idx = riv_df.index[riv_df[main_id_str] == first_node]
    last_idx = riv_df.index[riv_df[main_id_str] == last_node]

    # building new `riv_df` with new index values
    idx = pd.Index(first_idx.to_list() + riv_df.index.to_list())
    idx = idx.drop_duplicates(keep='first')
    idx = pd.Index(idx.to_list() + last_idx.to_list())
    idx = idx.drop_duplicates(keep='last')
    riv_df = riv_df.loc[idx]

    # reseting index, just to reassure
    riv_df.reset_index(drop=True, inplace=True)

    # extracting np.ndarrays
    new_seg_id = riv_df[main_id_str].to_numpy()
    new_ds_seg_id = riv_df[ds_main_id_str].to_numpy()

    return new_seg_id, new_ds_seg_id

def _prepare_landcover_mesh(
    landcover: pd.DataFrame,
    cat_dim: str,
    gru_dim: str,
    landcover_ds_name: str = 'landcover',
    dummy_col_name: int = None,
    dummy_col_value: int = 0,
) -> xr.Dataset:
    '''Implements necessary landcover manipulations by:
    1) removing columns with no contribution (sum=0),
    2) assigning `cat_dim` column as index,
    3) assigning a dummy variables as needed by MESH,
    4) QA/QC on the select parameters indicated in `cols`,
    5)
    '''
    if dummy_col_name is None:
        dummy_col_name = 9999

    # remove zero-sum columns
    landcover = landcover.loc[:, landcover.sum(axis=0) > 0]

    # removing non-digit characters from columns, if any
    landcover.columns = [int(re.sub('\D', '', c)) for c in landcover.columns]

    # set column name for landcover classes
    landcover.columns.name = gru_dim

    # adding necessary MESH dummy landcover variable - Hard-coded
    landcover.insert(len(landcover.columns), dummy_col_name, dummy_col_value)

    # making an xarray.DataArray
    landcover_da = landcover.stack().to_xarray()

    # converting to xr.Dataset
    landcover_ds = landcover_da.to_dataset(name=landcover_ds_name)

    return landcover_ds

def _fill_na_ds(
    ds: xr.Dataset,
    na_values: Dict[str, float],
) -> xr.Dataset:
    '''Replacing NA values within an xarray.Dataset
    object for different DataAarrays (corresponding to
    the keys of the `na_values`) with the corresponding
    values of `na_values`.
    '''
    # make a copy of the dataset
    ds = ds.copy()

    # replacing NAs
    for var, val in na_values.items():
        ds[var] = xr.where(ds[var] == np.nan, val, ds[var])

    return ds

def _downcast_to_int(
    ds: xr.Dataset,
    int_vars: Iterable[str],
) -> xr.Dataset:
    '''Downcasts the variables of `ds` to int32 type
    if they are not already int32.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset object of interest
    int_vars : Iterable[str]
        Iterable of variable names to be downcasted to int32

    Returns
    -------
    ds : xarray.Dataset
        Dataset with specified variables downcasted to int32
    '''
    # make a copy of the dataset
    ds = ds.copy()

    # downcast to int32
    for var in int_vars:
        if var in ds and ds[var].dtype != np.int32:
            ds[var] = ds[var].astype(np.int32)

    return ds

def _prepare_geodataframe_mesh(
    geodf: gpd.GeoDataFrame,
    cat_dim: str,
    geometry_dim: str = 'geometry',
) -> xr.Dataset:
    '''Implements necessary geodf manipulations by:
    1) removing the `geometry_dim` column of the
       geopandas.GeoDataFrame object and turning it
       into a pandas.DataFrame automatically
    2) setting index of the new DataFrame to the
       `cat_dim` indicating IDs for each element of
       `geodf`
    3) returning an xarray.Dataset
    '''
    # drop the `geometry_var` column
    geodf = geodf.copy().drop(columns=geometry_dim)

    return geodf.set_index(cat_dim).to_xarray()

def prepare_mesh_ddb(
    riv: gpd.GeoDataFrame,
    cat: gpd.GeoDataFrame,
    landcover: pd.DataFrame,
    cat_dim: str,
    gru_dim: str,
    hru_dim: str,
    gru_names: Sequence[str],
    include_vars: Dict[str, str],
    attr_local: Dict[str, Dict[str, str]],
    attr_global: Dict[str, str],
    min_values: Dict[str, float] = None,
    fill_na: Dict[str, float] = None,
    ordered_dims: Dict[str, Iterable] = None,
    ddb_units: Dict[str, str] = None,
    ddb_to_units: Dict[str, str] = None,
) -> xr.Dataset:
    """
    Prepares the drainage database (ddb) for the MESH model.

    This function applies a set of manipulations to river network and catchment
    geospatial data to construct the drainage database required by MESH.

    Parameters
    ----------
    riv : str, list of str, or geopandas.GeoDataFrame
        Path(s) to ESRI Shapefile(s) or a GeoDataFrame containing river
        LineString(s).
    cat : str, list of str, or geopandas.GeoDataFrame
        Path(s) to ESRI Shapefile(s) or a GeoDataFrame containing catchment
        Polygon(s) or MultiPolygon(s).
    landcover : pandas.DataFrame
        DataFrame of land use class fractions (values between 0 and 1) for each
        catchment. Columns are land cover classes; rows are catchment elements.
    cat_dim : str
        Name of the catchment dimension present in `cat`, `riv`, and `landcover`.
    gru_dim : str
        Name of the landcover class dimension after processing `landcover`.
    hru_dim : str
        Name of the hydrological response unit (HRU) dimension for output.
    gru_names : Sequence of str
        List of landcover class names, ordered to match processed `landcover`.
    include_vars : dict
        Dictionary mapping variable names in input data to output names in ddb.
    attr_local : dict
        Dictionary mapping output variable names to their attribute dictionaries.
    attr_global : dict
        Dictionary mapping global attribute names to their descriptions.
    min_values : dict, optional
        Minimum values for variables in ddb. Keys must match output variable names.
    fill_na : dict, optional
        Values to fill for NaNs in each variable of ddb. Keys are variable names.
    ordered_dims : dict, optional
        Dictionary mapping dimension names to ordered values for sorting ddb.
    ddb_units : dict, optional
        Dictionary mapping variable names to their units for quantification.
    ddb_to_units : dict, optional
        Dictionary mapping variable names to target units for conversion.

    Returns
    -------
    ddb : xarray.Dataset
        Drainage database as an xarray.Dataset containing all required variables
        and attributes for MESH model setup.

    """
    # define necessary variables
    geometry_var = 'geometry'  # geopandas.GeoDataFrame geometry column name
    landcover_var = 'landclass'  # landcover variable name in the object

    dummy_col = 9999  # MESH's dummy landcover class dimension/column name
    dummy_col_value = 0  # MESH's dummy landcover class value

    # extract order of `cat_dim`
    cat_dim_array = riv.loc[:, cat_dim].copy().to_numpy()

    # check `riv` and `cat` dtypes
    riv = _check_geo_obj_dtype(riv)
    cat = _check_geo_obj_dtype(cat)

    # calculate catchment areas in meters squared
    grid_area = _calculate_polygon_areas(
        gdf=cat,
        target_area_unit='m ** 2',
        equal_area_crs='ESRI:54009'
    )
    grid_area['area'] = grid_area['area'].pint.magnitude

    # check `landcover` and `coords` dtypes
    if not isinstance(landcover, pd.DataFrame):
        raise TypeError("`landcover` and `coords` must be of type "
                        "pandas.DataFrame")

    # make actual xarray.Dataset objects for `riv` and `cat`
    riv_ds, cat_ds, grid_area_ds = (_prepare_geodataframe_mesh(
            geodf,
            cat_dim=cat_dim,
            geometry_dim=geometry_var)
        for geodf in [riv, cat, grid_area])

    # make xarray.Dataset object for `landcover`
    landcover_ds = _prepare_landcover_mesh(
        landcover,
        cat_dim=cat_dim,
        gru_dim=gru_dim,
        landcover_ds_name=landcover_var,
        dummy_col_name=dummy_col,
        dummy_col_value=dummy_col_value)

    # add landcover classes names present in the setup to `ddb`
    # note that the last landcover class/name is a dummy one
    _gru_list = landcover_ds[gru_dim].to_numpy()
    # creating dictionary for final merging
    lc_names = {
        "coords": {
            gru_dim: {
                "dims": gru_dim,
                "data": _gru_list,
            },
        },
        "dims": gru_dim,
        "data_vars": {
            "landclass_names": {
                "dims": gru_dim,
                "data": [gru_names[c] for c in _gru_list[0:-1]] + ['Dump']
            },
        },
    }
    lc_names_da = xr.Dataset.from_dict(lc_names)

    # making a list of all xarray.Dataset objects
    ds_list = [
        landcover_ds,
        lc_names_da,
        riv_ds,
        cat_ds,
        grid_area_ds]

    # merging all xarray objects into one
    ddb = xr.combine_by_coords(ds_list, compat='override')

    # only select `include_vars` variable(s) while renaming
    # based on the _values_ (NOT KEYS) of the dictionary
    ddb = ddb.rename_vars(name_dict=include_vars)
    ddb = ddb[list(include_vars.values())]

    # fill NA values for `ddb`
    if fill_na is not None:
        ddb = _fill_na_ds(ds=ddb,
                          na_values=fill_na)

    # assigning local attributes for each variable
    if attr_local:
        for var, val in attr_local.items():
            if var in ddb:
                for attr, desc in val.items():
                    ddb[var].attrs[attr] = desc

    # assigning global attributes for `ddb`
    if attr_global:
        for attr, desc in attr_global.items():
            ddb.attrs[attr] = desc

    # check to see if all the keys included in the `units` dictionary are
    # found inside the `variables` sequence
    for k in ddb_units:
        if k not in ddb:
            raise ValueError(f"item {k} defined in `units` cannot be found"
                             " in `variables`")

    # if all elements of `variables` not found in `units`,
    # assign them to None
    for v in include_vars:
        if v not in ddb_units:
            ddb_units[v] = None

    # now assign the units - assuming `pint`'s default unit registery
    ddb = ddb.pint.quantify(units=ddb_units)

    # if `to_units` is defined
    if ddb_to_units:
        # if dimensionless values exists, pop them
        ddb_to_units = {k: v for k, v in ddb_to_units.items()
                        if v != 'dimensionless'}
        # implement the conversion
        ddb = ddb.pint.to(units=ddb_to_units)

    # print the netCDF file
    ddb = ddb.pint.dequantify()

    # sort dimensions of `ddb`
    if ordered_dims is not None:
        ddb = ddb.reindex(indexers=ordered_dims,
                          copy=True)

    # rename ddb dimension
    ddb = ddb.rename({cat_dim: hru_dim})

    # change the value of `gru` dimension
    ddb[gru_dim] = range(0, len(ddb[gru_dim]))

    # # assinging minimum values for `ddb`
    if min_values is not None:
        # only apply to variables that are present in `ddb`
        min_values = {k: v for k, v in min_values.items()
                      if k in ddb}

        # if min_values turns to be non-empty
        if min_values:
            for var, val in min_values.items():
                ddb[var] = ddb[var].where(ddb[var] > val,
                                          val,
                                          drop=False)

    # downcast specific variables to int32 if exist
    int_vars = ['IAK', 'Rank', 'Next']
    for var in int_vars:
        if var not in ddb:
            int_vars.remove(var)

    # calling the downcast function
    ddb = _downcast_to_int(ddb, int_vars=int_vars)

    # assure the order of `hru_dim` is correct
    ddb = ddb.loc[{hru_dim: cat_dim_array}].copy()

    # rename coordinate variable names to avoid confusion with dimension
    # names
    ddb = ddb.drop_vars(gru_dim)

    return ddb
