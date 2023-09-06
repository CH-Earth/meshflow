"""
containing modules to set up MESH models
"""

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

from hydrant.topology import river_graph

from typing import (
    Iterable,
    Tuple,
    Dict,
)
import re


def extract_rank_next(
    seg: Iterable,
    ds_seg: Iterable,
    outlet_value: int = -9999,
) -> Tuple[np.ndarray, ...]:
    '''Producing rank_var and next_var variables needed for 
    MESH modelling
    
    Parameters
    ----------
    seg : array-like or list-like
        The ordered list (or array) of segment IDs corresponding
        to river reaches presented in an area of interest
    ds_seg : array-like or list-like
        The ordered list (or array) of downstream segment IDs
        corresponding to those of `seg_id` elements
    outlet_value : int, [defaults to -9999]
        The outlet value assigned to `to_segment` indicating
        sinking from the system
    
    
    Returns
    -------
    rank_var: numpy.array of int
        The 'rank_var' of each segment ID produced based on
        MESH modelling standards
    next_var: numpy.array of int
        The 'next_var' variable indicating the downstream segment
        of river reaches corresponding to 'rank_var'
    seg_id: numpy.array of int
        The 'seg_id' that has been reordered to match values of
        `rank_var` and `next_var`.
    to_segment: numpy.array of int
        The 'to_segment' that has been reordered to match values
        of `rank_var` and `next_var`.
    
    
    Notes
    -----
    The function is mainly developed by Dr. Ala Bahrami at
    <ala.bahrami@usask.ca> and Cooper Albano <cooper.albano@usask.ca>
    as part of the North American MESH model workflow development.
    Minor changes have been implemented by Kasra Keshavarz
    <kasra.keshavarz1@ucalgary.ca>.

    The original workflow is located at the following link:
    https://github.com/MESH-Model/MESH-Scripts
    <last accessed on August 29th, 2023>
    '''

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


def _set_min_values(
    ds: xr.Dataset,
    min_values: Dict[str, float],
) -> xr.Dataset:
    '''Setting minimum values (values of the `min_values` 
    dictionary) for different variables (keys of the 
    `min_values` dictionary).
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset object of interest
    min_values : dict
        a dictionary with keys corresponding to any of the `ds`
        variables, and values corresponding to the minimums

    Returns
    -------
    ds : xarray.Dataset
        Dataset with minimum values set
    '''
    # make a copy of the dataset
    ds = ds.copy()

    # set the minimum values
    for var, val in min_values.items():
        ds[var] = xr.where(ds[var] <= val, val, ds[var])

    return ds


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


def _prepare_geodataframe_mesh(
    geodf: gpd.GeoDataFrame,
    cat_dim: str,
    geometry_dim: str ='geometry',
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
    include_vars: Dict[str, str],
    attr_local: Dict[str, Dict[str, str]],
    attr_global: Dict[str, str],
    min_values: Dict[str, float] = None,
    fill_na: Dict[str, float] = None,
    ordered_dims: Dict[str, Iterable] = None,
    ddb_units: Dict[str, str] = None,
    ddb_to_units: Dict[str, str] = None,
) -> xr.Dataset:
    '''Prepares the drainage database (ddb) of the MESH model.
    The function implements a set of ad-hoc manipulations on
    the river network and catchment geospatial data to prepare
    the ddb.

    Parameters
    ----------
    riv : str of ESRI Shapefile path, or list of paths, or geopandas.GeoDataFrame
        The path to the ESRI Shapefile of the geospatial object
        containing LineString(s) of the river network
    cat : str of ESRI Shapefile path, or list of paths, or geopandas.GeoDataFrame
        The path to the ESRI Shapefile of the geospatial object
        containing Polygon(s) and MultiPolygon(s) of the catchments
    landcover : pandas.DataFrame
        Dataframe of land use class fractions (i.e., between 0 and
        1) seen in each element of `cat`. Each column corresponds to
        the fraction of each land cover class and each row corres-
        ponds to each element of `cat`
    coords : pandas.DataFrame
        Dataframe of the `cat`'s centroid coordinates, including a
        column for the latitude and another for the longitude
    cat_dim : str
        The dimension name of the catchments available in `cat`,
        `riv`, `landcover`, and `coords`
    gru_dim : str
        The dimension name corresponding to the landcover classes
        after necessary manipulations on `landcover`
    include_vars : dict
        The keys correspond to the variables of `riv`, `cat`, or
        `landcover` to be included in the returned xarray.Dataset,
        and the values correspond to the rename values
    attr_local : dict
        The keys correspond to the renamed value of `include_vars`
        and the values consist of a dictionary with keys as attribute
        names and values of attribute description
    attr_global : dict
        The keys correspond to the global attribute names and values
        of attribute description
    outlet_value : int, optional [defaults to -9999]
        The outlet value for the 'nextdownid' variable shown in
        the `cols` dictionary indicating sinking from the system
    min_values : dict, optional
        Set the minimum values corresponding to the variables indicated
        as dictionary keys. Of course, the keys must have been included
        as values of the `include_vars` dictionary
    fill_na : dict, optional
        Fill the numpy.nan values found in each xarray.DataArray of
        `ddb`
    ordered_dims : Dict[str, array-like], optional
        Sorting dimensions of the `ddb` taken as keys of the `ordered_dims`
        and ordered values of each as their corresponding values

    Returns
    -------
    ddb : xarray.Dataset
        drainage database in an xarray.Dataset object with necessary information

    '''
    # define necessary variables
    geometry_var = 'geometry'  # geopandas.GeoDataFrame geometry column name
    landcover_var = 'landcover'  # landcover variable name in the object
    dummy_col = 9999  # MESH's dummy landcover class dimension/column name
    dummy_col_value = 0  # MESH's dummy landcover class value

    # check `riv` and `cat` dtypes
    riv = _check_geo_obj_dtype(riv)
    cat = _check_geo_obj_dtype(cat)

    # check `landcover` and `coords` dtypes
    if not isinstance(landcover, pd.DataFrame):
        raise TypeError("`landcover` and `coords` must be of type "
                        "pandas.DataFrame")

    # make actual xarray.Dataset objects for `riv` and `cat`
    riv_ds, cat_ds = (_prepare_geodataframe_mesh(geodf,
                                                 cat_dim=cat_dim,
                                                 geometry_dim=geometry_var,
                                                 )
                      for geodf in [riv, cat])

    # make xarray.Dataset object for `landcover`
    landcover_ds = _prepare_landcover_mesh(landcover,
                                           cat_dim=cat_dim,
                                           gru_dim=gru_dim,
                                           landcover_ds_name=landcover_var,
                                           dummy_col_name=dummy_col,
                                           dummy_col_value=dummy_col_value)

    # making a list of all xarray.Dataset objects
    ds_list = [riv_ds,
               cat_ds,
               landcover_ds]

    # merging all xarray objects into one
    ddb = xr.combine_by_coords(ds_list, compat='override')

    # only select `include_vars` variable(s) while renaming
    # based on the _values_ (NOT KEYS) of the dictionary
    ddb = ddb.rename_vars(name_dict=include_vars)
    ddb = ddb[list(include_vars.values())]

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
        ddb = ddb.pint.to(units=ddb_to_units)

    # print the netCDF file
    ddb = ddb.pint.dequantify()

    # fill NA values for `ddb`
    if fill_na is not None:
        ddb = _fill_na_ds(ds=ddb,
                          na_values=fill_na)

    # assinging minimum values for `ddb`
    if min_values is not None:
        ddb = _set_min_values(ds=ddb,
                              min_values=min_values)

    # sort dimensions of `ddb`
    if ordered_dims is not None:
        ddb = ddb.reindex(indexers=ordered_dims,
                          copy=True)

    # assigning local attributes for each variable
    if attr_local:
        for var, val in attr_local.items():
            for attr, desc in val.items():
                ddb[var].attrs[attr] = desc

    # assigning global attributes for `ddb`
    if attr_global:
        for attr, desc in attr_global.items():
            ddb.attrs[attr] = desc

    # rename ddb dimension
    ddb = ddb.rename({cat_dim: 'subbasin'})

    # change the value of `gru` dimension
    ddb['gru'] = range(0, len(ddb['gru']))

    return ddb
