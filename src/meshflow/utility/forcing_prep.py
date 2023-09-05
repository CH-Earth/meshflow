"""
Containing important functions for preparing "forcing" database for MESH
models
"""


import cdo  # binary required >2.2.1
import xarray as xr  # requires xarray >2023.7.0
import pint_xarray  # requires typing_extensions >4.7.1
import pint  # pint >0.22

from typing import (
    Sequence,
    Dict,
    Optional,
)


def mesh_forcing(
    path: str,
    variables: Sequence[str],
    units: Dict[str, str],
    unit_registry: pint.UnitRegistry = None,
    to_units: Optional[Dict[str, str]] = None,
) -> None:
    """Prepares a MESH forcing file.

    Parameters
    ----------
    path : str
        The path to input forcing files.
    variables : Sequence[str]
        A sequence of variable names to be included in the output file.
    units : Dict[str, str]
        A dictionary mapping variable names to their units.
    unit_registry : pint.UnitRegistry, optional
        A Pint unit registry for converting units, by default None.
    to_units : Dict[str, str], optional
        A dictionary mapping variable names to target units for conversion,
        by default None.
    return_xarray : bool, optional
        If True, return the resulting xarray.Dataset, by default False.

    Raises
    ------
    TypeError
        If `variables` is not a sequence of string values.
    ValueError
        If `units` associated with `variables` elements are not provided or if
        any variable defined in `units` cannot be found in `variables`.

    Returns
    -------
    xarray.Dataset
        Returns an xarray.Dataset containing the merged and converted data.
        Otherwise, None.

    Notes
    -----
    The function merges all the input forcing files into a single NetCDF file
    as MESH only reads one file. CDO is used for merging, but the function
    returns an xarray.Dataset.

    [FIXME]: The merge functionality could become more comprehensive in future
    versions.
    """

    # if variables dtype is not string, throw an exception
    if not isinstance(variables[0], str):
        raise TypeError("`variables` must be a sequence of string values")

    # if `units` is not provided, throw an exception
    if not units:
        raise ValueError("`units` associated with `variables` elements must"
                         " be provided")

    # merge all the input forcing files as MESH only reads on file
    # CDO is used due to its stability, however, an xarray.DataSet
    # is returned

    # [FIXME]: merge functionality could become more comprehensive in
    # future versions
    cdo_obj = cdo.Cdo()  # CDO object
    ds = cdo_obj.mergetime(input=path, returnXArray=variables)  # Mergeing

    # check to see if all the keys included in the `units` dictionary are
    # found inside the `variables` sequence
    for k in units:
        if k not in ds:
            raise ValueError(f"item {k} defined in `units` cannot be found"
                             " in `variables`")

    # if all elements of `variables` not found in `units`,
    # assign them to None
    for v in variables:
        if v not in units:
            units[v] = None

    # now assign the units
    ds = ds.pint.quantify(units=units, unit_registry=unit_registry)

    # if `to_units` is defined
    if to_units:
        ds = ds.pint.to(units=to_units)

    # print the netCDF file
    ds = ds.pint.dequantify()

    # rename easymore's output name
    var = [i for i in ds.dims.keys() if i != 'time']
    ds = ds.transpose().rename({k: 'subbasin' for k in var})

    # convert calendar to 'standard' based on MESH's input standard
    ds = ds.convert_calendar(calendar='standard',
                             dim='time',
                             align_on=None)

    # adding CRS variable
    ds['crs'] = 1

    return ds


def freq_long_name(
    freq_alias: 'str'
) -> str:
    """returning fullname of a offset alias based on pandas conventions

    Paramters
    ---------
    freq_alias: str
        Time offset alias which is usually a single character to represent
        time interval frequencies, such as 'H' for 'hours'

    Returns
    -------
    str
        fullname of the time offset
    """

    if not isinstance(freq_alias, str):
        raise TypeError(f"frequency value of \'{freq_alias}\' is not"
                        "acceptable")

    if freq_alias in ('H'):
        return 'hours'
    elif freq_alias in ('T', 'min'):
        return 'minutes'
    elif freq_alias in ('S'):
        return 'seconds'
    elif freq_alias in ('L', 'ms'):
        return 'milliseconds'
    else:
        raise ValueError(f"frequency value \'{freq_alias}\' is not"
                         "acceptable")
    return
