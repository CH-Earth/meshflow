
import cdo # binary required >2.2.1
import xarray as xr # requires xarray >2023.7.0
import pint_xarray # requires typing_extensions >4.7.1
import pint # pint >0.22

import os
from typing import (
    List,
    Sequence,
    Dict,
    Optional,
)


def mesh_forcing(
    path: str,
    variables: Sequence[str],
    output_path: str,
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
    output_path : str
        The path where the output file will be saved.
    units : Dict[str, str]
        A dictionary mapping variable names to their units.
    unit_registry : pint.UnitRegistry, optional
        A Pint unit registry for converting units, by default None.
    to_units : Dict[str, str], optional
        A dictionary mapping variable names to target units for conversion,
        by default None.
    output_file : str, optional
        The name of the output NetCDF file, by default 'MESH_forcing.nc'.
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
    
    # Create output path
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    
    # create log path
    log_dir = os.path.join(output_path, 'log')
    try:
        os.makedirs(log_dir)
    except:
        pass
    
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
    cdo_log = os.path.join(log_dir, 'merge_unitchange.log') # log file
    cdo_obj = cdo.Cdo(logging=True, logFile=cdo_log) # CDO object
    ds = cdo_obj.mergetime(input=path, returnXArray=variables) # Mergeing based on t-step
    
    # check to see if all the keys included in the `units` dictionary are
    # found inside the `variables` sequence
    for k, _ in units.items():
        if k not in ds.keys():
            raise ValueError(f"item {k} defined in `units` cannot be found" \
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

    return ds

