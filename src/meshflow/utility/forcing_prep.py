"""
Containing important functions for preparing "forcing" database for MESH
models
"""

# import third-party libraries
import cdo  # binary required >2.2.1
import xarray as xr  # requires xarray >2023.7.0
import pint_xarray  # requires typing_extensions >4.7.1
import pint  # pint >0.22

# import built-in libraries
from typing import (
    Sequence,
    Dict,
    Optional,
)

from zoneinfo import ZoneInfo
from datetime import datetime


def prepare_mesh_forcing(
    path: str,
    variables: Sequence[str],
    units: Dict[str, str],
    hru_dim: Optional[str] = None,
    unit_registry: pint.UnitRegistry = None,
    to_units: Optional[Dict[str, str]] = None,
    aggregate: bool = False,
    local_attrs: Optional[Dict[str, str]] = None,
    global_attrs: Optional[Dict[str, str]] = None,
) -> None:
    """
    Prepare a MESH forcing file by merging, converting, and annotating data.

    Parameters
    ----------
    path : str
        Path to input forcing files.
    variables : Sequence[str]
        Sequence of variable names to include in the output file.
    units : dict of str
        Dictionary mapping variable names to their units.
    hru_dim : str, optional
        Name of the HRU dimension to use in the output dataset.
    unit_registry : pint.UnitRegistry, optional
        Pint unit registry for unit conversion. Default is None.
    to_units : dict of str, optional
        Dictionary mapping variable names to target units for conversion.
    aggregate : bool, default False
        If True, merge multiple input files into one using CDO. If False,
        assumes input files are already in the correct format and reads them
        directly.
        This is useful for MESH, which only reads one file.
        If `aggregate` is False, the input files are assumed to be already
        in the correct chunk format and are read directly.
        If `aggregate` is True, CDO is used to merge the files.
        Note that CDO is a binary dependency and must be installed separately.
        The merged dataset is returned as an xarray.Dataset.
    local_attrs : dict of dict, optional
        Dictionary of local attributes for each forcing variable.
        The keys are variable names and the values are dictionaries
        of attributes to assign to each variable.
        Example: {'air_temperature': {'long_name': 'Air Temperature',
                                       'units': 'K'}}
    global_attrs : dict of str, optional
        Dictionary of global attributes for the output dataset. The keys are
        attribute names and the values are their descriptions.

    Returns
    -------
    xarray.Dataset
        Merged and converted dataset containing forcing variables.

    Raises
    ------
    TypeError
        If `variables` is not a sequence of string values.
    ValueError
        If units for variables are not provided, or if any variable in
        `units` cannot be found in the dataset.

    Notes
    -----
    - Merges all input forcing files into a single NetCDF file, as MESH only
      reads one file. CDO is used for merging, but the function returns an
      xarray.Dataset.
    - The `variables` sequence should include:
        'air_pressure', 'specific_humidity', 'air_temperature', 'wind_speed',
        'precipitation', 'shortwave_radiation', 'longwave_radiation'.
    - The `units` and `to_units` dictionaries must contain the same keys as
      `variables`.
    - The merge functionality may be expanded in future versions.
    """
    # if `units` is not provided, throw an exception
    if not units:
        raise ValueError("`units` associated with `variables` elements must"
                         " be provided")

    # merge all the input forcing files as MESH only reads on file
    # CDO is used due to its stability, however, an xarray.DataSet
    # is returned

    if aggregate:
        cdo_obj = cdo.Cdo()  # CDO object
        ds = cdo_obj.mergetime(input=path, returnXArray=variables)  # Mergeing
    else:
        # if `aggregate` is False, we assume that the input files are already
        # in proper chunk format and we just read the file
        ds = xr.open_dataset(path)

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

    # rename remapped output name
    var = [i for i in ds.dims.keys() if i != 'time']
    ds = ds.transpose().rename({var[0]: hru_dim})

    # convert calendar to 'standard' based on MESH's input standard
    ds = ds.convert_calendar(calendar='standard')

    # assigning local attributes for each variable
    if local_attrs:
        for var, val in local_attrs.items():
            for attr, desc in val.items():
                ds[var].attrs[attr] = desc

    # assigning global attributes for `ddb`
    if global_attrs:
        # empty global attribute dictionary first
        ds.attrs = {}
        # assign new global attributes
        for attr, desc in global_attrs.items():
            ds.attrs[attr] = desc

    return ds


def freq_long_name(
    freq_alias: str
) -> str:
    """
    Return the full name of a time offset alias based on pandas conventions.

    Parameters
    ----------
    freq_alias : str
        Time offset alias representing time interval frequency, such as
        'H' for 'hours', 'T' for 'minutes', etc.

    Returns
    -------
    str
        Full name of the time offset (e.g., 'hours', 'minutes').

    Raises
    ------
    TypeError
        If `freq_alias` is not a string.
    ValueError
        If `freq_alias` is not a recognized offset alias.

    Examples
    --------
    >>> freq_long_name('H')
    'hours'
    >>> freq_long_name('T')
    'minutes'
    """

    if not isinstance(freq_alias, str):
        raise TypeError(
            f"frequency value of '{freq_alias}' is not acceptable"
        )

    if freq_alias in ('H', 'h'):
        return 'hours'
    elif freq_alias in ('T', 'min'):
        return 'minutes'
    elif freq_alias in ('S',):
        return 'seconds'
    elif freq_alias in ('L', 'ms'):
        return 'milliseconds'
    else:
        raise ValueError(
            f"frequency value '{freq_alias}' is not acceptable"
        )


def calculate_time_difference(
    initial_time_zone: str,
    target_time_zone: str
) -> int | float:
    """
    Calculate the time difference in hours between two IANA time zones.

    Parameters
    ----------
    initial_time_zone : str
        IANA time zone name (e.g., 'America/Toronto').
    target_time_zone : str
        IANA time zone name (e.g., 'UTC', 'Europe/London').

    Returns
    -------
    float or int
        Time difference in hours between the two time zones. If the time
        zones are the same, returns 0.

    Raises
    ------
    AssertionError
        If either time zone argument is not a string.
    ValueError
        If the time zone format is incorrect or not valid.

    Notes
    -----
    Time zone naming follows IANA conventions:
    https://data.iana.org/time-zones/releases/tzdb-2025b.tar.lz

    Examples
    --------
    >>> calculate_time_difference('UTC', 'America/Toronto')
    -4.0
    >>> calculate_time_difference('America/Edmonton', 'America/Toronto')
    2.0
    """
    # Routine error checks
    assert isinstance(initial_time_zone, str), (
        "forcing time zone needs to be of dtype `str`."
    )
    assert isinstance(target_time_zone, str), (
        "target time zone needs to be of dtype `str`."
    )

    # Define the datetime you want to compare (current time in UTC)
    dt = datetime.now(tz=ZoneInfo("UTC"))

    # Specify the IANA time zones
    tz1 = ZoneInfo(initial_time_zone)
    tz2 = ZoneInfo(target_time_zone)

    # Convert to target time zones
    dt_tz1 = dt.astimezone(tz1)
    dt_tz2 = dt.astimezone(tz2)

    # Get the UTC offsets
    offset1 = dt_tz1.utcoffset()
    offset2 = dt_tz2.utcoffset()

    # Calculate the time difference
    diff = offset2 - offset1

    return diff.total_seconds() / 3600  # Convert seconds to hours
