"""
Utilities for building an observations xarray.Dataset compatible with MESH workflows.

The primary entry point is `build_observations_dataset`, which consumes a list of
observation definitions and returns a Dataset with dims (time, hru_dim) and one
DataArray per observation "type" (e.g., streamflow), filled at the subbasin(s)
that have observations.
"""

# third-party imports
import numpy as np
import pandas as pd
import xarray as xr

# built-in imports
import sqlite3
import datetime

from typing import (
    Any,
    Dict,
    Sequence,
    Union,
)

# custom type hints
try:
    from os import PathLike
except ImportError:  # <Python3.8
    from typing import Union
    PathLike = Union[str, bytes]

from numpy.typing import ArrayLike

def _get_the_daily_dataframe(
    df: pd.DataFrame,
    regex_str: str,
    col: str,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    This function extracts daily data from a HYDAT dataframe
    """
    
    # filter and trim columns
    df = df.filter(regex=regex_str, axis=1) # extract the columns
    df.columns = df.columns.str.replace(r'\D', '', regex=True) # remove non-digits
    df = df.stack(future_stack=True) # stack without dropping
    df.index.names = ['STATION_NUMBER', 'YEAR', 'MONTH', 'DAY'] # assign index names
    df = df.reset_index() # reset index to add another level
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']].astype(str).agg('-'.join, axis=1), errors='coerce') # define date column
    df.drop(columns=['YEAR', 'MONTH', 'DAY'], inplace=True) # drop unnecessary columns
    df.dropna(subset=['DATE'], inplace=True) # remove invalid dates
    df.set_index(keys=['STATION_NUMBER', 'DATE'], drop=True, inplace=True) # set index levels
    df.columns = [col] # assing column name
    
    # pivot data to look nice
    df = df.unstack(level='STATION_NUMBER')
    df = df.reorder_levels(order=[1,0], axis=1)
    
    return df

def _extract_station_daily_flow(
    station: str,
    connection: sqlite3.Connection,
    start_date: str = '1850-01-01',
    end_date: str = str(datetime.datetime.now().date()),
    *args,
    **kwargs,
) -> pd.DataFrame:
    """This function simply extracts data from the HYDAT sqlite3 database"""

    # read station data
    df = pd.read_sql_query(f"SELECT * FROM DLY_FLOWS WHERE STATION_NUMBER LIKE '%{station}%'", connection)

    # set index
    df.set_index(keys=['STATION_NUMBER', 'YEAR', 'MONTH'], drop=True, inplace=True)

    # get the FLOW and FLAG
    df_flow = _get_the_daily_dataframe(df, r'^FLOW\d', 'FLOW')
    df_flag = _get_the_daily_dataframe(df, r'^FLOW_.', 'FLAG')

    df = pd.concat([df_flow, df_flag], 
                   axis=1)
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, level=0, ascending=False, inplace=True)

    df = df.loc[start_date:end_date, :]
    
    return df

def _extract_station_coords(
    station: str,
    connection: sqlite3.Connection,
) -> dict:
    """returns `latitude` and `longitude` values for `station`"""
    
    # read the specs
    df = pd.read_sql_query(f"SELECT * FROM STATIONS WHERE STATION_NUMBER LIKE '%{station}%'", connection)
    
    # rename the columns to their lowercase equivalent
    df.rename(
        columns={
            'LATITUDE': 'latitude',
            'LONGITUDE': 'longitude',
        },
        inplace=True,
    )
    
    # making a dictionary out of the values
    vals_dict = df.loc[:, ['latitude', 'longitude']].to_dict(orient='list')
    
    # return the values in form of a dictionary
    vals_dict = {k:v[0] for k,v in vals_dict.items()}
    
    return vals_dict

def from_hydat(
    hydat_sqlite_path: PathLike, # type: ignore
    station_ids: Sequence[str],
) -> Dict[str, Any]:
    """
    Load hydrometric observations from a HYDAT SQLite
    database for specified station IDs.

    Parameters
    ----------
    hydat_sqlite_path : Pathlike
        Path to the HYDAT SQLite database file.
    station_ids : Sequence[str]
        List of station IDs to load observations for.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the loaded observations, keyed by station ID.
    """
    # connect to the HYDAT database
    conn = sqlite3.connect(hydat_sqlite_path)

    observations = []

    for station in station_ids:
        # extract daily flow data
        daily_flow_df = _extract_station_daily_flow(
            station=station,
            connection=conn,
        )

        # extract station coordinates
        station_coords = _extract_station_coords(
            station=station,
            connection=conn,
        )

        # fill the missing timestamps with NaNs
        freq = '1D' # hard-coded as HYDAT provides daily data
        full_index = pd.date_range(
            daily_flow_df.index.min(),
            daily_flow_df.index.max(),
            freq=freq,
            tz=daily_flow_df.index.tz
            )
        df_full = daily_flow_df.reindex(full_index)  # missing rows to `NaN`

        observations.append(
            {
                'type': 'QO',
                'location': station_coords,
                'timeseries': df_full[station]['FLOW'], # hard-coded `FLOW` value
                'unit': 'm3/s',
                'freq': '1D', # pandas offset alias for daily frequency
            }
        )
    # close the connection
    conn.close()

    return observations

def _build_dataset(
    dicts: Dict,
    ranks: ArrayLike,
    subbasins: ArrayLike,
    time=None,
    merge_strategy="mask"
) -> xr.Dataset:
    """
    Build an xarray.Dataset from observation dictionaries.

    Each observation dict should have:
    - type: str
        Variable name (e.g., 'QO').
    - location: dict, optional
        Mapping with keys 'latitude' and 'longitude'.
    - timeseries: pandas.Series
        Float values (NaN allowed) with a pandas.DatetimeIndex.
    - unit: str, optional
        Units for the variable (e.g., 'm3/s').
    - freq: str, optional
        Sampling frequency (e.g., '1D').
    - subbasin: int
        Actual subbasin identifier (not used as the dim coordinate).
    - rank: int
        Rank value to place on the 'subbasin' dimension.

    Parameters
    ----------
    dicts : sequence of dict
        Observation entries as described above.
    ranks : array-like of shape (N,)
        Rank values in the exact order to appear on the 'subbasin' dimension.
    subbasins : array-like of shape (N,)
        Actual subbasin IDs aligned one-to-one with ranks.
    time : pandas.DatetimeIndex or sequence of datetime-like, optional
        Global time coordinate. If None, uses the union of all series times.
    merge_strategy : {'mask', 'overwrite'}, default 'mask'
        - 'mask': only write non-NaN values when the same (var, rank) is encountered again.
        - 'overwrite': later entries overwrite earlier ones fully at those timestamps.

    Returns
    -------
    xarray.Dataset
        Dataset with dims ('subbasin', 'time'). Contains one data variable per
        observation 'type', plus 'latitude' and 'longitude'. Coordinates include
        'time', 'subbasin' (holding rank values), and 'subbasin_id' aligned with
        the 'subbasin' dimension.

    Raises
    ------
    TypeError
        If any timeseries does not have a pandas.DatetimeIndex.
    ValueError
        If a rank is not uniquely found in `ranks` or if conflicting units are detected.
    """
    ranks = np.asarray(ranks)
    subbasins = np.asarray(subbasins)
    N = ranks.size

    # 1) Build global time index
    if time is None:
        time = pd.DatetimeIndex([])
        for d in dicts:
            ts = d['timeseries']
            if not isinstance(ts.index, pd.DatetimeIndex):
                raise TypeError("All timeseries must have a DatetimeIndex")
            time = time.union(ts.index)
        time = time.sort_values()
    else:
        time = pd.DatetimeIndex(time)
    T = len(time)

    # 2) Prepare lat/lon variables (default NaN)
    lat = np.full(N, np.nan, dtype=float)
    lon = np.full(N, np.nan, dtype=float)

    # 3) Discover variable types and allocate arrays (NaN-initialized)
    var_types = list(dict.fromkeys(d['type'] for d in dicts))  # preserve first-seen order
    arrays = {v: np.full((N, T), np.nan, dtype=float) for v in var_types}
    units = {}   # per variable type
    freqs = {}   # per variable type (optional)

    # 4) Fill arrays per dict
    for d in dicts:
        var = d['type']
        ts = d['timeseries'].reindex(time)  # align to global time, missing -> NaN
        target_rank = int(d['rank'])
        idx = np.nonzero(ranks == target_rank)[0]
        if idx.size != 1:
            raise ValueError(f"Rank {target_rank} not found uniquely in ranks array.")
        row = idx[0]

        # Merge strategy for values
        new_vals = ts.to_numpy()
        if merge_strategy == "mask":
            # Only write where new data is not NaN
            mask = ~np.isnan(new_vals)
            arrays[var][row, mask] = new_vals[mask]
        elif merge_strategy == "overwrite":
            arrays[var][row, :] = new_vals
        else:
            raise ValueError("merge_strategy must be 'mask' or 'overwrite'.")

        # Units per variable type (ensure consistency)
        unit = d.get('unit')
        if unit is not None:
            if var in units and units[var] not in (None, unit):
                raise ValueError(f"Conflicting units for {var}: {units[var]} vs {unit}")
            units[var] = unit

        # Freq per variable type (optional, mark as mixed if conflicting)
        freq = d.get('freq')
        if freq is not None:
            if var in freqs and freqs[var] not in (None, freq):
                freqs[var] = "mixed"
            else:
                freqs[var] = freq

        # Latitude/Longitude (fill only if available and current is NaN)
        loc = d.get('location') or {}
        lat_v = loc.get('latitude', None)
        lon_v = loc.get('longitude', None)
        if lat_v is not None and np.isnan(lat[row]):
            lat[row] = float(lat_v)
        if lon_v is not None and np.isnan(lon[row]):
            lon[row] = float(lon_v)

    # 5) Compose Dataset
    data_vars = {}
    for var, arr in arrays.items():
        v_attrs = {}
        if units.get(var):
            v_attrs['units'] = units[var]
        if freqs.get(var):
            v_attrs['freq'] = freqs[var]
        data_vars[var] = (('subbasin', 'time'), arr, v_attrs)

    # Add latitude/longitude as variables over subbasin
    data_vars['latitude'] = (('subbasin',), lat, {'units': 'degrees_north'})
    data_vars['longitude'] = (('subbasin',), lon, {'units': 'degrees_east'})

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': time,
            # IMPORTANT: dim 'subbasin' holds RANK values
            'subbasin': ranks,
            # actual subbasin IDs aligned with the dim
            'subbasin_id': ('subbasin', subbasins),
        },
    )
    return ds