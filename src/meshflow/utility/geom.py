"""
Containing common modules to manipulate and interpret geometrical objects
such as Polygons, LineStrings, MultiPolygons, etc.
"""

# third-party libraries
import geopandas as gpd
import pandas as pd
import xarray as xr

import pint
import pyproj
import pint_pandas

from pyproj import CRS

# build-in libraries
import warnings


def extract_centroid(
    gdf: gpd.GeoDataFrame,
    obj_id: str,
    epsg: int = 4326,
) -> gpd.GeoDataFrame:
    """
    Extract centroid latitude and longitude for each polygon in a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries in the 'geometry' column.
    obj_id : str
        Name of the column containing unique IDs for each polygon element.
    epsg : int, optional
        EPSG code for the output coordinate reference system (default: 4326).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
            - obj_id: IDs for each polygon (copied from input)
            - lat: Centroid latitude in the specified CRS
            - lon: Centroid longitude in the specified CRS

    Notes
    -----
    - If the input GeoDataFrame has no CRS, EPSG:4326 is assumed.
    - Only polygon geometries are supported; other types may yield errors.
    - Centroids are calculated in an equal-area projection and reprojected
      to the requested CRS for accurate coordinates.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> gdf = gpd.GeoDataFrame({
    ...     'id': [1],
    ...     'geometry': [Polygon([(0,0), (1,0), (1,1), (0,1)])]
    ... }, crs='EPSG:4326')
    >>> extract_centroid(gdf, obj_id='id')
    """
    # if crs is missing
    if not gdf.crs:
        # set to epsg=4326
        gdf.set_crs(epsg=4326)
        # warn user of the assumption made
        warnings.warn("EPSG of `gdf` is missing and is assumed to be"
                      "4326")
        # epsg value assumed as well
        epsg = 4326

    # calculate centroid lat/lon values
    centroids_gdf = gdf.to_crs('+proj=cea').centroid.to_crs(epsg=epsg)
    coords_df = pd.DataFrame(gdf[obj_id].copy())
    coords_df['lat'] = centroids_gdf.y
    coords_df['lon'] = centroids_gdf.x

    return coords_df


def prepare_mesh_coords(
    coords: pd.DataFrame,
    cat_dim: str,
    hru_dim: str,
) -> xr.Dataset:
    """
    Convert a DataFrame of mesh coordinates to an xarray.Dataset with
    specified dimension names.

    Parameters
    ----------
    coords : pandas.DataFrame
        DataFrame containing mesh coordinates. Must include a column for
        catchment element IDs.
    cat_dim : str
        Name of the column representing catchment element IDs (e.g., 'COMID').
    hru_dim : str
        Desired name for the output dimension (e.g., 'hru').

    Returns
    -------
    xr.Dataset
        Dataset with coordinates indexed by `hru_dim`.

    Examples
    --------
    >>> import pandas as pd
    >>> import xarray as xr
    >>> df = pd.DataFrame({
    ...     'COMID': [1, 2],
    ...     'lat': [45.0, 46.0],
    ...     'lon': [-75.0, -76.0]
    ... })
    >>> ds = prepare_mesh_coords(df, cat_dim='COMID', hru_dim='hru')
    >>> print(ds)
    """
    coords = coords.copy().set_index(cat_dim).to_xarray()
    coords_ds = coords.rename({cat_dim: hru_dim})
    return coords_ds


def _calculate_polygon_areas(
    gdf: gpd.GeoDataFrame,
    target_area_unit: str = 'm ** 2',
    equal_area_crs: str = 'ESRI:54009'
) -> gpd.GeoDataFrame:
    """
    Calculate polygon areas in a GeoDataFrame with proper unit handling.

    The area calculation follows these steps:
        1. For geographic CRS (e.g., EPSG:4326), transforms to equal-area
           projection.
        2. For projected CRS, uses native units if detectable.
        3. Converts to target units using Pint's dimensional analysis.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing polygon geometries. If CRS is undefined,
        assumes EPSG:4326 (WGS84) with coordinates in degrees.
    target_area_unit : str, optional
        Output unit for area values. Common options:
            'm ** 2' (default), 'km ** 2', 'mi ** 2', 'acre'.
    equal_area_crs : str, optional
        Equal-area projection to use when input CRS is geographic. The default
        'ESRI:54009' (World Mollweide) provides areas in m ** 2. Alternative
        options include 'EPSG:3410' and 'EPSG:6933'.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of input with added 'area' column containing Pint quantities.
        The units will match ``target_area_unit`` when possible.

    Notes
    -----
    Area calculations for geographic CRS use an equal-area projection for
    accuracy. Unit conversions use Pint's dimensional analysis.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> gdf = gpd.GeoDataFrame(
    ...     geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
    ...     crs="EPSG:4326"
    ... )
    >>> result = calculate_polygon_areas(gdf, target_area_unit='km ** 2')
    """
    # Initialize Pint unit registry
    ureg = pint.UnitRegistry()

    # Make a copy of the input GeoDataFrame to avoid modifying the original
    result_gdf = gdf.copy()

    # Check if CRS is defined, default to EPSG:4326 if not
    if result_gdf.crs is None:
        wrn_msg = """No CRS defined for GeoDataFrame.
        Assuming EPSG:4326 (WGS84) for area calculation."""
        warnings.warn(wrn_msg)
        result_gdf.crs = 'EPSG:4326'

    # Transform to equal area CRS if original is geographic
    try:
        equal_area_gdf = result_gdf.to_crs(equal_area_crs)
    except pyproj.exceptions.CRSError:
        raise ValueError(f"Failed to transform to equal area CRS: {equal_area_crs}. "
                             "Please provide a valid equal area CRS.")

    # Try to determine the unit from the CRS
    try:
        crs_unit = CRS(equal_area_gdf.crs).axis_info[0].unit_name
        if crs_unit in ('metre', 'meter'):
            area_unit = ureg('m^2')
        elif crs_unit == 'US survey foot':
            area_unit = ureg('survey_foot^2')
        elif crs_unit == 'foot':
            area_unit = ureg('foot^2')
        else:
            warnings.warn(f"CRS has linear unit '{crs_unit}' which is not automatically "
                         "mapped to a Pint unit. Area values will be unitless.")
    except (AttributeError, IndexError):
        warnings.warn("Could not determine units from CRS. Area values will be unitless.")


    # Calculate areas in the equal area CRS
    areas = equal_area_gdf.geometry.area.astype(f'pint[{area_unit}]')

    # Convert to target unit if areas have units
    if hasattr(areas, 'pint'):
        try:
            target_unit = ureg(target_area_unit)
            converted_areas = areas.pint.to(target_unit)
        except:
            warnings.warn(f"Could not convert to target unit '{target_area_unit}'. "
                          "Keeping original units.")
            converted_areas = areas
    else:
        warnings.warn("Could not determine units from CRS. Area values will be unitless.")


    # Add area column to the result GeoDataFrame
    result_gdf['area'] = converted_areas


    return result_gdf