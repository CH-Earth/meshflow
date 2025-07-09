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
    Returning centroid latitude and longitude values of any given `gdf`
    only if the elements contain Polygons

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        geometrical object containing polygons shapes in the `geometry`
        column
    obj_id : str
        name of the column of the given `gdf` containing IDs of Polygon
        elements
    epsg : int, default to `4326`
        EPSG value of projection in which the latitue and longitude are
        needed

    Returns
    -------
    pandas.DataFrame
        containing `lat` and `lon` columns for each element of `obj_id`
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
    '''Implements necessary coords manipulations by:
    1) setting the index to the catchment element
       dimension variable `cat_dim`. For example, in
       MERIT-Basins geospatial data, the `cat_dim`
       refers to the ID of each element, i.e., COMID.
    2) returning an xarray.Dataset object
    '''
    # create coords xr.Dataset
    coords = coords.copy().set_index(cat_dim).to_xarray()
    # change dimension name
    coords_ds = coords.rename({cat_dim: hru_dim})

    return coords_ds


def _calculate_polygon_areas(
    gdf: gpd.GeoDataFrame,
    target_area_unit: str = 'm ** 2',
    equal_area_crs: str = 'ESRI:54009'
) -> gpd.GeoDataFrame:
    r"""Calculate polygon areas in a GeoDataFrame with proper unit handling.
 
    The area calculation follows these steps:
 
    1. For geographic CRS (e.g., EPSG:4326), transforms to equal-area projection
    2. For projected CRS, uses native units if detectable
    3. Converts to target units using Pint's dimensional analysis


    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing polygon geometries. If CRS is undefined,
        assumes EPSG:4326 (WGS84) with coordinates in degrees (\degree).
    target_area_unit : str, optional
        Output unit for area values. Common options:
        - 'm\ :sup:`2`\' for square meters (default)
        - 'km\ :sup:`2`\' for square kilometers
        - 'mi\ :sup:`2`\' for square miles
        - 'acre' for acres
    equal_area_crs : str, optional
        Equal-area projection to use when input CRS is geographic. The default
        'ESRI:54009' (World Mollweide) provides areas in m\ :sup:`2`\.
        Alternative options include:
        - 'EPSG:3410' (NSIDC Polar Stereographic North)
        - 'EPSG:6933' (WGS 84 / NSIDC EASE-Grid 2.0 Global)


    Returns
    -------
    geopandas.GeoDataFrame
        Copy of input with added 'area' column containing Pint quantities.
        The units will match ``target_area_unit`` when possible.


    Notes
    -----
    - Area calculations for geographic CRS use:
      \[
      A = \int \int \sqrt{g} \, d\phi \, d\lambda
      \]
      where \(g\) is the determinant of the metric tensor for the projection.
    - Unit conversions use Pint's dimensional analysis:
      \[
      1\ \text{m}^2 = 10^{-6}\ \text{km}^2 = 2.47105 \times 10^{-4}\ \text{acres}
      \]


    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> gdf = gpd.GeoDataFrame(
    ...     geometry=[Polygon([(0,0),(1,0),(1,1),(0,1)])],
    ...     crs="EPSG:4326"
    ... )
    >>> result = calculate_polygon_areas(gdf, target_area_unit='km²')
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