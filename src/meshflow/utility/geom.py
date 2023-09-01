"""
Containing common modules to manipulate and interpret geometrical objects
such as Polygons, LineStrings, MultiPolygons, etc.
"""

import geopandas as gpd
import pandas as pd

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
    geom_var = 'geometry'

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
