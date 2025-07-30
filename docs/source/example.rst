Example Instantiation for Bow River at Calgary
==============================================

This example demonstrates how to use MESHFlow with the provided example
datasets. The workflow uses the Python API and data files included in the
`examples/` directory.

**Folders and Files Used:**

- ``examples/example-setup-python/meshflow_bow_at_calgary.ipynb``: Jupyter
    notebook with a step-by-step example.
- ``examples/bow-at-calgary-attributes/``: Contains landcover and attribute
    CSV files.
- ``examples/bow-at-calgary-forcings/``: Contains NetCDF files with
    meteorological forcing data.
- ``examples/bow-at-calgary-shapefiles/``: Contains shapefiles for rivers
    and subbasins.

**Quick Start (Python):**

1. Open the notebook
     ``examples/example-setup-python/meshflow_bow_at_calgary.ipynb`` in Jupyter
     or VS Code.
2. Follow the cells to:
     - Load the required packages and MESHFlow.
     - Set up paths to the example data files.
     - Load attribute, forcing, and shapefile data.
     - Run MESHFlow processing steps.
     - Inspect and visualize results.

**Example Data Overview:**

- *Attributes*: ``bcalgary_NA_NALCMS_landcover_2020_30m.tif`` (landcover
    raster), ``bcalgary_stats_NA_NALCMS_landcover_2020_30m.csv`` (attribute
    table)
- *Forcings*: ``averaged_remapped_bcalgary_*.nc`` (NetCDF files with
    meteorological data)
- *Shapefiles*: ``bcalgary_rivers.*``, ``bcalgary_subbasins.*`` (vector data
    for rivers and subbasins)

For a full, runnable example, see the Jupyter notebook:
:download:`meshflow_bow_at_calgary.ipynb <../../examples/example-setup-python/meshflow_bow_at_calgary.ipynb>`


.. toctree::
   :maxdepth: 1
   :caption: Example Contents:

