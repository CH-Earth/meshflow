Quick Usage
===========

Installation
------------

To install ``MESHFlow`` from its GitHub repository, run:

.. code-block:: console
   :linenos:

   $ pip install git+https://github.com/CH-Earth/meshflow.git

General Usage
-------------

``MESHFlow`` provides multiple interfaces. You can use it directly
in Python by instantiating the main class:

.. code-block:: python
   :linenos:

   >>> from meshflow import MESHWorkflow
   >>> mesh_workflow_instance = MESHWorkflow()

Alternatively, you can use the Command Line Interface (CLI):

.. code-block:: console 
   :linenos:

   $ meshflow --help
   Usage: meshflow [OPTIONS]

   Run MESHWorkflow from a JSON configuration file.

   Options:
   --json PATH         Path to the JSON configuration file  [required]
   --output-path PATH  Output path for saving results (optional)
   --help              Show this message and exit.


Both interfaces support the use of a `JSON` configuration file to set up
the ``MESHWorkflow`` and create an instance tailored to your domain of interest.

.. note::

   In future versions, all elements accepted in the JSON configuration file
   will also be accepted directly via the CLI. This feature is currently a
   work in progress.


.. toctree::
   :maxdepth: 2
   :caption: Configuration Contents


Configuration
-------------
The options and arguments to the package is critical in configuring an accurate
instantiation of the model. In the following we describe the options available
to the user in the Python interface. Please note that similar options are available
in the CLI interface as well using the ``--json`` argument.


Geospatial Data
---------------
The ``MESHWorkflow`` class requires geospatial data to function correctly. The geospatial
data should follow a certain standard to allow the workflow to function properly. You
can provide this data in the form of a ``GeoDataFrame`` object or a path to the geospatial
file. The geospatial data should include minimum necessary attributes for your specific use case.

The following attributes are typically required:
    - ``riv``: a ``GeoDataFrame`` containing river network data, or a path to the river network file.
    - ``cat``: a ``GeoDataFrame`` containing catchment data, or a path to the catchment file.
    - ``landcover``: a ``DataFrame`` containing land cover data, or a path to the land cover file.

River Network and Relevant Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

River Network and Relevant Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``riv`` file/object typically contains information about the river network, such as river IDs,
geometry, and flow direction. The flow direction is essential for the model to understand
how water moves through the river network. ``main_id`` and ``ds_main_id`` variables are used
to identify the main river and its downstream counterpart, respectively, in the ``riv``
object (or file). Using these two variables, the flow direction is determined for model
instantiations.

Example of a ``riv`` GeoDataFrame available in this repository:


.. code-block:: python
   :linenos:

   import geopandas as gpd
   riv = gpd.read_file("meshflow/examples/bow-at-calgary-shapefiles/bcalgary_subbasins.shp")
   riv.head()
   # Example output:
   #    COMID   lengthkm  lengthdir  sinuosity     slope       uparea  order  NextDownID ...
   # 0 71027942 28.145110 20.588731   1.367015  0.001798  7857.020581      4       -9999 ...
   # 1 71027957 20.589551 13.558710   1.518548  0.001508  7669.065145      4    71027942 ...
   # 2 71027962  2.160029  1.774091   1.217541  0.002403  7406.819260      4    71027957 ...
   # 3 71027963  1.951061  1.625592   1.200215  0.001484  6803.660756      4    71027962 ...
   # 4 71027969  4.507794  3.805159   1.184653  0.002480  6713.480813      4    71027963 ...
   # ...
   # [169 rows x 17 columns]


In the ``riv`` object example above, the ``COMID`` column represents the unique
identifier for each river segment, while the ``NextDownID`` column indicates the
``COMID`` of the downstream river segment. This information is crucial for
establishing the connectivity of the river network and determining flow
directions. In setting up the ``MESHWorkflow``, the ``main_id`` and ``ds_main_id``
variables are set to ``COMID`` and ``NextDownID``, respectively, to ensure the
model can correctly interpret the river network and flow directions.

.. note::

   The ``riv`` object should not contain any duplicate ``main_id`` values, as this
   can lead to errors in the model instantiation. Ensure that the river network
   data is clean and properly formatted before using it in the ``MESHWorkflow``.

.. note::

   The ```main_id``` and ``ds_main_id`` values must be unique integers or strings.
   If your data contains non-unique values, you may need to preprocess it to
   ensure uniqueness before using it in the ``MESHWorkflow``.


Other Required Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~
Apart from ``main_id`` and ``ds_main_id``, the ``riv`` object must obtain a minimum
set of attributes to ensure the model can function correctly. These attributes
include:

- ``river_length``: The length of the river segment in kilometers. In this
  example, the ``lengthkm`` column is used to represent the length of the river
  segment.

- ``river_slope``: The slope of the river segment, which is essential for
  understanding the flow dynamics in routing water through the river network.
  In this example, the ``slope`` column is used to represent the slope of
  the river segment.

- ``river_class``: Optional, but can be used to categorize river segments
  into a maximum of 5 classes based on their characteristics. In this case,
  the ``river_class`` can be assigned to the ``order`` column in the ``riv``
  object.

The key information above must be present in the ``riv`` object to ensure
the ``MESHWorkflow`` can function correctly. To instruct ``MESHFlow`` on how to
use the ``riv`` object, you can pass them as a dictionary to the ``ddb_vars``
parameter when instantiating the ``MESHWorkflow`` class. For example:

.. code-block:: python
   :linenos:

   >>> from meshflow import MESHWorkflow
   >>> mesh_workflow_instance = MESHWorkflow(
   ...     riv=riv,  # or a path to the river network file
   ...     main_id="COMID",
   ...     ds_main_id="NextDownID",
   ...     ddb_vars={
   ...         "river_length": "lengthkm",
   ...         "river_slope": "slope",
   ...         "river_class": "order"
   ...     }
   ...     ...,
   ... )


Units
~~~~~
In addition to the information described so far, the units of the ``ddb_vars`` is
also material in ensuring the model workflow functions correctly. One can describe
the units of the ``ddb_vars`` using the ``ddb_units`` parameter when instantiating
the ``MESHWorkflow`` class. For example:

.. code-block:: python
   :linenos:

   >>> from meshflow import MESHWorkflow
   >>> mesh_workflow_instance = MESHWorkflow(
   ...     riv=riv,  # or a path to the river network file
   ...     main_id="COMID",
   ...     ds_main_id="NextDownID",
   ...     ddb_vars={
   ...         "river_length": "lengthkm",
   ...         "river_slope": "slope",
   ...         "river_class": "order"
   ...     },
   ...     ddb_units={
   ...         "river_length": "km",
   ...         "river_slope": "dimensionless",
   ...         "river_class": "dimensionless"
   ...     },
   ...     ...,
   ... )

The ``ddb_units`` parameter allows you to specify the units for each
variable in the ``ddb_vars`` dictionary. This is important for ensuring
that the model interprets the data correctly and performs the necessary
calculations based on the provided units. The units follow Pint's
syntax, which is a Python library for handling physical quantities with units.

.. note::

   The ``Pint`` library is used to handle units in ``MESHFlow``. Ensure that
   you have it installed in your environment. You can install it using:

   .. code-block:: console

      pip install pint
      pip install pint_pandas
      pip install pint_xarray

   The units specified in the ``ddb_units`` dictionary should be compatible
   with Pint's syntax. For a complete list, refer to the following link:
   `Pint's Unit Registry <https://github.com/hgrecco/pint/blob/master/pint/default_en.txt>`_.

Catchment Data and Relevant Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``cat`` object typically contains information about the catchment
areas, such as catchment IDs, geometry, and coordinate reference systems
(CRS) of the subbasins.

The ``cat`` can be a ``GeoDataFrame`` object or a path to the catchment
file. The catchment data should include the following attributes:

- ``main_id``: A unique identifier for each catchment area **that corresponds to the**
  ``main_id`` **in the** ``riv`` **object**. This is essential for linking the catchment
  areas to the river network.

- ``geometry``: The geometry of the catchment area, which is used to define the
  spatial extent of the catchment.

The `CRS` (Coordinate Reference System) of the catchment data is also
important and must accompany the catchment data. It is usually inherent to the
catchment data itself, but if you are providing a path to a file, you should
ensure that the CRS is specified in the file or provided separately.

.. warning::

   If the `CRS` is not provided, a default CRS value of `EPSG:4326` will be used.

.. warning::

   The ``main_id`` values should also be present in the forcing data.
   If they are not, the model will not be able to link the catchment areas
   to the forcing data, leading to errors in the model instantiation. The value
   is typically present as a dimension in relevant NetCDF files.

An example of a ``cat`` file is available in this repository. Here is an example
of how to load the catchment data:

.. code-block:: python
   :linenos:

   import geopandas as gpd
   cat = gpd.read_file("meshflow/examples/bow-at-calgary-shapefiles/bcalgary_subbasins.shp")
   cat.head()
   # Example output:
   #         COMID    unitarea  hillslope                         geometry  
   # 0    71027942  153.058954          0  POLYGON ((-114.31792 51.18208, ...  
   # 1    71027957   93.459491          0  MULTIPOLYGON (((-114.43708 51.1...  
   # 2    71027962    1.982459          0  POLYGON ((-114.50042 51.19375, ...  
   # 3    71027963    7.289865          0  POLYGON ((-114.50042 51.22542, ...  
   # 4    71027969   29.509403          0  POLYGON ((-114.54792 51.23708, ...  
   ...       ...         ...        ...                                 ...
   [169 rows x 4 columns]

Typically, no other information is required in the ``cat`` object.


Forcing Data
------------
The forcing data is essential for the model to simulate hydrological processes
accurately. Currently, this workflow only supports NetCDF files as forcing data.
The forcing data should include two main dimensions: ``time`` and ``main_id``.
The ``time`` dimension represents the time steps for which the forcing data is
available, while the ``main_id`` dimension corresponds to the unique identifiers
for the catchment areas, which should match the ``main_id`` values in the ``cat``
and ``riv`` objects.

.. note::

   We recommend using the `easymore <https://github.com/ShervanGharari/EASYMORE>`_
   package to create the forcing data in NetCDF format. This package simplifies
   the process of creating NetCDF files with the required dimensions and attributes.
   It can also calculate the average values for the forcing data based on the
   catchments defined in the ``cat`` object.

   You can install the `easymore` package using:

   .. code-block:: console

      pip install easymore

The following shows an example of attributes available in the forcing data:

.. code-block:: console

   $ ncdump -h averaged_remapped_bcalgary_1980010112.nc 
   netcdf averaged_remapped_bcalgary_1980010112 {
   dimensions:
      COMID = 169 ;
      time = UNLIMITED ; // (24 currently)
   variables:
      int time(time) ;
         time:long_name = "time" ;
         time:units = "hours since 1980-01-01 12:00:00" ;
         time:calendar = "gregorian" ;
         time:standard_name = "time" ;
         time:axis = "T" ;
      double latitude(COMID) ;
         latitude:long_name = "latitude" ;
         latitude:units = "degrees_north" ;
         latitude:standard_name = "latitude" ;
      double longitude(COMID) ;
         longitude:long_name = "longitude" ;
         longitude:units = "degrees_east" ;
         longitude:standard_name = "longitude" ;
      double COMID(COMID) ;
         COMID:long_name = "shape ID" ;
         COMID:units = "1" ;
      double RDRS_v2.1_P_P0_SFC(time, COMID) ;
         RDRS_v2.1_P_P0_SFC:_FillValue = -9999. ;
         RDRS_v2.1_P_P0_SFC:long_name = "Forecast: Surface pressure" ;
         RDRS_v2.1_P_P0_SFC:units = "mb" ;
      double RDRS_v2.1_P_HU_09944(time, COMID) ;
         RDRS_v2.1_P_HU_09944:_FillValue = -9999. ;
         RDRS_v2.1_P_HU_09944:long_name = "Forecast: Specific humidity" ;
         RDRS_v2.1_P_HU_09944:units = "kg kg**-1" ;
      double RDRS_v2.1_P_TT_09944(time, COMID) ;
         RDRS_v2.1_P_TT_09944:_FillValue = -9999. ;
         RDRS_v2.1_P_TT_09944:long_name = "Forecast: Air temperature" ;
         RDRS_v2.1_P_TT_09944:units = "deg_C" ;
      double RDRS_v2.1_P_UVC_09944(time, COMID) ;
         RDRS_v2.1_P_UVC_09944:_FillValue = -9999. ;
         RDRS_v2.1_P_UVC_09944:long_name = "Forecast: Wind Modulus (derived using UU and VV)" ;
         RDRS_v2.1_P_UVC_09944:units = "kts" ;
      double RDRS_v2.1_A_PR0_SFC(time, COMID) ;
         RDRS_v2.1_A_PR0_SFC:_FillValue = -9999. ;
         RDRS_v2.1_A_PR0_SFC:long_name = "Analysis: Quantity of precipitation" ;
         RDRS_v2.1_A_PR0_SFC:units = "m" ;
      double RDRS_v2.1_P_FB_SFC(time, COMID) ;
         RDRS_v2.1_P_FB_SFC:_FillValue = -9999. ;
         RDRS_v2.1_P_FB_SFC:long_name = "Forecast: Downward solar flux" ;
         RDRS_v2.1_P_FB_SFC:units = "W m**-2" ;
      double RDRS_v2.1_P_FI_SFC(time, COMID) ;
         RDRS_v2.1_P_FI_SFC:_FillValue = -9999. ;
         RDRS_v2.1_P_FI_SFC:long_name = "Forecast: Surface incoming infrared flux" ;
         RDRS_v2.1_P_FI_SFC:units = "W m**-2" ;

As you can see, the forcing data includes various variables, including
``RDRS_v2.1_P_P0_SFC`` for surface pressure, ``RDRS_v2.1_P_HU_09944`` for specific humidity,
``RDRS_v2.1_P_TT_09944`` for air temperature, ``RDRS_v2.1_P_UVC_09944`` for wind modulus,
``RDRS_v2.1_A_PR0_SFC`` for precipitation, ``RDRS_v2.1_P_FB_SFC`` for downward solar flux,
``RDRS_v2.1_P_FI_SFC`` for surface incoming infrared flux.

It also includes two main dimensions: ``time`` and ``COMID``. The ``time`` dimension
represents the time steps for which the forcing data is available, while the
``COMID`` dimension corresponds to the unique identifiers for the catchment areas,
which should match the ``main_id`` values in the ``cat`` and ``riv`` objects.

This example is provided using the
`RDRSv2.1 <https://datatool.readthedocs.io/en/latest/scripts/eccc-rdrs.html>`_ dataset,
and is processed using the ``easymore`` package to create the forcing data
in NetCDF format.


Forcing Variables and Associated Units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The units of the forcing data variables must be presented to the ``MESHWorkflow`` class
to ensure the model can interpret the data correctly. You can provide the units
using the ``forcing_units`` parameter when instantiating the ``MESHWorkflow`` class.

To identify variables of forcing data, a predefined list of variables is used and users
must ensure that the units of the variables in the forcing data match the
predefined list. The predefined list of variables is as follows:

- ``air_pressure``

- ``specific_humidity``

- ``air_temperature``

- ``wind_speed``

- ``precipitation``

- ``shortwave_radiation``

- ``longwave_radiation``

Using the ``forcing_vars`` parameters, first we identify the variables
of the forcing data. Furthermore, we use the ``forcing_units`` to identify
forcing units. In the example above, the variables are identified as follows:

.. code-block:: python
   :linenos:

   >>> from meshflow import MESHWorkflow
   >>> mesh_workflow_instance = MESHWorkflow(
   ...     ...
   ...     forcing_vars={
   ...         "air_pressure": "RDRS_v2.1_P_P0_SFC",
   ...         "specific_humidity": "RDRS_v2.1_P_HU_09944",
   ...         "air_temperature": "RDRS_v2.1_P_TT_09944",
   ...         "wind_speed": "RDRS_v2.1_P_UVC_09944",
   ...         "precipitation": "RDRS_v2.1_A_PR0_SFC",
   ...         "shortwave_radiation": "RDRS_v2.1_P_FB_SFC",
   ...         "longwave_radiation": "RDRS_v2.1_P_FI_SFC"
   ...     },
   ...     forcing_units={
   ...         "air_pressure": "millibar",
   ...         "specific_humidity": "kilogram / kilogram",
   ...         "air_temperature": "celsius",
   ...         "wind_speed": "knot",
   ...         "precipitation": "meter / hour",
   ...         "shortwave_radiation": "watt / meter ** 2",
   ...         "longwave_radiation": "watt / meter ** 2"
   ...     },
   ...     ...
   ... )


Land Cover Data
---------------
The land cover data is essential for the model to simulate hydrological processes
accurately for each Grouped Response Unit (GRU). Currently, only classifications
produced as part of the `gistool <https://gistool.readthedocs.io/en/latest/>`_
package are supported.

A typical land cover data should include the following format:

- rows: Each row represents each subbasin present in the domain. The values
  should correspond to the ``main_id`` values in the ``cat`` and ``riv``
  objects.

- columns: Each column represents a land cover class. The values should be
  integers representing the land cover class for each subbasin. The classes
  should be defined in the ``gistool`` package, which provides a set of
  predefined land cover classes. The package produces column heads by prefixing
  ``frac_`` to the land cover class number. For example, if the land cover
  class is 1, the column head should be ``frac_1``.

- values: The values in the land cover data should be floats representing the
  fraction of the land cover class in each subbasin. The values should be between
  0 and 1, representing the fraction of the land cover class in the subbasin.
  The sum of the fractions for each subbasin should be equal to 1.

The land cover data can be provided as a ``DataFrame`` object or a path to the
land cover file. The land cover data should be passed to the ``MESHWorkflow``
class using the ``landcover`` parameter. For example:

.. code-block:: python
   :linenos:

   >>> from meshflow import MESHWorkflow
   >>> mesh_workflow_instance = MESHWorkflow(
   ...     ...
   ...     landcover=landcover_df,  # or a path to the land cover file
   ...     ...
   ... )

.. note::

   If a `pandas.DataFrame` is provided, the index should be the values of the
   ``main_id`` in the ``cat`` and ``riv`` objects. The columns should be the land
   cover classes prefixed with ``frac_``, and the values should be the fractions
   of the land cover class in each subbasin. The values should be floats between
   0 and 1.

Landcover Names
^^^^^^^^^^^^^^^
The land cover classes can also be named for better understanding and
future reference. The names can be provided as a dictionary to the
``landcover_classes`` parameter when instantiating the ``MESHWorkflow`` class.

For example:

.. code-block:: python
   :linenos:

   >>> from meshflow import MESHWorkflow
   >>> mesh_workflow_instance = MESHWorkflow(
   ...     ...
   ...     landcover_classes={
   ...         0: "Unknown",
   ...         1: "Temperate or sub-polar needleleaf forest",
   ...         2: "Sub-polar taiga needleleaf forest",
   ...         3: "Tropical or sub-tropical broadleaf evergreen forest",
   ...         4: "Tropical or sub-tropical broadleaf deciduous forest",
   ...         5: "Temperate or sub-polar broadleaf deciduous forest",
   ...         6: "Mixed forest",
   ...         7: "Tropical or sub-tropical shrubland",
   ...         8: "Temperate or sub-polar shrubland",
   ...         9: "Tropical or sub-tropical grassland",
   ...         10: "Temperate or sub-polar grassland",
   ...         11: "Sub-polar or polar shrubland-lichen-moss",
   ...         12: "Sub-polar or polar grassland-lichen-moss",
   ...         13: "Sub-polar or polar barren-lichen-moss",
   ...         14: "Wetland",
   ...         15: "Cropland",
   ...         16: "Barren lands",
   ...         17: "Urban",
   ...         18: "Water",
   ...         19: "Snow and Ice"
   ...     },
   ...     ...
   ... )

The values defined above correspond to the land cover classes of the 
Landsat land cover classification system. The relevant documentation can be found
at `gistool documentation <https://gistool.readthedocs.io/en/latest/scripts/landsat.html>`_.


Additional Settings
-------------------

Settings Dictionary
^^^^^^^^^^^^^^^^^^^

The ``settings`` parameter allows you to provide a dictionary of options
to control various aspects of the workflow and simulation. This dictionary
is organized into several sections, each responsible for a different part
of the model configuration.

Example structure:

.. code-block:: python
   :linenos:

   settings = {
       "core": {...},
       "class_params": {...},
       "hydrology_params": {...},
       "run_options": {...},
   }

Therefore, the ``settings`` parameter consists of the following sections:

- ``core``: Controls the main simulation specifics, such as time periods,
  time zones, forcing files, and output paths.

- ``class_params``: Defines measurement heights for meteorological variables,
  copyright information, and GRU (Grouped Response Unit) classifications
  that `CLASS` accepts.

- ``hydrology_params``: Contains parameters related to hydrological processes,
  such as GRU dependent and independent hydrological variables, and routing
  options.

- ``run_options``: Specifies options for running the simulation, including
  parallelization settings, logging levels, and runtime configurations. Also,
  users can specify the presence of available physical processes
  in the model, such as the activation of the ``PBMS`` module.


Core Settings
^^^^^^^^^^^^^

The ``core`` section controls the main simulation parameters, such
as time periods, time zones, and output paths.

For example:

.. code-block:: python
   :linenos:

   "core": {
       "forcing_files": "multiple",  # Specify if multiple forcing files are used
       "forcing_start_date": "1980-01-01 00:00:00",  # Start date of forcing data
       "simulation_start_date": "1985-02-12 12:00:00",  # Start date of simulation
       "simulation_end_date": "2010-05-18 18:00:00",  # End date of simulation
       "forcing_time_zone": "UTC",  # Time zone of forcing data
       # "model_time_zone": "America/Edmonton",  # Optional: time zone for the model
       "output_path": "results",  # Path to save model outputs relative to model instance's path
   }


- ``forcing_files``: Specify if multiple forcing files are used (e.g., "multiple" or "single").

- ``forcing_start_date``: Start date of the forcing data (string, accepting
  various formats, such as ``YYYY-MM-DD HH:MM:SS``).

- ``simulation_start_date``: Start date of the simulation.

- ``simulation_end_date``: End date of the simulation.

- ``forcing_time_zone``: Time zone of the forcing data (e.g., "UTC").

- ``model_time_zone``: (Optional) Time zone for the model (e.g., "America/Edmonton").
  If not specified, the model will attempt to determine the time zone from the location
  of the subbasins.

- ``output_path``: (Optional) Path to save model outputs. Default value is ``results``.


Class Parameters
^^^^^^^^^^^^^^^^

The ``class_params`` section defines measurement heights for meteorological
variables, copyright information, and GRU (Grouped Response Unit) classifications
and parameters that `CLASS` accepts.

For example:

.. code-block:: python
   :linenos:

   >>> from meshflow import MESHWorkflow
   >>> mesh_workflow_instance = MESHWorkflow(
   ...     ...,
   ...     class_params={
   ...         "measurement_heights": {
   ...             "wind_speed": 40,
   ...             "specific_humidity": 40,
   ...             "air_temperature": 40,
   ...             "roughness_length": 50,
   ...         },
   ...         "copyright": {
   ...             "author": "University of Calgary",
   ...             "location": "University of Calgary",
   ...         },
   ...         "grus": {
   ...             0: "needleleaf",
   ...             1: {
   ...                 "class": "needleleaf",
   ...                 "LNZ0": -1.4,
   ...             },
   ...             2: "needleleaf",
   ...             3: "broadleaf",
   ...             4: "broadleaf",
   ...             5: {
   ...                 "class": "broadleaf",
   ...                 "ROOT": 0.5,
   ...             },
   ...             6: "broadleaf",
   ...             7: "grass",
   ...             8: "grass",
   ...             9: "grass",
   ...             10: "grass",
   ...             11: "grass",
   ...             12: "grass",
   ...             13: "grass",
   ...             14: "water",
   ...             15: "crops",
   ...             16: "barrenland",
   ...             17: "urban",
   ...             18: "water",
   ...             19: "water",
   ...         }
   ...     },
   ...     ...,
   ... )

- ``measurement_heights``: Dictionary specifying heights (in meters) for wind speed,
  specific humidity, air temperature, and roughness height.

- ``copyright``: (Optional). Dictionary with author and location information.

- ``grus``: Dictionary mapping GRU indices to land cover classes (e.g., "needleleaf",
  "broadleaf", "grass", "water", "crops", "barrenland", "urban").

``grus``
^^^^^^^^
All the land classes must be further categorized into different classes based on
`CLASS`'s assumptions. These classes are:

- ``needleleaf``,

- ``broadleaf``,

- ``grass``,

-  ``crops``,

- ``barrenland``, (or ``water`` or ``urban``)
 
One can use the ``grus`` dictionary inside ``class_params`` to define the GRU
class for each subbasin.

To further define ``CLASS`` parameters for each GRU, you can use a dictionary
for each class and define parameter values for each GRU. Otherwise, a filler
value (**not default**) will be used for each GRU. The filler values are not
accurate for process simulation, and users should use calibration workflows
to determine the accurate values for each GRU.

In the example provided above, the ``grus`` dictionary maps GRU indices
to land cover classes the ``CLASS`` model accepts.

.. note::
   
   The GRU indices should match the indices in the land cover data provided
   in the ``landcover`` parameter. The values should be integers representing
   the land cover class for each subbasin, and the classes should be defined
   in the ``gistool`` package.

.. note::
   
   The values of the ``grus`` dictionary can either be a string representing
   the land cover class or a dictionary containing parameters for each GRU.
   Or, it can be a dictionary containing parameters for each GRU class. If a
   dictionary is provided, the ``CLASS`` type can be defined using the
   ``class`` keyword in the dictionary. An instance is provided for class 5
   in the example above, where the ``ROOT`` parameter is set to 0.5. In another
   example, the ``LNZ0`` parameter is set to -1.4 for class 1. These parameters
   are specific to the GRU class and can be defined based on the user's
   requirements.
   
.. note:: 
   All relevant class parameters can be defined in the dictionary, and the
   ``CLASS`` model will use these parameters for the GRU class. If not provided,
   a default value will be used for the GRU class, which may not be accurate
   for process simulation. Users should use calibration workflows to determine
   the accurate values for each GRU class.

.. note::
   A list of CLASS parameters can be found in the
   `MESH documentation <https://mesh-model.atlassian.net/wiki/spaces/USER/pages/6390222/MESH_parameters_CLASS.ini>`_.


Hydrology Parameters
^^^^^^^^^^^^^^^^^^^^

The ``hydrology_params`` section allows you to specify parameters related
to hydrological processes and routing options for the model. This section
is structured to provide both routing parameters and GRU-dependent
hydrological parameters.

For example:

.. code-block:: python
   :linenos:

   >>> from meshflow import MESHWorkflow
   >>> mesh_workflow_instance = MESHWorkflow(
   ...     ...,
   ...     settings={
   ...         ...,
   ...         "hydrology_params": {
   ...             "routing": [
   ...                 {"R2N": 0.3},
   ...             ],
   ...             "hydrology": {
   ...                 5: {
   ...                     "ZSNL": 2
   ...                 },
   ...             },
   ...         },
   ...         ...,
   ...     },
   ... )


- ``routing``: A list of dictionaries specifying routing parameters.
  Each dictionary can define parameters such as ``R2N`` (e.g.,
  river-to-network routing coefficient) and other routing-related
  options as required by the model.

- ``hydrology``: A dictionary mapping GRU indices to hydrological
  parameters. Each key is a GRU index (e.g., 5), and the value is a
  dictionary of hydrological parameters for that GRU. For example,
  ``ZSNL`` can be set for a specific GRU class.

.. note::
   The available hydrological and routing parameters depend on the
   model configuration and should be set according to your simulation
   requirements. If a parameter is not provided, a default value may
   be used, which may not be optimal for your domain. Users are encouraged
   to consult the `model documentation <https://mesh-model.atlassian.net/wiki/spaces/USER/pages/6390211/MESH_parameters_hydrology.ini>`_
   for a full list of available hydrology and routing parameters.


Run Options
^^^^^^^^^^^

.. warning::
   Currently, the ``run_options`` section is not changeable, but it will be
   developed while adding water management options to the model.

Other Relevant Options
----------------------

The following options are also available in the ``settings`` parameter and
are briefly described below:

- ``forcing_local_attrs``: A dictionary describing local attributes for the
  forcing data.

- ``forcing_global_attrs``: A dictionary describing global attributes for the
  forcing data.

- ``ddb_local_attrs``: A dictionary describing local attributes for the
  drainage database (DDB).

- ``ddb_global_attrs``: A dictionary describing global attributes for the
  drainage database (DDB).

- ``ddb_min_values``: A dictionary describing minimum values for the drainage
  database (DDB).

- ``gru_dim``: The dimension name for the Grouped Response Unit (GRU).

- ``hru_dim``: The dimension name for the Hydrological Response Unit (HRU).

- ``outlet_value``: The value used to represent the outlet in the model
  (default is ``-9999``).
