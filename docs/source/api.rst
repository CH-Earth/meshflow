API Reference
=============

This section provides complete API documentation for all public classes,
functions, and modules in MESHflow.

.. currentmodule:: meshflow

Core API
---------

Main Workflow Class
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   MESHWorkflow

Utility Functions
-----------------

The utility modules provide specialized functions for various tasks.

Geometry Processing
~~~~~~~~~~~~~~~~~~~

Functions for handling geometric operations and spatial data processing.

.. currentmodule:: meshflow.utility.geom

.. autosummary::
   :toctree: generated/

   extract_centroid
   prepare_mesh_coords

.. automodule:: meshflow.utility.geom
   :members:
   :show-inheritance:
   :no-index:

Network Analysis
~~~~~~~~~~~~~~~~

Functions for river network processing and topology analysis.

.. currentmodule:: meshflow.utility.network

.. autosummary::
   :toctree: generated/

   extract_rank_next
   prepare_mesh_ddb

.. automodule:: meshflow.utility.network
   :members:
   :show-inheritance:
   :no-index:

Forcing Data Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for preparing meteorological forcing data for MESH models.

.. currentmodule:: meshflow.utility.forcing_prep

.. autosummary::
   :toctree: generated/

   prepare_mesh_forcing
   freq_long_name
   calculate_time_difference

.. automodule:: meshflow.utility.forcing_prep
   :members:
   :show-inheritance:
   :no-index:

Configuration Templating
~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for generating MESH configuration files from templates.

.. currentmodule:: meshflow.utility.templating

.. autosummary::
   :toctree: generated/

   render_class_template
   render_hydrology_template
   render_run_options_template
   deep_merge

.. automodule:: meshflow.utility.templating
   :members:
   :show-inheritance:
   :no-index:
