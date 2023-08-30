.. meshflow documentation master file, created by
   sphinx-quickstart on Wed Aug 30 10:56:29 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Meshflow's documentation!
====================================
``Meshflow`` aims to facilitate setting up MESH models for any temporal and
spatial domain of interest. This package is prepared in Python and can be
accessed from command line using its Command Line Interface. 

MESH (Modélisation Environnementale communautaire - Surface Hydrology) is
the hydrology land-surface scheme (HLSS) of Environment and Climate Change
Canada's (ECCC's) community environmental modelling system (Pietroniro et
al. 2007), and is complimentary to ECCC's GEM-Hydro modelling platform.
MESH allows different surface component models to coexist within the same
modelling framework so that they can easily be compared for the same
experiment using exactly the same forcings, interpolation procedures,
grid, time period, time step and output specifications. An important
feature of MESH is its ability to read atmospheric forcings from files
instead of obtaining them from an atmospheric model. This makes it
possible to test changes to the land surface schemes offline and to drive
the HLSS with forcing data from other sources such as direct observations
or reanalysis products.

Early stages and recent evolution of Environment Canada's
atmospheric-hydrologic-land-surface modelling system are described in
Pietroniro et al. (2007) and Wheater et al. (2022). A conceptual framework
for model development was initiated using different degrees of model
coupling that range from a linked model which requires separate
calibration of the atmospheric model and the hydrological model to a
complete two way coupled model Soulis et al. (2005). MESH evolved from the
WATCLASS model which links WATFLOOD routing model to the Canadian Land
Surface Scheme (CLASS), was used as a basis for coupling with both weather
and climate atmospheric models.

.. note::
   This project is under active development. Current release is
   `v0.1.0-dev0`. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   examples
   applications using Digital Research Alliance (DRA) HPCs



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
