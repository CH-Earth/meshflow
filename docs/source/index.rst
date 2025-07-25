.. meshflow documentation master file, created by
   sphinx-quickstart on Wed Aug 30 10:56:29 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MESHflow's documentation!
====================================

``MESHflow`` streamlines the setup of MESH models for any temporal or
spatial domain. Written in Python, it is accessible via a Command Line
Interface.

``MESH`` (Modélisation Environnementale communautaire - Surface Hydrology)
is the hydrology land-surface scheme (HLSS) of Environment and Climate
Change Canada's (ECCC's) community environmental modelling system
(Pietroniro et al. 2007) [#]_. It complements ECCC's GEM-Hydro modelling
platform. MESH supports multiple surface component models within a single
framework, enabling direct comparison using identical forcings,
interpolation, grids, periods, time steps, and output settings. A key
feature is its ability to read atmospheric forcings from files, allowing
offline testing and use of data from observations or reanalysis products.

The evolution of Environment Canada's atmospheric-hydrologic-land-surface
modelling system is detailed in Pietroniro et al. (2007) and Wheater et
al. (2022) [#]_. Model development began with varying degrees of coupling,
from linked models requiring separate calibration to fully coupled systems
(Soulis et al. 2005) [#]_. MESH originated from the WATCLASS model, which
integrated the WATFLOOD routing model with the Canadian Land Surface
Scheme (CLASS), and served as a foundation for coupling with weather and
climate models.

.. note::
   This project is under active development. Current release is
   `v0.1.0-dev3`. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   configuration


.. rubric:: Footnotes
.. [#] Pietroniro, A., V. Fortin, N. Kouwen, C. Neal, R. Turcotte, B. Davison, D. Verseghy et al. "Development of the MESH modelling system for hydrological ensemble forecasting of the Laurentian Great Lakes at the regional scale." Hydrology and Earth System Sciences 11, no. 4 (2007): 1279-1294.
.. [#] Wheater, Howard S., John W. Pomeroy, Alain Pietroniro, Bruce Davison, Mohamed Elshamy, Fuad Yassin, Prabin Rokaya et al. "Advances in modelling large river basins in cold regions with Modélisation Environmentale Communautaire—Surface and Hydrology (MESH), the Canadian hydrological land surface scheme." Hydrological Processes 36, no. 4 (2022): e14557.
.. [#] Soulis, E. D., N. Kouwen, Al Pietroniro, F. R. Seglenieks, K. R. Snelgrove, P. Pellerin, D. W. Shaw, and L. W. Martz. "A framework for hydrological modelling in MAGS." Prediction in Ungauged Basins: Approaches for Canada’s Cold Regions, edited by: Spence, C., Pomeroy, JW and Pietroniro, A., Canadian Water Resources Association (2005): 119-138.
