# Introduction
`MESHFlow` aims to facilitate setting up MESH models for any temporal and
spatial domain of interest. This package is prepared in Python and can be
accessed from the command line using its Command Line Interface.

MESH (Modélisation Environnementale communautaire - Surface Hydrology) is
the hydrology land-surface scheme (HLSS) of Environment and Climate Change
Canada’s (ECCC’s) community environmental modelling system (Pietroniro et
al. 2007), and is complimentary to ECCC’s GEM-Hydro modelling platform.
MESH allows different surface component models to coexist within the same
modelling framework so that they can easily be compared for the same
experiment using exactly the same forcings, interpolation procedures, grid,
time period, time step and output specifications. An important feature of
MESH is its ability to read atmospheric forcings from files instead of
obtaining them from an atmospheric model. This makes it possible to test
changes to the land surface schemes offline and to drive the HLSS with
forcing data from other sources such as direct observations or reanalysis
products.

Early stages and recent evolution of Environment Canada’s
atmospheric-hydrologic-land-surface modelling system are described in
Pietroniro et al. (2007) and Wheater et al. (2022). A conceptual framework
for model development was initiated using different degrees of model
coupling that range from a linked model which requires separate calibration
of the atmospheric model and the hydrological model to a complete two way
coupled model Soulis et al. (2005). MESH evolved from the WATCLASS model
which links WATFLOOD routing model to the Canadian Land Surface Scheme
(CLASS), was used as a basis for coupling with both weather and climate
atmospheric models.

# MESHFlow
`MESHFlow` is a Python package for automating MESH model setup.

Full documentation: [mesh-workflow.readthedocs.io](https://mesh-workflow.readthedocs.io)

# License
Copyright (C) 2023, University of Calgary

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
