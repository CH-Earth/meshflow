# Introduction 
MeshFlow aims to facilitate setting up MESH models for any temporal and spatial domain of interest. This package is prepared in Python and can be accessed from command line using its Command Line Interface.

MESH (Modélisation Environnementale communautaire - Surface Hydrology) is the hydrology land-surface scheme (HLSS) of Environment and Climate Change Canada’s (ECCC’s) community environmental modelling system (Pietroniro et al. 2007), and is complimentary to ECCC’s GEM-Hydro modelling platform. MESH allows different surface component models to coexist within the same modelling framework so that they can easily be compared for the same experiment using exactly the same forcings, interpolation procedures, grid, time period, time step and output specifications. An important feature of MESH is its ability to read atmospheric forcings from files instead of obtaining them from an atmospheric model. This makes it possible to test changes to the land surface schemes offline and to drive the HLSS with forcing data from other sources such as direct observations or reanalysis products.

Early stages and recent evolution of Environment Canada’s atmospheric-hydrologic-land-surface modelling system are described in Pietroniro et al. (2007) and Wheater et al. (2022). A conceptual framework for model development was initiated using different degrees of model coupling that range from a linked model which requires separate calibration of the atmospheric model and the hydrological model to a complete two way coupled model Soulis et al. (2005). MESH evolved from the WATCLASS model which links WATFLOOD routing model to the Canadian Land Surface Scheme (CLASS), was used as a basis for coupling with both weather and climate atmospheric models.

# Usage
## Installation
To install Meshflow directly from its GitHub repository:

```console
$ git clone https://github.com/kasra-keshavarz/meshflow.git
$ pip install meshflow/.
```

Or, simply use pip:
```console
$ pip install git+https://github.com/kasra-keshavarz/meshflow.git
```

## General Usage
Meshflow comes in several interfaces. It can either be called directly from Python by instantiating package’s main class:
```python
>>> from meshflow import MESHWorkflow
>>> exp1 = MESHWorkflow()
```

Or, it can be called using its Command Line Interface (CLI):
```console
$ meshflow
Usage: meshflow [OPTIONS] COMMAND [ARGS]...

  This package automates MESH model setup through a flexible workflow that can
  be set up using a JSON configuration file, a Command Line Interface (CLI) or
  directly inside a Python script/environment.


Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  cli   Run Meshflow using CLI
  conf  Run Meshflow using a JSON configuration file

  For bug reports, questions, and discussions open an issue at
  https://github.com/kasra-keshavarz/meshflow.git
```

# Documentation
Full documentation of `Meshflow` is located at its [readthedocs](https://mesh-workflow.readthedocs.io/en/latest/index.html) webpage.

# Support
Please open a new ticket on the Issues tab of the current repository in case of any problem.

# License
MESH Modelling Setup Workflow - `MeshFlow`<br>
Copyright (C) 2023, University of Calgary<br>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
