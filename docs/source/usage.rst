Usage
=====

Installation
------------

To install `Meshflow` directly from its GitHub repository:

.. code-block:: console 
   :linenos:

   $ git clone https://github.com/kasra-keshavarz/meshflow.git
   $ pip install meshflow/.

Or, simply use pip:

.. code-block:: console 
   :linenos:

   $ pip install git+https://github.com/kasra-keshavarz/meshflow.git


General Usage
-------------
`Meshflow` comes in several interfaces. It can either be called directly
from Python by instantiating package's main class:

.. code-block:: python
   :linenos:

   >>> from meshflow import MESHWorkflow
   >>> exp1 = MESHWorkflow()


Or, it can be called using its Command Line Interface (CLI):

.. code-block:: console 
   :linenos:

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

In both interfaces, you may use a `JSON` configuration file to set up the
`MESHWorkflow` and build a set up for a domain of interest.
