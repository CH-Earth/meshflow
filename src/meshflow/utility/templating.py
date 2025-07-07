"""
This module provides functions to generate textual configuration files
using Jinja2 templating engine for MESH model instantiations.
"""
# third-party libraries
from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
)

# built-in libraries
import os
import json
import copy

from typing import (
    Dict,
    Any,
    Sequence
)

from importlib import resources

# custom type hints
try:
    from os import PathLike
except ImportError:  # <Python3.8
    from typing import Union
    PathLike = Union[str, bytes]


# constants
TEMPLATE_CLASS = "MESH_parameters_CLASS.ini.jinja"
TEMPLATE_HYDROLOGY = "MESH_parameters_hydrology.ini.jinja"
TEMPLATE_RUN_OPTIONS = "MESH_input_run_options.ini.jinja"

DEFAULT_CLASS_HEADER = resources.files("meshflow.templates").joinpath("default_CLASS_header.json")
DEFAULT_CLASS_CASE = resources.files("meshflow.templates").joinpath("default_CLASS_case.json")
DEFAULT_CLASS_PARAMS = resources.files("meshflow.templates").joinpath("default_CLASS_parameters.json")
DEFAULT_HYDROLOGY_PARAMS = resources.files("meshflow.templates").joinpath("default_hydrology_parameters.json")
DEFAULT_RUN_OPTIONS = resources.files("meshflow.templates").joinpath("default_input_run_options.json")

DEFAULT_CLASS_LINES = resources.files("meshflow.templates").joinpath("default_CLASS_lines.json")
DEFAULT_CLASS_TYPES = resources.files("meshflow.templates").joinpath("default_CLASS_types.json")

# global variables and helper functions
def raise_helper(msg):
    """Jinja2 helper function to raise exceptions."""
    raise Exception(msg)
# Jinja2 environment setup
environment = Environment(
    loader=PackageLoader("meshflow", "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
    line_comment_prefix='##',
)
environment.globals['raise'] = raise_helper


def deep_merge(
    d1: Dict[str, Any],
    d2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge d2 dictionary into d1.
    """
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            deep_merge(d1[key], value)  # Recursive merge for nested dictionaries
        else:
            d1[key] = value  # Overwrite or add new key-value pair
    return d1


def render_class_template(
    class_info: Dict[str, Any],
    class_case: Dict[str, Any],
    class_grus: Sequence[Dict[str, Any]],
    default_header_path: PathLike = DEFAULT_CLASS_HEADER,
    default_params_path: PathLike = DEFAULT_CLASS_PARAMS,
    default_case_path: PathLike = DEFAULT_CLASS_CASE,
    default_lines_path: PathLike = DEFAULT_CLASS_LINES,
    default_types_path: PathLike = DEFAULT_CLASS_TYPES,
    template_class_jinja_path: PathLike = TEMPLATE_CLASS,
) -> str:
    """
    Render a class template using Jinja2.

    Parameters
    ----------
    class_params : dict
        A dictionary containing class parameters to be used in the template.
    default_header_path : PathLike
        Path to the template heading file.
    template_body_path : PathLike
        Path to the template body file.
    template_case_path : PathLike
        Path to the template case file.

    Returns
    -------
    str
        Rendered class template as a string.

    Raises
    ------
    FileNotFoundError
        If any of the template files do not exist.
    """
    # load the default values for each GRU
    with open(default_params_path, 'r') as file:
        data = json.load(file)
    with open(default_header_path, 'r') as file:
        info = json.load(file)
    with open(default_case_path, 'r') as file:
        case = json.load(file)
    with open(default_lines_path, 'r') as file:
        gru_lines = json.load(file)
    with open(default_types_path, 'r') as file:
        gru_types = json.load(file)

    # populate new dictionary for GRU blocks in the CLASS file
    populating_list = []

    # create a dictionary for GRU blocks
    gru_block = {"vars": []}

    for gru, params in class_grus.items():
        d = {}
        for param, param_value in params.items():
            if param == 'class':
                d['veg'] = {'class': param_value}
                continue
            param_line = gru_lines[param]
            param_type = gru_types[param]

            if param_type in d.keys():
                d[param_type].update({f'line{param_line}': {param: param_value}})
            else:
                d[param_type] = {f'line{param_line}': {param: param_value}}
        gru_block['vars'].append(d)

    # deep update GRU blocks
    for block in gru_block['vars']:
        new_data = copy.deepcopy(data)

        # deep merge
        it = deep_merge(new_data['class_defaults'], block)

        # update the block dictionary
        populating_list.append(it)

    # update the class parameters with the populated list
    # gru_block['vars'] is populated based on the assumption that
    # in Python3.7+, the dictionaries maintain insertion order
    # so the order of GRU blocks in the CLASS file is preserved
    # as per the order of class_grus dictionary
    # this is important for the MESH model to read the CLASS file correctly
    # and also needs consistency between various input files
    gru_block.update({'vars': populating_list})

    # update case block
    class_case = deep_merge(case, {"vars": {"case": class_case}})

    # update info block
    class_info = deep_merge(info, {"vars": class_info})

    # add formats, columns, and comments
    new_keys = ('formats', 'comments', 'columns')
    for key in new_keys:
        gru_block[key] = data[key]

    # create the template environment
    template = environment.get_template(template_class_jinja_path)

    # create content
    content = template.render(
        info_block=class_info,
        case_block=class_case,
        gru_block=gru_block,
        variables="vars",
        comments="comments",
        formats="formats",
        columns="columns",
    )

    return content


def render_hydrology_template(
    routing_params: Dict[str, Any] = {},
    hydrology_params: Dict[str, Any] = {},
    default_params_path: PathLike = DEFAULT_HYDROLOGY_PARAMS,
    template_hydrology_path: PathLike = TEMPLATE_HYDROLOGY,
) -> str:
    """
    Render a hydrology parameters ini template using Jinja2.

    Parameters
    ----------
    routing_params : dict
        A dictionary containing routing parameters to be used in the template.
    hydrology_params : dict
        A dictionary containing hydrology parameters to be used in the template.
    default_params_path : PathLike
        Path to the default hydrology parameters file.
    template_hydrology_path : PathLike
        Path to the hydrology template file.

    Returns
    -------
    str
        Rendered hydrology template as a string.

    Raises
    ------
    FileNotFoundError
        If any of the template files do not exist.
    """
    # load the default values for each GRU
    with open(default_params_path, 'r') as file:
        data = json.load(file)

    # components
    routing_defaults = data.get('routing')
    hydrology_defaults = data.get('hydrology')

    # deep update routing block
    for idx, routing_block in enumerate(routing_params):
        defaults = copy.deepcopy(routing_defaults)

        # deep merge
        defaults.update(routing_block)

        # update the dict (list of dicts)
        routing_params[idx].update(defaults)

    # deep update gru block
    for gru, gru_hydro_params in hydrology_params.items():
        defaults = copy.deepcopy(hydrology_defaults)

        # update default values
        defaults.update(gru_hydro_params)

        # update the dict
        hydrology_params[gru].update(defaults)

    # preparing the object in the way the Jinja2 template expects
    # the routing parameters are a list of sorted dictionaries
    # in other words, sorting is done based on the GRU classes
    hydrology_params = [{gru: hydrology_params[gru]} for gru in sorted(hydrology_params.keys())]

    # hydrology template file
    template = environment.get_template(template_hydrology_path)

    # create content
    content = template.render(
        routing_dict=routing_params,
        gru_dict=hydrology_params,
    )

    return content


def render_run_options_template(
    run_options_dict: Dict[str, Any],
    default_run_options_path: PathLike = DEFAULT_RUN_OPTIONS,
    template_run_options_path: PathLike = TEMPLATE_RUN_OPTIONS,
) -> str:
    """Render a run options template using Jinja2.

    Parameters
    ----------
    run_options : dict
        A dictionary containing run options to be used in the template.
    default_run_options_path : PathLike
        Path to the default run options file.
    template_run_options_path : PathLike
        Path to the run options template file.

    Returns
    -------
    str
        Rendered run options template as a string.

    Raises
    ------
    FileNotFoundError
        If any of the template files do not exist.
    """
    # load the default values for each GRU
    with open(default_run_options_path, 'r') as file:
        options = json.load(file)

    # deep update the dictionary
    options['settings'].update(deep_merge(options['settings'], run_options_dict))

    # options template file
    template = environment.get_template(template_run_options_path)

    # create content
    content = template.render(
        options_dict = options,
    )

    return content
