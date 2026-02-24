"""
Utility functions for generating MESH model configuration files
using the Jinja2 templating engine.

This module provides helpers to render CLASS, hydrology, and run options
configuration files from Python dictionaries, leveraging default templates
and parameters for consistent file generation.
"""
# third-party libraries
from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
)

import xarray as xr

# built-in libraries
import json
import copy
import warnings

from typing import (
    Dict,
    Any,
    Sequence,
    Tuple,
    Optional,
    Union
)

from importlib import resources

# import internal modules
from .utils import is_int
from ..templates.aliases import normalize_alias
from .default_parameters_attrs import parameters_local_attrs as LOCAL_ATTRS

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

# GWF-inspired default parameter names
# leading DEFAULT_CLASS_PARAMS to read the default keys
with open(DEFAULT_CLASS_PARAMS, 'r') as file:
    _default_class_params = json.load(file)
DEFAULT_GWF_PARAMS: set = {p.split('_')[0] for p in _default_class_params.keys() if p.endswith('_defaults')}

# helper functions
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
    default_header_path: PathLike = DEFAULT_CLASS_HEADER, # type: ignore
    default_params_path: PathLike = DEFAULT_CLASS_PARAMS, # type: ignore
    default_case_path: PathLike = DEFAULT_CLASS_CASE, # type: ignore
    default_lines_path: PathLike = DEFAULT_CLASS_LINES, # type: ignore
    default_types_path: PathLike = DEFAULT_CLASS_TYPES, # type: ignore
    template_class_jinja_path: PathLike = TEMPLATE_CLASS, # type: ignore
) -> str:
    """
    Render a CLASS configuration file using Jinja2 templates.

    Parameters
    ----------
    class_info : dict
        Dictionary containing metadata and info for the CLASS file header.
    class_case : dict
        Dictionary containing case-specific settings for the CLASS file.
    class_grus : Sequence[dict]
        Sequence of dictionaries, each representing GRU parameters.
    default_header_path : PathLike, optional
        Path to the default CLASS header JSON file.
    default_params_path : PathLike, optional
        Path to the default CLASS parameters JSON file.
    default_case_path : PathLike, optional
        Path to the default CLASS case JSON file.
    default_lines_path : PathLike, optional
        Path to the default CLASS lines JSON file.
    default_types_path : PathLike, optional
        Path to the default CLASS types JSON file.
    template_class_jinja_path : PathLike, optional
        Path to the Jinja2 template for the CLASS file.

    Returns
    -------
    str
        Rendered CLASS configuration file as a string.

    Raises
    ------
    FileNotFoundError
        If any of the template or default files do not exist.
    Exception
        If a Jinja2 template error occurs.
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

            # add the line number as a key to the parameter type dictionary
            # if the `param_type` key does not exist, add it first
            if param_type not in d.keys():
                d[param_type] = {}

            # ensure the line key exists
            if f'line{param_line}' not in d[param_type].keys():
                d[param_type][f'line{param_line}'] = {}  # ensure the key exists

            if param_type in d.keys():
                d[param_type][f'line{param_line}'].update({param: param_value})
            else:
                d[param_type][f'line{param_line}'] = {param: param_value}
        gru_block['vars'].append(d)

    # deep update GRU blocks
    for block in gru_block['vars']:
        # make a deep copy of the default data to avoid modifying the original
        new_data = copy.deepcopy(data)

        # select the default parameter set based on the 'class' key in the block
        # if the 'class' key is not present, use the default parameter set `class_fillers`
        class_category_name = block.get('veg').get('class', None)
        if class_category_name is not None:
            # normalize the class category name using aliases
            normalized_class_category_name = normalize_alias(class_category_name)

            # if the normalized class category name is not in the default data, raise an error
            if normalized_class_category_name not in DEFAULT_GWF_PARAMS:
                normalized_class_category_name = 'class_fillers'
            else:
                normalized_class_category_name += '_defaults'
        else:
            normalized_class_category_name = 'class_fillers'

        # warnings messages
        if normalized_class_category_name == 'class_fillers':
            warnings.warn(f"Using the default parameter set for `class_fillers` "
                          "for the GRU block with class assigned as "
                          f"{class_category_name}. Parameter values may not "
                          "have sound physical meanings.")
        else:
            warnings.warn("Using the Global Water Futures (GWF) default "
                          f"parameter set for class category `{class_category_name}`."
                          " The parameter set assigned is based on GWF's "
                          f"`{normalized_class_category_name}` category.")

        # deep merge
        it = deep_merge(new_data[normalized_class_category_name], block)

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

    return content if content.endswith('\n') else content + '\n'

def render_hydrology_template(
    routing_params: Dict[str, Any] = {},
    hydrology_params: Dict[str, Any] = {},
    default_params_path: PathLike = DEFAULT_HYDROLOGY_PARAMS, # type: ignore
    template_hydrology_path: PathLike = TEMPLATE_HYDROLOGY, # type: ignore
    parameters_ds: Optional[xr.Dataset] = None,
    hru_dim: Optional[str] = 'subbasin',
    gru_dim: Optional[str] = 'NGRU',
    return_ds: bool = False,
    *args,
    **kwargs,
) -> str | Tuple[str, xr.Dataset]:
    """
    Render a hydrology parameters INI template using Jinja2.

    Parameters
    ----------
    routing_params : dict, optional
        Dictionary containing routing parameters for the template.
    hydrology_params : dict, optional
        Dictionary containing hydrology parameters for the template.
    default_params_path : PathLike, optional
        Path to the default hydrology parameters JSON file.
    template_hydrology_path : PathLike, optional
        Path to the Jinja2 template for hydrology parameters.
    parameters_ds : xarray.Dataset, optional
        An optional xarray.Dataset containing hydrology-specific parameters
        that may be needed for rendering the template. If not provided,
        it will be generated based on the routing parameters if necessary.
    hru_dim : str, optional
        Name of the dimension representing HRUs (or subbasins) in
        the parameters dataset.
    gru_dim : str, optional
        Name of the dimension representing GRUs in the parameters dataset.
    return_ds : bool, optional
        Whether to return the xarray.Dataset containing hydrology-specific
        parameters along with the rendered template. If True, the function
        will return a tuple of (rendered_template, parameters_ds). If False,
        it will return only the rendered template.

    Returns
    -------
    str
        Rendered hydrology template as a string.
    xr.Dataset, optional
        If `return_ds` is True, also return the xarray.Dataset containing
        hydrology-specific parameters.

    Raises
    ------
    FileNotFoundError
        If any of the template or default files do not exist.
    Exception
        If a Jinja2 template error occurs.
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

    # if the keys are simply integers, sorting will be based on integer values
    if all(is_int(key) for key in hydrology_params.keys()):
        # make the keys integers for sorting
        hydrology_params = {int(gru): value for gru, value in hydrology_params.items()}
    # else, leave as is (string sorting -> bad idea)

    # preparing the object in the way the Jinja2 template expects
    # the routing parameters are a list of sorted dictionaries
    # in other words, sorting is done based on the GRU classes
    hydrology_params = [{gru: hydrology_params[gru]} for gru in sorted(hydrology_params.keys())]

    # hydrology template file
    template = environment.get_template(template_hydrology_path)

    # if process_details is provided in kwargs and is not None
    additional_kwargs: dict[str, list[str]] = {}
    if 'process_details' in kwargs and kwargs.get('process_details') is not None:
        additional_kwargs = kwargs.get('process_details')

    # if addtional_kwargs is not empty, we need to determine whether
    # a MESH_parameters.nc file is necessary or not.
    #   If necessary, we will have to generate an xarray.Dataset and 
    #   fit the hydrology specific parameters inside; we also need to
    #   remove these values from the MESH_parameters_hydrology.ini file
    #   to avoid redundancy and potential conflicts.
    #   Otherwise, we will just proceed with rendering the template as is.
    if additional_kwargs:
        # if only `pwr` and `flz` are provided in additional_kwargs, then
        # we need to adjust them inside the `routing_params` to assure that
        # the number of elements matches the number of subbasins, rather than
        # the number of river classes in the normal hydrology parameters.
        # this case typically happens when no routing mechanism is turned on
        # inside MESH.
        if set(additional_kwargs['routing']) == {'pwr', 'flz'}:
            # iterate over each routing block in `routing_params`
            # remove any keys other than `pwr` and `flz`
            for routing_block in routing_params:
                keys_to_remove = set(routing_block.keys()) - {'pwr', 'flz'}
                if keys_to_remove:
                    for key in keys_to_remove:
                        routing_block.pop(key, None)
            # we will need to find the number of subbasins (in the vector
            # case) to determine the dimensions of the xarray.Dataset. 
            # the computational_units dictionary should have the necessary
            # information for this. THe specific key would be 'subbasin'.
            try:
                num_subbasins: int = int(len(parameters_ds[hru_dim]))
            except Exception as e:
                raise Exception("Error determining the number of subbasins"
                                " from `computational_units`. Use `subbasin`"
                                " as the key: " + str(e))

            # after extracting the number of basins, one has to make sure
            # the `pwr` and `flz` parameters in the `routing_params`
            # are lists of the same length as the number of subbasins.
            if len(routing_params) != num_subbasins:
                warnings.warn("Since only `pwr` and `flz` parameters are available in "
                              "routing scheme (representing baseflow), the number of routing "
                              "blocks in `routing_params` should match the "
                              "number of subbasins. Therefore, adjusting the number of"
                              f" routing blocks to {num_subbasins}.")
                # repeating the last element of `routing_params` to match
                # the number of subbasins
                last_routing_block = routing_params[-1]
                while len(routing_params) < num_subbasins:
                    routing_params.append(copy.deepcopy(last_routing_block)) # type: ignore

            # now, we can adjust the xarray.Dataset for MESH_parameters.nc
            parameters = {
                var: {
                    'dims': hru_dim,
                    'data': [routing_block.get(var) for routing_block in routing_params], # type: ignore
                    'attrs': LOCAL_ATTRS.get(var, {}),
                } for var in additional_kwargs['routing']
            }
            dims = {hru_dim: num_subbasins}
            coords = {
                hru_dim: {
                    "dims": hru_dim,
                    "data": parameters_ds[hru_dim].data, # type: ignore
                },
            }
            ds = xr.Dataset.from_dict({
                'dims': dims,
                'coords': coords,
                'data_vars': parameters,
            })
            # update the `parameters_ds` variable to be passed to the template
            parameters_ds.update(ds) # type: xarray.Dataset

    # create content
    content = template.render(
        routing_dict=routing_params,
        gru_dict=hydrology_params,
        **additional_kwargs,
    )

    # assuring the content has a newline at the end for better formatting of the INI file
    content = content if content.endswith('\n') else content + '\n'

    if return_ds:
        return content, parameters_ds # type: ignore

    return content

def render_run_options_template(
    run_options_dict: Dict[str, Any],
    template_run_options_path: PathLike = TEMPLATE_RUN_OPTIONS, # type: ignore
) -> str:
    """
    Render a run options configuration file using Jinja2 templates.

    Parameters
    ----------
    run_options_dict : dict
        Dictionary containing run options to be used in the template.
    template_run_options_path : PathLike, optional
        Path to the Jinja2 template for run options.

    Returns
    -------
    str
        Rendered run options configuration file as a string.

    Raises
    ------
    FileNotFoundError
        If any of the template or default files do not exist.
    Exception
        If a Jinja2 template error occurs.
    """
    # options template file
    template = environment.get_template(template_run_options_path)

    # create content
    content = template.render(options_dict = run_options_dict)

    # return content
    if content.endswith('\n'):
        return content
    else:
        return content + '\n'
