"""Utility functions for MESHFlow."""
import ast
from typing import Dict, Any


def is_int(s: str) -> bool:
    try:
        int(s)  # Uses base 10 unless prefixes (0x, 0b, 0o) appear
        return True
    except ValueError:
        return False


def _parse_group_key(key: Any) -> Any:
    """Parse a potential group key into an iterable of items.

    Supports:
    - tuple: (1, 2, 3) — the only valid grouping key in a Python dict
    - string representation of tuple/list: "(1, 2, 3)" or "[1, 2, 3]"
    - plain value: returned as-is

    .. note::
       Lists cannot be dictionary keys in Python. Use tuples when defining
       grouped keys directly in Python dictionaries. String representations
       (e.g. from JSON) are parsed automatically.

    Parameters
    ----------
    key : any
        The dictionary key to parse.

    Returns
    -------
    any
        An iterable (tuple) if the key represents a group,
        otherwise the original key.
    """
    if isinstance(key, (list, tuple)):
        return key
    if isinstance(key, str):
        stripped = key.strip()
        if (stripped.startswith('[') and stripped.endswith(']')) or \
           (stripped.startswith('(') and stripped.endswith(')')):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return parsed
            except (ValueError, SyntaxError):
                pass
    return key


def expand_grouped_keys(d: Dict[Any, Any]) -> Dict[Any, Any]:
    """Expand dictionary keys that represent groups into individual entries.

    Given a dictionary where some keys may be tuples or string
    representations thereof, return a new dictionary where each item in a
    group key maps to the same value. Non-group keys are preserved unchanged.

    Parameters
    ----------
    d : dict
        Input dictionary potentially containing grouped keys.

    Returns
    -------
    dict
        New dictionary with all grouped keys expanded.

    Examples
    --------
    >>> expand_grouped_keys({1: {"a": 1}, (2, 3): {"a": 2}})
    {1: {'a': 1}, 2: {'a': 2}, 3: {'a': 2}}

    >>> expand_grouped_keys({"(1, 4, 17)": {"zsnl": 0.5}})
    {1: {'zsnl': 0.5}, 4: {'zsnl': 0.5}, 17: {'zsnl': 0.5}}

    .. note::
       Lists cannot be dictionary keys in Python. Use tuples for grouped
       keys in Python code, or string representations (e.g. from JSON).
    """
    expanded: Dict[Any, Any] = {}
    for key, value in d.items():
        parsed = _parse_group_key(key)
        if isinstance(parsed, (list, tuple)):
            for item in parsed:
                expanded[item] = value
        else:
            expanded[parsed] = value
    return expanded
