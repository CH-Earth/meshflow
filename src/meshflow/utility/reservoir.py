"""
Utilities for preparing MESH_input_reservoir.txt context data.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import geopandas as gpd
import pandas as pd

RESERVOIR_LINK_KEYS = ("reservoir_id", "basin_id")
RESERVOIR_ID_ALIASES = ("COMID", "comid", "main_id", "id", "basin_id")
RESERVOIR_PARAM_ID_ALIASES = (
    "reservoir_id",
    "reservoirid",
    "res_id",
    "reservoir",
    "id",
)
RESERVOIR_NAME_ALIASES = ("name", "reservoir_name", "res_name", "reservoir")
RESERVOIR_B1_ALIASES = ("b1", "B1", "WF_B1", "wf_b1")
RESERVOIR_B2_ALIASES = ("b2", "B2", "WF_B2", "wf_b2")
RESERVOIR_INPUT_FLAGS = (1, 3)
RESERVOIR_FILE_FORMATS = ("txt", "tb0")
RESERVOIR_FILE_FORMAT_DEFAULT = "txt"
RESERVOIR_NAME_GAP = 25
TB0_FILE_BANNER = "########################################"


def _expand_grouped_keys(mapping: Mapping[Any, Any]) -> Dict[Any, Any]:
    """Expand grouped dict keys; works when this module is loaded standalone."""
    try:
        from .utils import expand_grouped_keys
    except ImportError:  # pragma: no cover - unit-test standalone load
        import importlib.util

        utils_path = os.path.join(os.path.dirname(__file__), "utils.py")
        spec = importlib.util.spec_from_file_location(
            "_meshflow_utils_for_reservoir", utils_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        expand_grouped_keys = module.expand_grouped_keys
    return expand_grouped_keys(dict(mapping))


def normalize_lake_indicator(values: pd.Series) -> pd.Series:
    """
    Convert a lake indicator / lake-ID column to 0/1 flags.

    Non-lake sentinels are ``0``, ``-1``, and ``NaN``. Any other value is
    treated as a lake (for example a positive lake area or lake ID).
    """
    filled = values.fillna(0)
    return ((filled != 0) & (filled != -1)).astype("int32")


def degrees_to_mesh_minutes(degrees: float) -> float:
    """Convert decimal degrees to MESH north-south/east-west minutes."""
    return float(degrees) * 60.0


def normalize_reservoir_file_format(flag: Any) -> tuple[str, bool]:
    """
    Return a supported ``RESERVOIRFILEFLAG`` value.

    Parameters
    ----------
    flag : Any
        Requested reservoir file format.

    Returns
    -------
    tuple[str, bool]
        Normalized format (``txt`` or ``tb0``) and whether the input was
        invalid and corrected to the default.
    """
    if flag is None:
        return RESERVOIR_FILE_FORMAT_DEFAULT, False

    normalized = str(flag).strip().lower()
    if not normalized:
        return RESERVOIR_FILE_FORMAT_DEFAULT, False
    if normalized in RESERVOIR_FILE_FORMATS:
        return normalized, False

    return RESERVOIR_FILE_FORMAT_DEFAULT, True


def default_reservoir_name(basin_id: Union[int, str]) -> str:
    """Return the default reservoir name for a basin without CSV metadata."""
    return str(basin_id)


def mesh_reservoir_name(name: str) -> str:
    """Normalize reservoir names for MESH text files."""
    return " ".join(str(name).split())[:12]


def tb0_coeff(value: float) -> str:
    """Format a reservoir coefficient for ``MESH_input_reservoir.tb0``."""
    value = float(value)
    if value == 0.0:
        return "0.0E+00"
    text = f"{value:.2E}".replace("e", "E")
    if text.endswith("E+00") and len(text) > 4 and text[3] == "0":
        return text[:3] + text[4:]
    return text


def tb0_location(value: float) -> str:
    """Format a decimal-degree location for ``MESH_input_reservoir.tb0``."""
    return f"{float(value):g}"


def tb0_reach_area(value: float) -> str:
    """Format a reach area in square metres for ``MESH_input_reservoir.tb0``."""
    return str(int(round(float(value))))


def format_tb0_column_lines(column_rows: Sequence[Sequence[str]]) -> List[str]:
    """
    Left-align column values to even widths with at least one space between.

    Each column's width is the widest value in that column across all rows.
    Values are left-aligned within their column width and joined with a
    single space so columns stay vertically aligned.
    """
    if not column_rows:
        return []

    n_cols = len(column_rows[0])
    if any(len(row) != n_cols for row in column_rows):
        raise ValueError("All tb0 column metadata rows must have the same length.")

    widths = [
        max((len(row[i]) for row in column_rows), default=0)
        for i in range(n_cols)
    ]
    return [
        " ".join(value.ljust(widths[i]) for i, value in enumerate(row)).rstrip()
        for row in column_rows
    ]


def build_tb0_column_metadata(
    reservoirs: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    """Build left-aligned ``:ColumnMetaData`` value lines for the tb0 file."""
    if not reservoirs:
        return {
            "column_type": "",
            "column_units": "",
            "column_name": "",
            "column_model": "",
            "column_location_x": "",
            "column_location_y": "",
            "coeff1": "",
            "coeff2": "",
            "reach_area": "",
        }

    rows = [
        ["float"] * len(reservoirs),
        ["m3/s"] * len(reservoirs),
        [mesh_reservoir_name(reservoir["name"]) for reservoir in reservoirs],
        ["LAKE"] * len(reservoirs),
        [tb0_location(reservoir["lon"]) for reservoir in reservoirs],
        [tb0_location(reservoir["lat"]) for reservoir in reservoirs],
        [tb0_coeff(reservoir["b1"]) for reservoir in reservoirs],
        [tb0_coeff(reservoir["b2"]) for reservoir in reservoirs],
        [tb0_reach_area(reservoir["reach_area"]) for reservoir in reservoirs],
    ]
    (
        column_type,
        column_units,
        column_name,
        column_model,
        column_location_x,
        column_location_y,
        coeff1,
        coeff2,
        reach_area,
    ) = format_tb0_column_lines(rows)

    return {
        "column_type": column_type,
        "column_units": column_units,
        "column_name": column_name,
        "column_model": column_model,
        "column_location_x": column_location_x,
        "column_location_y": column_location_y,
        "coeff1": coeff1,
        "coeff2": coeff2,
        "reach_area": reach_area,
    }


def mesh_timestep_delta_t_hours(timestep_minutes: float) -> str:
    """Convert MESH ``TIMESTEPFLAG`` minutes to tb0 ``:DeltaT`` hours."""
    hours = float(timestep_minutes) / 60.0
    if hours == int(hours):
        return str(int(hours))
    return f"{hours:g}"


def fortran_i5(value: int) -> str:
    """Format an integer using Fortran ``i5``."""
    return f"{int(value):5d}"


def fortran_f7_1(value: float) -> str:
    """Format a real using Fortran ``f7.1``."""
    return f"{float(value):7.1f}"


def fortran_g10_3(value: float) -> str:
    """Format a real using Fortran ``g10.3``.

    Uses fixed decimals for typical magnitudes and scientific notation (e.g.
    ``3.50E-14``) when values are too small, too large, or would round to zero
    in fixed format.
    """
    value = float(value)
    if value == 0.0:
        return f"{0.0:10.3f}"

    abs_val = abs(value)
    fixed = f"{value:10.3f}"
    if abs_val < 1e-3 or abs_val >= 1e7 or float(fixed) == 0.0:
        return f"{value:10.2E}"
    return fixed


def fortran_a12(value: str) -> str:
    """Format a string using Fortran ``a12``."""
    return f"{str(value)[:12]:12s}"


def fortran_i2(value: int) -> str:
    """Format an integer using Fortran ``i2``."""
    return f"{int(value):2d}"


def format_reservoir_header(n_reservoirs: int) -> str:
    """Return the MESH reservoir file header line."""
    return (
        f"{fortran_i5(n_reservoirs)} {fortran_i5(0)} {fortran_i5(0)}"
    )


def format_reservoir_line(
    lat_min: float,
    lon_min: float,
    b1: float,
    b2: float,
    name: str,
    ireach_num: int,
    location_flag: int = 0,
) -> str:
    """
    Return one aligned, space-separated reservoir record line.

    Kept for tests and direct formatting checks. Production output is rendered
    through ``MESH_input_reservoir.txt.jinja``.
    """
    if location_flag == 1:
        lat_field = fortran_f7_1(lat_min)
        lon_field = fortran_f7_1(lon_min)
    else:
        lat_field = fortran_i5(int(round(lat_min)))
        lon_field = fortran_i5(int(round(lon_min)))

    return (
        f"{lat_field} {lon_field} {fortran_g10_3(b1)} {fortran_g10_3(b2)}"
        f"{(' ' * RESERVOIR_NAME_GAP)}{fortran_a12(mesh_reservoir_name(name))} "
        f"{fortran_i2(ireach_num)}"
    )


def parse_reservoir_coefficient_link(
    column_config: Optional[Mapping[str, Any]],
    main_id: str,
) -> tuple[str, Optional[str]]:
    """
    Resolve how reservoir coefficients are joined to catchments.

    ``column_config`` may contain exactly one of:

    - ``reservoir_id``: CSV column matched to ``ddb_vars['reservoir_id']``
      (default when no link key is given)
    - ``basin_id``: CSV column matched to the catchment ``main_id``

    Legacy ``id_col`` entries imply ``basin_id`` linking.
    """
    if not column_config:
        return "reservoir_id", None

    found: List[tuple[str, str]] = []
    for link_key in RESERVOIR_LINK_KEYS:
        if link_key in column_config:
            value = column_config[link_key]
            if value in (None, ""):
                raise ValueError(
                    f"`{link_key}` in `reservoir_coefficient_columns` must be "
                    "a parameter-file column name."
                )
            found.append((link_key, str(value)))

    if len(found) > 1:
        keys = [key for key, _ in found]
        raise ValueError(
            "Specify only one reservoir coefficient link key in "
            f"`reservoir_coefficient_columns`; got {keys}."
        )
    if found:
        return found[0]

    if "id_col" in column_config:
        value = column_config["id_col"]
        if value in (None, ""):
            raise ValueError(
                "`id_col` in `reservoir_coefficient_columns` must be a "
                "parameter-file column name."
            )
        return "basin_id", str(value)

    return "reservoir_id", None


def _resolve_column(
    columns: Sequence[str],
    aliases: Sequence[str],
    explicit: Optional[str] = None,
) -> Optional[str]:
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(
                f"Column `{explicit}` not found. Available columns: {list(columns)}"
            )
        return explicit

    for alias in aliases:
        if alias in columns:
            return alias

    return None


def read_reservoir_coefficients(
    csv_path: str,
    main_id: str,
    link_key: str = "reservoir_id",
    id_col: Optional[str] = None,
    name_col: Optional[str] = None,
    b1_col: Optional[str] = None,
    b2_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read reservoir power-curve coefficients from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the coefficient CSV file.
    main_id : str
        Name of the basin ID column on the catchment data (e.g. ``COMID``).
    link_key : str, optional
        Join field for coefficient lookup: ``reservoir_id`` (default) or
        ``basin_id``.
    id_col, name_col, b1_col, b2_col : str, optional
        Explicit column names. When omitted, common aliases are detected.
    """
    if link_key not in RESERVOIR_LINK_KEYS:
        raise ValueError(
            f"`link_key` must be one of {RESERVOIR_LINK_KEYS}; got `{link_key}`."
        )

    coeff_df = pd.read_csv(csv_path)
    columns = list(coeff_df.columns)

    if link_key == "basin_id":
        id_aliases = (main_id, *RESERVOIR_ID_ALIASES)
        id_label = "basin ID"
    else:
        id_aliases = RESERVOIR_PARAM_ID_ALIASES
        id_label = "reservoir ID"

    resolved_id = _resolve_column(columns, id_aliases, explicit=id_col)
    if resolved_id is None:
        raise ValueError(
            f"Could not find a {id_label} column in `{csv_path}`. "
            f"Expected one of {id_aliases}."
        )

    resolved_name = _resolve_column(columns, RESERVOIR_NAME_ALIASES, explicit=name_col)
    resolved_b1 = _resolve_column(columns, RESERVOIR_B1_ALIASES, explicit=b1_col)
    resolved_b2 = _resolve_column(columns, RESERVOIR_B2_ALIASES, explicit=b2_col)

    if resolved_b1 is None or resolved_b2 is None:
        raise ValueError(
            f"Could not find B1/B2 coefficient columns in `{csv_path}`. "
            f"Expected columns like {RESERVOIR_B1_ALIASES} and {RESERVOIR_B2_ALIASES}."
        )

    out = coeff_df[[resolved_id, resolved_b1, resolved_b2]].copy()
    out.columns = [link_key, "b1", "b2"]
    if resolved_name is not None:
        out["name"] = coeff_df[resolved_name]
    else:
        out["name"] = pd.NA

    out = out.drop_duplicates(subset=[link_key], keep="first")
    return out.set_index(link_key, drop=False)


def _resolve_param_value(
    entry: Mapping[str, Any],
    aliases: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first matching alias value from a parameter entry."""
    lower_map = {str(key).lower(): value for key, value in entry.items()}
    for alias in aliases:
        if alias in entry:
            return entry[alias]
        lowered = alias.lower()
        if lowered in lower_map:
            return lower_map[lowered]
    return default


def normalize_reservoir_param_entry(
    entry: Optional[Mapping[str, Any]] = None,
    *,
    fill_missing: bool = True,
) -> Dict[str, Any]:
    """
    Normalize a reservoir parameter entry to ``b1``, ``b2``, and optional
    ``name``.

    Accepts common aliases such as ``B1`` / ``WF_B1`` and ``B2`` / ``WF_B2``.
    When ``fill_missing`` is True (default), missing coefficients default to
    ``0.0``. When False, only explicitly provided fields are returned so
    partial overlays can update a single coefficient during calibration.
    """
    if entry is None:
        entry = {}
    if not isinstance(entry, Mapping):
        raise ValueError("Reservoir parameter entries must be dictionaries.")

    lower_keys = {str(key).lower() for key in entry.keys()}
    has_b1 = any(alias.lower() in lower_keys or alias in entry for alias in RESERVOIR_B1_ALIASES)
    has_b2 = any(alias.lower() in lower_keys or alias in entry for alias in RESERVOIR_B2_ALIASES)
    has_name = any(alias.lower() in lower_keys or alias in entry for alias in RESERVOIR_NAME_ALIASES)

    normalized: Dict[str, Any] = {}

    if has_b1 or fill_missing:
        raw_b1 = _resolve_param_value(entry, RESERVOIR_B1_ALIASES, default=0.0)
        if raw_b1 is None or (isinstance(raw_b1, float) and pd.isna(raw_b1)):
            raw_b1 = 0.0
        normalized["b1"] = float(raw_b1)

    if has_b2 or fill_missing:
        raw_b2 = _resolve_param_value(entry, RESERVOIR_B2_ALIASES, default=0.0)
        if raw_b2 is None or (isinstance(raw_b2, float) and pd.isna(raw_b2)):
            raw_b2 = 0.0
        normalized["b2"] = float(raw_b2)

    if has_name:
        raw_name = _resolve_param_value(entry, RESERVOIR_NAME_ALIASES, default=None)
        if raw_name is not None and not (isinstance(raw_name, float) and pd.isna(raw_name)):
            name = str(raw_name).strip()
            if name:
                normalized["name"] = name

    return normalized


def coefficients_to_reservoir_params(
    coefficients: pd.DataFrame,
    link_key: str = "reservoir_id",
) -> Dict[Any, Dict[str, Any]]:
    """Convert a coefficient DataFrame into a calibratable parameter dict."""
    if link_key not in RESERVOIR_LINK_KEYS:
        raise ValueError(
            f"`link_key` must be one of {RESERVOIR_LINK_KEYS}; got `{link_key}`."
        )
    if coefficients is None or coefficients.empty:
        return {}
    if link_key not in coefficients.columns:
        raise ValueError(
            f"Coefficient table is missing link column `{link_key}`."
        )

    params: Dict[Any, Dict[str, Any]] = {}
    for _, row in coefficients.drop_duplicates(subset=[link_key], keep="first").iterrows():
        link_id = row[link_key]
        entry = {
            "b1": row["b1"] if "b1" in row.index else 0.0,
            "b2": row["b2"] if "b2" in row.index else 0.0,
        }
        if "name" in row.index and not pd.isna(row["name"]):
            entry["name"] = row["name"]
        params[link_id] = normalize_reservoir_param_entry(entry)
    return params


def reservoir_params_to_coefficients(
    reservoirs: Mapping[Any, Any],
    link_key: str = "reservoir_id",
) -> pd.DataFrame:
    """
    Convert a calibratable reservoir parameter dict to a coefficient
    DataFrame accepted by :func:`prepare_reservoir_context`.
    """
    if link_key not in RESERVOIR_LINK_KEYS:
        raise ValueError(
            f"`link_key` must be one of {RESERVOIR_LINK_KEYS}; got `{link_key}`."
        )

    expanded = _expand_grouped_keys(reservoirs) if reservoirs else {}
    rows: List[Dict[str, Any]] = []
    for link_id, entry in expanded.items():
        normalized = normalize_reservoir_param_entry(
            entry if isinstance(entry, Mapping) else {}
        )
        row: Dict[str, Any] = {
            link_key: link_id,
            "b1": normalized["b1"],
            "b2": normalized["b2"],
            "name": normalized.get("name", pd.NA),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[link_key, "b1", "b2", "name"]).set_index(
            link_key, drop=False
        )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=[link_key], keep="first")
    return out.set_index(link_key, drop=False)


def merge_reservoir_params(
    base: Optional[Mapping[Any, Any]] = None,
    overlay: Optional[Mapping[Any, Any]] = None,
) -> Dict[Any, Dict[str, Any]]:
    """
    Merge reservoir parameter dictionaries.

    ``overlay`` values replace matching keys in ``base`` (string/int soft
    match). Grouped keys are expanded like CLASS ``grus``. Partial overlay
    entries only update the fields they provide.
    """
    merged: Dict[Any, Dict[str, Any]] = {}
    for link_id, entry in _expand_grouped_keys(base or {}).items():
        merged[link_id] = normalize_reservoir_param_entry(
            entry if isinstance(entry, Mapping) else {}
        )

    for link_id, entry in _expand_grouped_keys(overlay or {}).items():
        normalized = normalize_reservoir_param_entry(
            entry if isinstance(entry, Mapping) else {},
            fill_missing=False,
        )
        existing_key = None
        if link_id in merged:
            existing_key = link_id
        else:
            for key in merged:
                if str(key) == str(link_id):
                    existing_key = key
                    break
        if existing_key is None:
            merged[link_id] = normalize_reservoir_param_entry(normalized)
        else:
            merged[existing_key] = {**merged[existing_key], **normalized}
    return merged


def seed_reservoir_params_from_catchments(
    cat: gpd.GeoDataFrame,
    main_id: str,
    lake_col: str = "IREACH",
    coeff_link_key: str = "reservoir_id",
    coeff_link_col: Optional[str] = None,
) -> Dict[Any, Dict[str, Any]]:
    """
    Seed zero-coefficient entries for every lake catchment.

    Keys use ``coeff_link_col`` when linking on ``reservoir_id``, otherwise
    ``main_id`` when linking on ``basin_id``.
    """
    if coeff_link_key not in RESERVOIR_LINK_KEYS:
        raise ValueError(
            f"`coeff_link_key` must be one of {RESERVOIR_LINK_KEYS}; "
            f"got `{coeff_link_key}`."
        )
    if lake_col not in cat.columns:
        raise ValueError(
            f"Lake column `{lake_col}` not found in catchment data. "
            f"Available columns: {list(cat.columns)}"
        )

    if coeff_link_col is None:
        coeff_link_col = main_id if coeff_link_key == "basin_id" else "reservoir_id"
    if coeff_link_col not in cat.columns:
        raise ValueError(
            f"Coefficient link column `{coeff_link_col}` not found in "
            f"catchment data. Available columns: {list(cat.columns)}"
        )

    lakes = (
        cat.loc[cat[lake_col] > 0, [main_id, coeff_link_col]]
        .drop_duplicates(subset=[main_id])
    )
    params: Dict[Any, Dict[str, Any]] = {}
    for row in lakes.itertuples(index=False):
        basin_id = getattr(row, main_id)
        link_id = getattr(row, coeff_link_col)
        default_name = default_reservoir_name(
            link_id if coeff_link_key == "reservoir_id" else basin_id
        )
        params[link_id] = {
            "b1": 0.0,
            "b2": 0.0,
            "name": default_name,
        }
    return params


def _resolve_reach_area(
    lake_area: Any,
    subbasin_area: Any,
    basin_id: Union[int, str],
) -> float:
    """Return lake reach area, preferring ``lake_area`` over subbasin area."""
    if lake_area is not None and not pd.isna(lake_area):
        value = float(lake_area)
        if value > 0.0:
            return value

    if subbasin_area is None or pd.isna(subbasin_area):
        raise ValueError(
            f"Could not resolve reach area for basin `{basin_id}`. "
            "Provide a positive `lake_area` value or subbasin area."
        )

    return float(subbasin_area)


def _lookup_mapping_value(
    mapping: Mapping[Any, Any],
    key: Union[int, str],
) -> Any:
    value = mapping.get(key)
    if value is not None:
        return value

    for map_key, map_value in mapping.items():
        if str(map_key) == str(key):
            return map_value

    return None


def _lookup_coefficient(
    coeff_lookup: Dict[Any, Dict[str, Any]],
    basin_id: Union[int, str],
) -> Optional[Dict[str, Any]]:
    coeff = _lookup_mapping_value(coeff_lookup, basin_id)
    if isinstance(coeff, dict):
        return coeff
    return None


def prepare_reservoir_context(
    cat: gpd.GeoDataFrame,
    coords: pd.DataFrame,
    main_id: str,
    lake_col: str = "IREACH",
    coefficients: Optional[pd.DataFrame] = None,
    location_flag: int = 0,
    use_power_coefficients: bool = True,
    lake_area_col: Optional[str] = None,
    subbasin_areas: Optional[Mapping[Any, Any]] = None,
    coeff_link_key: str = "reservoir_id",
    coeff_link_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the Jinja context for reservoir input files.

    Lakes are taken from ``lake_col`` values greater than zero and written in
    ascending ``IREACH`` order. Locations use catchment centroids converted
    from decimal degrees to MESH minutes (degrees × 60).

    Parameters
    ----------
    cat : geopandas.GeoDataFrame
        Catchment data containing ``main_id`` and ``lake_col``.
    coords : pandas.DataFrame
        Centroid coordinates with ``main_id``, ``lat``, and ``lon`` in decimal
        degrees.
    main_id : str
        Basin identifier column shared by ``cat`` and ``coords``.
    lake_col : str, optional
        Lake numbering column written by ``_compute_ireach`` (default
        ``IREACH``).
    coefficients : pandas.DataFrame, optional
        Coefficient table indexed by ``coeff_link_key``. When a lake is
        missing from this table, ``WF_B1`` and ``WF_B2`` are set to zero.
    coeff_link_key : str, optional
        Catchment field used to join coefficients: ``reservoir_id`` (default)
        or ``basin_id``. Only used when ``coefficients`` is provided.
    coeff_link_col : str, optional
        Catchment column for coefficient lookup. Defaults to ``reservoir_id``
        or the ``ddb_vars['reservoir_id']`` mapping for ``reservoir_id``
        linking, and ``main_id`` for ``basin_id`` linking.
    location_flag : int, optional
        MESH ``LOCATIONFLAG`` value. ``0`` writes integer ``i5`` locations;
        ``1`` writes real ``f7.1`` locations.
    use_power_coefficients : bool, optional
        When ``False`` (e.g. ``RESERVOIRFLAG`` 3), ``WF_B1`` and ``WF_B2`` are
        always zero. Names may still be taken from ``coefficients`` when
        provided.
    lake_area_col : str, optional
        Catchment column with lake reach area in square metres. When absent or
        invalid for a lake, the corresponding subbasin area is used.
    subbasin_areas : mapping, optional
        Subbasin areas keyed by ``main_id``. Typically taken from
        ``GridArea`` in the drainage database.

    Returns
    -------
    dict
        Context dictionary with keys ``n_reservoirs``, ``location_flag``, and
        ``reservoirs``.
    """
    if location_flag not in (0, 1):
        raise ValueError("`location_flag` must be 0 or 1.")
    if coeff_link_key not in RESERVOIR_LINK_KEYS:
        raise ValueError(
            f"`coeff_link_key` must be one of {RESERVOIR_LINK_KEYS}; "
            f"got `{coeff_link_key}`."
        )
    needs_coeff_link = coefficients is not None
    if needs_coeff_link:
        if coeff_link_col is None:
            coeff_link_col = (
                main_id if coeff_link_key == "basin_id" else "reservoir_id"
            )
        if coeff_link_col not in cat.columns:
            raise ValueError(
                f"Coefficient link column `{coeff_link_col}` not found in "
                f"catchment data. Available columns: {list(cat.columns)}"
            )
    if lake_col not in cat.columns:
        raise ValueError(
            f"Lake column `{lake_col}` not found in catchment data. "
            f"Available columns: {list(cat.columns)}"
        )
    if lake_area_col is not None and lake_area_col not in cat.columns:
        raise ValueError(
            f"Lake area column `{lake_area_col}` not found in catchment data. "
            f"Available columns: {list(cat.columns)}"
        )

    lake_columns = [main_id, lake_col]
    if needs_coeff_link and coeff_link_col not in lake_columns:
        lake_columns.append(coeff_link_col)
    if lake_area_col is not None:
        lake_columns.append(lake_area_col)

    lakes = (
        cat.loc[cat[lake_col] > 0, lake_columns]
        .drop_duplicates(subset=[main_id])
        .merge(coords[[main_id, "lat", "lon"]], on=main_id, how="left")
        .sort_values(lake_col)
    )

    coeff_lookup: Dict[Any, Dict[str, Any]] = {}
    if coefficients is not None:
        coeff_lookup = coefficients.set_index(coeff_link_key, drop=False).to_dict(
            "index"
        )

    area_lookup: Dict[Any, Any] = {}
    if subbasin_areas is not None:
        area_lookup = dict(subbasin_areas)
    elif isinstance(cat, gpd.GeoDataFrame) and "geometry" in cat.columns:
        from .geom import _calculate_polygon_areas

        area_gdf = _calculate_polygon_areas(
            cat[[main_id, "geometry"]].copy(),
            target_area_unit="m ** 2",
        )
        area_lookup = dict(
            zip(
                area_gdf[main_id],
                area_gdf["area"].astype(float),
            )
        )

    resolve_reach_area = lake_area_col is not None or bool(area_lookup)

    reservoirs: List[Dict[str, Any]] = []
    for row in lakes.itertuples(index=False):
        basin_id = getattr(row, main_id)
        if needs_coeff_link:
            coeff_lookup_id = getattr(row, coeff_link_col)
            coeff = _lookup_coefficient(coeff_lookup, coeff_lookup_id)
            default_name = default_reservoir_name(
                coeff_lookup_id if coeff_link_key == "reservoir_id" else basin_id
            )
        else:
            coeff = None
            default_name = default_reservoir_name(basin_id)
        ireach_num = int(getattr(row, lake_col))
        lat_deg = float(getattr(row, "lat"))
        lon_deg = float(getattr(row, "lon"))
        lat_min = degrees_to_mesh_minutes(lat_deg)
        lon_min = degrees_to_mesh_minutes(lon_deg)
        lake_area = (
            getattr(row, lake_area_col)
            if lake_area_col is not None
            else None
        )
        reach_area = None
        if resolve_reach_area:
            reach_area = _resolve_reach_area(
                lake_area=lake_area,
                subbasin_area=_lookup_mapping_value(area_lookup, basin_id),
                basin_id=basin_id,
            )
        if not use_power_coefficients:
            b1 = 0.0
            b2 = 0.0
            if coeff is None:
                name = default_name
            else:
                raw_name = coeff.get("name", pd.NA)
                if pd.isna(raw_name) or str(raw_name).strip() == "":
                    name = default_name
                else:
                    name = str(raw_name)
        elif coeff is None:
            b1 = 0.0
            b2 = 0.0
            name = default_name
        else:
            b1 = float(coeff.get("b1", 0.0) or 0.0)
            b2 = float(coeff.get("b2", 0.0) or 0.0)
            raw_name = coeff.get("name", pd.NA)
            if pd.isna(raw_name) or str(raw_name).strip() == "":
                name = default_name
            else:
                name = str(raw_name)

        reservoirs.append(
            {
                "basin_id": basin_id,
                "lat": lat_deg,
                "lon": lon_deg,
                "lat_min": lat_min,
                "lon_min": lon_min,
                "b1": b1,
                "b2": b2,
                "name": name,
                "ireach_num": ireach_num,
                "reach_area": reach_area,
            }
        )

    return {
        "n_reservoirs": len(reservoirs),
        "location_flag": location_flag,
        "reservoirs": reservoirs,
    }


def prepare_reservoir_inflows_context(
    reservoir_context: Dict[str, Any],
    start_time: str,
    delta_t: str,
    meshflow_version: str,
    creation_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the Jinja context for ``MESH_input_reservoir.tb0``."""
    missing_areas = [
        reservoir["basin_id"]
        for reservoir in reservoir_context.get("reservoirs", [])
        if reservoir.get("reach_area") is None
    ]
    if missing_areas:
        raise ValueError(
            "Could not resolve reach area for reservoir basins "
            f"{missing_areas}. Map `lake_area` in `ddb_vars` or ensure "
            "subbasin `GridArea` is available."
        )

    return {
        **reservoir_context,
        "start_time": start_time,
        "delta_t": delta_t,
        "meshflow_version": meshflow_version,
        "creation_date": creation_date or date.today().isoformat(),
        "tb0_banner": TB0_FILE_BANNER,
        "tb0_columns": build_tb0_column_metadata(
            reservoir_context.get("reservoirs", [])
        ),
    }


def write_reservoir_output_files(
    output_dir: str,
    reservoir_text: Optional[str] = None,
    reservoir_inflows_text: Optional[str] = None,
    blank_stub: Optional[str] = None,
) -> None:
    """
    Write the active reservoir input file and remove the unused format.

    Prefer ``reservoir_inflows_text`` (``.tb0``) when set, otherwise
    ``reservoir_text`` (``.txt``). If neither is provided, write
    ``blank_stub`` as ``MESH_input_reservoir.txt``.
    """
    reservoir_txt_path = os.path.join(output_dir, "MESH_input_reservoir.txt")
    reservoir_tb0_path = os.path.join(output_dir, "MESH_input_reservoir.tb0")

    if reservoir_inflows_text:
        with open(reservoir_tb0_path, "w") as handle:
            handle.write(reservoir_inflows_text)
        if os.path.isfile(reservoir_txt_path):
            os.remove(reservoir_txt_path)
        return

    content = reservoir_text if reservoir_text is not None else blank_stub
    if content is None:
        raise ValueError(
            "Provide reservoir_text, reservoir_inflows_text, or blank_stub."
        )

    with open(reservoir_txt_path, "w") as handle:
        handle.write(content)
    if os.path.isfile(reservoir_tb0_path):
        os.remove(reservoir_tb0_path)

