"""Unit tests for MESH_input_reservoir.txt generation."""

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "meshflow"
UTILITY = SRC / "utility"
TEMPLATES = SRC / "templates"


def _load_reservoir_module():
    spec = importlib.util.spec_from_file_location(
        "meshflow_reservoir", UTILITY / "reservoir.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["meshflow_reservoir"] = module
    spec.loader.exec_module(module)
    return module


reservoir = _load_reservoir_module()

degrees_to_mesh_minutes = reservoir.degrees_to_mesh_minutes
default_reservoir_name = reservoir.default_reservoir_name
format_reservoir_header = reservoir.format_reservoir_header
format_reservoir_line = reservoir.format_reservoir_line
fortran_i5 = reservoir.fortran_i5
fortran_f7_1 = reservoir.fortran_f7_1
normalize_lake_indicator = reservoir.normalize_lake_indicator
prepare_reservoir_context = reservoir.prepare_reservoir_context
prepare_reservoir_inflows_context = reservoir.prepare_reservoir_inflows_context
read_reservoir_coefficients = reservoir.read_reservoir_coefficients
parse_reservoir_coefficient_link = reservoir.parse_reservoir_coefficient_link


def test_normalize_lake_indicator_accepts_flags_area_and_lake_ids():
    values = pd.Series([0, 1, -1, 42, 1.5e6, None])
    flags = normalize_lake_indicator(values)
    assert list(flags) == [0, 1, 0, 1, 1, 0]
def render_reservoir_template(context: dict) -> str:
    """Render the production Jinja template using the same filters as meshflow."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        trim_blocks=True,
        lstrip_blocks=True,
        line_comment_prefix="##",
    )
    env.filters["fortran_i5"] = reservoir.fortran_i5
    env.filters["fortran_f7_1"] = reservoir.fortran_f7_1
    env.filters["fortran_g10_3"] = reservoir.fortran_g10_3
    env.filters["fortran_a12"] = reservoir.fortran_a12
    env.filters["fortran_i2"] = reservoir.fortran_i2
    env.filters["mesh_reservoir_name"] = reservoir.mesh_reservoir_name
    env.filters["tb0_coeff"] = reservoir.tb0_coeff
    env.filters["tb0_location"] = reservoir.tb0_location
    env.filters["tb0_reach_area"] = reservoir.tb0_reach_area

    content = env.get_template("MESH_input_reservoir.txt.jinja").render(**context)
    return content if content.endswith("\n") else content + "\n"


def render_reservoir_inflows_template(context: dict) -> str:
    """Render the production inflows Jinja template using the same filters as meshflow."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        trim_blocks=True,
        lstrip_blocks=True,
        line_comment_prefix="##",
    )
    env.filters["mesh_reservoir_name"] = reservoir.mesh_reservoir_name
    env.filters["tb0_coeff"] = reservoir.tb0_coeff
    env.filters["tb0_location"] = reservoir.tb0_location
    env.filters["tb0_reach_area"] = reservoir.tb0_reach_area

    content = env.get_template("MESH_input_reservoir.tb0.jinja").render(**context)
    return content if content.endswith("\n") else content + "\n"


def _render_reservoir_file(**kwargs) -> str:
    context = prepare_reservoir_context(**kwargs)
    return render_reservoir_template(context)


def test_normalize_reservoir_file_format_defaults_to_txt():
    assert reservoir.normalize_reservoir_file_format(None) == ("txt", False)
    assert reservoir.normalize_reservoir_file_format("txt") == ("txt", False)
    assert reservoir.normalize_reservoir_file_format("TB0") == ("tb0", False)


def test_mesh_timestep_delta_t_hours_matches_model_timestep():
    assert reservoir.mesh_timestep_delta_t_hours(60) == "1"
    assert reservoir.mesh_timestep_delta_t_hours(30) == "0.5"
    assert reservoir.mesh_timestep_delta_t_hours(120) == "2"


def test_normalize_reservoir_file_format_warns_on_invalid_values():
    assert reservoir.normalize_reservoir_file_format("csv") == ("txt", True)
    assert reservoir.normalize_reservoir_file_format("") == ("txt", False)


def test_degrees_to_mesh_minutes():
    assert degrees_to_mesh_minutes(51.178) == 3070.68


def test_reservoir_header_matches_dummy_file():
    assert format_reservoir_header(0) == "    0     0     0"
    assert format_reservoir_header(3) == "    3     0     0"


def test_blank_reservoir_template_replaces_default_settings_file():
    """Blank stub comes from Jinja, not default_settings/MESH_input_reservoir.txt."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        trim_blocks=True,
        lstrip_blocks=True,
        line_comment_prefix="##",
    )
    env.filters["fortran_i5"] = reservoir.fortran_i5
    env.filters["fortran_f7_1"] = reservoir.fortran_f7_1
    env.filters["fortran_g10_3"] = reservoir.fortran_g10_3
    env.filters["fortran_a12"] = reservoir.fortran_a12
    env.filters["fortran_i2"] = reservoir.fortran_i2
    env.filters["mesh_reservoir_name"] = reservoir.mesh_reservoir_name

    content = env.get_template("MESH_input_reservoir.txt.jinja").render(
        n_reservoirs=0,
        location_flag=0,
        reservoirs=[],
    )
    if not content.endswith("\n"):
        content += "\n"

    assert content.startswith("    0     0     0\n")
    assert content.rstrip("\n") == "    0     0     0"


def test_format_reservoir_line_locationflag_zero():
    line = format_reservoir_line(
        lat_min=3066.0,
        lon_min=-6906.0,
        b1=0.0,
        b2=0.0,
        name="102",
        ireach_num=1,
        location_flag=0,
    )

    assert line == (
        " 3066 -6906      0.000      0.000"
        "                         102           1"
    )


def test_format_reservoir_line_locationflag_one():
    line = format_reservoir_line(
        lat_min=3066.68,
        lon_min=-6906.12,
        b1=0.15,
        b2=0.25,
        name="Ghost Lake",
        ireach_num=2,
        location_flag=1,
    )

    assert line == (
        " 3066.7 -6906.1      0.150      0.250"
        "                         Ghost Lake    2"
    )


def test_jinja_reservoir_file_strict_locationflag_zero():
    cat = pd.DataFrame(
        {
            "COMID": [101, 102, 103],
            "IREACH": [0, 1, 2],
        }
    )
    coords = pd.DataFrame(
        {
            "COMID": [101, 102, 103],
            "lat": [51.0, 51.1, 51.2],
            "lon": [-115.0, -115.1, -115.2],
        }
    )

    text = _render_reservoir_file(
        cat=cat,
        coords=coords,
        main_id="COMID",
        location_flag=0,
    )

    lines = text.rstrip("\n").splitlines()
    assert lines[0] == "    2     0     0"
    assert lines[1] == format_reservoir_line(
        lat_min=3066.0,
        lon_min=-6906.0,
        b1=0.0,
        b2=0.0,
        name=default_reservoir_name(102),
        ireach_num=1,
        location_flag=0,
    )


def test_jinja_reservoir_file_uses_csv_coefficients(tmp_path):
    cat = pd.DataFrame(
        {
            "COMID": [101],
            "IREACH": [1],
            "reservoir_id": ["R-101"],
        }
    )
    coords = pd.DataFrame({"COMID": [101], "lat": [50.0], "lon": [-114.0]})
    csv_path = tmp_path / "reservoirs.csv"
    csv_path.write_text(
        "reservoir_id,name,b1,b2\n"
        "R-101,Ghost Lake,0.15,0.25\n"
    )
    coefficients = read_reservoir_coefficients(str(csv_path), main_id="COMID")

    text = _render_reservoir_file(
        cat=cat,
        coords=coords,
        main_id="COMID",
        coefficients=coefficients,
        coeff_link_col="reservoir_id",
        location_flag=1,
    )

    lines = text.rstrip("\n").splitlines()
    assert lines[0] == "    1     0     0"
    assert lines[1] == (
        " 3000.0 -6840.0      0.150      0.250"
        "                         Ghost Lake    1"
    )
    assert default_reservoir_name(999) == "999"


def test_jinja_reservoir_file_zeros_coefficients_for_flag_three(tmp_path):
    cat = pd.DataFrame(
        {
            "COMID": [101],
            "IREACH": [1],
            "reservoir_id": ["R-101"],
        }
    )
    coords = pd.DataFrame({"COMID": [101], "lat": [50.0], "lon": [-114.0]})
    csv_path = tmp_path / "reservoirs.csv"
    csv_path.write_text(
        "reservoir_id,name,b1,b2\n"
        "R-101,Ghost Lake,0.15,0.25\n"
    )
    coefficients = read_reservoir_coefficients(str(csv_path), main_id="COMID")

    context = prepare_reservoir_context(
        cat=cat,
        coords=coords,
        main_id="COMID",
        coefficients=coefficients,
        coeff_link_col="reservoir_id",
        location_flag=0,
        use_power_coefficients=False,
    )
    text = render_reservoir_template(context)

    assert context["reservoirs"][0]["b1"] == 0.0
    assert context["reservoirs"][0]["b2"] == 0.0
    assert context["reservoirs"][0]["name"] == "Ghost Lake"
    assert "      0.000      0.000                         " in text
    assert "      0.150" not in text


def test_prepare_reservoir_context_uses_lake_area_when_available():
    cat = pd.DataFrame(
        {
            "COMID": [101],
            "IREACH": [1],
            "lake_area": [11600000.0],
        }
    )
    coords = pd.DataFrame({"COMID": [101], "lat": [51.21], "lon": [-114.7]})

    context = prepare_reservoir_context(
        cat=cat,
        coords=coords,
        main_id="COMID",
        lake_area_col="lake_area",
        subbasin_areas={101: 5000000.0},
    )

    assert context["reservoirs"][0]["reach_area"] == 11600000.0


def test_prepare_reservoir_context_falls_back_to_subbasin_area():
    cat = pd.DataFrame({"COMID": [101], "IREACH": [1]})
    coords = pd.DataFrame({"COMID": [101], "lat": [51.21], "lon": [-114.7]})

    context = prepare_reservoir_context(
        cat=cat,
        coords=coords,
        main_id="COMID",
        subbasin_areas={101: 7402235.0},
    )

    assert context["reservoirs"][0]["reach_area"] == 7402235.0


def test_jinja_reservoir_inflows_matches_example_layout(tmp_path):
    cat = pd.DataFrame(
        {
            "COMID": [101],
            "IREACH": [1],
            "lake_area": [11600000.0],
            "reservoir_id": ["R-101"],
        }
    )
    coords = pd.DataFrame({"COMID": [101], "lat": [51.21], "lon": [-114.7]})
    csv_path = tmp_path / "reservoirs.csv"
    csv_path.write_text(
        "reservoir_id,name,b1,b2\n"
        "R-101,Ghost,3.5e-14,2.0\n"
    )
    coefficients = read_reservoir_coefficients(str(csv_path), main_id="COMID")

    reservoir_context = prepare_reservoir_context(
        cat=cat,
        coords=coords,
        main_id="COMID",
        coefficients=coefficients,
        coeff_link_col="reservoir_id",
        lake_area_col="lake_area",
        subbasin_areas={101: 5000000.0},
    )
    inflows_context = prepare_reservoir_inflows_context(
        reservoir_context,
        start_time="1980/01/01 00:00",
        delta_t="1",
        meshflow_version="v0.2.0-dev",
    )
    text = render_reservoir_inflows_template(inflows_context)

    assert ":FileType tb0  ASCII" in text
    assert ":WrittenBy          MESHflow" in text
    assert ":Version            v0.2.0-dev" in text
    assert f":CreationDate       {date.today().isoformat()}" in text
    assert ":StartTime          1980/01/01 00:00" in text
    assert ":DeltaT             1" in text
    assert ":Application" not in text
    assert ":RoutingDeltaT" not in text
    assert ":FillFlag" not in text
    assert text.splitlines()[0] == "########################################"
    assert "   :ColumnName         Ghost" in text
    assert "   :ColumnLocationX    -114.7" in text
    assert "   :ColumnLocationY    51.21" in text
    assert "   :Coeff1             3.50E-14" in text
    assert "   :Coeff2             2.0E+00" in text
    assert "   :ReachArea          11600000" in text
    assert text.count(":ColumnMetaData") == 1
    assert text.count(":EndColumnMetaData") == 1
    assert text.rstrip().endswith(":endHeader")


def test_jinja_reservoir_inflows_uses_single_column_metadata_block(tmp_path):
    cat = pd.DataFrame(
        {
            "COMID": [101, 102],
            "IREACH": [1, 2],
            "lake_area": [11600000.0, 5000000.0],
            "reservoir_id": ["R-101", "R-102"],
        }
    )
    coords = pd.DataFrame(
        {
            "COMID": [101, 102],
            "lat": [51.21, 51.30],
            "lon": [-114.7, -115.0],
        }
    )
    csv_path = tmp_path / "reservoirs.csv"
    csv_path.write_text(
        "reservoir_id,name,b1,b2\n"
        "R-101,Ghost,3.5e-14,2.0\n"
        "R-102,Lake2,1.0,1.0\n"
    )
    coefficients = read_reservoir_coefficients(str(csv_path), main_id="COMID")

    reservoir_context = prepare_reservoir_context(
        cat=cat,
        coords=coords,
        main_id="COMID",
        coefficients=coefficients,
        coeff_link_col="reservoir_id",
        lake_area_col="lake_area",
        subbasin_areas={101: 5000000.0, 102: 4000000.0},
    )
    inflows_context = prepare_reservoir_inflows_context(
        reservoir_context,
        start_time="1980/01/01 00:00",
        delta_t="1",
        meshflow_version="v0.2.0-dev",
    )
    text = render_reservoir_inflows_template(inflows_context)

    assert text.count(":ColumnMetaData") == 1
    assert text.count(":EndColumnMetaData") == 1
    assert "   :ColumnName         Ghost    Lake2" in text
    assert "   :ColumnLocationX    -114.7   -115" in text
    assert "   :ColumnLocationY    51.21    51.3" in text
    assert "   :Coeff1             3.50E-14 1.0E+00" in text
    assert "   :Coeff2             2.0E+00  1.0E+00" in text
    assert "   :ReachArea          11600000 5000000" in text


def test_format_tb0_column_lines_left_aligns_to_widest_per_column():
    lines = reservoir.format_tb0_column_lines(
        [
            ["a", "bb"],
            ["ccc", "d"],
            ["ee", "fffff"],
        ]
    )
    assert lines == [
        "a   bb",
        "ccc d",
        "ee  fffff",
    ]
    # At least one space between columns, even when a value fills its width.
    assert "ccc d" == lines[1]
    assert lines[0].index("bb") == lines[2].index("f")


def test_tb0_coeff_matches_mesh_example_format():
    assert reservoir.tb0_coeff(3.5e-14) == "3.50E-14"
    assert reservoir.tb0_coeff(2.0) == "2.0E+00"
    assert reservoir.tb0_coeff(0.0) == "0.0E+00"


def test_fortran_g10_3_uses_scientific_for_extreme_values():
    assert len(reservoir.fortran_g10_3(0.15)) == 10
    assert reservoir.fortran_g10_3(0.15) == "     0.150"
    assert reservoir.fortran_g10_3(2.0) == "     2.000"
    assert reservoir.fortran_g10_3(3.5e-14) == "  3.50E-14"
    assert reservoir.fortran_g10_3(0.000005166) == "  5.17E-06"
    assert reservoir.fortran_g10_3(21568734) == "  2.16E+07"
    assert "E" in reservoir.fortran_g10_3(3.5e-14)
    assert "e" not in reservoir.fortran_g10_3(3.5e-14)


def test_parse_reservoir_coefficient_link_defaults_to_reservoir_id():
    assert parse_reservoir_coefficient_link({}, "COMID") == ("reservoir_id", None)
    assert parse_reservoir_coefficient_link({"name_col": "name"}, "COMID") == (
        "reservoir_id",
        None,
    )


def test_parse_reservoir_coefficient_link_accepts_explicit_keys():
    assert parse_reservoir_coefficient_link({"basin_id": "COMID"}, "COMID") == (
        "basin_id",
        "COMID",
    )
    assert parse_reservoir_coefficient_link(
        {"reservoir_id": "res_id"}, "COMID"
    ) == ("reservoir_id", "res_id")


def test_read_reservoir_coefficients_defaults_to_reservoir_id(tmp_path):
    csv_path = tmp_path / "reservoirs.csv"
    csv_path.write_text(
        "reservoir_id,name,b1,b2\n"
        "R-101,Ghost Lake,0.15,0.25\n"
    )
    coefficients = read_reservoir_coefficients(str(csv_path), main_id="COMID")

    assert list(coefficients.columns) == ["reservoir_id", "b1", "b2", "name"]
    assert coefficients.loc["R-101", "name"] == "Ghost Lake"


def test_read_reservoir_coefficients_links_on_basin_id_when_explicit(tmp_path):
    csv_path = tmp_path / "reservoirs.csv"
    csv_path.write_text(
        "COMID,name,b1,b2\n"
        "101,Ghost Lake,0.15,0.25\n"
    )
    coefficients = read_reservoir_coefficients(
        str(csv_path),
        main_id="COMID",
        link_key="basin_id",
    )

    assert list(coefficients.columns) == ["basin_id", "b1", "b2", "name"]
    assert coefficients.loc[101, "name"] == "Ghost Lake"


def test_read_reservoir_coefficients_links_on_reservoir_id(tmp_path):
    csv_path = tmp_path / "reservoirs.csv"
    csv_path.write_text(
        "res_id,name,b1,b2\n"
        "R-101,Ghost Lake,0.15,0.25\n"
    )
    coefficients = read_reservoir_coefficients(
        str(csv_path),
        main_id="COMID",
        link_key="reservoir_id",
        id_col="res_id",
    )

    assert list(coefficients.columns) == ["reservoir_id", "b1", "b2", "name"]
    assert coefficients.loc["R-101", "name"] == "Ghost Lake"


def test_prepare_reservoir_context_links_coefficients_by_reservoir_id(tmp_path):
    cat = pd.DataFrame(
        {
            "COMID": [101],
            "IREACH": [1],
            "reservoir_id": ["R-101"],
        }
    )
    coords = pd.DataFrame({"COMID": [101], "lat": [50.0], "lon": [-114.0]})
    csv_path = tmp_path / "reservoirs.csv"
    csv_path.write_text(
        "res_id,name,b1,b2\n"
        "R-101,Ghost Lake,0.15,0.25\n"
    )
    coefficients = read_reservoir_coefficients(
        str(csv_path),
        main_id="COMID",
        link_key="reservoir_id",
        id_col="res_id",
    )

    context = prepare_reservoir_context(
        cat=cat,
        coords=coords,
        main_id="COMID",
        coefficients=coefficients,
        coeff_link_key="reservoir_id",
        coeff_link_col="reservoir_id",
        location_flag=1,
    )

    assert context["reservoirs"][0]["name"] == "Ghost Lake"
    assert context["reservoirs"][0]["b1"] == 0.15
    assert context["reservoirs"][0]["b2"] == 0.25


def test_write_reservoir_files_tb0_does_not_leave_blank_txt(tmp_path):
    blank = tmp_path / "MESH_input_reservoir.txt"
    blank.write_text("    0     0     0\n")

    reservoir.write_reservoir_output_files(
        str(tmp_path),
        reservoir_inflows_text=":FileType tb0  ASCII\n:endHeader\n",
    )

    assert (tmp_path / "MESH_input_reservoir.tb0").read_text().startswith(":FileType tb0")
    assert not blank.exists()


def test_write_reservoir_files_txt_removes_stale_tb0(tmp_path):
    stale = tmp_path / "MESH_input_reservoir.tb0"
    stale.write_text(":FileType tb0  ASCII\n")

    reservoir.write_reservoir_output_files(
        str(tmp_path),
        reservoir_text="    1     0     0\n",
    )

    assert (tmp_path / "MESH_input_reservoir.txt").read_text().startswith("    1")
    assert not stale.exists()


def test_write_reservoir_files_blank_stub_when_disabled(tmp_path):
    reservoir.write_reservoir_output_files(
        str(tmp_path),
        blank_stub="    0     0     0\n",
    )

    assert (tmp_path / "MESH_input_reservoir.txt").read_text().startswith("    0")
    assert not (tmp_path / "MESH_input_reservoir.tb0").exists()


def test_normalize_reservoir_param_entry_accepts_aliases():
    normalized = reservoir.normalize_reservoir_param_entry(
        {"WF_B1": 0.15, "B2": 0.25, "reservoir_name": "Ghost Lake"}
    )
    assert normalized == {"b1": 0.15, "b2": 0.25, "name": "Ghost Lake"}


def test_coefficients_round_trip_through_reservoir_params(tmp_path):
    csv_path = tmp_path / "reservoirs.csv"
    csv_path.write_text(
        "reservoir_id,name,b1,b2\n"
        "R-101,Ghost Lake,0.15,0.25\n"
    )
    coefficients = read_reservoir_coefficients(str(csv_path), main_id="COMID")
    params = reservoir.coefficients_to_reservoir_params(
        coefficients, link_key="reservoir_id"
    )
    assert params["R-101"]["b1"] == 0.15
    assert params["R-101"]["b2"] == 0.25
    assert params["R-101"]["name"] == "Ghost Lake"

    round_trip = reservoir.reservoir_params_to_coefficients(
        params, link_key="reservoir_id"
    )
    assert float(round_trip.loc["R-101", "b1"]) == 0.15
    assert float(round_trip.loc["R-101", "b2"]) == 0.25


def test_merge_reservoir_params_overlay_overrides_csv_seed():
    base = {
        "R-101": {"b1": 0.15, "b2": 0.25, "name": "Ghost Lake"},
        "R-102": {"b1": 1.0, "b2": 2.0},
    }
    overlay = {
        "R-101": {"b1": 0.18},
        ("R-102", "R-103"): {"b1": 0.5, "b2": 0.6},
    }
    merged = reservoir.merge_reservoir_params(base, overlay)
    assert merged["R-101"]["b1"] == 0.18
    assert merged["R-101"]["b2"] == 0.25
    assert merged["R-101"]["name"] == "Ghost Lake"
    assert merged["R-102"]["b1"] == 0.5
    assert merged["R-102"]["b2"] == 0.6
    assert merged["R-103"]["b1"] == 0.5


def test_seed_reservoir_params_from_catchments():
    cat = pd.DataFrame(
        {
            "COMID": [101, 102, 103],
            "IREACH": [1, 0, 2],
            "reservoir_id": ["R-101", "R-102", "R-103"],
        }
    )
    params = reservoir.seed_reservoir_params_from_catchments(
        cat=cat,
        main_id="COMID",
        coeff_link_col="reservoir_id",
    )
    assert set(params) == {"R-101", "R-103"}
    assert params["R-101"] == {"b1": 0.0, "b2": 0.0, "name": "R-101"}


def test_jinja_reservoir_file_uses_dict_coefficients():
    cat = pd.DataFrame(
        {
            "COMID": [101],
            "IREACH": [1],
            "reservoir_id": ["R-101"],
        }
    )
    coords = pd.DataFrame({"COMID": [101], "lat": [50.0], "lon": [-114.0]})
    coefficients = reservoir.reservoir_params_to_coefficients(
        {"R-101": {"b1": 0.15, "b2": 0.25, "name": "Ghost Lake"}},
        link_key="reservoir_id",
    )

    text = _render_reservoir_file(
        cat=cat,
        coords=coords,
        main_id="COMID",
        coefficients=coefficients,
        coeff_link_col="reservoir_id",
        location_flag=1,
    )

    lines = text.rstrip("\n").splitlines()
    assert lines[1] == (
        " 3000.0 -6840.0      0.150      0.250"
        "                         Ghost Lake    1"
    )


def test_calibrated_dict_updates_rendered_coefficients():
    cat = pd.DataFrame(
        {
            "COMID": [101],
            "IREACH": [1],
            "reservoir_id": ["R-101"],
        }
    )
    coords = pd.DataFrame({"COMID": [101], "lat": [50.0], "lon": [-114.0]})
    params = {
        "R-101": {"b1": 0.15, "b2": 0.25, "name": "Ghost Lake"},
    }

    # Calibration iteration: mutate dict in place, then re-render.
    params["R-101"]["b1"] = 0.18
    params["R-101"]["b2"] = 0.22
    coefficients = reservoir.reservoir_params_to_coefficients(
        params, link_key="reservoir_id"
    )
    context = prepare_reservoir_context(
        cat=cat,
        coords=coords,
        main_id="COMID",
        coefficients=coefficients,
        coeff_link_col="reservoir_id",
        location_flag=1,
    )

    assert context["reservoirs"][0]["b1"] == 0.18
    assert context["reservoirs"][0]["b2"] == 0.22
