"""
End-to-end tests for meshflow based on the Bow River at Calgary examples.

These tests exercise the two primary example workflows:
1. ``nogrouping`` – routing parameters given as a flat list.
2. ``grouping``   – routing/hydrology parameters use grouped (tuple) keys.

Both instantiate ``MESHWorkflow``, run the full pipeline, and save results.
"""

import os
import warnings

import meshflow
import pytest


# ---------------------------------------------------------------------------
# Shared example data paths
# ---------------------------------------------------------------------------
EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples")
)

RIV_PATH = os.path.join(
    EXAMPLES_DIR, "bow-at-calgary-shapefiles", "bcalgary_rivers.shp"
)
CAT_PATH = os.path.join(
    EXAMPLES_DIR, "bow-at-calgary-shapefiles", "bcalgary_subbasins.shp"
)
LC_PATH = os.path.join(
    EXAMPLES_DIR,
    "bow-at-calgary-attributes",
    "bcalgary_stats_NA_NALCMS_landcover_2020_30m.csv",
)
FORCING_DIR = os.path.join(EXAMPLES_DIR, "bow-at-calgary-forcings")

LANDCOVER_CLASSES = {
    0: "Unknown",
    1: "Temperate or sub-polar needleleaf forest",
    2: "Sub-polar taiga needleleaf forest",
    3: "Tropical or sub-tropical broadleaf evergreen forest",
    4: "Tropical or sub-tropical broadleaf deciduous forest",
    5: "Temperate or sub-polar broadleaf deciduous forest",
    6: "Mixed forest",
    7: "Tropical or sub-tropical shrubland",
    8: "Temperate or sub-polar shrubland",
    9: "Tropical or sub-tropical grassland",
    10: "Temperate or sub-polar grassland",
    11: "Sub-polar or polar shrubland-lichen-moss",
    12: "Sub-polar or polar grassland-lichen-moss",
    13: "Sub-polar or polar barren-lichen-moss",
    14: "Wetland",
    15: "Cropland",
    16: "Barren lands",
    17: "Urban",
    18: "Water",
    19: "Snow and Ice",
}

FORCING_VARS = {
    "air_pressure": "RDRS_v2.1_P_P0_SFC",
    "specific_humidity": "RDRS_v2.1_P_HU_09944",
    "air_temperature": "RDRS_v2.1_P_TT_09944",
    "wind_speed": "RDRS_v2.1_P_UVC_09944",
    "precipitation": "RDRS_v2.1_A_PR0_SFC",
    "shortwave_radiation": "RDRS_v2.1_P_FB_SFC",
    "longwave_radiation": "RDRS_v2.1_P_FI_SFC",
}

FORCING_UNITS = {
    "air_pressure": "millibar",
    "specific_humidity": "kilogram / kilogram",
    "air_temperature": "celsius",
    "wind_speed": "knot",
    "precipitation": "meter / hour",
    "shortwave_radiation": "watt / meter ** 2",
    "longwave_radiation": "watt / meter ** 2",
}

DDB_VARS = {
    "river_slope": "slope",
    "river_length": "lengthkm",
    "river_class": "order",
    "maxup": "maximum_up",
}

DDB_UNITS = {
    "river_slope": "dimensionless",
    "river_length": "kilometer",
    "river_class": "dimensionless",
}

COMMON_SETTINGS = {
    "core": {
        "forcing_files": "multiple",
        "forcing_start_date": "1980-01-01 00:00:00",
        "simulation_start_date": "1985-02-12 12:00:00",
        "simulation_end_date": "2010-05-18 18:00:00",
        "forcing_time_zone": "UTC",
        "output_path": "results",
        "landcover_mode": "fractional",
    },
    "class_params": {
        "measurement_heights": {
            "wind_speed": 40,
            "specific_humidity": 40,
            "air_temperature": 40,
            "roughness_length": 50,
        },
        "copyright": {
            "author": "University of Calgary",
            "location": "University of Calgary",
        },
        "grus": {
            0: "needleleaf",
            1: {
                "class": "needleleaf",
                "LNZ0": -1.4,
            },
            2: "needleleaf",
            3: "broadleaf",
            4: "broadleaf",
            5: {
                "class": "broadleaf",
                "rOot": 4,
            },
            6: [
                {
                    "class": "needleleaf",
                    "fcan": 0.5,
                    "lnz0": 0.1,
                    "rsmn": 275.13,
                },
                {
                    "class": "broadleaf",
                    "fcan": 0.5,
                    "lnz0": 0.21,
                },
            ],
            (7, 8, 9, 10, 11, 12): "grass",
            13: "grass",
            14: "water",
            15: "crops",
            16: "barrenland",
            17: "urban",
            18: "water",
            19: "snow",
        },
    },
    "run_options": {
        "flags": {
            "etc": {
                "RUNMODE": "noroute",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    """Return the configuration dictionary common to both examples."""
    return {
        "riv": RIV_PATH,
        "cat": CAT_PATH,
        "landcover": LC_PATH,
        "forcing_files": FORCING_DIR,
        "forcing_vars": FORCING_VARS,
        "forcing_units": FORCING_UNITS,
        "main_id": "COMID",
        "ds_main_id": "NextDownID",
        "landcover_classes": LANDCOVER_CLASSES,
        "ddb_vars": DDB_VARS,
        "ddb_units": DDB_UNITS,
        "settings": COMMON_SETTINGS,
        "outlet_value": -9999,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoGroupingExample:
    """Tests that mirror ``meshflow_bow_at_calgary-nogrouping.ipynb``."""

    @pytest.fixture
    def config(self, base_config):
        cfg = base_config.copy()
        cfg["settings"] = {
            **COMMON_SETTINGS,
            "hydrology_params": {
                "routing": [
                    {
                        "R2N": 0.3,
                        "pwr": 0.72,
                    },
                    {
                        "pwr": 0.32,
                        "flz": 0.13,
                    },
                ],
                "hydrology": {
                    5: {
                        "ZSNL": 2,
                    },
                },
            },
        }
        return cfg

    def test_instantiation(self, config):
        """MESHWorkflow should instantiate without errors."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            workflow = meshflow.MESHWorkflow(**config)
        assert isinstance(workflow, meshflow.MESHWorkflow)

    def test_run_and_save(self, config, tmp_path):
        """Full run() + save() should complete and emit expected artefacts."""
        output_dir = tmp_path / "outputs_nogrouping"
        # Match the example notebook: run(save_path=output_dir) places
        # forcing files under output_dir / "forcings" automatically.
        save_path = output_dir

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            workflow = meshflow.MESHWorkflow(**config)
            workflow.run(save_path=str(save_path))
            workflow.save(output_dir=str(output_dir))

        # Core artefacts produced by the example
        assert (output_dir / "MESH_parameters_CLASS.ini").exists()
        assert (output_dir / "MESH_parameters_hydrology.ini").exists()
        assert (output_dir / "MESH_input_run_options.ini").exists()
        assert (output_dir / "MESH_drainage_database.nc").exists()
        assert (output_dir / "MESH_parameters.nc").exists()
        # Forcing files saved into the sub-directory
        forcing_files = list((save_path / "forcings").glob("*.nc"))
        assert len(forcing_files) > 0

    def test_routing_and_hydrology_values(self, config, tmp_path):
        """Routing/hydrology parameters should match the flat-list input."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            workflow = meshflow.MESHWorkflow(**config)
            workflow.run(save_path=str(tmp_path / "forcings"))

        # routing given as a flat list: index 0 -> first dict, index 1 -> second dict
        routing = workflow.hydrology_dict["routing"]
        assert routing[0]["pwr"] == 0.72
        assert routing[1]["pwr"] == 0.32
        assert routing[1]["flz"] == 0.13

        # hydrology given as integer-key dict
        hydrology = workflow.hydrology_dict["hydrology"]
        assert hydrology[5]["zsnl"] == 2


class TestGroupingExample:
    """Tests that mirror ``meshflow_bow_at_calgary-grouping.ipynb``."""

    @pytest.fixture
    def config(self, base_config):
        cfg = base_config.copy()
        cfg["settings"] = {
            **COMMON_SETTINGS,
            "hydrology_params": {
                "routing": {
                    (0, 2): {
                        "pwr": 0.2,
                        "flz": 0.3,
                    },
                },
                "hydrology": {
                    (5, 6): {
                        "ZSNL": 2,
                    },
                },
            },
        }
        return cfg

    def test_instantiation(self, config):
        """MESHWorkflow should instantiate without errors."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            workflow = meshflow.MESHWorkflow(**config)
        assert isinstance(workflow, meshflow.MESHWorkflow)

    def test_run_and_save(self, config, tmp_path):
        """Full run() + save() should complete and emit expected artefacts."""
        output_dir = tmp_path / "outputs_grouping"
        # Match the example notebook: run(save_path=output_dir) places
        # forcing files under output_dir / "forcings" automatically.
        save_path = output_dir

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            workflow = meshflow.MESHWorkflow(**config)
            workflow.run(save_path=str(save_path))
            workflow.save(output_dir=str(output_dir))

        # Core artefacts produced by the example
        assert (output_dir / "MESH_parameters_CLASS.ini").exists()
        assert (output_dir / "MESH_parameters_hydrology.ini").exists()
        assert (output_dir / "MESH_input_run_options.ini").exists()
        assert (output_dir / "MESH_drainage_database.nc").exists()
        assert (output_dir / "MESH_parameters.nc").exists()
        # Forcing files saved into the sub-directory
        forcing_files = list((save_path / "forcings").glob("*.nc"))
        assert len(forcing_files) > 0

    def test_grouping_expansion(self, config, tmp_path):
        """Tuple keys in routing/hydrology should expand to individual entries."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            workflow = meshflow.MESHWorkflow(**config)
            workflow.run(save_path=str(tmp_path / "forcings"))

        # routing: (0, 2) -> both classes 0 and 2 get the same params
        routing = workflow.hydrology_dict["routing"]
        assert routing[0]["pwr"] == 0.2
        assert routing[0]["flz"] == 0.3
        assert routing[2]["pwr"] == 0.2
        assert routing[2]["flz"] == 0.3
        # class 1 should keep default values because it was not provided
        assert routing[1]["pwr"] == 2.5
        assert routing[1]["flz"] == 1e-05

        # hydrology: (5, 6) -> both GRUs 5 and 6 get the same params
        hydrology = workflow.hydrology_dict["hydrology"]
        assert hydrology[5]["zsnl"] == 2
        assert hydrology[6]["zsnl"] == 2
