# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Component metadata: the components-tree scanner and the toolbox tab grouping.

The scanner is run on a small components tree written to a temporary folder,
named after real ``KNOWN_COMPONENTS`` entries, so the tests do not depend on
the contents of the real tree.
"""

import pytest

from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from fastga_he.gui.power_train_builder.power_train_builder_metadata import (
    _get_performance_component_names,
)
from fastga_he.gui.power_train_builder.power_train_builder_toolbox_config_table import (
    _group_icons_by_tab,
    _with_unknown_component,
)

FIRST_COMPONENT = KNOWN_COMPONENTS[0]
SECOND_COMPONENT = next(
    component
    for component in KNOWN_COMPONENTS
    if component["components_type"] != FIRST_COMPONENT["components_type"]
)
KNOWN_TYPES = {component["components_type"] for component in KNOWN_COMPONENTS}


def _component_folder(root, folder_name, init_source, constants_source=None):
    folder = root / folder_name
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text(init_source)
    if constants_source is not None:
        (folder / "constants.py").write_text(constants_source)


def _performances_import(component) -> str:
    return f"from .performances import Performances{component['OM_components_name']}\n"


# ---------------------------------------------------------------------------
# Components-tree scanner
# ---------------------------------------------------------------------------


def test_scanner_reads_the_constants_of_every_registered_component(tmp_path):
    _component_folder(
        tmp_path,
        "first",
        _performances_import(FIRST_COMPONENT),
        "POSSIBLE_POSITION = ['in_the_wing', 'in_the_nose']\n"
        "POSSIBLE_OPTION = {'mode': ['a', 'b'], 'flag': [True, False]}\n",
    )
    _component_folder(
        tmp_path / "group" / "sub",
        "second",
        _performances_import(SECOND_COMPONENT),
        "POSSIBLE_POSITION = ['in_the_back']\n",
    )
    _component_folder(
        tmp_path,
        "noise",
        "from .sizing import SizingSomething\n"
        "from .performances import PerformancesNotARegisteredComponent\n"
        "import os\n",
        "POSSIBLE_POSITION = ['should_not_be_used']\n",
    )

    positions, options = _get_performance_component_names(tmp_path)

    first_type = FIRST_COMPONENT["components_type"]
    second_type = SECOND_COMPONENT["components_type"]
    assert positions[first_type] == ["in_the_wing", "in_the_nose"]
    assert options[first_type] == {"mode": ["a", "b"], "flag": [True, False]}
    # Nested folders are found too; a constant the file lacks is empty
    assert positions[second_type] == ["in_the_back"]
    assert options[second_type] == {}
    # Other imports and unregistered Performances classes are ignored
    assert "should_not_be_used" not in [p for values in positions.values() for p in values]
    # Every known type gets an entry, empty when it is not in the tree
    assert set(positions) >= KNOWN_TYPES and set(options) >= KNOWN_TYPES
    for missing_type in KNOWN_TYPES - {first_type, second_type}:
        assert positions[missing_type] == [] and options[missing_type] == {}


@pytest.mark.parametrize(
    "constants_source",
    [
        pytest.param(None, id="no_constants_file"),
        pytest.param("", id="constants_not_defined"),
        # An enum reference cannot be evaluated without importing the module, and
        # the module is not in sys.modules: the entry stays empty instead of failing
        pytest.param(
            "POSSIBLE_POSITION = [Position.FRONT]\nPOSSIBLE_OPTION = {'mode': Mode.ALL}\n",
            id="non_literal_constants_of_a_module_not_imported",
        ),
    ],
)
def test_scanner_falls_back_to_empty_entries(tmp_path, constants_source):
    _component_folder(tmp_path, "first", _performances_import(FIRST_COMPONENT), constants_source)

    positions, options = _get_performance_component_names(tmp_path)

    assert positions[FIRST_COMPONENT["components_type"]] == []
    assert options[FIRST_COMPONENT["components_type"]] == {}


# ---------------------------------------------------------------------------
# Toolbox tabs
# ---------------------------------------------------------------------------


def _component(icon, type_class):
    return {"icon_for_network_graph": icon, "components_type_class": type_class}


@pytest.mark.parametrize(
    "components, expected",
    [
        pytest.param(
            [
                _component("battery", "source"),
                _component("cable", ["connector", "source"]),
                _component("bus_bar", "connector"),
            ],
            {"source": ["battery", "cable"], "connector": ["cable", "bus_bar"]},
            id="one_tab_per_type_class_in_order",
        ),
        pytest.param(
            [
                _component("dc_load", "propulsive_load"),
                # In a list, propulsive_load adds no tab of its own
                _component("propeller", ["propulsor", "propulsive_load"]),
            ],
            {"load": ["dc_load"], "propulsor": ["propeller"]},
            id="propulsive_load_alone_goes_to_the_load_tab",
        ),
        pytest.param(
            [
                _component("e_motor", "propulsor_driver"),
                _component("e_motor", "propulsor_driver"),
            ],
            {"propulsor_driver": ["e_motor"]},
            id="an_icon_is_listed_once_per_tab",
        ),
    ],
)
def test_toolbox_tabs(components, expected):
    assert _group_icons_by_tab(components) == expected


def test_unknown_component_closes_its_tabs():
    icons_by_tab = {
        "source": ["battery"],
        "connector": ["cable"],
        "propulsor_driver": ["e_motor"],
        "propulsor": ["propeller"],
        "tank": ["fuel_tank"],
        "load": ["dc_load"],
    }

    tabs = _with_unknown_component(icons_by_tab)

    # Last in the connector, propulsor, load, tank and source tabs, nowhere else
    assert tabs == {
        "source": ["battery", "unknown"],
        "connector": ["cable", "unknown"],
        "propulsor_driver": ["e_motor"],
        "propulsor": ["propeller", "unknown"],
        "tank": ["fuel_tank", "unknown"],
        "load": ["dc_load", "unknown"],
    }
    # A tab is never created for the unknown component alone
    assert _with_unknown_component({"source": ["battery"]}) == {"source": ["battery", "unknown"]}
