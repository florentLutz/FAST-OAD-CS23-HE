# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Component metadata: the components-tree scanner and the toolbox tab grouping.

The scanner is run on a small components tree written to a temporary folder,
named after real ``KNOWN_COMPONENTS`` entries, so the tests do not depend on
the contents of the real tree.
"""

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


def _scan_first_component(tmp_path, constants_source) -> tuple:
    """
    Scan a tree holding only the first registered component.

    :param constants_source: Text of its ``constants.py``, or ``None`` for no such file.
    :return: ``(possible positions, possible options)`` found for it.
    """
    _component_folder(tmp_path, "first", _performances_import(FIRST_COMPONENT), constants_source)
    positions, options = _get_performance_component_names(tmp_path)
    first_type = FIRST_COMPONENT["components_type"]
    return positions[first_type], options[first_type]


def test_scanner_without_constants_file_gives_empty_entries(tmp_path):
    assert _scan_first_component(tmp_path, None) == ([], {})


def test_scanner_without_constants_defined_gives_empty_entries(tmp_path):
    assert _scan_first_component(tmp_path, "") == ([], {})


def test_scanner_with_non_literal_constants_gives_empty_entries(tmp_path):
    # An enum reference cannot be evaluated without importing the module, and
    # the module is not in sys.modules: the entry stays empty instead of failing
    constants_source = (
        "POSSIBLE_POSITION = [Position.FRONT]\nPOSSIBLE_OPTION = {'mode': Mode.ALL}\n"
    )

    assert _scan_first_component(tmp_path, constants_source) == ([], {})


# ---------------------------------------------------------------------------
# Toolbox tabs
# ---------------------------------------------------------------------------


def _component(icon, type_class):
    return {"icon_for_network_graph": icon, "components_type_class": type_class}


def test_toolbox_has_one_tab_per_type_class_in_order():
    components = [
        _component("battery", "source"),
        # Made-up: a cable is only a connector, but this shows a component in two tabs
        _component("cable", ["connector", "source"]),
        _component("bus_bar", "connector"),
    ]

    assert _group_icons_by_tab(components) == {
        "source": ["battery", "cable"],
        "connector": ["cable", "bus_bar"],
    }


def test_toolbox_puts_propulsive_load_alone_in_the_load_tab():
    components = [
        # An electric motor (PMSM, SM_PMSM)
        _component("e_motor", "propulsive_load"),
        # In a list (ICE, turboshaft), propulsive_load adds no tab of its own
        _component("ice", ["source", "propulsive_load"]),
    ]

    assert _group_icons_by_tab(components) == {"load": ["e_motor"], "source": ["ice"]}


def test_toolbox_lists_an_icon_once_per_tab():
    # PMSM and SM_PMSM share the e_motor icon
    components = [
        _component("e_motor", "propulsive_load"),
        _component("e_motor", "propulsive_load"),
    ]

    assert _group_icons_by_tab(components) == {"load": ["e_motor"]}


def test_unknown_component_closes_its_tabs():
    icons_by_tab = {
        "source": ["battery"],
        "connector": ["cable"],
        # Made-up: every real tab gets the unknown component, so a fake one
        # shows that the other tabs do not
        "other_class": ["other_icon"],
        "propulsor": ["propeller"],
        "tank": ["fuel_tank"],
        "load": ["dc_load"],
    }

    tabs = _with_unknown_component(icons_by_tab)

    # Last in the connector, propulsor, load, tank and source tabs, nowhere else
    assert tabs == {
        "source": ["battery", "unknown"],
        "connector": ["cable", "unknown"],
        "other_class": ["other_icon"],
        "propulsor": ["propeller", "unknown"],
        "tank": ["fuel_tank", "unknown"],
        "load": ["dc_load", "unknown"],
    }
    # A tab is never created for the unknown component alone
    assert _with_unknown_component({"source": ["battery"]}) == {"source": ["battery", "unknown"]}
