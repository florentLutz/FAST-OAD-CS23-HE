# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Save / load tests: the YAML writer and reader must round-trip a design.

The design is built through the GUI entry points, saved with the handler's own
save path and loaded back with its own load path.
"""

import json

from ruamel.yaml import YAML

from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from fastga_he.gui.power_train_builder.power_train_network_reader import PowerTrainYAMLReader
from fastga_he.gui.power_train_builder.power_train_network_writer import SOURCE_SIDE_KEYWORDS
from .conftest import icon_with_options


def _known_component(component_type: str) -> dict:
    return next(
        known_component
        for known_component in KNOWN_COMPONENTS
        if known_component["components_type"] == component_type
    )


def _port_count_attributes(component_type: str) -> tuple[str, str]:
    """``(source_side, target_side)`` number_of_* attributes of a variable-port type."""
    port_attributes = [
        attribute_name
        for attribute_name in _known_component(component_type).get("attributes") or []
        if "number_of_" in attribute_name
    ]
    source_attribute = next(
        attribute_name
        for attribute_name in port_attributes
        if any(keyword in attribute_name for keyword in SOURCE_SIDE_KEYWORDS)
    )
    target_attribute = next(
        attribute_name for attribute_name in port_attributes if attribute_name != source_attribute
    )
    return source_attribute, target_attribute


# Columns that must survive a save / load unchanged ("url", "w", "h" are
# regenerated from the icon and the current icon size)
_ROUND_TRIP_COLUMNS = (
    "name",
    "node_type",
    "icon_type",
    "position",
    "n_sources",
    "n_targets",
    "symmetry_name",
    "symmetry_node_index",
)


def _build_electric_chain(canvas) -> dict:
    """
    Battery -> cable -> motor -> propeller, plus a mirrored motor / propeller pair.

    :return: Dictionary mapping a short role name to the node index.
    """
    nodes = {
        "battery": canvas.place("battery", 100, 100),
        "cable": canvas.place("cable", 100, 300),
        "bus": canvas.place("bus_bar", 300, 300),
        "motor": canvas.place("e_motor", 100, 500),
        "propeller": canvas.place("propeller", 100, 700),
        "motor_mirror": canvas.place("e_motor", 500, 500),
        "propeller_mirror": canvas.place("propeller", 500, 700),
    }
    canvas.connect(("target", nodes["battery"], "1"), ("source", nodes["cable"], "1"))
    # Clicked source-first: the edge is stored from the source side
    canvas.connect(("source", nodes["bus"], "1"), ("target", nodes["cable"], "1"))
    canvas.connect(("target", nodes["bus"], "1"), ("source", nodes["motor"], "1"))
    canvas.connect(("source", nodes["motor_mirror"], "1"), ("target", nodes["bus"], "2"))
    canvas.connect(("target", nodes["motor"], "1"), ("source", nodes["propeller"], "1"))
    canvas.connect(
        ("target", nodes["motor_mirror"], "1"), ("source", nodes["propeller_mirror"], "1")
    )
    return nodes


def _apply_symmetry(canvas, node_index: int, peer_name: str):
    canvas.select(node_index)
    canvas.state.symmetry_select.value = peer_name
    canvas.handler.apply_node_configurations()


def test_round_trip_keeps_components_connections_and_layout(canvas, tmp_path):
    """
    Test that a design saved to YAML and loaded back has the same components, connections, and layout.
    """
    nodes = _build_electric_chain(canvas)
    _apply_symmetry(canvas, nodes["motor"], "e_motor_2")
    expected_columns = {column: canvas.node_column(column) for column in _ROUND_TRIP_COLUMNS}
    expected_xy = list(zip(canvas.node_column("x"), canvas.node_column("y"), strict=True))
    expected_edges = sorted(canvas.edges())
    yaml_path = tmp_path / "design.yml"

    # save
    assert canvas.handler._save_canvas_state(str(yaml_path))
    # restart a new session
    canvas.handler._on_new_design()
    assert canvas.node_column("name") == []
    # load design
    canvas.handler._load_canvas_state(str(yaml_path))

    for column, expected_values in expected_columns.items():
        assert canvas.node_column(column) == expected_values, column
    assert list(zip(canvas.node_column("x"), canvas.node_column("y"), strict=True)) == expected_xy
    assert sorted(canvas.edges()) == expected_edges


def test_round_trip_keeps_options(canvas, tmp_path):
    """
    Test that a design saved to YAML and loaded back has the same options.
    """
    motor = canvas.place(icon_with_options(canvas.state), 100, 500)
    canvas.select(motor)
    options_before = json.loads(canvas.node_column("options")[motor])
    assert options_before
    yaml_path = tmp_path / "design.yml"

    canvas.handler._save_canvas_state(str(yaml_path))
    canvas.handler._load_canvas_state(str(yaml_path))

    assert json.loads(canvas.node_column("options")[motor]) == options_before


def test_saved_file_has_fast_oad_sections_and_port_indices(canvas, tmp_path):
    _build_electric_chain(canvas)
    yaml_path = tmp_path / "design.yml"
    # save the design to a YAML file
    canvas.handler._save_canvas_state(str(yaml_path))
    # reload the yml file
    saved = YAML(typ="safe").load(yaml_path.read_text())

    assert set(saved["power_train_components"]) == set(canvas.node_column("name"))
    node_types = dict(zip(canvas.node_column("name"), canvas.node_column("node_type"), strict=True))
    assert (
        saved["power_train_components"]["battery_1"]["id"]
        == _known_component(node_types["battery_1"])["id"]
    )
    # Variable-port components carry their port counts as number_of_* options
    source_attribute, target_attribute = _port_count_attributes(node_types["bus_bar_1"])
    bus_options = saved["power_train_components"]["bus_bar_1"]["options"]
    assert bus_options[source_attribute] == 3 and bus_options[target_attribute] == 3
    connections = saved["component_connections"]
    # The bus bar has several target ports, so its endpoints carry the port index
    assert {"source": "e_motor_1", "target": ["bus_bar_1", 1]} in connections
    assert {"source": "e_motor_2", "target": ["bus_bar_1", 2]} in connections
    assert {"source": ["bus_bar_1", 1], "target": "cable_1"} in connections
    assert "canvas_layout" in saved


def test_final_version_has_no_layout_and_reloads_with_automatic_layout(canvas, tmp_path):
    _build_electric_chain(canvas)
    expected_edges = sorted(canvas.edges())
    yaml_path = tmp_path / "final.yml"

    canvas.handler._save_canvas_state(str(yaml_path), include_canvas_layout=False)
    assert "canvas_layout" not in YAML(typ="safe").load(yaml_path.read_text())
    canvas.handler._load_canvas_state(str(yaml_path))

    names = canvas.node_column("name")
    y_by_name = dict(zip(names, canvas.node_column("y"), strict=True))
    # Energy storage at the bottom, propulsors at the top
    assert y_by_name["battery_1"] < y_by_name["e_motor_1"] < y_by_name["propeller_1"]
    assert len(set(zip(canvas.node_column("x"), canvas.node_column("y"), strict=True))) == len(
        names
    )
    assert sorted(canvas.edges()) == expected_edges


def test_empty_design_round_trip(canvas, tmp_path):
    yaml_path = tmp_path / "empty.yml"

    assert canvas.handler._save_canvas_state(str(yaml_path))
    canvas.handler._load_canvas_state(str(yaml_path))

    assert canvas.node_column("name") == []
    assert canvas.edges() == []


def test_reader_handles_hand_written_yaml(builder_state, tmp_path):
    bus_type = builder_state.icon_to_component_types["bus_bar"][0]
    battery_type = builder_state.icon_to_component_types["battery"][0]
    source_attribute, target_attribute = _port_count_attributes(bus_type)
    yaml_path = tmp_path / "hand_written.yml"
    yaml_path.write_text(
        "title: hand written\n"
        "power_train_components:\n"
        "  battery_1:\n"
        f"    id: {_known_component(battery_type)['id']}\n"
        "  bus_1:\n"
        f"    id: {_known_component(bus_type)['id']}\n"
        "    options:\n"
        f"      {source_attribute}: 1\n"
        f"      {target_attribute}: 2\n"
        "  mystery_1:\n"
        "    id: fastga_he.pt_component.does_not_exist\n"
        "component_connections:\n"
        "  - source: bus_1\n"
        "    target: battery_1\n"
        "  - source: '[bus_1, 2]'\n"
        "    target: mystery_1\n"
    )

    canvas_state = PowerTrainYAMLReader(builder_state, 60).read(yaml_path)

    nodes_data = canvas_state["components"]
    # Unknown component ids are skipped, not fatal
    assert nodes_data["name"] == ["battery_1", "bus_1"]
    # Variable-port components read their port counts from the number_of_* options
    assert nodes_data["n_sources"][1] == 1 and nodes_data["n_targets"][1] == 2
    # The connection to the skipped component is dropped
    assert len(canvas_state["connections"]["starting_node_index"]) == 1


def test_reader_rejects_nothing_on_empty_file(builder_state, tmp_path):
    yaml_path = tmp_path / "blank.yml"
    yaml_path.write_text("")

    canvas_state = PowerTrainYAMLReader(builder_state, 60).read(yaml_path)

    assert canvas_state["components"]["name"] == []
    assert canvas_state["watcher_file_path"] == ""


def test_failed_load_is_reported_in_the_status_line_and_keeps_the_canvas(canvas, tmp_path):
    canvas.place("battery", 100, 100)
    broken_path = tmp_path / "broken.yml"
    broken_path.write_text("power_train_components: [unclosed\n")

    canvas.handler._load_canvas_state(str(broken_path))

    assert "Could not load broken.yml" in canvas.state.status_div.text
    assert canvas.node_column("name") == ["battery_1"]


def test_missing_file_is_reported_in_the_status_line(canvas, tmp_path):
    canvas.handler._load_canvas_state(str(tmp_path / "missing.yml"))

    assert "file not found" in canvas.state.status_div.text


def test_non_yaml_file_is_refused_and_keeps_the_canvas(canvas, tmp_path):
    canvas.place("battery", 100, 100)
    json_path = tmp_path / "old_backup.json"
    json_path.write_text('{"components": {}, "connections": {}}')

    canvas.handler._load_canvas_state(str(json_path))

    assert "not a YAML file" in canvas.state.status_div.text
    assert canvas.node_column("name") == ["battery_1"]
