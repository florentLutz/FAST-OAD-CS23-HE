# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Save / load tests: the YAML writer and reader must round-trip a design.

The design is built through the GUI entry points, saved with the handler's own
save path and loaded back with its own load path.

The watcher CSV path is set in the "final version?" overlay and written to the
YAML. There, the file dialogs are replaced by a fixed path, and the save is
started the way the overlay buttons start it (see ``_JS_SAVE_MODE_CLICK`` in
the toolbox): first commit the field's ``value_input`` into its ``value``, then
change ``browse_save_trigger``. BokehJS keeps two properties for a text field:
``value_input`` follows every keystroke, ``value`` only changes on Enter /
blur. The tests type into the field the same way, so a path that Python sets
in ``value`` alone (and that the commit step would then overwrite) is caught.
"""

import itertools
import json

from ruamel.yaml import YAML

from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from fastga_he.gui.power_train_builder.power_train_builder_state import (
    SAVE_MODE_EDITABLE,
    SAVE_MODE_FINAL,
)
from fastga_he.gui.power_train_builder.power_train_network_reader import PowerTrainYAMLReader
from fastga_he.gui.power_train_builder.power_train_network_writer import port_side_of
from .conftest import find_icon_with_options


def _known_component(component_type: str) -> dict:
    return next(
        known_component
        for known_component in KNOWN_COMPONENTS
        if known_component["components_type"] == component_type
    )


def _port_count_attributes(component_type: str) -> tuple[str, str]:
    """``(source_side, target_side)`` number_of_* attributes of a variable-port type."""
    attribute_by_side = {
        port_side_of(attribute_name): attribute_name
        for attribute_name in _known_component(component_type).get("attributes") or []
    }
    return attribute_by_side["source"], attribute_by_side["target"]


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
    canvas.apply()


def _options_of_every_node(canvas) -> list[dict]:
    """Options of every node, decoded from their JSON strings."""
    return [json.loads(options) for options in canvas.node_column("options")]


def test_round_trip_keeps_components_connections_options_and_layout(canvas, tmp_path):
    """
    A design saved to YAML and loaded back has the same components, connections,
    options and layout.
    """
    nodes = _build_electric_chain(canvas)
    _apply_symmetry(canvas, nodes["motor"], "e_motor_2")
    # Plus a component with options to choose from, if the registry has one
    icon_key = find_icon_with_options(canvas.state)
    if icon_key is not None:
        with_options = canvas.place(icon_key, 700, 100)
        assert _options_of_every_node(canvas)[with_options]

    expected_columns = {column: canvas.node_column(column) for column in _ROUND_TRIP_COLUMNS}
    expected_options = _options_of_every_node(canvas)
    expected_xy = list(zip(canvas.node_column("x"), canvas.node_column("y"), strict=True))
    expected_edges = sorted(canvas.edges())
    yaml_path = tmp_path / "design.yml"

    # save
    assert canvas.save(yaml_path)
    # restart a new session
    canvas.new_design()
    assert canvas.node_column("name") == []
    # load design
    canvas.load(yaml_path)

    for column, expected_values in expected_columns.items():
        assert canvas.node_column(column) == expected_values, column
    assert _options_of_every_node(canvas) == expected_options
    assert list(zip(canvas.node_column("x"), canvas.node_column("y"), strict=True)) == expected_xy
    assert sorted(canvas.edges()) == expected_edges


def test_saved_file_has_fast_oad_sections_and_port_indices(canvas, tmp_path):
    _build_electric_chain(canvas)
    yaml_path = tmp_path / "design.yml"
    # save the design to a YAML file
    canvas.save(yaml_path)
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

    canvas.save(yaml_path, final=True)
    assert "canvas_layout" not in YAML(typ="safe").load(yaml_path.read_text())
    canvas.load(yaml_path)

    names = canvas.node_column("name")
    y_by_name = dict(zip(names, canvas.node_column("y"), strict=True))
    # Energy storage at the bottom, propulsors at the top
    assert y_by_name["battery_1"] < y_by_name["e_motor_1"] < y_by_name["propeller_1"]
    assert len(set(zip(canvas.node_column("x"), canvas.node_column("y"), strict=True))) == len(
        names
    )
    assert sorted(canvas.edges()) == expected_edges


def test_empty_design_and_empty_file_load_as_an_empty_canvas(canvas, builder_state, tmp_path):
    yaml_path = tmp_path / "empty.yml"
    assert canvas.save(yaml_path)
    canvas.load(yaml_path)

    assert canvas.node_column("name") == []
    assert canvas.edges() == []

    # A blank file is read as an empty design, not rejected
    blank_path = tmp_path / "blank.yml"
    blank_path.write_text("")
    canvas_state = PowerTrainYAMLReader(builder_state, 60).read(blank_path)

    assert canvas_state["components"]["name"] == []
    assert canvas_state["watcher_file_path"] == ""


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
        "  mystery_2:\n"
        "    id: fastga_he.pt_component.does_not_exist\n"
        "component_connections:\n"
        # Listed before mystery_1 is typed: kept once the bus has typed it
        "  - source: mystery_2\n"
        "    target: '[mystery_1, 2]'\n"
        "  - source: bus_1\n"
        "    target: battery_1\n"
        "  - source: '[bus_1, 2]'\n"
        "    target: mystery_1\n"
        # Dropped: a component connected to itself, and a battery side without ports
        "  - source: bus_1\n"
        "    target: '[bus_1, 2]'\n"
        "  - source: battery_1\n"
        "    target: bus_1\n"
    )

    canvas_state = PowerTrainYAMLReader(builder_state, 60).read(yaml_path)

    nodes_data = canvas_state["components"]
    # An unregistered component id is shown as an unknown component, not dropped
    assert nodes_data["name"] == ["battery_1", "bus_1", "mystery_1", "mystery_2"]
    assert nodes_data["icon_type"][2] == "unknown"
    assert nodes_data["node_type"][2] == "does_not_exist"
    # Variable-port components read their port counts from the number_of_* options
    assert nodes_data["n_sources"][1] == 1 and nodes_data["n_targets"][1] == 2
    # Its connections are kept too, undefined ports typed through the chain from the bus
    assert len(canvas_state["connections"]["starting_node_index"]) == 3
    assert nodes_data["target_port_type"][2] == nodes_data["source_port_type"][3] == "electricity"
    assert nodes_data["source_port_type"][2] == "undefined"


def _load_into_a_one_battery_design(canvas, file_path):
    """Place a battery, then try to load *file_path* over it."""
    canvas.place("battery", 100, 100)
    canvas.load(file_path)


def test_invalid_yaml_is_reported_and_keeps_the_canvas(canvas, tmp_path):
    file_path = tmp_path / "broken.yml"
    file_path.write_text("power_train_components: [unclosed\n")

    _load_into_a_one_battery_design(canvas, file_path)

    assert "Could not load broken.yml" in canvas.state.status_div.text
    assert canvas.node_column("name") == ["battery_1"]


def test_missing_file_is_reported_and_keeps_the_canvas(canvas, tmp_path):
    _load_into_a_one_battery_design(canvas, tmp_path / "missing.yml")

    assert "file not found" in canvas.state.status_div.text
    assert canvas.node_column("name") == ["battery_1"]


def test_file_that_is_not_yaml_is_reported_and_keeps_the_canvas(canvas, tmp_path):
    file_path = tmp_path / "old_backup.json"
    file_path.write_text('{"components": {}, "connections": {}}')

    _load_into_a_one_battery_design(canvas, file_path)

    assert "not a YAML file" in canvas.state.status_div.text
    assert canvas.node_column("name") == ["battery_1"]


def test_unknown_component_round_trip(canvas, tmp_path):
    """
    An unknown component is saved with the id built from its type and exactly its
    options; the editable file also keeps its port counts and port types, the final
    version sizes its ports from its connections.
    """
    battery = canvas.place("battery", 100, 100)
    unknown = canvas.place("unknown", 300, 300)
    canvas.select(unknown)
    state = canvas.state
    state.unknown_type_input.value = "my_part"
    state.unknown_position_input.value = "in_the_nose"
    canvas.add_unknown_option("efficiency", "0.95")
    state.source_count_spinner.value = 2
    state.unknown_symmetry_input.value = "right_part"
    canvas.apply()
    state.source_port_type_select.value = "electricity"
    canvas.connect(("source", unknown, "2"), ("target", battery, "1"))
    expected_columns = {
        column: canvas.node_column(column)
        for column in _ROUND_TRIP_COLUMNS + ("options", "source_port_type", "target_port_type")
    }

    yaml_path = tmp_path / "design.yml"
    assert canvas.save(yaml_path)
    saved = YAML(typ="safe").load(yaml_path.read_text())
    assert saved["power_train_components"]["unknown_1"] == {
        "id": "fastga_he.pt_component.my_part",
        "options": {"efficiency": 0.95},
        "position": "in_the_nose",
        "symmetrical": "right_part",
    }
    assert {"source": ["unknown_1", 2], "target": "battery_1"} in saved["component_connections"]

    canvas.new_design()
    canvas.load(yaml_path)
    for column, expected_values in expected_columns.items():
        assert canvas.node_column(column) == expected_values, column
    assert canvas.edges() == [((battery, "target", "1"), (unknown, "source", "2"))]

    # A final version has no canvas_layout: the port types come from the
    # connections (none on the target side), and the source side gets as many
    # ports as its connections use
    final_path = tmp_path / "final.yml"
    canvas.save(final_path, final=True)
    canvas.load(final_path)
    assert canvas.edges() == [((battery, "target", "1"), (unknown, "source", "2"))]
    assert canvas.node_column("source_port_type")[unknown] == "electricity"
    assert canvas.node_column("target_port_type")[unknown] == "undefined"
    assert canvas.node_column("n_sources")[unknown] == 2
    assert canvas.node_column("n_targets")[unknown] == 1


# ---------------------------------------------------------------------------
# Watcher CSV path
# ---------------------------------------------------------------------------

WATCHER_PATH = "results/pt_watcher.csv"

# Stands in for the Date.now() of the overlay's JS: every save is a new trigger value
_click_stamps = itertools.count()


def _type_in_watcher_field(canvas, text: str, commit: bool = False):
    """
    Type *text* in the watcher field, as the browser reports it.

    :param commit: ``True`` to also press Enter / leave the field, which copies
        the typed text into ``value``.
    """
    canvas.state.watcher_path_input.value_input = text
    if commit:
        canvas.state.watcher_path_input.value = text


def _save_through_overlay(canvas, monkeypatch, yaml_path, save_mode=SAVE_MODE_EDITABLE):
    """Save as the overlay buttons do, with the save dialog answering *yaml_path*."""
    monkeypatch.setattr(
        canvas.handler, "_open_tkinter_dialog", lambda *args, **kwargs: str(yaml_path)
    )
    watcher_path_input = canvas.state.watcher_path_input
    # Same steps as _JS_SAVE_MODE_CLICK
    watcher_path_input.value = watcher_path_input.value_input
    canvas.state.browse_save_trigger.value = f"{save_mode}:{next(_click_stamps)}"
    return YAML(typ="safe").load(yaml_path.read_text())


def _browse_watcher(canvas, monkeypatch, chosen_path: str):
    """Pick *chosen_path* with the overlay's Browse button."""
    monkeypatch.setattr(canvas.handler, "_open_tkinter_dialog", lambda *args, **kwargs: chosen_path)
    canvas.state.browse_watcher_trigger.value = str(next(_click_stamps))


def _watcher_field(canvas) -> tuple[str, str]:
    """``(value, value_input)`` of the watcher field."""
    return canvas.state.watcher_path_input.value, canvas.state.watcher_path_input.value_input


def test_watcher_widgets_are_in_the_save_overlay(canvas):
    # The overlay and every widget laid out inside it, at any depth: the loop
    # also visits the children appended to the list while it runs. Only
    # "children" is followed, not references(), which would also find widgets
    # merely passed to a CustomJS callback
    overlay_widgets = [canvas.state.save_version_overlay]
    for widget in overlay_widgets:
        overlay_widgets.extend(getattr(widget, "children", []))

    assert canvas.state.watcher_path_input in overlay_widgets


def test_typed_watcher_path_is_written_to_the_yaml(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    _type_in_watcher_field(canvas, WATCHER_PATH, commit=True)

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")

    assert saved["watcher_file_path"] == WATCHER_PATH
    assert "canvas_layout" in saved
    assert canvas.state.save_button.button_type == "success"


def test_watcher_path_still_being_typed_is_written_to_the_yaml(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    # No Enter / blur before clicking the save button: only value_input holds the text
    _type_in_watcher_field(canvas, WATCHER_PATH, commit=False)

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")

    assert saved["watcher_file_path"] == WATCHER_PATH
    assert "canvas_layout" in saved
    assert canvas.state.save_button.button_type == "success"


def test_watcher_path_is_written_to_a_final_version(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    _type_in_watcher_field(canvas, WATCHER_PATH, commit=True)

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml", SAVE_MODE_FINAL)

    assert saved["watcher_file_path"] == WATCHER_PATH
    assert "canvas_layout" not in saved
    assert canvas.state.save_button.button_type == "success"


def test_empty_watcher_field_leaves_the_key_out_of_the_yaml(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")

    assert "watcher_file_path" not in saved
    assert "canvas_layout" in saved
    assert canvas.state.save_button.button_type == "success"


def test_editing_the_watcher_path_marks_the_design_unsaved(canvas, monkeypatch, tmp_path):
    _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")
    assert canvas.state.save_button.button_type == "success"

    _type_in_watcher_field(canvas, WATCHER_PATH, commit=True)

    assert canvas.state.save_button.button_type == "warning"


def test_browsed_watcher_path_is_saved_until_typed_over(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)

    _browse_watcher(canvas, monkeypatch, "/data/pt_watcher.csv")
    # The overlay buttons commit value_input, so it must hold the path too
    assert _watcher_field(canvas) == ("/data/pt_watcher.csv", "/data/pt_watcher.csv")
    assert canvas.state.watcher_file_path == "/data/pt_watcher.csv"
    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "browsed.yml")
    assert saved["watcher_file_path"] == "/data/pt_watcher.csv"

    _type_in_watcher_field(canvas, WATCHER_PATH)
    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "typed.yml")
    assert saved["watcher_file_path"] == WATCHER_PATH


def test_loaded_watcher_path_is_shown_and_kept_on_resave(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    _type_in_watcher_field(canvas, WATCHER_PATH, commit=True)
    yaml_path = tmp_path / "design.yml"
    _save_through_overlay(canvas, monkeypatch, yaml_path)
    canvas.new_design()
    assert _watcher_field(canvas) == ("", "")

    canvas.load(yaml_path)

    # Shown without marking the design unsaved
    assert _watcher_field(canvas) == (WATCHER_PATH, WATCHER_PATH)
    assert canvas.state.save_button.button_type == "success"
    # Saved again without touching the field
    resaved = _save_through_overlay(canvas, monkeypatch, tmp_path / "resaved.yml")
    assert resaved["watcher_file_path"] == WATCHER_PATH
