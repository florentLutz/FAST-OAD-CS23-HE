# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Unit tests for the builder's pure helpers: the shared state helpers, component
metadata, config-panel formatting, UI triggers and the undo / redo history.

Each shared helper replaced logic that used to be written out in several
places, so these tests pin down the single behaviour every caller relies on.
"""

from types import SimpleNamespace

import bokeh.models as bkmodel

from fastga_he.gui.power_train_builder import power_train_builder_metadata as metadata
from fastga_he.gui.power_train_builder.power_train_builder_handler_config_panel import (
    PRIMARY_PORT_LABEL_COLOR,
    PRIMARY_SOURCE_PORT_TYPE_KEYWORDS,
    PRIMARY_TARGET_PORT_TYPE_KEYWORDS,
    _option_values_to_strings,
    _port_label_stylesheets,
    _strings_to_option_values,
)
from fastga_he.gui.power_train_builder.power_train_builder_handler_io import _flip_trigger
from fastga_he.gui.power_train_builder.power_train_builder_history import (
    EditHistoryStack,
    make_snapshot,
)
from fastga_he.gui.power_train_builder.power_train_builder_metadata import (
    component_type_classes,
    icon_data_url,
)
from fastga_he.gui.power_train_builder.power_train_builder_state import (
    DIALOG_POSITION_STYLES,
    DIALOG_STYLES,
    EDGE_COLUMNS,
    PORT_TYPE_COLORS,
    UNDEFINED_PORT_COLOR,
    _EMPTY,
    _NO_SYMMETRY_PEER,
    append_entry,
    default_options,
    default_position,
    drop_entry,
    edge_end_ports,
    edge_indices_not_touching_port,
    empty_columns,
    link_symmetry,
    ports_compatible,
    unlink_symmetry,
)
from fastga_he.gui.power_train_builder.power_train_builder_toolbox_config_table import _js_flip
from fastga_he.gui.power_train_builder.power_train_network_writer import (
    port_side_of,
    unknown_component_id,
    unknown_component_type,
)


def _edges(*end_pairs) -> dict:
    """Edge data from ``((kind, node, label), (kind, node, label))`` pairs."""
    edge_data = empty_columns(EDGE_COLUMNS)
    for starting_port, ending_port in end_pairs:
        edge_data = append_entry(
            edge_data,
            EDGE_COLUMNS,
            {
                "xs": [0.0, 1.0],
                "ys": [0.0, 1.0],
                "color": "blue",
                **{
                    f"{end}_{field}": value
                    for end, port in (("starting", starting_port), ("ending", ending_port))
                    for field, value in zip(
                        ("port_kind", "node_index", "port_label"), port, strict=True
                    )
                },
            },
        )
    return edge_data


# ---------------------------------------------------------------------------
# Shared state helpers
# ---------------------------------------------------------------------------


def test_append_and_drop_entry_return_new_column_data():
    column_data = {"name": ["a", "b"], "x": [1.0, 2.0]}

    appended = append_entry(column_data, ("name", "x"), {"name": "c", "x": 3.0})
    dropped = drop_entry(column_data, 0)

    assert appended == {"name": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]}
    assert dropped == {"name": ["b"], "x": [2.0]}
    # The input is never changed, and a column missing from it is started
    assert column_data == {"name": ["a", "b"], "x": [1.0, 2.0]}
    assert append_entry({}, ("name",), {"name": "a"}) == {"name": ["a"]}


def test_defaults_of_a_new_component_are_the_first_choices():
    definition = {"mode": ["percent_split", "power_share"], "flag": [True, False], "none": []}

    # Options without any possible value are left out
    assert default_options(definition) == {"mode": "percent_split", "flag": True}
    assert default_position(["in_the_front", "in_the_nose"]) == "in_the_front"
    assert default_position([]) == _EMPTY


def test_edges_are_read_end_by_end():
    edge_data = _edges(
        (("source", 1, "1"), ("target", 0, "1")),
        (("target", 0, "1"), ("source", 2, "1")),  # target port clicked first
        (("source", 1, "2"), ("target", 3, "1")),
    )

    # Each edge keeps the order its ports were clicked in
    assert edge_end_ports(edge_data)[1] == (("target", 0, "1"), ("source", 2, "1"))
    assert edge_end_ports(empty_columns(EDGE_COLUMNS)) == []
    # A port is found at either end: target port 1 of node 0 ends edge 0 and starts edge 1
    assert edge_indices_not_touching_port(edge_data, 0, "1", "target") == [2]
    # Same node and label on the other side touches nothing
    assert edge_indices_not_touching_port(edge_data, 0, "1", "source") == [0, 1, 2]


def test_link_then_unlink_symmetry():
    names, indices = [_EMPTY] * 3, [_NO_SYMMETRY_PEER] * 3

    link_symmetry(names, indices, ["motor_1", "motor_2", "motor_3"], 0, 2)
    assert names == ["motor_3", _EMPTY, "motor_1"]
    assert indices == [2, _NO_SYMMETRY_PEER, 0]

    # Only the listed node is reset; its peer still points at it
    unlink_symmetry(names, indices, [0])
    assert names == [_EMPTY, _EMPTY, "motor_1"]
    assert indices == [_NO_SYMMETRY_PEER, _NO_SYMMETRY_PEER, 0]


def test_port_side_of_number_of_attributes():
    # Inputs and tanks count the source ports
    assert port_side_of("number_of_inputs") == "source"
    assert port_side_of("number_of_tanks") == "source"
    # Outputs, engines and power sources count the target ports
    assert port_side_of("number_of_outputs") == "target"
    assert port_side_of("number_of_engines") == "target"
    assert port_side_of("number_of_power_sources") == "target"


def test_port_side_of_other_attributes():
    # number_of_, but no side keyword
    assert port_side_of("number_of_cells") is None
    # Side keyword, but not number_of_
    assert port_side_of("voltage_inputs") is None


def test_unknown_component_helpers():
    electric, mechanical = PORT_TYPE_COLORS["electricity"], PORT_TYPE_COLORS["mechanical"]

    # Ports connect only on the same, defined energy type
    assert ports_compatible(electric, electric)
    assert not ports_compatible(electric, mechanical)
    assert not ports_compatible(UNDEFINED_PORT_COLOR, mechanical)
    assert not ports_compatible(UNDEFINED_PORT_COLOR, UNDEFINED_PORT_COLOR)

    # A typed type gets the FAST-OAD_CS23-HE prefix, a full id is kept, and reading undoes both
    assert unknown_component_id("my_part") == "fastga_he.pt_component.my_part"
    assert unknown_component_id("other.pt_component.part") == "other.pt_component.part"
    assert unknown_component_id("") == "fastga_he.pt_component.unknown"
    for component_type in ("my_part", "other.pt_component.part"):
        assert unknown_component_type(unknown_component_id(component_type)) == component_type
    assert unknown_component_type(None) == "unknown"


# ---------------------------------------------------------------------------
# Component metadata
# ---------------------------------------------------------------------------


def test_component_type_classes_wraps_a_single_class_in_a_list():
    known_component = {"components_type_class": "propulsor"}

    assert component_type_classes(known_component) == ["propulsor"]
    # The entry itself is not changed
    assert known_component["components_type_class"] == "propulsor"


def test_component_type_classes_keeps_a_list_of_classes():
    known_component = {"components_type_class": ["source", "connector"]}

    assert component_type_classes(known_component) == ["source", "connector"]
    # The entry itself is not changed
    assert known_component["components_type_class"] == ["source", "connector"]


def test_icon_data_url_reads_each_icon_once(monkeypatch, tmp_path):
    icon_path = tmp_path / "icon.png"
    icon_path.write_bytes(b"\x89PNG")
    encoded_urls = []

    def fake_url_to_base64(url):
        encoded_urls.append(url)
        return "data:image/png;base64,AAAA"

    # Replace the real function with a fake that records the URL
    monkeypatch.setattr(metadata, "_url_to_base64", fake_url_to_base64)
    # Empty icon cache for this test; monkeypatch restores the real one afterwards
    monkeypatch.setattr(metadata, "_ICON_DATA_URLS", {})

    first = icon_data_url(icon_path)
    second = icon_data_url(icon_path)

    assert first == second == "data:image/png;base64,AAAA"
    assert encoded_urls == ["file://" + str(icon_path.resolve())]


# ---------------------------------------------------------------------------
# Config panel formatting
# ---------------------------------------------------------------------------


def _round_trip(value):
    """An option value turned into its display string, then read back."""
    display_string = _option_values_to_strings(value)
    assert isinstance(display_string, str)
    return _strings_to_option_values(display_string)


def test_option_values_round_trip_through_display_strings():
    """Every option value survives the trip to its display string and back, type included."""
    # "is" checks the type too: True must not come back as 1
    assert _round_trip(True) is True
    assert _round_trip(False) is False

    integer = _round_trip(3)
    assert integer == 3 and type(integer) is int

    floating_point = _round_trip(2.5)
    assert floating_point == 2.5 and type(floating_point) is float

    assert _round_trip("percent_split") == "percent_split"


def _is_highlighted(node_type: str, port_label: str, keywords) -> bool:
    """
    Whether a port label of the Connections panel is highlighted as the primary port.

    The primary port gets exactly one stylesheet, with the highlight colour;
    every other port keeps the default style (no stylesheet).
    """
    stylesheets = _port_label_stylesheets(node_type, port_label, keywords)
    is_highlighted = len(stylesheets) == 1
    assert len(stylesheets) <= 1
    assert (PRIMARY_PORT_LABEL_COLOR in "".join(stylesheets)) is is_highlighted
    return is_highlighted


def test_primary_port_labels_are_highlighted():
    # Source port 1 of a splitter or a gearbox
    assert _is_highlighted("dc_splitter", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS)
    assert _is_highlighted("gearbox", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS)
    # Target port 1 of a planetary gear
    assert _is_highlighted("planetary_gear", "1", PRIMARY_TARGET_PORT_TYPE_KEYWORDS)


def test_other_port_labels_keep_the_default_style():
    # Another port number
    assert not _is_highlighted("dc_splitter", "2", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS)
    # The other side
    assert not _is_highlighted("dc_splitter", "1", PRIMARY_TARGET_PORT_TYPE_KEYWORDS)
    assert not _is_highlighted("planetary_gear", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS)
    # A component without a primary port
    assert not _is_highlighted("pmsm", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS)


# ---------------------------------------------------------------------------
# UI constants and triggers
# ---------------------------------------------------------------------------


def test_dialog_style_and_trigger_helpers():
    # Every dialog is placed by the shared position styles
    assert DIALOG_POSITION_STYLES.items() <= DIALOG_STYLES.items()

    # Python flips a trigger between "0" and "1", so every flip is a change
    trigger = bkmodel.TextInput(value="0")
    _flip_trigger(trigger)
    assert trigger.value == "1"
    _flip_trigger(trigger)
    assert trigger.value == "0"

    # The JS twin does the same on the named trigger
    assert _js_flip("browse_load_trigger") == (
        "browse_load_trigger.value = browse_load_trigger.value === '1' ? '0' : '1';"
    )


# ---------------------------------------------------------------------------
# Undo / redo history
# ---------------------------------------------------------------------------


def test_undo_then_redo_returns_snapshots_in_order():
    history_stack = EditHistoryStack()
    history_stack.push_undo({"step": 1})
    history_stack.push_undo({"step": 2})

    assert history_stack.can_undo and not history_stack.can_redo
    assert history_stack.undo() == {"step": 2}

    history_stack.push_redo({"step": 3})
    assert history_stack.can_redo
    assert history_stack.redo() == {"step": 3}


def test_only_a_new_edit_clears_the_redo_stack():
    history_stack = EditHistoryStack()
    history_stack.push_redo({"step": 2})

    # Redo saves the current state for undo, keeping the other redo steps
    history_stack.push_undo_keep_redo({"step": 1})
    assert history_stack.can_undo and history_stack.can_redo

    # A new edit makes the undone steps unreachable
    history_stack.push_undo({"step": 3})
    assert not history_stack.can_redo


def test_empty_or_cleared_history_has_nothing_to_undo_or_redo():
    history_stack = EditHistoryStack()
    assert history_stack.undo() is None
    assert history_stack.redo() is None
    assert len(history_stack) == 0

    history_stack.push_undo({"step": 1})
    history_stack.push_redo({"step": 2})
    history_stack.clear()

    assert not history_stack.can_undo and not history_stack.can_redo


def test_max_depth_drops_oldest_snapshot():
    history_stack = EditHistoryStack(max_depth=3)
    for step in range(5):
        history_stack.push_undo({"step": step})

    assert len(history_stack) == 3
    remaining_steps = [history_stack.undo()["step"] for _ in range(3)]
    assert remaining_steps == [4, 3, 2]


def test_snapshot_is_independent_of_live_data():
    nodes_data = {"name": ["battery_1"], "x": [1.0]}
    state = SimpleNamespace(
        placed_nodes_source=SimpleNamespace(data=nodes_data),
        edge_source=SimpleNamespace(data={"xs": [[0.0, 1.0]]}),
        placed_counter={"battery": 1},
    )

    # Take a snapshot, then change the live data
    snapshot = make_snapshot(state)
    nodes_data["name"].append("cable_1")
    state.edge_source.data["xs"][0].append(2.0)
    state.placed_counter["battery"] = 5

    assert snapshot["nodes"]["name"] == ["battery_1"]
    assert snapshot["edges"]["xs"] == [[0.0, 1.0]]
    assert snapshot["placed_counter"] == {"battery": 1}
