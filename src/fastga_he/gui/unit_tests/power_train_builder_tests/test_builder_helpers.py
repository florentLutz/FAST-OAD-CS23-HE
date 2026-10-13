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
import pytest

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
    append_row,
    default_options,
    default_position,
    drop_row,
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
        edge_data = append_row(
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


def test_append_and_drop_row_return_new_column_data():
    column_data = {"name": ["a", "b"], "x": [1.0, 2.0]}

    appended = append_row(column_data, ("name", "x"), {"name": "c", "x": 3.0})
    dropped = drop_row(column_data, 0)

    assert appended == {"name": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]}
    assert dropped == {"name": ["b"], "x": [2.0]}
    # The input is never changed, and a column missing from it is started
    assert column_data == {"name": ["a", "b"], "x": [1.0, 2.0]}
    assert append_row({}, ("name",), {"name": "a"}) == {"name": ["a"]}


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


@pytest.mark.parametrize(
    "attribute_name, expected",
    [
        ("number_of_inputs", "source"),
        ("number_of_tanks", "source"),
        ("number_of_outputs", "target"),
        ("number_of_engines", "target"),
        ("number_of_power_sources", "target"),
        ("number_of_cells", None),  # number_of_, but no side keyword
        ("voltage_inputs", None),  # side keyword, but not number_of_
    ],
)
def test_port_side_of(attribute_name, expected):
    assert port_side_of(attribute_name) == expected


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


@pytest.mark.parametrize(
    "type_class, expected",
    [("propulsor", ["propulsor"]), (["source", "connector"], ["source", "connector"])],
)
def test_component_type_classes_is_always_a_list(type_class, expected):
    known_component = {"components_type_class": type_class}

    assert component_type_classes(known_component) == expected
    # The entry itself is not changed
    assert known_component["components_type_class"] == type_class


def test_icon_data_url_reads_each_icon_once(monkeypatch, tmp_path):
    icon_path = tmp_path / "icon.png"
    icon_path.write_bytes(b"\x89PNG")
    encoded_urls = []

    def fake_url_to_base64(url):
        encoded_urls.append(url)
        return "data:image/png;base64,AAAA"

    monkeypatch.setattr(metadata, "_url_to_base64", fake_url_to_base64)
    icon_data_url.cache_clear()
    try:
        first = icon_data_url(icon_path)
        second = icon_data_url(icon_path)
    finally:
        icon_data_url.cache_clear()

    assert first == second == "data:image/png;base64,AAAA"
    assert encoded_urls == ["file://" + str(icon_path.resolve())]


# ---------------------------------------------------------------------------
# Config panel formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False, 3, 2.5, "percent_split"])
def test_option_value_round_trips_through_display_string(value):
    """Every option value survives the trip to its display string and back, type included."""
    display_string = _option_values_to_strings(value)

    assert isinstance(display_string, str)
    assert _strings_to_option_values(display_string) == value
    assert type(_strings_to_option_values(display_string)) is type(value)


@pytest.mark.parametrize(
    "node_type, port_label, keywords, is_primary",
    [
        ("dc_splitter", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS, True),
        ("gearbox", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS, True),
        ("planetary_gear", "1", PRIMARY_TARGET_PORT_TYPE_KEYWORDS, True),
        ("dc_splitter", "2", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS, False),
        ("dc_splitter", "1", PRIMARY_TARGET_PORT_TYPE_KEYWORDS, False),
        ("planetary_gear", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS, False),
        ("pmsm", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS, False),
    ],
)
def test_only_the_primary_port_label_is_highlighted(node_type, port_label, keywords, is_primary):
    stylesheets = _port_label_stylesheets(node_type, port_label, keywords)

    # The primary port gets one stylesheet with the highlight colour; others keep the default style
    assert len(stylesheets) == int(is_primary)
    assert (PRIMARY_PORT_LABEL_COLOR in "".join(stylesheets)) is is_primary


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
