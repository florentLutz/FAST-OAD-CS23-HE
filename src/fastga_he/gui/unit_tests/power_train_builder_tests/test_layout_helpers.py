# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""Unit tests for the automatic-layout helpers and the reader's connection helpers."""

import networkx as nx
import pytest

from fastga_he.gui.power_train_builder.power_train_network_layout import (
    _LAYOUT_STEP_MIN_ICONS_X,
    _LAYOUT_STEP_MIN_ICONS_Y,
    _centered_slots,
    _grid_positions,
    _hierarchy_layers,
    _layout_area,
    _order_rows_by_ports,
    _port_offset,
    _spread,
)
from fastga_he.gui.power_train_builder.power_train_network_reader import (
    _parse_endpoint,
    _type_unknown_ports_from_connections,
)

_ICON_SIZE = 50

_CHAIN_TYPES = {
    "battery": "battery_pack",
    "cable": "dc_line",
    "motor": "pmsm",
    "propeller": "propeller",
}


# ---------------------------------------------------------------------------
# _hierarchy_layers
# ---------------------------------------------------------------------------


def _layers(edges: list, node_types: dict, propulsors: set, from_propulsor: bool = False) -> dict:
    """Hierarchy layers of the graph with *node_types* as nodes and *edges* as edges."""
    graph = nx.DiGraph()
    graph.add_nodes_from(node_types)
    graph.add_edges_from(edges)
    return _hierarchy_layers(graph, node_types, propulsors, from_propulsor)


def test_hierarchy_layers_counted_from_storage_when_propulsors_are_farthest():
    edges = [("battery", "cable"), ("cable", "motor"), ("motor", "propeller")]

    assert _layers(edges, _CHAIN_TYPES, {"propeller"}) == {
        "propeller": 0,
        "motor": 1,
        "cable": 2,
        "battery": 3,
    }


def test_hierarchy_layers_switch_to_propulsors_when_another_branch_is_farther():
    # The dangling load is as far from the battery as the propeller is
    edges = [
        ("battery", "bus"),
        ("bus", "motor"),
        ("motor", "propeller"),
        ("bus", "cable"),
        ("cable", "load"),
    ]
    node_types = {
        "battery": "battery_pack",
        "bus": "dc_bus",
        "motor": "pmsm",
        "propeller": "propeller",
        "cable": "dc_line",
        "load": "dc_load",
    }

    assert _layers(edges, node_types, {"propeller"}) == {
        "propeller": 0,
        "motor": 1,
        "bus": 2,
        "battery": 3,
        "cable": 3,
        "load": 4,
    }


def test_hierarchy_layers_from_propulsor_flag_forces_the_propulsor_reference():
    edges = [("battery", "motor"), ("motor", "propeller")]
    node_types = {"battery": "battery_pack", "motor": "pmsm", "propeller": "propeller"}

    assert _layers(edges, node_types, {"propeller"}, from_propulsor=True) == {
        "propeller": 0,
        "motor": 1,
        "battery": 2,
    }


def test_hierarchy_layers_put_disconnected_nodes_in_an_extra_bottom_layer():
    edges = [("battery", "motor"), ("motor", "propeller")]
    node_types = {
        "battery": "battery_pack",
        "motor": "pmsm",
        "propeller": "propeller",
        "lonely_cable": "dc_line",
    }

    assert _layers(edges, node_types, {"propeller"}) == {
        "propeller": 0,
        "motor": 1,
        "battery": 2,
        "lonely_cable": 3,
    }


def test_hierarchy_layers_without_storage_are_counted_from_propulsors():
    node_types = {"motor": "pmsm", "propeller": "propeller"}

    assert _layers([("motor", "propeller")], node_types, {"propeller"}) == {
        "propeller": 0,
        "motor": 1,
    }


def test_hierarchy_layers_without_storage_or_propulsor_are_all_one_layer():
    node_types = {"motor": "pmsm", "propeller": "propeller"}

    assert _layers([("motor", "propeller")], node_types, set()) == {"motor": 0, "propeller": 0}


def test_hierarchy_layers_of_an_empty_graph():
    assert _layers([], {}, set()) == {}


# ---------------------------------------------------------------------------
# _spread, _layout_area, _grid_positions
# ---------------------------------------------------------------------------


def test_spread_is_centered_and_capped_by_step_max():
    assert _spread([0, 1], low=0.0, high=100.0, step_min=0.0, step_max=10.0) == [45.0, 55.0]


def test_spread_fills_the_range():
    assert _spread([0, 1, 2], low=0.0, high=100.0, step_min=0.0, step_max=1000.0) == [
        0.0,
        50.0,
        100.0,
    ]


def test_spread_puts_a_single_value_at_the_center():
    assert _spread([7], low=0.0, high=10.0, step_min=0.0, step_max=5.0) == [5.0]


def test_spread_step_min_wins_over_step_max():
    assert _spread([0, 1], low=0.0, high=100.0, step_min=30.0, step_max=10.0) == [35.0, 65.0]


def test_spread_too_wide_goes_past_both_ends():
    assert _spread([0, 1, 2], low=0.0, high=100.0, step_min=80.0, step_max=1000.0) == [
        -30.0,
        50.0,
        130.0,
    ]


def test_spread_too_wide_with_high_anchor_keeps_the_highest_value_at_high():
    assert _spread(
        [0, 1, 2], low=0.0, high=100.0, step_min=80.0, step_max=1000.0, anchor="high"
    ) == [-60.0, 20.0, 100.0]


def test_layout_area_collapses_on_tiny_window():
    tiny_area = {"x0": 0.0, "x1": 10.0, "y0": 0.0, "y1": 10.0, "per_px": 1.0}

    x_min, x_max, y_min, y_max = _layout_area(tiny_area)

    assert x_min == x_max == 5.0
    assert y_min == y_max == 5.0


def test_grid_positions_are_unique_and_inside_area():
    area = (0.0, 800.0, 0.0, 600.0)

    for num_positions in (0, 1, 7, 30):
        positions = _grid_positions(num_positions, area, _ICON_SIZE)

        assert len(positions) == num_positions
        assert len(set(positions)) == num_positions
        assert all(0.0 <= x <= 800.0 and 0.0 <= y <= 600.0 for x, y in positions)


def test_grid_positions_keep_the_minimum_spacing_and_grow_downward():
    area = (0.0, 800.0, 0.0, 600.0)

    positions = _grid_positions(100, area, _ICON_SIZE)

    xs = sorted({x for x, _ in positions})
    ys = sorted({y for _, y in positions}, reverse=True)
    # Too many to fit: the minimum steps are used, in both directions
    assert min(b - a for a, b in zip(xs, xs[1:])) == pytest.approx(
        _LAYOUT_STEP_MIN_ICONS_X * _ICON_SIZE
    )
    assert min(a - b for a, b in zip(ys, ys[1:])) == pytest.approx(
        _LAYOUT_STEP_MIN_ICONS_Y * _ICON_SIZE
    )
    # The top row stays at the top of the area; the rows that do not fit go below it
    assert ys[0] == 600.0
    assert ys[-1] < 0.0


# ---------------------------------------------------------------------------
# _centered_slots, _port_offset, _order_rows_by_ports
# ---------------------------------------------------------------------------


def test_centered_slots():
    assert _centered_slots(0) == []
    assert _centered_slots(1) == [0.0]
    assert _centered_slots(2) == [-0.5, 0.5]
    assert _centered_slots(3) == [-1.0, 0.0, 1.0]


def test_port_offset():
    # A single port sits on the component
    assert _port_offset("1", 1, "source") == 0.0
    # Source ports go from left to right
    assert _port_offset("1", 3, "source") == -0.25
    assert _port_offset("3", 3, "source") == 0.25
    # Target ports go from right to left
    assert _port_offset("1", 3, "target") == 0.25
    assert _port_offset("3", 3, "target") == -0.25


def _two_row_design(n_sources: dict, n_targets: dict, edges: list) -> tuple[dict, dict]:
    """
    Node and edge columns of a small design, as the reader builds them.

    :param n_sources: Source port count of every component, by name.
    :param n_targets: Target port count of every component, by name.
    :param edges: ``((name, kind, label), (name, kind, label))`` per edge.
    :return: ``(nodes_data, edges_data)``.
    """
    names = list(n_sources)
    nodes_data = {
        "name": names,
        "n_sources": [n_sources[name] for name in names],
        "n_targets": [n_targets[name] for name in names],
    }
    edges_data = {
        f"{end}_{column}": []
        for end in ("starting", "ending")
        for column in ("node_index", "port_kind", "port_label")
    }
    for edge in edges:
        for end, (name, kind, label) in zip(("starting", "ending"), edge):
            edges_data[f"{end}_node_index"].append(names.index(name))
            edges_data[f"{end}_port_kind"].append(kind)
            edges_data[f"{end}_port_label"].append(label)
    return nodes_data, edges_data


def test_order_rows_by_ports_leaves_a_layout_without_crossings_unchanged():
    # Source port 1 of "gear" (on the left) goes to "left", port 2 to "right"
    nodes_data, edges_data = _two_row_design(
        n_sources={"gear": 2, "left": 0, "right": 0},
        n_targets={"gear": 0, "left": 1, "right": 1},
        edges=[
            (("gear", "source", "1"), ("left", "target", "1")),
            (("gear", "source", "2"), ("right", "target", "1")),
        ],
    )
    raw_positions = {"gear": (0.0, 1), "left": (-0.5, 0), "right": (0.5, 0)}

    assert _order_rows_by_ports(raw_positions, nodes_data, edges_data) is raw_positions


def test_order_rows_by_ports_swaps_components_whose_edges_cross_at_source_ports():
    # Source port 1 of "gear" is on its left but leads to the component placed on the right
    nodes_data, edges_data = _two_row_design(
        n_sources={"gear": 2, "a": 0, "b": 0},
        n_targets={"gear": 0, "a": 1, "b": 1},
        edges=[
            (("gear", "source", "1"), ("b", "target", "1")),
            (("gear", "source", "2"), ("a", "target", "1")),
        ],
    )
    raw_positions = {"gear": (0.0, 1), "a": (-0.5, 0), "b": (0.5, 0)}

    assert _order_rows_by_ports(raw_positions, nodes_data, edges_data) == {
        "gear": (0.0, 1),
        "a": (0.5, 0),
        "b": (-0.5, 0),
    }


def test_order_rows_by_ports_swaps_components_whose_edges_cross_at_target_ports():
    # Target port 1 of "bus" is on its right but comes from the component placed on the left
    nodes_data, edges_data = _two_row_design(
        n_sources={"a": 1, "b": 1, "bus": 0},
        n_targets={"a": 0, "b": 0, "bus": 2},
        edges=[
            (("a", "source", "1"), ("bus", "target", "1")),
            (("b", "source", "1"), ("bus", "target", "2")),
        ],
    )
    raw_positions = {"a": (-0.5, 1), "b": (0.5, 1), "bus": (0.0, 0)}

    assert _order_rows_by_ports(raw_positions, nodes_data, edges_data) == {
        "a": (0.5, 1),
        "b": (-0.5, 1),
        "bus": (0.0, 0),
    }


# ---------------------------------------------------------------------------
# _type_unknown_ports_from_connections
# ---------------------------------------------------------------------------


def _unknown_components(port_types: list) -> dict:
    """
    Node columns of unknown components with the given port types.

    :param port_types: ``(source port type, target port type)`` per component.
    :return: ``nodes_data`` with the columns the port typing reads.
    """
    return {
        "name": [f"unknown_{index}" for index in range(len(port_types))],
        "icon_type": ["unknown"] * len(port_types),
        "source_port_type": [source_type for source_type, _ in port_types],
        "target_port_type": [target_type for _, target_type in port_types],
    }


def test_type_unknown_ports_spreads_through_a_side_with_several_connections():
    # 0 has a defined target side. 1's source side is connected to it and to
    # 2's target side, so both undefined sides get its energy type
    nodes_data = _unknown_components(
        [("undefined", "electricity"), ("undefined", "undefined"), ("undefined", "undefined")]
    )
    endpoints = [(1, "1", 0, "1"), (1, "1", 2, "1")]

    _type_unknown_ports_from_connections(nodes_data, endpoints)

    assert nodes_data["source_port_type"] == ["undefined", "electricity", "undefined"]
    assert nodes_data["target_port_type"] == ["electricity", "undefined", "electricity"]


def test_type_unknown_ports_leaves_sides_connected_to_nothing_defined():
    nodes_data = _unknown_components([("undefined", "undefined"), ("undefined", "undefined")])
    endpoints = [(0, "1", 1, "1")]

    _type_unknown_ports_from_connections(nodes_data, endpoints)

    assert nodes_data["source_port_type"] == ["undefined", "undefined"]
    assert nodes_data["target_port_type"] == ["undefined", "undefined"]


def test_type_unknown_ports_uses_the_first_connection_when_types_conflict():
    # 2's source side is connected to an electrical port first, then a mechanical one
    nodes_data = _unknown_components(
        [("undefined", "electricity"), ("undefined", "mechanical"), ("undefined", "undefined")]
    )
    endpoints = [(2, "1", 0, "1"), (2, "1", 1, "1")]

    _type_unknown_ports_from_connections(nodes_data, endpoints)

    assert nodes_data["source_port_type"][2] == "electricity"


# ---------------------------------------------------------------------------
# _parse_endpoint
# ---------------------------------------------------------------------------


def test_parse_endpoint():
    # A plain name is port 1
    assert _parse_endpoint("motor_1") == ("motor_1", "1")
    # [name, N] as a YAML list, or as a quoted string
    assert _parse_endpoint(["bus_1", 2]) == ("bus_1", "2")
    assert _parse_endpoint("[bus_1, 3]") == ("bus_1", "3")


def test_parse_endpoint_rejects_malformed_endpoints():
    assert _parse_endpoint(["bus_1"]) is None
    assert _parse_endpoint(None) is None
