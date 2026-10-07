# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""Unit tests for the reader's automatic-layout helpers."""

import networkx as nx
import pytest

from ..power_train_network_reader import (
    _grid_positions,
    _hierarchy_layers,
    _layout_area,
    _parse_endpoint,
    _spread,
)


def _chain_graph(*names) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    for start, end in zip(names, names[1:], strict=False):
        graph.add_edge(start, end)
    return graph


# ---------------------------------------------------------------------------
# _hierarchy_layers
# ---------------------------------------------------------------------------


def test_layers_count_from_storage_when_propulsors_are_farthest():
    graph = _chain_graph("battery", "cable", "motor", "propeller")
    node_types = {
        "battery": "battery_pack",
        "cable": "dc_line",
        "motor": "pmsm",
        "propeller": "propeller",
    }

    layers = _hierarchy_layers(graph, node_types, {"propeller"})

    assert layers == {"propeller": 0, "motor": 1, "cable": 2, "battery": 3}


def test_layers_switch_to_propulsors_when_another_branch_is_farther():
    # The dangling load is as far from the battery as the propeller is
    graph = _chain_graph("battery", "bus", "motor", "propeller")
    graph.add_edges_from([("bus", "cable"), ("cable", "load")])
    node_types = {
        "battery": "battery_pack",
        "bus": "dc_bus",
        "motor": "pmsm",
        "propeller": "propeller",
        "cable": "dc_line",
        "load": "dc_load",
    }

    layers = _hierarchy_layers(graph, node_types, {"propeller"})

    assert layers["propeller"] == 0
    assert layers["motor"] == 1 and layers["bus"] == 2


def test_from_propulsor_flag_forces_propulsor_reference():
    graph = _chain_graph("battery", "motor", "propeller")
    node_types = {"battery": "battery_pack", "motor": "pmsm", "propeller": "propeller"}

    layers = _hierarchy_layers(graph, node_types, {"propeller"}, from_propulsor=True)

    assert layers == {"propeller": 0, "motor": 1, "battery": 2}


def test_disconnected_nodes_go_to_an_extra_bottom_layer():
    graph = _chain_graph("battery", "motor", "propeller")
    graph.add_node("lonely_cable")
    node_types = {
        "battery": "battery_pack",
        "motor": "pmsm",
        "propeller": "propeller",
        "lonely_cable": "dc_line",
    }

    layers = _hierarchy_layers(graph, node_types, {"propeller"})

    assert layers["lonely_cable"] == max(layers.values())
    assert layers["lonely_cable"] > layers["battery"]


@pytest.mark.parametrize("propulsor_names", [set(), {"propeller"}])
def test_layers_cover_every_node_without_storage(propulsor_names):
    graph = _chain_graph("motor", "propeller")
    node_types = {"motor": "pmsm", "propeller": "propeller"}

    layers = _hierarchy_layers(graph, node_types, propulsor_names)

    assert set(layers) == {"motor", "propeller"}


def test_layers_of_empty_graph():
    assert _hierarchy_layers(nx.DiGraph(), {}, set()) == {}


# ---------------------------------------------------------------------------
# _spread, _layout_area, _grid_positions
# ---------------------------------------------------------------------------


def test_spread_is_centred_and_capped_by_step_max():
    spread_values = _spread([0, 1], low=0.0, high=100.0, step_max=10.0)

    assert spread_values == [45.0, 55.0]


def test_spread_fills_range_when_step_max_allows():
    spread_values = _spread([0, 1, 2], low=0.0, high=100.0, step_max=1000.0)

    assert spread_values == [0.0, 50.0, 100.0]


def test_spread_of_a_single_value_is_the_centre():
    assert _spread([7], low=0.0, high=10.0, step_max=5.0) == [5.0]


def test_layout_area_collapses_on_tiny_window():
    tiny_area = {"x0": 0.0, "x1": 10.0, "y0": 0.0, "y1": 10.0, "per_px": 1.0}

    x_min, x_max, y_min, y_max = _layout_area(tiny_area)

    assert x_min == x_max == 5.0
    assert y_min == y_max == 5.0


@pytest.mark.parametrize("num_positions", [0, 1, 7, 30])
def test_grid_positions_are_unique_and_inside_area(num_positions):
    area = (0.0, 800.0, 0.0, 600.0)

    positions = _grid_positions(num_positions, area)

    assert len(positions) == num_positions
    assert len(set(positions)) == num_positions
    assert all(0.0 <= x <= 800.0 and 0.0 <= y <= 600.0 for x, y in positions)


# ---------------------------------------------------------------------------
# _parse_endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        ("motor_1", ("motor_1", "1")),
        (["bus_1", 2], ("bus_1", "2")),
        ("[bus_1, 3]", ("bus_1", "3")),
        (["bus_1"], None),
        (None, None),
    ],
)
def test_parse_endpoint(endpoint, expected):
    assert _parse_endpoint(endpoint) == expected
