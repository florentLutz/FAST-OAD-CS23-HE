# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""Unit tests for the reader's automatic-layout helpers."""

import networkx as nx
import pytest

from fastga_he.gui.power_train_builder.power_train_network_reader import (
    _grid_positions,
    _hierarchy_layers,
    _layout_area,
    _parse_endpoint,
    _spread,
)

_CHAIN_TYPES = {
    "battery": "battery_pack",
    "cable": "dc_line",
    "motor": "pmsm",
    "propeller": "propeller",
}


# ---------------------------------------------------------------------------
# _hierarchy_layers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edges, node_types, propulsors, from_propulsor, expected",
    [
        pytest.param(
            [("battery", "cable"), ("cable", "motor"), ("motor", "propeller")],
            _CHAIN_TYPES,
            {"propeller"},
            False,
            {"propeller": 0, "motor": 1, "cable": 2, "battery": 3},
            id="counted_from_storage_when_propulsors_are_farthest",
        ),
        pytest.param(
            # The dangling load is as far from the battery as the propeller is
            [
                ("battery", "bus"),
                ("bus", "motor"),
                ("motor", "propeller"),
                ("bus", "cable"),
                ("cable", "load"),
            ],
            {
                "battery": "battery_pack",
                "bus": "dc_bus",
                "motor": "pmsm",
                "propeller": "propeller",
                "cable": "dc_line",
                "load": "dc_load",
            },
            {"propeller"},
            False,
            {"propeller": 0, "motor": 1, "bus": 2, "battery": 3, "cable": 3, "load": 4},
            id="switches_to_propulsors_when_another_branch_is_farther",
        ),
        pytest.param(
            [("battery", "motor"), ("motor", "propeller")],
            {"battery": "battery_pack", "motor": "pmsm", "propeller": "propeller"},
            {"propeller"},
            True,
            {"propeller": 0, "motor": 1, "battery": 2},
            id="from_propulsor_flag_forces_the_propulsor_reference",
        ),
        pytest.param(
            [("battery", "motor"), ("motor", "propeller")],
            {
                "battery": "battery_pack",
                "motor": "pmsm",
                "propeller": "propeller",
                "lonely_cable": "dc_line",
            },
            {"propeller"},
            False,
            {"propeller": 0, "motor": 1, "battery": 2, "lonely_cable": 3},
            id="disconnected_nodes_go_to_an_extra_bottom_layer",
        ),
        pytest.param(
            [("motor", "propeller")],
            {"motor": "pmsm", "propeller": "propeller"},
            {"propeller"},
            False,
            {"propeller": 0, "motor": 1},
            id="no_storage_counted_from_propulsors",
        ),
        pytest.param(
            [("motor", "propeller")],
            {"motor": "pmsm", "propeller": "propeller"},
            set(),
            False,
            {"motor": 0, "propeller": 0},
            id="no_storage_no_propulsor_all_in_one_layer",
        ),
        pytest.param([], {}, set(), False, {}, id="empty_graph"),
    ],
)
def test_hierarchy_layers(edges, node_types, propulsors, from_propulsor, expected):
    graph = nx.DiGraph()
    graph.add_nodes_from(node_types)
    graph.add_edges_from(edges)

    assert _hierarchy_layers(graph, node_types, propulsors, from_propulsor) == expected


# ---------------------------------------------------------------------------
# _spread, _layout_area, _grid_positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, low, high, step_max, expected",
    [
        pytest.param([0, 1], 0.0, 100.0, 10.0, [45.0, 55.0], id="centred_and_capped_by_step_max"),
        pytest.param([0, 1, 2], 0.0, 100.0, 1000.0, [0.0, 50.0, 100.0], id="fills_the_range"),
        pytest.param([7], 0.0, 10.0, 5.0, [5.0], id="single_value_at_the_centre"),
    ],
)
def test_spread(values, low, high, step_max, expected):
    assert _spread(values, low=low, high=high, step_max=step_max) == expected


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
