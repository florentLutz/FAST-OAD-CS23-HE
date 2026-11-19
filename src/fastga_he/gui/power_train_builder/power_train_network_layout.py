# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Automatic canvas positions of the components of a loaded powertrain.

Used by :class:`~.power_train_network_reader.PowerTrainYAMLReader` for the
components of a file that have no ``canvas_layout`` entry (a file saved as
"final version", a hand-written YAML, or one produced before that section
existed). :func:`place_nodes_without_layout` is the entry point:

1. :func:`_hierarchy_layers` puts every component in a layer, with the same
   rules as the network viewer (energy storage at the bottom, propulsors at
   the top).
2. :class:`~.layout_generation.HierarchicalLayout` orders each layer (Tutte
   crossing reduction), then :func:`_order_rows_by_ports` reorders the rows
   so that fewer edges cross once ports are taken into account.
3. :func:`_spread_x` / :func:`_spread_y` map the abstract layout onto the
   canvas area of the user's window (:func:`_layout_area`).

A simple grid (:func:`_grid_positions`) is kept as a last-resort fallback if
the hierarchical layout cannot be computed.
"""

import logging
from typing import NamedTuple

import networkx as nx
import numpy as np

from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from ..layout_generation import HierarchicalLayout
from .power_train_builder_metadata import component_type_classes
from .power_train_builder_state import (
    CANVAS_FIT_NODE_MARGIN,
    CANVAS_WORLD_X_RANGE,
    CANVAS_WORLD_Y_RANGE,
)
from .power_train_network_writer import KEY_CANVAS_LAYOUT

_LOGGER = logging.getLogger(__name__)


# Automatic layouts are spread over the canvas area of the user's current
# browser window (BuilderState.canvas_area, reported by the browser; with the
# window maximized this is the maximized canvas), inset by these margins in
# screen pixels. The top margin clears the Undo / Redo buttons (top-left) and
# the legend (top-right). Margins never go below CANVAS_FIT_NODE_MARGIN (data
# units), so the loaded design fits the current view without re-zooming
_LAYOUT_MARGIN_PX = {"left": 50.0, "right": 50.0, "top": 110.0, "bottom": 50.0}

# Largest gap between two neighbouring components, as a fraction of the layout
# area's width / height. Keeps small designs (e.g. two layers) compact and
# centred instead of pushed to opposite edges of a large screen
_LAYOUT_STEP_MAX_FRACTION_X = 0.25
_LAYOUT_STEP_MAX_FRACTION_Y = 0.30

# Smallest gap between two neighboring components, in icon sizes. A row needs
# room for the icon, its ports above and below, and its name; a column for its
# name and its ports. If the design needs more room than the window, it grows
# past the window instead of overlapping (the view then zooms out to show it all)
_LAYOUT_STEP_MIN_ICONS_X = 2.2
_LAYOUT_STEP_MIN_ICONS_Y = 2.6

# Port ordering: the ports on one side of a component are spread over this
# width, in slots (from -1/4 to +1/4). It is less than 1, so the port order
# only decides between components attached to the same neighbor
_PORT_ORDER_SPREAD = 0.5

# Port ordering: sort the rows top-down, bottom-up, then top-down again. The
# bottom-up pass lets the port order of a component low in the design (e.g. a
# bus with many inputs) reach the rows above it; the last pass then settles the
# rows below. True = top-down
_PORT_ORDER_PASSES = (True, False, True)

# Area assumed when the browser has not reported one (headless use / tests):
# the 800 x 950 design world at 1 data unit per pixel
DEFAULT_CANVAS_AREA = dict(
    x0=float(CANVAS_WORLD_X_RANGE[0]),
    x1=float(CANVAS_WORLD_X_RANGE[1]),
    y0=float(CANVAS_WORLD_Y_RANGE[0]),
    y1=float(CANVAS_WORLD_Y_RANGE[1]),
    per_px=1.0,
)

# Energy-storage component types or type classes used as the hierarchy
# reference, matching the viewer's get_component_distance(["tank", "battery_pack"])
_ENERGY_STORAGE_KEYWORDS = ("tank", "battery_pack")


# Default of the viewer's from_propulsor argument: when True, layers are always
# counted from the propulsors instead of first trying the energy storage
_LAYOUT_FROM_PROPULSOR = False


def place_nodes_without_layout(
    nodes_data: dict,
    edges_data: dict,
    nodes_without_layout: list,
    propulsor_names: set,
    canvas_area: dict,
    icon_size: float,
) -> None:
    """
    Position the nodes that have no ``canvas_layout`` entry, in place.

    Uses the network viewer's :class:`HierarchicalLayout` (top-to-bottom,
    Tutte crossing reduction) on the whole graph, spread over the canvas
    area of the user's window (see :func:`_layout_area`), then keeps the
    generated position only for the nodes that need one. Falls back to a
    grid over the same area if the layout cannot be computed.

    :param nodes_data: Node columns built by the reader; ``x`` / ``y`` edited in place.
    :param edges_data: Edge columns built by the reader.
    :param nodes_without_layout: Indices of the nodes to position.
    :param propulsor_names: Set of component names that are propulsors.
    :param canvas_area: ``{"x0", "x1", "y0", "y1", "per_px"}`` of the user's
        window (:data:`DEFAULT_CANVAS_AREA` when the browser has not reported one).
    :param icon_size: Icon size, in canvas data units; sets the minimum steps.
    """
    layout_area = _layout_area(canvas_area)
    try:
        positions = _hierarchical_canvas_positions(
            nodes_data, edges_data, propulsor_names, layout_area, icon_size
        )
        method = "hierarchical layout"
    except Exception:  # pragma: no cover - defensive: never block a load on layout
        _LOGGER.exception("Hierarchical layout failed; placing components on a grid.")
        positions = None

    if positions is None:
        grid = _grid_positions(len(nodes_without_layout), layout_area, icon_size)
        positions = {
            nodes_data["name"][node_index]: position
            for node_index, position in zip(nodes_without_layout, grid, strict=True)
        }
        method = "grid"

    for node_index in nodes_without_layout:
        nodes_data["x"][node_index], nodes_data["y"][node_index] = positions[
            nodes_data["name"][node_index]
        ]

    _LOGGER.info(
        "%d component(s) had no %s entry; positioned with the %s.",
        len(nodes_without_layout),
        KEY_CANVAS_LAYOUT,
        method,
    )


def _hierarchy_layers(
    graph: nx.DiGraph,
    node_types: dict,
    propulsor_names: set,
    from_propulsor: bool = _LAYOUT_FROM_PROPULSOR,
) -> dict:
    """
    Assign every node a layer index (0 = top of the canvas).

    Follows the viewer's ``GraphBuilder._get_hierarchy_layers``. Layers are first
    computed from the graph distance to energy storage (tanks / batteries),
    flipped so storage ends up at the bottom: ``layer = max_distance - distance``.
    This is kept only if every node farthest from storage is a propulsor, so the
    propulsors form the top layer. Otherwise (e.g. a branch ending on a
    component that is not a propulsor), or when *from_propulsor* is set, the
    layers are the graph distance from the propulsors instead, which puts every
    propulsor in the top layer.

    Unlike the viewer, a work-in-progress design may have no storage, no
    propulsor, or disconnected parts. If neither reference reaches every node,
    the one reaching more nodes is used and the remaining nodes go to an extra
    bottom layer, so the layout always covers the whole graph.

    :param graph: NetworkX DiGraph of the powertrain.
    :param node_types: Dictionary mapping component names to their component type.
    :param propulsor_names: Set of component names that are propulsors.
    :param from_propulsor: Count the layers from the propulsors straight away.
    :return: Dictionary mapping component names to their layer index.
    """
    # Distances ignore the edge direction, as in get_component_distance
    undirected_graph = graph.to_undirected()
    num_nodes = graph.number_of_nodes()

    layers_from_energy, propulsors_not_on_top = _layers_from_energy_storage(
        undirected_graph, node_types, propulsor_names
    )
    # The viewer switches to the propulsor reference as soon as a node at the
    # maximum distance from storage is not a propulsor
    from_propulsor = from_propulsor or propulsors_not_on_top

    layers_from_propulsor = {}
    if propulsor_names and (from_propulsor or len(layers_from_energy) < num_nodes):
        layers_from_propulsor = dict(
            nx.multi_source_dijkstra_path_length(undirected_graph, propulsor_names)
        )

    layers = _chosen_layers(layers_from_energy, layers_from_propulsor, num_nodes, from_propulsor)
    return _with_unreached_nodes_at_the_bottom(graph, layers)


def _layers_from_energy_storage(
    undirected_graph: nx.Graph, node_types: dict, propulsor_names: set
) -> tuple[dict, bool]:
    """
    Layers counted from the energy storage, which ends up at the bottom.

    ``layer = max_distance - distance``, where *distance* is the graph distance
    to the nearest tank / battery.

    :param undirected_graph: The powertrain graph, edge direction ignored.
    :param node_types: Dictionary mapping component names to their component type.
    :param propulsor_names: Set of component names that are propulsors.
    :return: ``(layers, propulsors_not_on_top)``: the layer of every node the
        storage reaches (empty without storage), and whether a node farthest
        from storage is not a propulsor, in which case the viewer counts the
        layers from the propulsors instead.
    """
    energy_storage_types = _energy_storage_component_types()
    energy_storage_nodes = [
        node_name
        for node_name, node_type in node_types.items()
        if node_type in energy_storage_types
    ]
    if not energy_storage_nodes:
        return {}, False

    distance_from_energy = nx.multi_source_dijkstra_path_length(
        undirected_graph, energy_storage_nodes
    )
    max_distance = max(distance_from_energy.values())
    layers = {
        node_name: max_distance - distance for node_name, distance in distance_from_energy.items()
    }
    propulsors_not_on_top = any(
        distance == max_distance and node_name not in propulsor_names
        for node_name, distance in distance_from_energy.items()
    )
    return layers, propulsors_not_on_top


def _energy_storage_component_types() -> set:
    """
    Component types counted as energy storage when computing hierarchy layers.

    A component type is energy storage if the type itself, or one of its type
    classes, is in :data:`_ENERGY_STORAGE_KEYWORDS`: the same matching rule as
    the viewer's ``FASTGAHEPowerTrainConfigurator.get_component_distance``.

    :return: Set of component types.
    """
    energy_storage_types = set()
    for known_component in KNOWN_COMPONENTS:
        component_type = known_component["components_type"]
        type_classes = component_type_classes(known_component)
        if component_type in _ENERGY_STORAGE_KEYWORDS or any(
            keyword in type_classes for keyword in _ENERGY_STORAGE_KEYWORDS
        ):
            energy_storage_types.add(component_type)
    return energy_storage_types


def _chosen_layers(
    layers_from_energy: dict, layers_from_propulsor: dict, num_nodes: int, from_propulsor: bool
) -> dict:
    """
    Pick the layering reference: storage, propulsors, or the one reaching more nodes.

    :param layers_from_energy: Layers counted from the energy storage.
    :param layers_from_propulsor: Layers counted from the propulsors.
    :param num_nodes: Number of nodes in the graph.
    :param from_propulsor: Whether the propulsor reference is required.
    :return: The chosen layers (may miss nodes neither reference reaches).
    """
    if len(layers_from_energy) == num_nodes and not from_propulsor:
        return layers_from_energy
    if len(layers_from_propulsor) == num_nodes:
        return layers_from_propulsor
    # Neither reference covers the whole graph: keep the one that covers more
    if len(layers_from_energy) >= len(layers_from_propulsor):
        return layers_from_energy
    return layers_from_propulsor


def _with_unreached_nodes_at_the_bottom(graph: nx.DiGraph, layers: dict) -> dict:
    """
    Complete *layers* with the nodes it misses, all in one extra bottom layer.

    :param graph: The powertrain graph.
    :param layers: Layer of the nodes the chosen reference reaches.
    :return: Integer layer of every node of *graph*: the reached nodes first, in
        the order of *layers*, then the others.
    """
    extra_layer = max(layers.values()) + 1 if layers else 0
    completed_layers = {node_name: int(layer) for node_name, layer in layers.items()}
    for node_name in graph.nodes:
        completed_layers.setdefault(node_name, extra_layer)
    return completed_layers


def _layout_area(canvas_area: dict) -> tuple[float, float, float, float]:
    """
    Area available to generated layouts: the window's canvas area inset by
    :data:`_LAYOUT_MARGIN_PX` (converted to data units with the area's
    ``per_px``, and never less than ``CANVAS_FIT_NODE_MARGIN``).

    :param canvas_area: ``{"x0", "x1", "y0", "y1", "per_px"}`` in data units.
    :return: ``(x_min, x_max, y_min, y_max)``.
    """
    per_px = canvas_area["per_px"]

    def inset(side: str) -> float:
        """Margin on *side* in data units, never below ``CANVAS_FIT_NODE_MARGIN``."""
        return max(CANVAS_FIT_NODE_MARGIN, _LAYOUT_MARGIN_PX[side] * per_px)

    # Rounded inwards to the 0.1 data-unit precision of saved positions
    x_min = float(np.ceil((canvas_area["x0"] + inset("left")) * 10) / 10)
    x_max = float(np.floor((canvas_area["x1"] - inset("right")) * 10) / 10)
    y_min = float(np.ceil((canvas_area["y0"] + inset("bottom")) * 10) / 10)
    y_max = float(np.floor((canvas_area["y1"] - inset("top")) * 10) / 10)
    # Degenerate (tiny window): collapse onto the centre instead of inverting
    if x_max < x_min:
        x_min = x_max = (canvas_area["x0"] + canvas_area["x1"]) / 2
    if y_max < y_min:
        y_min = y_max = (canvas_area["y0"] + canvas_area["y1"]) / 2
    return x_min, x_max, y_min, y_max


def _spread(
    values: list,
    low: float,
    high: float,
    step_min: float,
    step_max: float,
    anchor: str = "center",
) -> list:
    """
    Map evenly spaced layout coordinates onto the range ``[low, high]``.

    The step between neighbors is as large as the range allows, but stays
    between ``step_min`` and ``step_max``. The result is centered in the range.
    If even ``step_min`` does not fit, the result is larger than the range:
    it overflows on both sides, or only below *low* with ``anchor="high"``.

    :param values: Layout coordinates, one unit apart.
    :param low: Lower end of the range.
    :param high: Upper end of the range.
    :param step_min: Smallest step between neighbors.
    :param step_max: Largest step between neighbors.
    :param anchor: ``"center"``, or ``"high"`` to keep the highest value at
        *high* when the result does not fit.
    :return: The mapped coordinates, in the order of *values*.
    """
    lowest, highest = min(values), max(values)
    span = highest - lowest
    room = high - low

    # 1. Step: fill the range, within [step_min, step_max]. The minimum wins
    # over the maximum when they conflict (a narrow window), so components
    # never overlap; np.clip would do the opposite
    step = max(step_min, min(step_max, room / span)) if span else 0.0

    # 2. Reference point: the center, or the top when it does not fit
    if anchor == "high" and span * step > room:
        origin, origin_value = high, highest
    else:
        origin, origin_value = (low + high) / 2, (lowest + highest) / 2

    # 3. Every value at its distance from the reference point
    return [round(origin + (float(value) - origin_value) * step, 1) for value in values]


def _spread_x(values: list, layout_area: tuple, icon_size: float) -> list:
    """
    Map layout columns onto the width of *layout_area*, centered (see :func:`_spread`).

    :param values: Column coordinates of the layout.
    :param layout_area: ``(x_min, x_max, y_min, y_max)``.
    :param icon_size: Icon size, in canvas data units; sets the smallest step.
    :return: The x coordinates. They overflow on both sides if the columns do not fit.
    """
    x_min, x_max, _, _ = layout_area
    return _spread(
        values,
        x_min,
        x_max,
        _LAYOUT_STEP_MIN_ICONS_X * icon_size,
        _LAYOUT_STEP_MAX_FRACTION_X * (x_max - x_min),
    )


def _spread_y(values: list, layout_area: tuple, icon_size: float) -> list:
    """
    Map layout rows onto the height of *layout_area* (see :func:`_spread`).

    :param values: Row coordinates of the layout; the top row has the largest.
    :param layout_area: ``(x_min, x_max, y_min, y_max)``.
    :param icon_size: Icon size, in canvas data units; sets the smallest step.
    :return: The y coordinates. If the rows do not fit, the top row stays at
        the top of the area and the other rows go below it.
    """
    _, _, y_min, y_max = layout_area
    return _spread(
        values,
        y_min,
        y_max,
        _LAYOUT_STEP_MIN_ICONS_Y * icon_size,
        _LAYOUT_STEP_MAX_FRACTION_Y * (y_max - y_min),
        anchor="high",
    )


class _EdgeEnd(NamedTuple):
    """One end of an edge, for the port ordering."""

    # Component at that end
    component: str
    # Where the port is along its side of the component (see _port_offset)
    port_offset: float


def _order_rows_by_ports(raw_positions: dict, nodes_data: dict, edges_data: dict) -> dict:
    """
    Reorder the components within their rows so that fewer edges cross.

    :class:`HierarchicalLayout` ignores ports, so two components attached to
    ports 1 and 2 of the same neighbor can end up the wrong way round, and
    their edges cross.

    * A layout without crossings is returned unchanged.
    * Otherwise, each row is sorted by where its components connect in the
      neighboring row, ports included (see :func:`_sort_rows`).
    * The new order is kept only if it has fewer crossings, so the result is
      never worse than the input.

    Components stay in their row and the row keeps its slots; only the order
    changes.

    :param raw_positions: ``{component: (slot, layout_row)}`` from
        :meth:`HierarchicalLayout.generate_networkx_layout`; the top row has
        the largest *layout_row*.
    :param nodes_data: Node columns (names and port counts).
    :param edges_data: Edge columns (end components, port kinds and labels).
    :return: ``{component: (slot, layout_row)}`` with the new slots.
    """
    components_by_row, component_row_index, component_slot = _split_into_rows(raw_positions)
    edge_end_pairs = _edge_end_pairs(nodes_data, edges_data)
    initial_crossings = _row_crossings(component_row_index, component_slot, edge_end_pairs)
    if initial_crossings == 0:
        return raw_positions

    component_far_ends = _component_far_ends(edge_end_pairs)
    fewest_crossings, best_component_slot = initial_crossings, dict(component_slot)
    for downward in _PORT_ORDER_PASSES:
        _sort_rows(
            components_by_row,
            component_row_index,
            component_slot,
            component_far_ends,
            downward,
        )
        crossings = _row_crossings(component_row_index, component_slot, edge_end_pairs)
        if crossings < fewest_crossings:
            fewest_crossings, best_component_slot = crossings, dict(component_slot)

    _LOGGER.debug(
        "Port ordering: %d edge crossing(s) between rows, %d before.",
        fewest_crossings,
        initial_crossings,
    )
    return {
        component: (best_component_slot[component], layout_row)
        for component, (_, layout_row) in raw_positions.items()
    }


def _split_into_rows(raw_positions: dict) -> tuple[list, dict, dict]:
    """
    Group the components of a layout by row.

    :param raw_positions: ``{component: (slot, layout_row)}``; the top row has
        the largest *layout_row*.
    :return: ``(components_by_row, component_row_index, component_slot)``:

        * the components of each row, top row first, each row from left to right;
        * the row index of each component (0 = top row);
        * the slot of each component.
    """
    layout_row_slots_and_names: dict = {}
    for component, (slot, layout_row) in raw_positions.items():
        layout_row_slots_and_names.setdefault(layout_row, []).append((slot, component))
    components_by_row = [
        [component for _, component in sorted(layout_row_slots_and_names[layout_row])]
        for layout_row in sorted(layout_row_slots_and_names, reverse=True)
    ]
    component_row_index = {
        component: row_index
        for row_index, row_components in enumerate(components_by_row)
        for component in row_components
    }
    component_slot = {component: slot for component, (slot, _) in raw_positions.items()}
    return components_by_row, component_row_index, component_slot


def _edge_end_pairs(nodes_data: dict, edges_data: dict) -> list[tuple[_EdgeEnd, _EdgeEnd]]:
    """
    Both ends of each edge, for the port ordering.

    :param nodes_data: Node columns (names and port counts).
    :param edges_data: Edge columns (end components, port kinds and labels).
    :return: One ``(starting end, ending end)`` pair per edge.
    """
    return [
        (
            _edge_end(nodes_data, edges_data, edge_index, "starting"),
            _edge_end(nodes_data, edges_data, edge_index, "ending"),
        )
        for edge_index in range(len(edges_data["starting_node_index"]))
    ]


def _edge_end(nodes_data: dict, edges_data: dict, edge_index: int, end: str) -> _EdgeEnd:
    """
    One end of an edge: its component, and where its port is on that component.

    :param nodes_data: Node columns (names and port counts).
    :param edges_data: Edge columns (end components, port kinds and labels).
    :param edge_index: Zero-based index of the edge.
    :param end: ``"starting"`` or ``"ending"``.
    :return: The edge end.
    """
    node_index = edges_data[f"{end}_node_index"][edge_index]
    port_kind = edges_data[f"{end}_port_kind"][edge_index]
    port_count_column = "n_sources" if port_kind == "source" else "n_targets"
    return _EdgeEnd(
        component=nodes_data["name"][node_index],
        port_offset=_port_offset(
            edges_data[f"{end}_port_label"][edge_index],
            nodes_data[port_count_column][node_index],
            port_kind,
        ),
    )


def _port_offset(port_label, port_count, port_kind: str) -> float:
    """
    Where a port is along its side of the component, relative to the component, in slots.

    The ports are in the order ``PortEdgeMixin.compute_ports`` draws them:
    source ports (below the icon) from left to right, target ports (above
    the icon) from right to left. They are spread evenly over
    :data:`_PORT_ORDER_SPREAD`, centered on the component.

    :param port_label: Port label (``"1"``, ``"2"``, ...).
    :param port_count: Number of ports on that side.
    :param port_kind: ``"source"`` or ``"target"``.
    :return: The offset: negative on the left of the component, 0 for a single port.
    """
    port_count = int(port_count or 1)
    if port_count <= 1:
        return 0.0
    offset = ((int(port_label) - 1) / (port_count - 1) - 0.5) * _PORT_ORDER_SPREAD
    return offset if port_kind == "source" else -offset


def _component_far_ends(edge_end_pairs: list) -> dict:
    """
    For each component, the other end of each of its edges.

    :param edge_end_pairs: From :func:`_edge_end_pairs`.
    :return: ``{component: [far end, ...]}``, one far end per edge.
    """
    component_far_ends: dict = {}
    for starting_end, ending_end in edge_end_pairs:
        component_far_ends.setdefault(starting_end.component, []).append(ending_end)
        component_far_ends.setdefault(ending_end.component, []).append(starting_end)
    return component_far_ends


def _centered_slots(count: int) -> list[float]:
    """
    The slots of a row of *count* components: one unit apart, centered on 0.

    These are the slots :class:`HierarchicalLayout` uses for each layer.

    :param count: Number of components in the row.
    :return: ``[-(count - 1) / 2, ..., (count - 1) / 2]``.
    """
    return [-(count - 1) / 2 + slot_index for slot_index in range(count)]


def _sort_rows(
    components_by_row: list,
    component_row_index: dict,
    component_slot: dict,
    component_far_ends: dict,
    downward: bool,
) -> None:
    """
    Sort each row by where its components connect in the neighboring row, one row after another.

    The rows are sorted in order, top-down or bottom-up. The neighboring row is
    the row sorted just before: the one above when going top-down, the one
    below when going bottom-up. Each component is placed at the average
    position of the ports it connects to there (see
    :func:`_mean_far_port_position`). Ties keep the current order.

    *components_by_row* and *component_slot* are updated in place.

    :param components_by_row: Components of each row, top row first, each row
        from left to right.
    :param component_row_index: Row index of each component.
    :param component_slot: Current slot of each component.
    :param component_far_ends: From :func:`_component_far_ends`.
    :param downward: ``True`` to sort the rows top-down, ``False`` bottom-up.
    """
    row_count = len(components_by_row)
    if downward:
        # The top row stays as it is; each row below it is sorted against the row above
        row_indices = range(1, row_count)
        neighbor_row_offset = -1
    else:
        # The bottom row stays as it is; each row above it is sorted against the row below
        row_indices = reversed(range(row_count - 1))
        neighbor_row_offset = +1

    for row_index in row_indices:
        neighbor_row_index = row_index + neighbor_row_offset
        row_components = components_by_row[row_index]

        # Sort key of each component: where it connects in the neighboring row,
        # then its current slot, so that ties keep the current order
        component_sort_key = {}
        for component in row_components:
            far_port_position = _mean_far_port_position(
                component,
                neighbor_row_index,
                component_row_index,
                component_slot,
                component_far_ends,
            )
            component_sort_key[component] = (far_port_position, component_slot[component])
        sorted_components = sorted(row_components, key=lambda name: component_sort_key[name])

        # The row keeps its slots; its components take them in their new order
        row_slots = _centered_slots(len(sorted_components))
        for component, slot in zip(sorted_components, row_slots):
            component_slot[component] = slot
        components_by_row[row_index] = sorted_components


def _mean_far_port_position(
    component: str,
    neighbor_row_index: int,
    component_row_index: dict,
    component_slot: dict,
    component_far_ends: dict,
) -> float:
    """
    Average position of the ports a component connects to in another row.

    :param component: Component name.
    :param neighbor_row_index: Index of the other row.
    :param component_row_index: Row index of each component.
    :param component_slot: Current slot of each component.
    :param component_far_ends: From :func:`_component_far_ends`.
    :return: The average position, or the component's own slot if it has no
        edge to that row.
    """
    far_port_positions = [
        _port_position(far_end, component_slot)
        for far_end in component_far_ends.get(component, [])
        if component_row_index[far_end.component] == neighbor_row_index
    ]
    if not far_port_positions:
        return component_slot[component]
    return sum(far_port_positions) / len(far_port_positions)


def _port_position(edge_end: _EdgeEnd, component_slot: dict) -> float:
    """
    Position of the port at one end of an edge: the component's slot plus the port offset.

    :param edge_end: The edge end.
    :param component_slot: Current slot of each component.
    :return: The position, in slots.
    """
    return component_slot[edge_end.component] + edge_end.port_offset


def _row_crossings(component_row_index: dict, component_slot: dict, edge_end_pairs: list) -> int:
    """
    Count the edge crossings between neighboring rows.

    Only edges that join two neighboring rows are counted. Edges that meet at
    the same port do not cross.

    :param component_row_index: Row index of each component.
    :param component_slot: Slot of each component.
    :param edge_end_pairs: From :func:`_edge_end_pairs`.
    :return: The number of crossings.
    """
    # Port connections in the gap below each row, as
    # (upper port position, lower port position)
    row_port_connection_edges: dict = {}
    for first_end, second_end in edge_end_pairs:
        first_row_index = component_row_index[first_end.component]
        second_row_index = component_row_index[second_end.component]
        # Only edges between two neighboring rows are compared
        if abs(first_row_index - second_row_index) != 1:
            continue

        # The upper end is in the row with the smaller index (0 = top row). It
        # is not always the source end: an edge can go up from its source
        if first_row_index < second_row_index:
            upper_end, lower_end, upper_row_index = first_end, second_end, first_row_index
        else:
            upper_end, lower_end, upper_row_index = second_end, first_end, second_row_index

        row_port_connection_edges.setdefault(upper_row_index, []).append(
            (
                _port_position(upper_end, component_slot),
                _port_position(lower_end, component_slot),
            )
        )

    crossings = 0
    # Each gap between two neighboring rows: only connections in the same gap can cross
    for port_connection_edges in row_port_connection_edges.values():
        # Each connection of the gap...
        for first_index in range(len(port_connection_edges)):
            # ...against each connection after it, so (A, B) is not counted again as (B, A)
            for second_index in range(first_index + 1, len(port_connection_edges)):
                if _edges_cross(
                    port_connection_edges[first_index], port_connection_edges[second_index]
                ):
                    crossings += 1
    return crossings


def _edges_cross(first_port_connection_edge: tuple, second_port_connection_edge: tuple) -> bool:
    """
    Whether two edges between the same two rows cross.

    They cross when their ends are in opposite orders in the two rows: the
    first is left of the second in the upper row but right of it in the lower
    row, or the reverse. Edges that meet at the same port do not cross.

    :param first_port_connection_edge: ``(upper port position, lower port position)`` of one edge.
    :param second_port_connection_edge: The same for the other edge.
    :return: ``True`` if they cross.
    """
    first_upper, first_lower = first_port_connection_edge
    second_upper, second_lower = second_port_connection_edge
    return (first_upper - second_upper) * (first_lower - second_lower) < 0


def _hierarchical_canvas_positions(
    nodes_data: dict,
    edges_data: dict,
    propulsor_names: set,
    layout_area: tuple,
    icon_size: float,
    from_propulsor: bool = _LAYOUT_FROM_PROPULSOR,
) -> dict:
    """
    Run :class:`HierarchicalLayout` on the loaded design and spread the result
    over *layout_area* (see :func:`_layout_area`).

    Neighboring components are never closer than the minimum steps
    (:data:`_LAYOUT_STEP_MIN_ICONS_X` and ``_Y``, in icon sizes). A design that
    does not fit keeps its top row at the top of the area, and grows downward
    and sideways from the center.

    :param nodes_data: Node columns built by ``_build_nodes``.
    :param edges_data: Edge columns built by ``_build_edges``.
    :param propulsor_names: Set of component names that are propulsors.
    :param layout_area: ``(x_min, x_max, y_min, y_max)`` available to the layout.
    :param icon_size: Icon size, in canvas data units; sets the minimum steps.
    :param from_propulsor: Count the hierarchy layers from the propulsors straight away.
    :return: ``{component_name: (x, y)}`` for every node.
    """
    names = list(nodes_data["name"])
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    for start, end in zip(
        edges_data["starting_node_index"], edges_data["ending_node_index"], strict=True
    ):
        graph.add_edge(names[start], names[end])

    node_types = dict(zip(names, nodes_data["node_type"], strict=True))
    layers = _hierarchy_layers(graph, node_types, propulsor_names, from_propulsor)
    # Put the layers in file order, so that a file always opens the same way:
    # the layout keeps the order of its top and bottom rows, and the order
    # _hierarchy_layers returns changes between runs (it comes from a set)
    layers = {name: layers[name] for name in names}

    # The viewer's layout works in abstract units: x = slot inside a layer
    # (centred on 0, step 1), y = layer row (top row has the largest y in "TB")
    raw_positions = HierarchicalLayout(
        graph, orientation="TB", node_layer_dict=layers, sorting=True
    ).generate_networkx_layout()
    raw_positions = _order_rows_by_ports(raw_positions, nodes_data, edges_data)

    layout_names = list(raw_positions)
    xs = _spread_x([raw_positions[name][0] for name in layout_names], layout_area, icon_size)
    ys = _spread_y([raw_positions[name][1] for name in layout_names], layout_area, icon_size)
    return {name: (x, y) for name, x, y in zip(layout_names, xs, ys, strict=True)}


def _grid_positions(count: int, layout_area: tuple, icon_size: float) -> list[tuple[float, float]]:
    """
    Evenly spaced positions over *layout_area*, row by row from the top-left.

    The number of columns follows the area's aspect ratio (no fixed limit),
    so the grid fills wide and tall canvases alike. Like the hierarchical
    layout, it keeps the minimum steps, and grows downward if it does not fit.

    :param count: Number of positions to generate.
    :param layout_area: ``(x_min, x_max, y_min, y_max)`` available to the grid.
    :param icon_size: Icon size, in canvas data units; sets the minimum steps.
    :return: List of ``(x, y)`` tuples.
    """
    if count <= 0:
        return []
    x_min, x_max, y_min, y_max = layout_area
    width, height = x_max - x_min, y_max - y_min
    aspect = width / height if height > 0 else 1.0
    columns = min(count, max(1, int(np.round(np.sqrt(count * aspect)))))
    rows = int(np.ceil(count / columns))

    column_xs = _spread_x(list(range(columns)), layout_area, icon_size)
    row_ys = _spread_y([-row for row in range(rows)], layout_area, icon_size)
    return [(column_xs[i % columns], row_ys[i // columns]) for i in range(count)]
