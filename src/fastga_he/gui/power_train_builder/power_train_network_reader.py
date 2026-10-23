# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
YAML de-serialiser for the powertrain builder.

:class:`PowerTrainYAMLReader` is the inverse of
:class:`~.power_train_network_writer.PowerTrainYAMLWriter`: it reads a powertrain
YAML configuration file and rebuilds the canvas state (nodes + edges) the
builder needs to display and edit it.

The YAML is the single source of truth. Everything the canvas needs is
derived from the powertrain configurator sections of the file:

* ``power_train_components`` -> node name, component type (from ``id``),
  icon, position, options, symmetry peer, and port counts (variable-port
  components read theirs back from the ``number_of_*`` options the writer
  emitted; every other type uses its default count).
* ``component_connections`` -> edges (``[name, N]`` = port ``N``; a plain
  name = port ``1``).
* ``watcher_file_path`` -> the per time step performance parameter csv for the powertrain
components.

Only the canvas (x, y) of each component is GUI-specific; it is read from the
optional ``canvas_layout`` section. Components without a layout entry (a file
saved as "final version", a hand-written YAML, or one produced before this
section existed) are positioned with the same
:class:`~.layout_generation.HierarchicalLayout` used by the network viewer
(energy storage at the bottom, propulsors at the top), so *any* valid
powertrain YAML can be opened in the builder. A simple grid is kept as a
last-resort fallback if that layout cannot be computed.

Port coordinates and edge ``xs`` / ``ys`` are *not* computed here; the
handler's ``_rebuild_all_ports`` recomputes them from the node positions and
port counts once the data is loaded.
"""

import json
import logging
import re
from pathlib import Path

import networkx as nx
import numpy as np
from ruamel.yaml import YAML

from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from ..layout_generation import HierarchicalLayout
from .power_train_builder_state import (
    CANVAS_FIT_NODE_MARGIN,
    CANVAS_WORLD_X_RANGE,
    CANVAS_WORLD_Y_RANGE,
    EDGE_COLUMNS,
    NODE_COLUMNS,
    VARIABLE_PORT_DEFAULT,
    _EMPTY,
    _NO_SYMMETRY_PEER,
    BUILDER_ICONS_CONFIG,
    PORT_TYPE_COLORS,
    UNDEFINED_PORT_TYPE,
    UNKNOWN_DEFAULT_PORT_COUNT,
    UNKNOWN_ICON_KEY,
    default_options,
    default_position,
    empty_columns,
    is_unknown_node,
    link_symmetry,
    node_port_colors,
    ports_compatible,
)
from .power_train_builder_metadata import component_type_classes, icon_data_url
from .power_train_network_writer import (
    KEY_CANVAS_LAYOUT,
    KEY_PT_COMPONENTS,
    KEY_PT_CONNECTIONS,
    KEY_PT_WATCHER,
    port_side_of,
    unknown_component_type,
)

_LOGGER = logging.getLogger(__name__)

# KNOWN_COMPONENTS entry for each FAST-OAD component id
# (e.g. "fastga_he.pt_component.battery_pack")
_KNOWN_COMPONENT_BY_ID = {component["id"]: component for component in KNOWN_COMPONENTS}

# Matches a "[name, N]" endpoint that was written as a quoted string
_QUOTED_ENDPOINT_PATTERN = re.compile(r"^\s*\[\s*([^,\]]+?)\s*,\s*(\d+)\s*\]\s*$")

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

# Area assumed when the browser has not reported one (headless use / tests):
# the 800 x 950 design world at 1 data unit per pixel
_DEFAULT_CANVAS_AREA = dict(
    x0=float(CANVAS_WORLD_X_RANGE[0]),
    x1=float(CANVAS_WORLD_X_RANGE[1]),
    y0=float(CANVAS_WORLD_Y_RANGE[0]),
    y1=float(CANVAS_WORLD_Y_RANGE[1]),
    per_px=1.0,
)

# Energy-storage component types or type classes used as the hierarchy
# reference, matching the viewer's get_component_distance(["tank", "battery_pack"])
_ENERGY_STORAGE_KEYWORDS = ("tank", "battery_pack")


def _matching_component_types(references: tuple) -> set:
    """
    Component types whose type, or one of whose type classes, is in *references*.

    Same matching rule as ``FASTGAHEPowerTrainConfigurator.get_component_distance``,
    which accepts either a component type or a component type class.

    :param references: Component types or component type classes to match.
    :return: Set of matching component types.
    """
    matching_types = set()
    for known_component in KNOWN_COMPONENTS:
        if known_component["components_type"] in references or any(
            reference in component_type_classes(known_component) for reference in references
        ):
            matching_types.add(known_component["components_type"])
    return matching_types


# Component types counted as energy storage when computing hierarchy layers
_ENERGY_STORAGE_COMPONENT_TYPES = _matching_component_types(_ENERGY_STORAGE_KEYWORDS)

# Default of the viewer's from_propulsor argument: when True, layers are always
# counted from the propulsors instead of first trying the energy storage
_LAYOUT_FROM_PROPULSOR = False


class PowerTrainYAMLReader:
    """
    Rebuild builder canvas data from a powertrain YAML configuration file.

    :param state: :class:`~.power_train_builder_state.BuilderState`, used only
        for its read-only per-type metadata (``possible_position``,
        ``possible_options``, ``default_source_count``,
        ``default_target_count``).
    :param icon_size: Pixel size (width = height) of placed icons.
    """

    def __init__(self, state, icon_size: int):
        """
        Keep the per-type metadata the reader needs from *state*.

        :param state: :class:`~.power_train_builder_state.BuilderState` instance.
        :param icon_size: Pixel size (width = height) of placed icons.
        """
        self.possible_position = state.possible_position
        self.possible_options = state.possible_options
        self.default_source_count = state.default_source_count
        self.default_target_count = state.default_target_count
        self.icon_size = icon_size
        # Canvas area of the user's window (None until the browser reports it)
        self.canvas_area = getattr(state, "canvas_area", None) or _DEFAULT_CANVAS_AREA

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def read(self, file_path: str | Path) -> dict:
        """
        Parse *file_path* and return the canvas state it describes.

        :param file_path: Path to a powertrain ``.yml`` / ``.yaml`` file.
        :return: ``{"components": {col: [...]}, "connections": {col: [...]},
            "watcher_file_path": str}``, ready for
            ``IOMixin._restore_canvas_from_dict``.
        :raises OSError: If the file cannot be opened.
        :raises ruamel.yaml.YAMLError: If the file is not valid YAML.
        """
        with open(file_path, encoding="utf-8") as yaml_file:
            yaml_data = YAML(typ="safe").load(yaml_file) or {}

        components = yaml_data.get(KEY_PT_COMPONENTS) or {}
        canvas_layout = yaml_data.get(KEY_CANVAS_LAYOUT) or {}

        nodes_data, name_to_index, nodes_without_layout, propulsor_names = self._build_nodes(
            components, canvas_layout
        )
        edges_data = self._build_edges(
            yaml_data.get(KEY_PT_CONNECTIONS) or [], nodes_data, name_to_index
        )
        _fit_unknown_port_counts(nodes_data, edges_data)

        if nodes_without_layout:
            self._place_nodes_without_layout(
                nodes_data, edges_data, nodes_without_layout, propulsor_names
            )

        watcher_path = yaml_data.get(KEY_PT_WATCHER) or ""

        return {
            "components": nodes_data,
            "connections": edges_data,
            "watcher_file_path": str(watcher_path),
        }

    # -----------------------------------------------------------------------
    # Nodes
    # -----------------------------------------------------------------------

    def _build_nodes(self, components: dict, canvas_layout: dict) -> tuple:
        """
        Build ``placed_nodes_source``-shaped data from ``power_train_components``.

        Nodes without a ``canvas_layout`` entry get a temporary (0, 0) position;
        :meth:`_place_nodes_without_layout` replaces it once edges are known.

        :return: ``(nodes_data, name_to_index, nodes_without_layout, propulsor_names)``.
        """
        nodes_data = empty_columns(NODE_COLUMNS)
        name_to_index: dict = {}
        symmetry_requests: list = []  # (node_index, peer_name)
        nodes_without_layout: list = []
        propulsor_names: set = set()

        for raw_name, raw_component in components.items():
            name = str(raw_name)
            component = raw_component or {}
            layout_entry = canvas_layout.get(name)

            known_component = _drawable_known_component(name, component)
            if known_component is None:
                node_row = _unknown_node_row(component, layout_entry)
            else:
                node_row = self._known_node_row(component, known_component)
                if "propulsor" in component_type_classes(known_component):
                    propulsor_names.add(name)

            node_index = len(nodes_data["name"])
            name_to_index[name] = node_index

            x, y = self._layout_position(layout_entry)
            if x is None:
                nodes_without_layout.append(node_index)
                x, y = 0.0, 0.0  # replaced by _place_nodes_without_layout

            node_row.update(
                x=x,
                y=y,
                url=icon_data_url(BUILDER_ICONS_CONFIG[node_row["icon_type"]]["icon_path"]),
                w=self.icon_size,
                h=self.icon_size,
                name=name,
                symmetry_name=_EMPTY,
                symmetry_node_index=_NO_SYMMETRY_PEER,
            )
            for column_name in NODE_COLUMNS:
                nodes_data[column_name].append(node_row[column_name])

            if component.get("symmetrical"):
                symmetry_requests.append((node_index, str(component["symmetrical"])))

        _link_symmetry_requests(nodes_data, name_to_index, symmetry_requests)

        return nodes_data, name_to_index, nodes_without_layout, propulsor_names

    def _known_node_row(self, component: dict, known_component: dict) -> dict:
        """
        Type, icon, position, options and port counts of a known component of the file.

        :param component: Its entry under ``power_train_components``.
        :param known_component: Its ``KNOWN_COMPONENTS`` entry.
        :return: Those columns of its ``placed_nodes_source`` row.
        """
        component_type = known_component["components_type"]
        file_options = component.get("options") or {}
        n_sources, n_targets = self._port_counts(component_type, known_component, file_options)
        return {
            "node_type": component_type,
            "icon_type": known_component["icon_for_network_graph"],
            "position": self._position(component_type, component.get("position")),
            "options": self._options_json(component_type, file_options),
            "n_sources": n_sources,
            "n_targets": n_targets,
            # Known components take their port colours from their icon
            "source_port_type": _EMPTY,
            "target_port_type": _EMPTY,
        }

    # -----------------------------------------------------------------------
    # Automatic layout (no canvas_layout entry)
    # -----------------------------------------------------------------------

    def _place_nodes_without_layout(
        self, nodes_data: dict, edges_data: dict, nodes_without_layout: list, propulsor_names: set
    ) -> None:
        """
        Position the nodes that have no ``canvas_layout`` entry, in place.

        Uses the network viewer's :class:`HierarchicalLayout` (top-to-bottom,
        Tutte crossing reduction) on the whole graph, spread over the canvas
        area of the user's window (see :func:`_layout_area`), then keeps the
        generated position only for the nodes that need one. Falls back to a
        grid over the same area if the layout cannot be computed.
        """
        area = _layout_area(self.canvas_area)
        try:
            positions = _hierarchical_canvas_positions(
                nodes_data, edges_data, propulsor_names, area
            )
            method = "hierarchical layout"
        except Exception:  # pragma: no cover - defensive: never block a load on layout
            _LOGGER.exception("Hierarchical layout failed; placing components on a grid.")
            positions = None

        if positions is None:
            grid = _grid_positions(len(nodes_without_layout), area)
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

    @staticmethod
    def _layout_position(layout_entry) -> tuple:
        """
        Return ``(x, y)`` from a ``canvas_layout`` entry, or ``(None, None)``.

        :param layout_entry: Value stored under the component name in ``canvas_layout``.
        :return: Tuple of canvas coordinates, or ``(None, None)`` if the entry is invalid.
        """
        if not isinstance(layout_entry, dict):
            return None, None
        try:
            return float(layout_entry["x"]), float(layout_entry["y"])
        except (KeyError, TypeError, ValueError):
            return None, None

    def _position(self, component_type: str, file_position) -> str:
        """
        Position from the file, falling back to the type's default.

        :param component_type: Component type key used to look up ``possible_position``.
        :param file_position: Position string read from the YAML file, if any.
        :return: The position string stored on the canvas node.
        """
        if file_position:
            return str(file_position)
        return default_position(self.possible_position.get(component_type, []))

    def _options_json(self, component_type: str, file_options: dict) -> str:
        """
        Options JSON in the same shape the GUI stores at placement / Apply time.

        Starts from the type's defaults, then overrides with the values found
        in the file. Only options the GUI knows (``possible_options``) are
        kept; ``number_of_*`` keys are represented by the port counts instead
        and are re-emitted by the writer on the next save.

        :param component_type: Component type key used to look up ``possible_options``.
        :param file_options: Options dictionary read from the YAML file.
        :return: JSON string of the option names and their values.
        """
        definition = self.possible_options.get(component_type, {})
        options = default_options(definition)
        options.update({name: value for name, value in file_options.items() if name in definition})
        return json.dumps(options)

    def _port_counts(
        self, component_type: str, known_component: dict, file_options: dict
    ) -> tuple[int, int]:
        """
        Port counts for a node: the type's defaults, except for variable-port
        components whose ``number_of_*`` options (written by
        ``PowerTrainYAMLWriter.add_component`` with the same keyword rules) override them.

        :param component_type: Component type key used to look up the default port counts.
        :param known_component: Matching entry of ``KNOWN_COMPONENTS``.
        :param file_options: Options dictionary read from the YAML file.
        :return: Tuple ``(n_sources, n_targets)``.
        """
        n_sources = int(self.default_source_count.get(component_type, 0))
        n_targets = int(self.default_target_count.get(component_type, 0))

        if n_sources != VARIABLE_PORT_DEFAULT:  # same "variable-port" test as the writer
            return n_sources, n_targets

        attributes = known_component.get("attributes")
        if not isinstance(attributes, list):
            return n_sources, n_targets

        for attribute_name in attributes:
            if "number_of_" not in attribute_name or attribute_name not in file_options:
                continue
            try:
                count = int(file_options[attribute_name])
            except (TypeError, ValueError):
                continue
            port_side = port_side_of(attribute_name)
            if port_side == "source":
                n_sources = count
            elif port_side == "target":
                n_targets = count

        return n_sources, n_targets

    # -----------------------------------------------------------------------
    # Edges
    # -----------------------------------------------------------------------

    def _build_edges(self, connections: list, nodes_data: dict, name_to_index: dict) -> dict:
        """
        Build ``edge_source``-shaped data from ``component_connections``.

        Edges are always stored source -> target (the convention the config
        panel already uses), and only between ports that may be connected on
        the canvas: the same, defined energy type at both ends. An undefined
        port side of an unknown component first takes the energy type of what
        it is connected to (see :func:`_type_unknown_ports_from_connections`).
        ``xs`` / ``ys`` are placeholders; they are recomputed by
        ``_rebuild_all_ports`` / ``_rebuild_edges`` after load, which also drops
        any edge whose port index exceeds the node's count.

        :param connections: List of ``{"source": ..., "target": ...}`` dictionaries.
        :param nodes_data: Node columns built by :meth:`_build_nodes`, whose
            unknown port types may be set here.
        :param name_to_index: Dictionary mapping component names to their node index.
        :return: Dictionary of edge columns.
        """
        edges_data = empty_columns(EDGE_COLUMNS)
        endpoints = _connection_endpoints(connections, name_to_index)
        _type_unknown_ports_from_connections(nodes_data, endpoints)

        for source_index, source_label, target_index, target_label in endpoints:
            source_color = node_port_colors(nodes_data, source_index)[0]
            target_color = node_port_colors(nodes_data, target_index)[1]
            # A side without colour has no ports at all
            if source_color is None or not ports_compatible(source_color, target_color):
                _LOGGER.warning(
                    "Skipping connection %s -> %s: their ports cannot be connected.",
                    nodes_data["name"][source_index],
                    nodes_data["name"][target_index],
                )
                continue

            edges_data["xs"].append([0.0, 0.0])
            edges_data["ys"].append([0.0, 0.0])
            edges_data["color"].append(source_color)
            edges_data["starting_node_index"].append(source_index)
            edges_data["starting_port_label"].append(source_label)
            edges_data["starting_port_kind"].append("source")
            edges_data["ending_node_index"].append(target_index)
            edges_data["ending_port_label"].append(target_label)
            edges_data["ending_port_kind"].append("target")

        return edges_data


# ============================================================================
# Helpers
# ============================================================================


def _drawable_known_component(name: str, component: dict) -> dict | None:
    """
    ``KNOWN_COMPONENTS`` entry of a component of the file, if the canvas can draw it.

    A component with an unregistered ``id``, or whose icon the builder does not
    have, is shown as an unknown component (see :func:`_unknown_node_row`).

    :param name: Component name in the file.
    :param component: Its entry under ``power_train_components``.
    :return: The ``KNOWN_COMPONENTS`` entry, or ``None`` for an unknown component.
    """
    known_component = _KNOWN_COMPONENT_BY_ID.get(component.get("id"))
    if known_component is None:
        _LOGGER.info("Component %s (id %r) is shown as unknown.", name, component.get("id"))
        return None
    icon_key = known_component["icon_for_network_graph"]
    if icon_key not in BUILDER_ICONS_CONFIG:
        _LOGGER.info("Component %s: no icon for %r, shown as unknown.", name, icon_key)
        return None
    return known_component


def _connection_endpoints(connections: list, name_to_index: dict) -> list[tuple]:
    """
    Node index and port label of both ends of every valid connection of the file.

    Malformed connections, connections to a component that is not in the file
    and connections of a component to itself are reported and left out.

    :param connections: List of ``{"source": ..., "target": ...}`` dictionaries.
    :param name_to_index: Dictionary mapping component names to their node index.
    :return: ``(source_index, source_label, target_index, target_label)`` of each.
    """
    endpoints = []
    for connection in connections:
        if not isinstance(connection, dict):
            continue
        source = _parse_endpoint(connection.get("source"))
        target = _parse_endpoint(connection.get("target"))
        if source is None or target is None:
            _LOGGER.warning("Skipping malformed connection %r.", connection)
            continue

        (source_name, source_label), (target_name, target_label) = source, target
        if source_name not in name_to_index or target_name not in name_to_index:
            _LOGGER.warning(
                "Skipping connection %s -> %s: unknown component.", source_name, target_name
            )
            continue
        if source_name == target_name:
            _LOGGER.warning("Skipping connection of %s to itself.", source_name)
            continue
        endpoints.append(
            (name_to_index[source_name], source_label, name_to_index[target_name], target_label)
        )
    return endpoints


# Port type of each defined energy colour, to type an unknown port from what it connects to
_PORT_TYPE_OF_COLOR = {
    color: port_type
    for port_type, color in PORT_TYPE_COLORS.items()
    if port_type != UNDEFINED_PORT_TYPE
}


def _type_unknown_ports_from_connections(nodes_data: dict, endpoints: list) -> None:
    """
    Give every undefined port side of an unknown component the energy type it is connected to.

    A file without ``canvas_layout`` does not store the port types of its
    unknown components, but its connections imply them. The first connection
    to a defined port decides; repeated until nothing changes, so a chain of
    unknown components is typed from its known end. A side connected to
    nothing defined stays undefined, and its connections are then refused.

    :param nodes_data: Node columns built by ``_build_nodes``, edited in place.
    :param endpoints: Connections from :func:`_connection_endpoints`.
    """
    is_typing = True
    while is_typing:
        is_typing = False
        for source_index, _, target_index, _ in endpoints:
            source_color = node_port_colors(nodes_data, source_index)[0]
            target_color = node_port_colors(nodes_data, target_index)[1]
            is_typing |= _type_undefined_side(nodes_data, source_index, "source", target_color)
            is_typing |= _type_undefined_side(nodes_data, target_index, "target", source_color)


def _type_undefined_side(nodes_data: dict, node_index: int, side: str, far_color) -> bool:
    """
    Set one undefined port side of an unknown component to the energy type of *far_color*.

    :param nodes_data: Node columns, edited in place.
    :param node_index: Zero-based index of the node.
    :param side: ``"source"`` or ``"target"``.
    :param far_color: Colour of the port at the other end of a connection of that side.
    :return: ``True`` if the side was typed.
    """
    column = f"{side}_port_type"
    if (
        not is_unknown_node(nodes_data, node_index)
        or nodes_data[column][node_index] != UNDEFINED_PORT_TYPE
        or far_color not in _PORT_TYPE_OF_COLOR
    ):
        return False
    nodes_data[column][node_index] = _PORT_TYPE_OF_COLOR[far_color]
    return True


def _unknown_node_row(component: dict, layout_entry) -> dict:
    """
    Type, icon, position, options and port settings of an unknown component of the file.

    Its options are kept exactly as written. Its port counts and port types
    come from its ``canvas_layout`` entry when the file was saved for later
    edits; otherwise the port types are undefined and the port counts are left
    to :func:`_fit_unknown_port_counts` (``None`` here).

    :param component: Its entry under ``power_train_components``.
    :param layout_entry: Its ``canvas_layout`` entry, if any.
    :return: Those columns of its ``placed_nodes_source`` row.
    """
    layout_entry = layout_entry if isinstance(layout_entry, dict) else {}
    return {
        "node_type": unknown_component_type(component.get("id")),
        "icon_type": UNKNOWN_ICON_KEY,
        "position": str(component.get("position") or _EMPTY),
        "options": json.dumps(component.get("options") or {}),
        "n_sources": _layout_port_count(layout_entry, "n_sources"),
        "n_targets": _layout_port_count(layout_entry, "n_targets"),
        "source_port_type": _layout_port_type(layout_entry, "source_port_type"),
        "target_port_type": _layout_port_type(layout_entry, "target_port_type"),
    }


def _layout_port_count(layout_entry: dict, key: str) -> int | None:
    """Port count stored in a ``canvas_layout`` entry, or ``None`` if missing or invalid."""
    try:
        count = int(layout_entry[key])
    except (KeyError, TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _layout_port_type(layout_entry: dict, key: str) -> str:
    """Port type stored in a ``canvas_layout`` entry, or undefined if missing or invalid."""
    port_type = layout_entry.get(key)
    return port_type if port_type in PORT_TYPE_COLORS else UNDEFINED_PORT_TYPE


def _fit_unknown_port_counts(nodes_data: dict, edges_data: dict) -> None:
    """
    Give every unknown component without stored port counts enough ports, in place.

    Each side gets as many ports as the highest port number its connections
    use, and at least :data:`UNKNOWN_DEFAULT_PORT_COUNT`.

    :param nodes_data: Node columns built by ``_build_nodes``, edited in place.
    :param edges_data: Edge columns built by ``_build_edges``.
    """
    highest_port_used: dict = {}
    for end in ("starting", "ending"):
        for node_index, label, kind in zip(
            edges_data[f"{end}_node_index"],
            edges_data[f"{end}_port_label"],
            edges_data[f"{end}_port_kind"],
            strict=True,
        ):
            key = (node_index, kind)
            highest_port_used[key] = max(highest_port_used.get(key, 0), int(label))
    for node_index in range(len(nodes_data["name"])):
        for column, kind in (("n_sources", "source"), ("n_targets", "target")):
            if nodes_data[column][node_index] is None:
                nodes_data[column][node_index] = max(
                    UNKNOWN_DEFAULT_PORT_COUNT, highest_port_used.get((node_index, kind), 0)
                )


def _link_symmetry_requests(nodes_data: dict, name_to_index: dict, symmetry_requests: list) -> None:
    """
    Link every ``symmetrical`` pair of the file on both nodes, in place.

    The writer emits ``symmetrical`` on only one node of each pair, but the
    canvas stores the link on both sides. A peer missing from the file is
    reported and the link ignored, except on an unknown component, which keeps
    the name without a peer on the canvas (as its free-text box allows).

    :param nodes_data: Node columns built by ``_build_nodes``, edited in place.
    :param name_to_index: Dictionary mapping component names to their node index.
    :param symmetry_requests: ``(node_index, peer_name)`` of every ``symmetrical`` entry.
    """
    for node_index, peer_name in symmetry_requests:
        peer_index = name_to_index.get(peer_name)
        if peer_index is None and is_unknown_node(nodes_data, node_index):
            nodes_data["symmetry_name"][node_index] = peer_name
            continue
        if peer_index is None:
            _LOGGER.warning(
                "Symmetry peer %s of %s not found; link ignored.",
                peer_name,
                nodes_data["name"][node_index],
            )
            continue
        link_symmetry(
            nodes_data["symmetry_name"],
            nodes_data["symmetry_node_index"],
            nodes_data["name"],
            node_index,
            peer_index,
        )


def _parse_endpoint(endpoint) -> tuple[str, str] | None:
    """
    Convert a YAML connection endpoint into ``(component_name, port_label)``.

    * ``[name, N]`` (YAML flow list) -> ``(name, "N")``
    * ``"[name, N]"`` (same, but quoted) -> ``(name, "N")``
    * ``name`` -> ``(name, "1")``

    Port labels are strings to match ``source_port_source["label"]``.

    :param endpoint: Raw ``source`` or ``target`` value of a YAML connection.
    :return: Tuple ``(component_name, port_label)``, or ``None`` if the endpoint is malformed.
    """
    if isinstance(endpoint, (list, tuple)) and len(endpoint) == 2:
        try:
            return str(endpoint[0]), str(int(endpoint[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(endpoint, str) and endpoint:
        quoted_match = _QUOTED_ENDPOINT_PATTERN.match(endpoint)
        if quoted_match:
            return quoted_match.group(1), str(int(quoted_match.group(2)))
        return endpoint, "1"
    return None


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
    energy_storage_nodes = [
        node_name
        for node_name, node_type in node_types.items()
        if node_type in _ENERGY_STORAGE_COMPONENT_TYPES
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


def _spread(values: list, low: float, high: float, step_max: float) -> list:
    """
    Map abstract coordinates (evenly spaced, any origin) onto ``[low, high]``:
    as spread out as the range allows, but at most ``step_max`` per unit, and
    centred in the range.

    :param values: Abstract coordinates to map.
    :param low: Lower bound of the target range.
    :param high: Upper bound of the target range.
    :param step_max: Largest allowed distance between two neighbouring abstract units.
    :return: List of mapped coordinates, in the same order as *values*.
    """
    span = max(values) - min(values)
    step = min(step_max, (high - low) / span) if span else 0.0
    middle = (max(values) + min(values)) / 2
    centre = (low + high) / 2
    return [
        min(high, max(low, round(centre + (float(value) - middle) * step, 1))) for value in values
    ]


def _hierarchical_canvas_positions(
    nodes_data: dict,
    edges_data: dict,
    propulsor_names: set,
    area: tuple,
    from_propulsor: bool = _LAYOUT_FROM_PROPULSOR,
) -> dict:
    """
    Run :class:`HierarchicalLayout` on the loaded design and spread the result
    over *area* (see :func:`_layout_area`).

    :param nodes_data: Node columns built by ``_build_nodes``.
    :param edges_data: Edge columns built by ``_build_edges``.
    :param propulsor_names: Set of component names that are propulsors.
    :param area: Tuple ``(x_min, x_max, y_min, y_max)`` available to the layout.
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

    # The viewer's layout works in abstract units: x = slot inside a layer
    # (centred on 0, step 1), y = layer row (top row has the largest y in "TB")
    raw_positions = HierarchicalLayout(
        graph, orientation="TB", node_layer_dict=layers, sorting=True
    ).generate_networkx_layout()

    x_min, x_max, y_min, y_max = area
    layout_names = list(raw_positions)
    xs = _spread(
        [raw_positions[name][0] for name in layout_names],
        x_min,
        x_max,
        _LAYOUT_STEP_MAX_FRACTION_X * (x_max - x_min),
    )
    ys = _spread(
        [raw_positions[name][1] for name in layout_names],
        y_min,
        y_max,
        _LAYOUT_STEP_MAX_FRACTION_Y * (y_max - y_min),
    )
    return {name: (x, y) for name, x, y in zip(layout_names, xs, ys, strict=True)}


def _grid_positions(count: int, area: tuple) -> list[tuple[float, float]]:
    """
    Evenly spaced positions over *area*, row by row from the top-left.

    The number of columns follows the area's aspect ratio (no fixed limit),
    so the grid fills wide and tall canvases alike.

    :param count: Number of positions to generate.
    :param area: Tuple ``(x_min, x_max, y_min, y_max)`` available to the grid.
    :return: List of ``(x, y)`` tuples.
    """
    if count <= 0:
        return []
    x_min, x_max, y_min, y_max = area
    width, height = x_max - x_min, y_max - y_min
    aspect = width / height if height > 0 else 1.0
    columns = min(count, max(1, int(np.round(np.sqrt(count * aspect)))))
    rows = int(np.ceil(count / columns))

    column_xs = _spread(list(range(columns)), x_min, x_max, _LAYOUT_STEP_MAX_FRACTION_X * width)
    row_ys = _spread(
        [-row for row in range(rows)], y_min, y_max, _LAYOUT_STEP_MAX_FRACTION_Y * height
    )
    return [(column_xs[i % columns], row_ys[i // columns]) for i in range(count)]
