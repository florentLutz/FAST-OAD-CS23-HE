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
section existed) are positioned automatically by
:func:`~.power_train_network_layout.place_nodes_without_layout`, so *any* valid
powertrain YAML can be opened in the builder.

Port coordinates and edge ``xs`` / ``ys`` are *not* computed here; the
handler's ``_rebuild_all_ports`` recomputes them from the node positions and
port counts once the data is loaded.
"""

import json
import logging
import re
from pathlib import Path

import networkx as nx
from ruamel.yaml import YAML

from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from .power_train_builder_state import (
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
    split_component_id,
    unknown_nodes_without_port,
)
from .power_train_builder_metadata import component_type_classes, icon_data_url
from .power_train_network_layout import DEFAULT_CANVAS_AREA, place_nodes_without_layout
from .power_train_network_writer import (
    KEY_CANVAS_LAYOUT,
    KEY_PT_COMPONENTS,
    KEY_PT_CONNECTIONS,
    KEY_PT_WATCHER,
    port_side_of,
)

_LOGGER = logging.getLogger(__name__)

# KNOWN_COMPONENTS entry for each FAST-OAD component id
# (e.g. "fastga_he.pt_component.battery_pack")
_KNOWN_COMPONENT_BY_ID = {component["id"]: component for component in KNOWN_COMPONENTS}

# Matches a "[name, N]" endpoint that was written as a quoted string
_QUOTED_ENDPOINT_PATTERN = re.compile(r"^\s*\[\s*([^,\]]+?)\s*,\s*(\d+)\s*\]\s*$")

# Port type of each defined energy colour, to type an unknown port from what it connects to
_PORT_TYPE_OF_COLOR = {
    color: port_type
    for port_type, color in PORT_TYPE_COLORS.items()
    if port_type != UNDEFINED_PORT_TYPE
}


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
        self.canvas_area = getattr(state, "canvas_area", None) or DEFAULT_CANVAS_AREA

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
        for name in unknown_nodes_without_port(nodes_data):
            _LOGGER.warning(
                "Unknown component %s has neither a source nor a target port; "
                "it cannot be connected to anything.",
                name,
            )

        if nodes_without_layout:
            place_nodes_without_layout(
                nodes_data,
                edges_data,
                nodes_without_layout,
                propulsor_names,
                self.canvas_area,
                self.icon_size,
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
        :func:`~.power_train_network_layout.place_nodes_without_layout` replaces
        it once edges are known.

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
                node_entry = _unknown_node_entry(component, layout_entry)
            else:
                node_entry = self._known_node_entry(component, known_component)
                if "propulsor" in component_type_classes(known_component):
                    propulsor_names.add(name)

            node_index = len(nodes_data["name"])
            name_to_index[name] = node_index

            x, y = self._layout_position(layout_entry)
            if x is None:
                nodes_without_layout.append(node_index)
                x, y = 0.0, 0.0  # replaced by place_nodes_without_layout

            node_entry.update(
                x=x,
                y=y,
                url=icon_data_url(BUILDER_ICONS_CONFIG[node_entry["icon_type"]]["icon_path"]),
                w=self.icon_size,
                h=self.icon_size,
                name=name,
                symmetry_name=_EMPTY,
                symmetry_node_index=_NO_SYMMETRY_PEER,
            )
            for column_name in NODE_COLUMNS:
                nodes_data[column_name].append(node_entry[column_name])

            if component.get("symmetrical"):
                symmetry_requests.append((node_index, str(component["symmetrical"])))

        _link_symmetry_requests(nodes_data, name_to_index, symmetry_requests)

        return nodes_data, name_to_index, nodes_without_layout, propulsor_names

    def _known_node_entry(self, component: dict, known_component: dict) -> dict:
        """
        Type, icon, position, options and port counts of a known component of the file.

        :param component: Its entry under ``power_train_components``.
        :param known_component: Its ``KNOWN_COMPONENTS`` entry.
        :return: Those columns of its ``placed_nodes_source`` entry.
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
            # Their id comes from KNOWN_COMPONENTS
            "id_prefix": _EMPTY,
        }

    # -----------------------------------------------------------------------
    # Canvas layout section
    # -----------------------------------------------------------------------
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

        # No options for users
        attributes = known_component.get("attributes")
        if not isinstance(attributes, list):
            return n_sources, n_targets

        # Having options but no options related to port counts
        elif n_sources != VARIABLE_PORT_DEFAULT or n_targets != VARIABLE_PORT_DEFAULT:
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
    have, is shown as an unknown component (see :func:`_unknown_node_entry`).

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


def _type_unknown_ports_from_connections(nodes_data: dict, endpoints: list) -> None:
    """
    Give every undefined port side of an unknown component the energy type it is connected to.

    A file without ``canvas_layout`` does not store the port types of its
    unknown components, but its connections imply them: connected ports carry
    the same energy type. So the undefined sides joined by connections form
    groups, and each group takes the energy type of the defined ports it is
    connected to; if there are several, the one of the first connection in the
    file. A group connected to nothing defined stays undefined, and its
    connections are then refused.

    :param nodes_data: Node columns built by ``_build_nodes``, edited in place.
    :param endpoints: Connections from :func:`_connection_endpoints`.
    """
    # Every port side, (node_index, "source" / "target"), joined to the sides it
    # is connected to. Each link remembers the first connection that made it
    port_side_graph = nx.Graph()
    for connection_index, (source_index, _, target_index, _) in enumerate(endpoints):
        source_side, target_side = (source_index, "source"), (target_index, "target")
        if not port_side_graph.has_edge(source_side, target_side):
            port_side_graph.add_edge(source_side, target_side, connection_index=connection_index)

    undefined_sides = [side for side in port_side_graph if _is_undefined_side(nodes_data, side)]
    for group in nx.connected_components(port_side_graph.subgraph(undefined_sides)):
        group_port_type = _connected_port_type(nodes_data, port_side_graph, group)
        if group_port_type is None:
            continue
        for node_index, side in group:
            nodes_data[f"{side}_port_type"][node_index] = group_port_type


def _is_undefined_side(nodes_data: dict, port_side: tuple) -> bool:
    """
    Whether a port side belongs to an unknown component and has no energy type yet.

    :param nodes_data: Node columns.
    :param port_side: ``(node_index, "source" / "target")``.
    :return: ``True`` for an undefined side of an unknown component.
    """
    node_index, side = port_side
    return (
        is_unknown_node(nodes_data, node_index)
        and nodes_data[f"{side}_port_type"][node_index] == UNDEFINED_PORT_TYPE
    )


def _connected_port_type(nodes_data: dict, side_graph, group: set) -> str | None:
    """
    Energy type of the defined ports a group of undefined sides is connected to.

    :param nodes_data: Node columns.
    :param side_graph: Port sides joined by connections, each link carrying the
        index of the first connection that made it.
    :param group: Undefined port sides joined to each other by connections.
    :return: The energy type of the first connection (in file order) from the
        group to a defined port, or ``None`` if the group reaches none.
    """
    first_connection_index, port_type = None, None
    for group_side in group:
        for far_side, link in side_graph[group_side].items():
            far_node_index, far_side_name = far_side
            far_color = node_port_colors(nodes_data, far_node_index)[
                0 if far_side_name == "source" else 1
            ]
            if far_color not in _PORT_TYPE_OF_COLOR:
                continue
            if first_connection_index is None or link["connection_index"] < first_connection_index:
                first_connection_index = link["connection_index"]
                port_type = _PORT_TYPE_OF_COLOR[far_color]
    return port_type


def _unknown_node_entry(component: dict, layout_entry) -> dict:
    """
    Type, icon, position, options and port settings of an unknown component of the file.

    Its id is split at its last dot into its id prefix and its component type
    (see :func:`~.power_train_builder_state.split_component_id`). Its options
    are kept exactly as written. Its port counts and port types
    come from its ``canvas_layout`` entry when the file was saved for later
    edits; otherwise the port types are undefined and the port counts are left
    to :func:`_fit_unknown_port_counts` (``None`` here).

    :param component: Its entry under ``power_train_components``.
    :param layout_entry: Its ``canvas_layout`` entry, if any.
    :return: Those columns of its ``placed_nodes_source`` entry.
    """
    layout_entry = layout_entry if isinstance(layout_entry, dict) else {}
    id_prefix, component_type = split_component_id(component.get("id"))
    return {
        "node_type": component_type,
        "id_prefix": id_prefix,
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
