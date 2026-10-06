# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
YAML de-serialiser for the powertrain builder.

:class:`PowerTrainYAMLReader` is the inverse of
:class:`~.power_train_network_writer.PowerTrainYAML`: it reads a powertrain
YAML configuration file and rebuilds the canvas state (nodes + edges) the
builder needs to display and edit it.

The YAML is the single source of truth. Everything the canvas needs is
derived from the FAST-OAD sections of the file:

* ``power_train_components`` -> node name, component type (from ``id``),
  icon, position, options, symmetry peer, and port counts (variable-port
  components read theirs back from the ``number_of_*`` options the writer
  emitted; every other type uses its default count).
* ``component_connections`` -> edges (``[name, N]`` = port ``N``; a plain
  name = port ``1``).
* ``watcher_file_path`` -> kept so that re-saving the design preserves it.

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
import math
import re
from pathlib import Path

import networkx as nx
from ruamel.yaml import YAML

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG, DEFAULT_COLOR, _url_to_base64
from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from ..layout_generation import HierarchicalLayout
from .power_train_builder_state import (
    CANVAS_FIT_NODE_MARGIN,
    CANVAS_WORLD_X_RANGE,
    CANVAS_WORLD_Y_RANGE,
    _EMPTY,
    _NO_SYMMETRY_PEER,
)
from .power_train_network_writer import (
    KEY_CANVAS_LAYOUT,
    KEY_PT_COMPONENTS,
    KEY_PT_CONNECTIONS,
    KEY_PT_WATCHER,
    SOURCE_SIDE_KEYWORDS,
    TARGET_SIDE_KEYWORDS,
)

_LOGGER = logging.getLogger(__name__)

# KNOWN_COMPONENTS entry for each FAST-OAD component id
# (e.g. "fastga_he.pt_component.battery_pack").
_KNOWN_COMPONENT_BY_ID = {component["id"]: component for component in KNOWN_COMPONENTS}

# Matches a "[name, N]" endpoint that was written as a quoted string.
_QUOTED_ENDPOINT_PATTERN = re.compile(r"^\s*\[\s*([^,\]]+?)\s*,\s*(\d+)\s*\]\s*$")

# Automatic layouts are spread over the canvas area of the user's current
# browser window (BuilderState.canvas_area, reported by the browser; with the
# window maximized this is the maximized canvas), inset by these margins in
# screen pixels. The top margin clears the Undo / Redo buttons (top-left) and
# the legend (top-right). Margins never go below CANVAS_FIT_NODE_MARGIN (data
# units), so the loaded design fits the current view without re-zooming.
_LAYOUT_MARGIN_PX = {"left": 50.0, "right": 50.0, "top": 110.0, "bottom": 50.0}

# Largest gap between two neighbouring components, as a fraction of the layout
# area's width / height. Keeps small designs (e.g. two layers) compact and
# centred instead of pushed to opposite edges of a large screen.
_LAYOUT_STEP_MAX_FRACTION_X = 0.25
_LAYOUT_STEP_MAX_FRACTION_Y = 0.30

# Area assumed when the browser has not reported one (headless use / tests):
# the 800 x 950 design world at 1 data unit per pixel.
_DEFAULT_CANVAS_AREA = dict(
    x0=float(CANVAS_WORLD_X_RANGE[0]),
    x1=float(CANVAS_WORLD_X_RANGE[1]),
    y0=float(CANVAS_WORLD_Y_RANGE[0]),
    y1=float(CANVAS_WORLD_Y_RANGE[1]),
    per_px=1.0,
)

# Energy-storage component types used as the hierarchy reference, matching
# the viewer's get_component_distance(["tank", "battery_pack"]).
_ENERGY_STORAGE_KEYWORDS = ("tank", "battery_pack")


_NODE_COLUMNS = (
    "x",
    "y",
    "url",
    "w",
    "h",
    "name",
    "node_type",
    "icon_type",
    "position",
    "options",
    "n_sources",
    "n_targets",
    "symmetry_name",
    "symmetry_node_index",
)
_EDGE_COLUMNS = (
    "xs",
    "ys",
    "color",
    "starting_node_index",
    "starting_port_label",
    "starting_port_kind",
    "ending_node_index",
    "ending_port_label",
    "ending_port_kind",
)


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
        self.possible_position = state.possible_position
        self.possible_options = state.possible_options
        self.default_source_count = state.default_source_count
        self.default_target_count = state.default_target_count
        self.icon_size = icon_size
        # Canvas area of the user's window (None until the browser reports it).
        self.canvas_area = getattr(state, "canvas_area", None) or _DEFAULT_CANVAS_AREA
        self._icon_url_cache: dict = {}

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
        nodes_data = {column: [] for column in _NODE_COLUMNS}
        name_to_index: dict = {}
        symmetry_requests: list = []  # (node_index, peer_name)
        nodes_without_layout: list = []
        propulsor_names: set = set()

        for raw_name, component in components.items():
            name = str(raw_name)
            component = component or {}

            known_component = _KNOWN_COMPONENT_BY_ID.get(component.get("id"))
            if known_component is None:
                _LOGGER.warning("Skipping component %s: unknown id %r.", name, component.get("id"))
                continue

            component_type = known_component["components_type"]
            icon_key = known_component["icon_for_network_graph"]
            if icon_key not in ICONS_CONFIG:
                _LOGGER.warning("Skipping component %s: no icon for %r.", name, icon_key)
                continue

            file_options = component.get("options") or {}
            n_sources, n_targets = self._port_counts(component_type, known_component, file_options)

            node_index = len(nodes_data["name"])
            name_to_index[name] = node_index

            x, y = self._layout_position(canvas_layout.get(name))
            if x is None:
                nodes_without_layout.append(node_index)
                x, y = 0.0, 0.0  # replaced by _place_nodes_without_layout

            type_class = known_component.get("components_type_class")
            type_classes = type_class if isinstance(type_class, list) else [type_class]
            if "propulsor" in type_classes:
                propulsor_names.add(name)

            nodes_data["x"].append(x)
            nodes_data["y"].append(y)
            nodes_data["url"].append(self._icon_url(icon_key))
            nodes_data["w"].append(self.icon_size)
            nodes_data["h"].append(self.icon_size)
            nodes_data["name"].append(name)
            nodes_data["node_type"].append(component_type)
            nodes_data["icon_type"].append(icon_key)
            nodes_data["position"].append(self._position(component_type, component.get("position")))
            nodes_data["options"].append(self._options_json(component_type, file_options))
            nodes_data["n_sources"].append(n_sources)
            nodes_data["n_targets"].append(n_targets)
            nodes_data["symmetry_name"].append(_EMPTY)
            nodes_data["symmetry_node_index"].append(_NO_SYMMETRY_PEER)

            if component.get("symmetrical"):
                symmetry_requests.append((node_index, str(component["symmetrical"])))

        # The writer emits "symmetrical" on only one node of each pair, but the
        # canvas stores the link on both sides.
        for node_index, peer_name in symmetry_requests:
            peer_index = name_to_index.get(peer_name)
            if peer_index is None:
                _LOGGER.warning(
                    "Symmetry peer %s of %s not found; link ignored.",
                    peer_name,
                    nodes_data["name"][node_index],
                )
                continue
            nodes_data["symmetry_name"][node_index] = peer_name
            nodes_data["symmetry_node_index"][node_index] = peer_index
            nodes_data["symmetry_name"][peer_index] = nodes_data["name"][node_index]
            nodes_data["symmetry_node_index"][peer_index] = node_index

        return nodes_data, name_to_index, nodes_without_layout, propulsor_names

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
                nodes_data["name"][node_index]: grid[i]
                for i, node_index in enumerate(nodes_without_layout)
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
        """Return ``(x, y)`` from a ``canvas_layout`` entry, or ``(None, None)``."""
        if not isinstance(layout_entry, dict):
            return None, None
        try:
            return float(layout_entry["x"]), float(layout_entry["y"])
        except (KeyError, TypeError, ValueError):
            return None, None

    def _position(self, component_type: str, file_position) -> str:
        """Position from the file, falling back to the type's default."""
        if file_position:
            return str(file_position)
        choices = self.possible_position.get(component_type, [])
        return choices[0] if choices else _EMPTY

    def _options_json(self, component_type: str, file_options: dict) -> str:
        """
        Options JSON in the same shape the GUI stores at placement / Apply time.

        Starts from the type's defaults, then overrides with the values found
        in the file. Only options the GUI knows (``possible_options``) are
        kept; ``number_of_*`` keys are represented by the port counts instead
        and are re-emitted by the writer on the next save.
        """
        definition = self.possible_options.get(component_type, {})
        options = {name: values[0] for name, values in definition.items() if values}
        options.update({name: value for name, value in file_options.items() if name in definition})
        return json.dumps(options)

    def _port_counts(
        self, component_type: str, known_component: dict, file_options: dict
    ) -> tuple[int, int]:
        """
        Port counts for a node: the type's defaults, except for variable-port
        components whose ``number_of_*`` options (written by
        ``PowerTrainYAML.add_component`` with the same keyword rules) override them.
        """
        n_sources = int(self.default_source_count.get(component_type, 0))
        n_targets = int(self.default_target_count.get(component_type, 0))

        if n_sources != 3:  # same "variable-port" test as the writer
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
            if any(keyword in attribute_name for keyword in SOURCE_SIDE_KEYWORDS):
                n_sources = count
            elif any(keyword in attribute_name for keyword in TARGET_SIDE_KEYWORDS):
                n_targets = count

        return n_sources, n_targets

    def _icon_url(self, icon_key: str) -> str:
        """Base64 data-URL of an icon (same encoding as ``_place_new_node``), cached."""
        if icon_key not in self._icon_url_cache:
            icon_path = ICONS_CONFIG[icon_key]["icon_path"]
            file_url = "file://" + str(Path(icon_path).resolve())
            self._icon_url_cache[icon_key] = _url_to_base64(file_url)
        return self._icon_url_cache[icon_key]

    # -----------------------------------------------------------------------
    # Edges
    # -----------------------------------------------------------------------

    def _build_edges(self, connections: list, nodes_data: dict, name_to_index: dict) -> dict:
        """
        Build ``edge_source``-shaped data from ``component_connections``.

        Edges are always stored source -> target (the convention the config
        panel already uses). ``xs`` / ``ys`` are placeholders; they are
        recomputed by ``_rebuild_all_ports`` / ``_rebuild_edges`` after load,
        which also drops any edge whose port index exceeds the node's count.
        """
        edges_data = {column: [] for column in _EDGE_COLUMNS}

        for connection in connections:
            if not isinstance(connection, dict):
                continue
            source = _parse_endpoint(connection.get("source"))
            target = _parse_endpoint(connection.get("target"))
            if source is None or target is None:
                _LOGGER.warning("Skipping malformed connection %r.", connection)
                continue

            source_name, source_label = source
            target_name, target_label = target
            if source_name not in name_to_index or target_name not in name_to_index:
                _LOGGER.warning(
                    "Skipping connection %s -> %s: unknown component.", source_name, target_name
                )
                continue

            source_index = name_to_index[source_name]
            target_index = name_to_index[target_name]
            icon_config = ICONS_CONFIG.get(nodes_data["icon_type"][source_index], {})

            edges_data["xs"].append([0.0, 0.0])
            edges_data["ys"].append([0.0, 0.0])
            edges_data["color"].append(icon_config.get("source_color") or DEFAULT_COLOR)
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


def _parse_endpoint(endpoint) -> tuple[str, str] | None:
    """
    Convert a YAML connection endpoint into ``(component_name, port_label)``.

    * ``[name, N]`` (YAML flow list) -> ``(name, "N")``
    * ``"[name, N]"`` (same, but quoted) -> ``(name, "N")``
    * ``name`` -> ``(name, "1")``

    Port labels are strings to match ``source_port_source["label"]``.
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


def _hierarchy_layers(graph: nx.DiGraph, node_types: dict, propulsor_names: set) -> dict:
    """
    Assign every node a layer index (0 = top of the canvas).

    Mirrors the viewer's ``GraphBuilder._get_hierarchy_layers``: layers are
    the graph distance from energy storage (tanks / batteries), flipped so
    storage ends up at the bottom and propulsors at the top. If that does not
    reach every node (e.g. no storage component yet, or a disconnected part
    of a work-in-progress design), the distance from the propulsors is used
    instead when it covers more nodes. Nodes reached by neither go to an
    extra bottom layer so the layout always covers the whole graph.
    """
    undirected = graph.to_undirected()

    energy_nodes = [
        name
        for name, node_type in node_types.items()
        if any(keyword in node_type for keyword in _ENERGY_STORAGE_KEYWORDS)
    ]
    from_energy = {}
    if energy_nodes:
        distance = nx.multi_source_dijkstra_path_length(undirected, energy_nodes)
        max_distance = max(distance.values())
        from_energy = {name: max_distance - d for name, d in distance.items()}

    from_propulsor = {}
    if propulsor_names and len(from_energy) < graph.number_of_nodes():
        from_propulsor = dict(nx.multi_source_dijkstra_path_length(undirected, propulsor_names))

    layers = from_energy if len(from_energy) >= len(from_propulsor) else from_propulsor

    unreached = [name for name in graph.nodes if name not in layers]
    if unreached:
        extra_layer = max(layers.values()) + 1 if layers else 0
        for name in unreached:
            layers[name] = extra_layer

    return {name: int(layer) for name, layer in layers.items()}


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
        return max(CANVAS_FIT_NODE_MARGIN, _LAYOUT_MARGIN_PX[side] * per_px)

    # Rounded inwards to the 0.1 data-unit precision of saved positions.
    x_min = math.ceil((canvas_area["x0"] + inset("left")) * 10) / 10
    x_max = math.floor((canvas_area["x1"] - inset("right")) * 10) / 10
    y_min = math.ceil((canvas_area["y0"] + inset("bottom")) * 10) / 10
    y_max = math.floor((canvas_area["y1"] - inset("top")) * 10) / 10
    # Degenerate (tiny window): collapse onto the centre instead of inverting.
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
    """
    span = max(values) - min(values)
    step = min(step_max, (high - low) / span) if span else 0.0
    middle = (max(values) + min(values)) / 2
    centre = (low + high) / 2
    return [
        min(high, max(low, round(centre + (float(value) - middle) * step, 1))) for value in values
    ]


def _hierarchical_canvas_positions(
    nodes_data: dict, edges_data: dict, propulsor_names: set, area: tuple
) -> dict:
    """
    Run :class:`HierarchicalLayout` on the loaded design and spread the result
    over *area* (see :func:`_layout_area`).

    :return: ``{component_name: (x, y)}`` for every node.
    """
    names = list(nodes_data["name"])
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    for start, end in zip(edges_data["starting_node_index"], edges_data["ending_node_index"]):
        graph.add_edge(names[start], names[end])

    node_types = dict(zip(names, nodes_data["node_type"]))
    layers = _hierarchy_layers(graph, node_types, propulsor_names)

    # The viewer's layout works in abstract units: x = slot inside a layer
    # (centred on 0, step 1), y = layer row (top row has the largest y in "TB").
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
    return {name: (x, y) for name, x, y in zip(layout_names, xs, ys)}


def _grid_positions(count: int, area: tuple) -> list[tuple[float, float]]:
    """
    Evenly spaced positions over *area*, row by row from the top-left.

    The number of columns follows the area's aspect ratio (no fixed limit),
    so the grid fills wide and tall canvases alike.
    """
    if count <= 0:
        return []
    x_min, x_max, y_min, y_max = area
    width, height = x_max - x_min, y_max - y_min
    aspect = width / height if height > 0 else 1.0
    columns = min(count, max(1, round(math.sqrt(count * aspect))))
    rows = math.ceil(count / columns)

    column_xs = _spread(list(range(columns)), x_min, x_max, _LAYOUT_STEP_MAX_FRACTION_X * width)
    row_ys = _spread(
        [-row for row in range(rows)], y_min, y_max, _LAYOUT_STEP_MAX_FRACTION_Y * height
    )
    return [(column_xs[i % columns], row_ys[i // columns]) for i in range(count)]