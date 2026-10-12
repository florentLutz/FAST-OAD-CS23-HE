# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Shared state, layout constants, and port-geometry utilities for the powertrain builder.

:class:`BuilderState` is the single source of truth for every Bokeh widget and
data source used by the builder. All other modules receive it by reference so
they all operate on the same objects.

:func:`compute_ports` is the geometric helper that places port balls evenly
around a circular node icon.

:func:`copy_column_data` is the small helper every mutating handler method
uses to take an independent, per-column copy of a ``ColumnDataSource.data``
dict before mutating and reassigning it (see its docstring for why the copy
step itself is required, not just a style preference).
"""

from dataclasses import dataclass, field

import bokeh.models as bkmodel

from fastga_he.gui.power_train_network_viewer import (
    ELECTRICITY_CURRENT_COLOR_CODE,
    FUEL_FLOW_COLOR_CODE,
    ICON_FOLDER_PATH,
    ICONS_CONFIG,
    MECHANICAL_POWER_COLOR_CODE,
)

# ============================================================================
# Layout constants
# ============================================================================

PALETTE_WIDTH = 300
ROW_HEIGHT = 52
ICON_SIZE = 45

# Port appearance
PORT_RADIUS = 8  # Radius of each port ball in data-units (approximately pixels).
NODE_RADIUS = ICON_SIZE / 2  # Half icon size used as the orbit base.

# Canvas "world": the design area that is always kept fully visible, in data
# units. The canvas itself is sized to the browser window at runtime and its
# view is fitted (uniform x / y scale) so this whole area — plus every placed
# component — is on screen. Positions saved in YAML files use these units
CANVAS_WORLD_X_RANGE = (0, 800)
CANVAS_WORLD_Y_RANGE = (0, 950)
# Horizontal space kept free right of the canvas for the config panel (its
# 380 px content + padding / gaps), so the page never scrolls sideways when a
# node is selected and the panel appears
CONFIG_PANEL_RESERVED_WIDTH = 425
# Approximate x (px) of the canvas' left edge: toolbox width + its padding / gaps
CANVAS_LEFT_OFFSET = PALETTE_WIDTH + 35

# Placement shared by the floating dialogs (startup, "final version?",
# unsaved changes): centred over the canvas area at any window size, instead
# of fixed pixel offsets tuned for one screen
DIALOG_POSITION_STYLES = {
    "position": "absolute",
    "left": f"calc(50% + {(CANVAS_LEFT_OFFSET - CONFIG_PANEL_RESERVED_WIDTH) // 2}px)",
    "top": "50%",
    "transform": "translate(-50%, -50%)",
}
# Full look of those dialogs: a dark rounded panel above the canvas (and below
# nothing: z-index 100 is the top layer of the page)
DIALOG_STYLES = {
    "background": "rgba(30,30,40,0.97)",
    "border": "2px solid #444",
    "border-radius": "16px",
    "padding": "10px 40px 30px 40px",
    **DIALOG_POSITION_STYLES,
    "z-index": "100",
    "box-shadow": "0 8px 32px rgba(0,0,0,0.6)",
}

# Status line under the toolbox when no component is armed and delete mode is off
STATUS_IDLE_HTML = "<i style='color:#aaa;font-size:14pt'>Select a component</i>"

# Data units kept between the outermost components and the edge of the fitted
# view. Also the minimum edge margin of generated layouts, so loading a design
# laid out automatically does not make the view zoom out
CANVAS_FIT_NODE_MARGIN = 60.0

# Smallest canvas size (px) used when the window is very small
CANVAS_MIN_WIDTH = 480
CANVAS_MIN_HEIGHT = 420

# Node icon size (canvas data units) and the range offered by the size slider
# in the toolbox header. Everything drawn around a node (hover ring, selection
# ring, name label, port orbit) scales by  current_size / DEFAULT_NODE_ICON_SIZE
DEFAULT_NODE_ICON_SIZE = 50
NODE_ICON_SIZE_MIN = 20
NODE_ICON_SIZE_MAX = 80
# Screen-pixel sizes of the rings / label offset at the default icon size
HOVER_RING_BASE_SIZE = ICON_SIZE + 10
SELECTION_RING_BASE_SIZE = ICON_SIZE + 14
NODE_LABEL_BASE_OFFSET = -ICON_SIZE // 2 - 2

# Button appearance
BUTTON_DEFAULT_COLOR_TYPE = "light"  # Unselected button state.
BUTTON_SELECTED_COLOR_TYPE = "primary"  # Selected button state (blue highlight).

NODE_SELECT_COLOR = "#FFD700"  # Gold colour for the selected-node overlay (semi-transparent).
_EMPTY = ""
# Sentinel stored in placed_nodes_source["symmetry_node_index"] for a node
# that has no symmetry peer selected
_NO_SYMMETRY_PEER = -1

# Default port count of a variable-port component (one with "number_of_*"
# attributes) on each side. It also marks such a component: its port counts
# are editable in the config panel and saved as number_of_* options
VARIABLE_PORT_DEFAULT = 3

# Save modes sent by the "final version?" overlay through browse_save_trigger
# (value = "<mode>:<timestamp>"): a final file has no canvas_layout section
SAVE_MODE_FINAL = "final"
SAVE_MODE_EDITABLE = "editable"
SAVE_MODE_CANCEL = "cancel"


# ============================================================================
# ColumnDataSource copy helper
# ============================================================================


def copy_column_data(data: dict) -> dict:
    """
    Return an independent, per-column-list copy of a ``ColumnDataSource.data`` dict.

    Bokeh's ``ColumnDataSource`` only detects changes on whole-column or
    whole-``.data`` reassignment — it does **not** track in-place mutation of
    a column's existing list (``.pop()``, ``.append()``, index assignment).
    Mutating ``source.data['x']`` directly would silently desync the browser
    from the server: the Python-side data would be correct, but the glyph on
    screen wouldn't update. Every mutating operation in this codebase
    therefore follows the same copy -> mutate copy -> single
    ``source.data = new_data`` reassignment pattern, and this helper is the
    shared "copy" half of it.

    :param data: The live ``.data`` dict of a ``ColumnDataSource``.
    :return: A new dict with the same keys, each mapped to a new ``list``
        containing the same values (a shallow copy per column).
    """
    return {key: list(values) for key, values in data.items()}


# ============================================================================
# Column layouts of the canvas data sources
# ============================================================================

# Columns of placed_nodes_source, one row per placed component. "options" is a
# JSON-encoded dict of option_name -> value, "n_sources" / "n_targets" the
# current port counts, "symmetry_name" / "symmetry_node_index" the
# symmetry peer (_EMPTY / _NO_SYMMETRY_PEER when there is none), and
# "source_port_type" / "target_port_type" the energy type of each port side of
# an unknown component (a PORT_TYPE_COLORS key; _EMPTY for known components,
# whose port colours come from their icon)
NODE_COLUMNS = (
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
    "source_port_type",
    "target_port_type",
)

# ============================================================================
# Unknown component
# ============================================================================

# A component the builder has no definition for (not in KNOWN_COMPONENTS):
# placed from the toolbox or loaded from a file, and configured by hand in the
# config panel. It is recognised by its icon, so whatever the user types as
# its component type never makes it pick up a known type's options or ports
UNKNOWN_ICON_KEY = "unknown"
UNKNOWN_COMPONENT_TYPE = "unknown"  # default type of a new unknown component
# Toolbox tabs the unknown component is listed in, always last
UNKNOWN_COMPONENT_TABS = ("connector", "propulsor", "load", "tank", "source")
# Default port counts of a new unknown component (each side configurable, 0 allowed)
UNKNOWN_DEFAULT_PORT_COUNT = 1

# Energy type of each port side of an unknown component, chosen in the config
# panel. "undefined" ports are drawn in the gray of unknown.png and cannot be
# connected until an energy type is chosen for them
UNDEFINED_PORT_TYPE = "undefined"
UNDEFINED_PORT_COLOR = "#666666"
PORT_TYPE_COLORS = {
    UNDEFINED_PORT_TYPE: UNDEFINED_PORT_COLOR,
    "electricity": ELECTRICITY_CURRENT_COLOR_CODE,
    "mechanical": MECHANICAL_POWER_COLOR_CODE,
    "fuel": FUEL_FLOW_COLOR_CODE,
}
# (value, label) choices of the port-type selects, labelled as in the legend
PORT_TYPE_CHOICES = [
    (UNDEFINED_PORT_TYPE, "Undefined"),
    ("electricity", "Electrical Current"),
    ("mechanical", "Mechanical Power"),
    ("fuel", "Fuel Flow"),
]

# Icons of the builder: every icon of the network viewer, then the unknown
# component. Its port colours are per node (see node_port_colors), so it has
# none here
BUILDER_ICONS_CONFIG = {
    **ICONS_CONFIG,
    UNKNOWN_ICON_KEY: {
        "icon_path": ICON_FOLDER_PATH / "unknown.png",
        "source_color": None,
        "target_color": None,
    },
}


def is_unknown_node(nodes_data: dict, node_index: int) -> bool:
    """
    Whether node *node_index* is an unknown component.

    :param nodes_data: ``placed_nodes_source.data`` (or a copy of it).
    :param node_index: Zero-based index of the node.
    :return: ``True`` for an unknown component.
    """
    return nodes_data["icon_type"][node_index] == UNKNOWN_ICON_KEY


def node_port_colors(nodes_data: dict, node_index: int) -> tuple[str | None, str | None]:
    """
    Energy-type colour of the source ports and of the target ports of a node.

    A known component takes the colours of its icon; an unknown component the
    port types chosen in its config panel.

    :param nodes_data: ``placed_nodes_source.data`` (or a copy of it).
    :param node_index: Zero-based index of the node.
    :return: ``(source_color, target_color)``; ``None`` for a side of a known
        component that has no ports.
    """
    if is_unknown_node(nodes_data, node_index):
        return tuple(
            PORT_TYPE_COLORS.get(nodes_data[column][node_index], UNDEFINED_PORT_COLOR)
            for column in ("source_port_type", "target_port_type")
        )
    icon_config = BUILDER_ICONS_CONFIG.get(nodes_data["icon_type"][node_index], {})
    # None on a side the component has no ports on
    return icon_config.get("source_color"), icon_config.get("target_color")


def ports_compatible(first_color: str, second_color: str) -> bool:
    """
    Whether two ports of these colours may be connected.

    Ports connect when they carry the same, defined energy type: an undefined
    port of an unknown component connects to nothing until a type is chosen.

    :param first_color: Colour of one port.
    :param second_color: Colour of the other port.
    :return: ``True`` if they may be connected.
    """
    return first_color == second_color and UNDEFINED_PORT_COLOR not in (first_color, second_color)


# Columns of edge_source, one row per connection between two ports
EDGE_COLUMNS = (
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

# Columns of source_port_source and target_port_source, one row per port ball
PORT_COLUMNS = (
    "x",
    "y",
    "color",
    "label",
    "node_index",
    "node_name",
    "node_type",
    "fill_alpha",
    "line_alpha",
    "connected",
)


def empty_columns(column_names: tuple) -> dict:
    """
    Return ``ColumnDataSource.data`` with every column in *column_names* empty.

    :param column_names: One of :data:`NODE_COLUMNS`, :data:`EDGE_COLUMNS` or
        :data:`PORT_COLUMNS`.
    :return: Dictionary mapping each column name to a new empty list.
    """
    return {column_name: [] for column_name in column_names}


def append_row(column_data: dict, column_names: tuple, row: dict) -> dict:
    """
    Return a copy of *column_data* with one more row at the end.

    :param column_data: ``ColumnDataSource.data`` (or a copy of it).
    :param column_names: Columns of the result, e.g. :data:`NODE_COLUMNS`.
    :param row: One value per column in *column_names*.
    :return: New column data; *column_data* itself is not changed.
    """
    return {
        column_name: list(column_data.get(column_name, [])) + [row[column_name]]
        for column_name in column_names
    }


def drop_row(column_data: dict, row_index: int) -> dict:
    """
    Return a copy of *column_data* without the row at *row_index*.

    :param column_data: ``ColumnDataSource.data`` (or a copy of it).
    :param row_index: Zero-based row to leave out.
    :return: New column data; *column_data* itself is not changed.
    """
    return {
        column_name: [value for position, value in enumerate(values) if position != row_index]
        for column_name, values in column_data.items()
    }


# ============================================================================
# Per-type defaults of a new component
# ============================================================================


def default_options(options_definition: dict) -> dict:
    """
    Default value of every option of a component type: the first possible value.

    :param options_definition: ``possible_options[component_type]``, i.e.
        ``{option_name: [possible values, …]}``.
    :return: ``{option_name: first value}``; options without values are left out.
    """
    return {
        option_name: option_values[0]
        for option_name, option_values in options_definition.items()
        if option_values
    }


def default_position(position_choices: list) -> str:
    """
    Default position of a component type: the first possible position, or ``""``.

    :param position_choices: ``possible_position[component_type]``.
    :return: The default position string.
    """
    return position_choices[0] if position_choices else _EMPTY


# ============================================================================
# Edge selection helpers
# ============================================================================


def select_edges(edge_data: dict, kept_edge_indices: list) -> dict:
    """
    Return a copy of *edge_data* holding only the edges at *kept_edge_indices*.

    :param edge_data: ``edge_source.data`` (or a copy of it).
    :param kept_edge_indices: Positions of the edges to keep, in order.
    :return: New edge data with the same columns.
    """
    return {
        column_name: [column_values[edge_position] for edge_position in kept_edge_indices]
        for column_name, column_values in edge_data.items()
    }


def edge_end_ports(edge_data: dict) -> list[tuple[tuple, tuple]]:
    """
    The two ports of every edge, as ``(starting_port, ending_port)`` pairs.

    Each port is a ``(kind, node_index, label)`` key, the same key the port
    data sources are indexed by. An edge starts at whichever port was clicked
    first, so the starting port can be a source or a target.

    :param edge_data: ``edge_source.data`` (or a copy of it).
    :return: One pair per edge, in edge order.
    """
    return [
        tuple(
            (
                edge_data[f"{end}_port_kind"][edge_position],
                edge_data[f"{end}_node_index"][edge_position],
                edge_data[f"{end}_port_label"][edge_position],
            )
            for end in ("starting", "ending")
        )
        for edge_position in range(len(edge_data.get("starting_node_index", [])))
    ]


def edge_indices_not_touching_node(edge_data: dict, node_index: int) -> list:
    """
    Positions of the edges that are not attached to node *node_index*.

    :param edge_data: ``edge_source.data`` (or a copy of it).
    :param node_index: Zero-based index of the node in ``placed_nodes_source``.
    :return: List of edge positions, in order.
    """
    return [
        edge_position
        for edge_position, (starting_node_index, ending_node_index) in enumerate(
            zip(
                edge_data.get("starting_node_index", []),
                edge_data.get("ending_node_index", []),
                strict=True,
            )
        )
        if node_index not in (starting_node_index, ending_node_index)
    ]


def edge_indices_not_touching_port(
    edge_data: dict, node_index: int, port_label: str, port_kind: str
) -> list:
    """
    Positions of the edges that are not attached to one port.

    :param edge_data: ``edge_source.data`` (or a copy of it).
    :param node_index: Zero-based index of the node that owns the port.
    :param port_label: Port label (``"1"``, ``"2"``, ...).
    :param port_kind: ``"source"`` or ``"target"``.
    :return: List of edge positions, in order.
    """
    port_key = (port_kind, node_index, port_label)
    return [
        edge_position
        for edge_position, end_ports in enumerate(edge_end_ports(edge_data))
        if port_key not in end_ports
    ]


# ============================================================================
# Symmetry link helpers
# ============================================================================


def link_symmetry(
    symmetry_names: list, symmetry_indices: list, node_names: list, first: int, second: int
) -> None:
    """
    Make nodes *first* and *second* each other's symmetry peer, in place.

    :param symmetry_names: ``symmetry_name`` column, edited in place.
    :param symmetry_indices: ``symmetry_node_index`` column, edited in place.
    :param node_names: ``name`` column, giving the peer names to store.
    :param first: Zero-based index of one node of the pair.
    :param second: Zero-based index of the other node.
    """
    for node_position, peer_position in ((first, second), (second, first)):
        symmetry_names[node_position] = node_names[peer_position]
        symmetry_indices[node_position] = peer_position


def unlink_symmetry(symmetry_names: list, symmetry_indices: list, node_positions) -> None:
    """
    Reset the symmetry entries at *node_positions*, in place.

    Only the given nodes are reset; their peers keep pointing at them unless
    they are listed too.

    :param symmetry_names: ``symmetry_name`` column, edited in place.
    :param symmetry_indices: ``symmetry_node_index`` column, edited in place.
    :param node_positions: Positions of the nodes to unlink.
    """
    for node_position in node_positions:
        symmetry_names[node_position] = _EMPTY
        symmetry_indices[node_position] = _NO_SYMMETRY_PEER


# ============================================================================
# Shared mutable state
# ============================================================================


@dataclass
class BuilderState:
    """
    Holds references to all Bokeh widgets and data sources of the powertrain builder.

    Passed by reference between :class:`ComponentPaletteConfigurationTableBuilder`,
    :class:`BuilderHandler`, and :class:`PowertrainBuilderLauncher` so that
    all parties share the same widget instances.

    Fields are grouped below by the part of the UI they belong to. Two
    naming conventions recur throughout and are worth knowing up front:

    * Any field ending in ``_source`` is a Bokeh ``ColumnDataSource`` and
      **must** be mutated via the copy -> mutate -> reassign pattern
      described in :func:`copy_column_data` — never in place.
    * Dicts keyed by component type (``possible_position``,
      ``possible_options``, ``default_source_count``,
      ``default_target_count``) all use the same key: the
      ``components_type`` string from ``KNOWN_COMPONENTS`` (e.g.
      ``"battery_pack"``), *not* the toolbox icon key.
    """

    # Toolbox / palette
    # One Button per entry in ICONS_CONFIG, in the same order
    buttons: list = field(default_factory=list)
    # icon_key of the toolbox button currently armed for placement, or None
    selected_component: str = field(default=None)
    # icon_key -> count of components of that icon placed so far, used to
    # generate unique default names ("battery_pack_1", "battery_pack_2", …)
    placed_counter: dict = field(default_factory=dict)
    delete_button: bkmodel.Button = field(default=None)
    delete_mode: bool = field(default=False)
    save_button: bkmodel.Button = field(default=None)
    end_session_button: bkmodel.Button = field(default=None)
    status_div: bkmodel.Div = field(default=None)
    # Slider next to the "Components" title that resizes every node icon
    # (see PlacementMixin._on_node_size_slider_change)
    node_size_slider: bkmodel.Slider = field(default=None)
    # Canvas renderers / annotation resized together with the icons; set by
    # the launcher once the canvas is built (None in headless use / tests)
    hover_ring_renderer: object = field(default=None)
    selection_ring_renderer: object = field(default=None)
    node_label_set: object = field(default=None)

    # Window-fitted canvas
    # Data units per screen pixel of the current fitted view (1.0 = the
    # original fixed 800 x 950 canvas). Sent from the browser through
    # view_scale_input; used to keep screen-pixel glyphs (hover / selection
    # rings, name labels) matched to the icons, which are sized in data units
    view_scale: float = field(default=1.0)
    # Visible canvas area of the current browser window, in data units, when
    # showing an empty design: {"x0", "x1", "y0", "y1", "per_px"}. Reported by
    # the browser with view_scale; None until then (headless use / tests), in
    # which case the 800 x 950 world at 1 unit / px is assumed. Automatic
    # layouts (YAML without canvas_layout) are spread over this area
    canvas_area: dict = field(default=None)
    # Hidden TextInput written by the browser-side fit callback (JS -> Python),
    # as JSON: {"k": view_scale, "area": [x0, x1, y0, y1], "area_k": per_px}
    view_scale_input: bkmodel.TextInput = field(default=None)
    # Hidden TextInput flipped by Python after a load / New Design so the
    # browser re-fits the view to the (new) components (Python -> JS)
    fit_view_trigger: bkmodel.TextInput = field(default=None)

    # Canvas node data
    # One row per placed component: position, icon URL, type, options JSON,
    # port counts, and symmetry link. Drives every per-node glyph of the
    # canvas: the icons, the hover ring and its tooltip, and the name labels
    placed_nodes_source: bkmodel.ColumnDataSource = field(default=None)
    # Semi-transparent highlight glyph drawn under the currently selected node
    selected_node_overlay_source: bkmodel.ColumnDataSource = field(default=None)
    # Row index into placed_nodes_source of the node currently open in the
    # config panel, or None if nothing is selected
    selected_node_index: int = field(default=None)

    # Ports
    # One row per port ball currently drawn on the canvas, keyed by
    # node_index (matching a row in placed_nodes_source) + label. Rebuilt
    # from scratch by PortEdgeMixin._rebuild_all_ports() on every node
    # add / move / delete / port-count change
    source_port_source: bkmodel.ColumnDataSource = field(default=None)
    target_port_source: bkmodel.ColumnDataSource = field(default=None)
    # Spinners for editable port counts; only shown for components whose
    # default source/target count is VARIABLE_PORT_DEFAULT (the "variable
    # port count" rule in power_train_builder_metadata._build_port_count_defaults),
    # and for unknown components
    source_count_spinner: bkmodel.Spinner = field(default=None)
    target_count_spinner: bkmodel.Spinner = field(default=None)
    # Energy type of the source / target ports of an unknown component (shown
    # for unknown components only). Unlike the other panel fields, a change
    # applies at once, without Apply
    source_port_type_select: bkmodel.Select = field(default=None)
    target_port_type_select: bkmodel.Select = field(default=None)
    # Column section wrapping the spinners and port-type selects above;
    # visible only when the selected node's ports are editable
    port_count_section: object = field(default=None)
    # First port clicked in the two-click canvas port-connection workflow
    # (see PortEdgeMixin._handle_port_tap); None if no click is pending
    awaiting_second_port_click: dict = field(default=None)
    # Single-point glyph source that highlights the port stored in
    # awaiting_second_port_click while the second click is awaited
    awaiting_second_port_click_source: bkmodel.ColumnDataSource = field(default=None)

    # Edges (permanent connections)
    # One row per committed connection between a source port and a target
    # port. Unlike the port sources, edges persist across
    # _rebuild_all_ports() calls (their xs/ys are recomputed in place by
    # PortEdgeMixin._rebuild_edges(), not rebuilt from scratch)
    edge_source: bkmodel.ColumnDataSource = field(default=None)
    # Snapshot of edge_source.data taken by the FIRST port-count-spinner
    # preview tick of an edit session (see
    # PortEdgeMixin._sync_edge_preview_backup). The next *real* (non-preview)
    # rebuild, or real edit, restores this and clears it, so an
    # edge hidden while previewing a reduced port count comes back exactly
    # as it was if the user never clicks Apply. None when no preview is
    # currently in progress
    edge_preview_backup: dict = field(default=None)

    # Config / edit panel
    # The whole config panel column (name, type, position, options,
    # connections, symmetry, Apply button) — toggled visible when a node is
    # selected on the canvas, hidden otherwise
    config_panel_column: object = field(default=None)
    name_input: bkmodel.TextInput = field(default=None)
    type_select: bkmodel.Select = field(default=None)
    position_select: bkmodel.Select = field(default=None)
    symmetry_select: bkmodel.Select = field(default=None)
    apply_button: bkmodel.Button = field(default=None)
    # Free-text counterparts of type_select, position_select and
    # symmetry_select, shown instead of them for an unknown component (which
    # has no list of possible values); committed on Apply like the selects
    unknown_type_input: bkmodel.TextInput = field(default=None)
    unknown_position_input: bkmodel.TextInput = field(default=None)
    unknown_symmetry_input: bkmodel.TextInput = field(default=None)

    # Options sub-section: options_table is the static wrapper (section
    # header + options_rows_column); options_rows_column is the dynamic
    # Column whose children (one TextInput label + Select value per option)
    # are rebuilt by ConfigPanelMixin._refresh_options_table() every time the
    # selected node's type changes. Apply reads the values straight from
    # those Select widgets. For an unknown component both the option name and
    # its value are TextInputs, and add_option_button (shown for unknown
    # components only) adds an empty row
    options_table: object = field(default=None)
    options_rows_column: object = field(default=None)
    add_option_button: bkmodel.Button = field(default=None)

    # Connections sub-section: the dynamic Column of per-port rows rebuilt by
    # ConfigPanelMixin._refresh_connections_table() for the selected node.
    # There is no backing CDS here — rows are read directly from edge_source
    # and panel_pending_connections
    connections_rows_column: object = field(default=None)

    # Dashed preview edges drawn while a connection is queued in the config
    # panel but not yet committed via Apply (distinct from the two-click
    # canvas workflow above, which writes straight to edge_source)
    panel_preview_edge_source: bkmodel.ColumnDataSource = field(default=None)
    # (starting_port_dict, ending_port_dict) pairs queued in the config
    # panel; flushed into edge_source, and this list cleared, by
    # ConfigPanelMixin.apply_node_configurations()
    panel_pending_connections: list = field(default_factory=list)

    # Per-component-type metadata (built once at startup)
    # All four dicts below are keyed by components_type string (see class
    # docstring), built once by ComponentPaletteBuilder.build() from
    # KNOWN_COMPONENTS / each component's constants.py, and treated as
    # read-only lookup tables afterwards
    default_source_count: dict = field(default_factory=dict)
    """components_type -> default number of source ports for that type."""
    default_target_count: dict = field(default_factory=dict)
    """components_type -> default number of target ports for that type."""
    possible_position: dict = field(default_factory=dict)
    """components_type -> ordered list of valid POSSIBLE_POSITION strings (first = default)."""
    possible_options: dict = field(default_factory=dict)
    """components_type -> {option_name: [possible values, …]} (first value = default)."""
    # icon_key -> list of components_type strings that use that icon; index
    # [0] is treated as "the" default component type for a freshly placed
    # icon (see PlacementMixin._place_new_node). Keyed the OPPOSITE way round
    # from the four dicts above — read the name carefully
    icon_to_component_types: dict = field(default_factory=dict)

    # Save / load
    # "Is this the final version?" overlay — shown by the Save button (and by
    # Save & Exit). "Final Version" saves without the GUI-only canvas_layout
    # section; "Edit Later" keeps it (see IOMixin._on_browse_save)
    save_version_overlay: object = field(default=None)
    # Path of the powertrain watcher CSV, written as watcher_file_path in the
    # saved YAML (empty: not written). Edited through watcher_path_input or its
    # Browse button, and filled from the file on load (see IOMixin)
    watcher_file_path: str = field(default="")
    # Text field showing / editing watcher_file_path in the "final version?" overlay
    watcher_path_input: bkmodel.TextInput = field(default=None)
    # Toggle triggers: JS changes the value of these hidden TextInputs to fire
    # the corresponding Python on_change callback (used to bridge JS-driven
    # UI actions, like a native OS file dialog, back into Python)
    browse_load_trigger: bkmodel.TextInput = field(default=None)
    browse_save_trigger: bkmodel.TextInput = field(default=None)
    browse_watcher_trigger: bkmodel.TextInput = field(default=None)

    # Startup overlay
    new_design_button: bkmodel.Button = field(default=None)
    load_design_button: bkmodel.Button = field(default=None)
    # The whole startup overlay column — hidden (not just its two buttons) on dismiss
    startup_overlay: object = field(default=None)

    # End-session flow
    # "Unsaved changes" confirmation overlay — shown by End Session when
    # save_button is yellow (i.e. there are unsaved edits)
    unsaved_exit_overlay: object = field(default=None)
    # Toggle trigger used by the overlay's "Save & Exit" button to open the
    # save flow before exiting
    end_session_save_trigger: bkmodel.TextInput = field(default=None)
    # Toggle trigger fired by JS when it is safe to actually stop the server.
    # Using a trigger (instead of on_click) means Python only runs
    # _end_session when JS has already decided it is appropriate — the
    # Python on_click callback on end_session_button is never registered,
    # avoiding unconditional firing
    end_session_trigger: bkmodel.TextInput = field(default=None)
    # Trigger flipped by Python to make the browser call window.close().
    # Uses js_on_change so the JS callback runs when the server pushes the value
    close_window_trigger: bkmodel.TextInput = field(default=None)

    # Undo / redo
    # EditHistoryStack instance (set by BuilderHandler.__init__)
    edit_history_stack: object = field(default=None)
    # Bokeh Button widgets for Undo and Redo, placed above the canvas
    undo_button: bkmodel.Button = field(default=None)
    redo_button: bkmodel.Button = field(default=None)
