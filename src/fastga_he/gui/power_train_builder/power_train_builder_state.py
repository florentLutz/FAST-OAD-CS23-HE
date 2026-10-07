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
# current port counts, and "symmetry_name" / "symmetry_node_index" the
# symmetry peer (_EMPTY / _NO_SYMMETRY_PEER when there is none)
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
)

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

# Columns of hover_source, the node subset shown in the hover tooltip
HOVER_COLUMNS = ("x", "y", "name", "node_type")


def empty_columns(column_names: tuple) -> dict:
    """
    Return ``ColumnDataSource.data`` with every column in *column_names* empty.

    :param column_names: One of :data:`NODE_COLUMNS`, :data:`EDGE_COLUMNS`,
        :data:`PORT_COLUMNS` or :data:`HOVER_COLUMNS`.
    :return: Dictionary mapping each column name to a new empty list.
    """
    return {column_name: [] for column_name in column_names}


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
    port_key = (node_index, port_label, port_kind)
    return [
        edge_position
        for edge_position in range(len(edge_data.get("starting_node_index", [])))
        if port_key
        not in (
            (
                edge_data["starting_node_index"][edge_position],
                edge_data["starting_port_label"][edge_position],
                edge_data["starting_port_kind"][edge_position],
            ),
            (
                edge_data["ending_node_index"][edge_position],
                edge_data["ending_port_label"][edge_position],
                edge_data["ending_port_kind"][edge_position],
            ),
        )
    ]


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
    # port counts, and symmetry link. The primary data source driving the
    # canvas's image_url glyph
    placed_nodes_source: bkmodel.ColumnDataSource = field(default=None)
    # Invisible glyph source used only to drive Bokeh HoverTool tooltips
    # (name / node_type) over each placed node; kept in lockstep with
    # placed_nodes_source but is not itself rendered
    hover_source: bkmodel.ColumnDataSource = field(default=None)
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
    # default source/target count is 3 (the "variable port count" rule in
    # power_train_builder_metadata._build_port_count_defaults)
    source_count_spinner: bkmodel.Spinner = field(default=None)
    target_count_spinner: bkmodel.Spinner = field(default=None)
    # Column section wrapping the two spinners above; visible only when the
    # selected node's ports are editable
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
    # PortEdgeMixin._rebuild_all_ports's override_counts parameter). The
    # next *real* (non-preview) rebuild restores this and clears it, so an
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

    # Options sub-section: options_table is the static wrapper (section
    # header + options_rows_column); options_rows_column is the dynamic
    # Column whose children (one TextInput label + Select value per option)
    # are rebuilt by ConfigPanelMixin._refresh_options_table() every time the
    # selected node's type changes; options_source is the backing CDS kept in
    # sync with those widgets so their current values can be read back in bulk
    options_table: object = field(default=None)
    options_rows_column: object = field(default=None)
    options_source: bkmodel.ColumnDataSource = field(default=None)

    # Connections sub-section: connections_table_widget is the static
    # wrapper column; connections_rows_column is the dynamic Column of
    # per-connection rows rebuilt by
    # ConfigPanelMixin._refresh_connections_table() for the selected node.
    # There is no backing CDS here — rows are read directly from edge_source
    connections_table_widget: bkmodel.ColumnDataSource = field(default=None)
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
