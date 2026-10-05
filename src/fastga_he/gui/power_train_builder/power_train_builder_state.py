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

# Button appearance
BUTTON_DEFAULT_COLOR_TYPE = "light"  # Unselected button state.
BUTTON_SELECTED_COLOR_TYPE = "primary"  # Selected button state (blue highlight).

NODE_SELECT_COLOR = "#FFD700"  # Gold colour for the selected-node overlay (semi-transparent).
_EMPTY = ""
# Sentinel stored in placed_nodes_source["symmetry_node_index"] for a node
# that has no symmetry peer selected.
_NO_SYMMETRY_PEER = -1


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

    # ── Toolbox / palette ───────────────────────────────────────────────────
    # One Button per entry in ICONS_CONFIG, in the same order.
    buttons: list = field(default_factory=list)
    # icon_key of the toolbox button currently armed for placement, or None.
    selected_component: str = field(default=None)
    # icon_key -> count of components of that icon placed so far, used to
    # generate unique default names ("battery_pack_1", "battery_pack_2", …).
    placed_counter: dict = field(default_factory=dict)
    delete_button: bkmodel.Button = field(default=None)
    delete_mode: bool = field(default=False)
    save_button: bkmodel.Button = field(default=None)
    end_session_button: bkmodel.Button = field(default=None)
    status_div: bkmodel.Div = field(default=None)

    # ── Canvas node data ─────────────────────────────────────────────────────
    # One row per placed component: position, icon URL, type, options JSON,
    # port counts, and symmetry link. The primary data source driving the
    # canvas's image_url glyph.
    placed_nodes_source: bkmodel.ColumnDataSource = field(default=None)
    # Invisible glyph source used only to drive Bokeh HoverTool tooltips
    # (name / node_type) over each placed node; kept in lockstep with
    # placed_nodes_source but is not itself rendered.
    hover_source: bkmodel.ColumnDataSource = field(default=None)
    # Semi-transparent highlight glyph drawn under the currently selected node.
    selected_node_overlay_source: bkmodel.ColumnDataSource = field(default=None)
    # Row index into placed_nodes_source of the node currently open in the
    # config panel, or None if nothing is selected.
    selected_node_index: int = field(default=None)

    # ── Ports ────────────────────────────────────────────────────────────────
    # One row per port ball currently drawn on the canvas, keyed by
    # node_index (matching a row in placed_nodes_source) + label. Rebuilt
    # from scratch by PortEdgeMixin._rebuild_all_ports() on every node
    # add / move / delete / port-count change.
    source_port_source: bkmodel.ColumnDataSource = field(default=None)
    target_port_source: bkmodel.ColumnDataSource = field(default=None)
    # Spinners for editable port counts; only shown for components whose
    # default source/target count is 3 (the "variable port count" rule in
    # power_train_builder_metadata._build_port_count_defaults).
    source_count_spinner: bkmodel.Spinner = field(default=None)
    target_count_spinner: bkmodel.Spinner = field(default=None)
    # Column section wrapping the two spinners above; visible only when the
    # selected node's ports are editable.
    port_count_section: object = field(default=None)
    # First port clicked in the two-click canvas port-connection workflow
    # (see PortEdgeMixin._handle_port_tap); None if no click is pending.
    awaiting_second_port_click: dict = field(default=None)
    # Single-point glyph source that highlights the port stored in
    # awaiting_second_port_click while the second click is awaited.
    awaiting_second_port_click_source: bkmodel.ColumnDataSource = field(default=None)

    # ── Edges (permanent connections) ───────────────────────────────────────
    # One row per committed connection between a source port and a target
    # port. Unlike the port sources, edges persist across
    # _rebuild_all_ports() calls (their xs/ys are recomputed in place by
    # PortEdgeMixin._rebuild_edges(), not rebuilt from scratch).
    edge_source: bkmodel.ColumnDataSource = field(default=None)
    # Snapshot of edge_source.data taken by the FIRST port-count-spinner
    # preview tick of an edit session (see
    # PortEdgeMixin._rebuild_all_ports's override_counts parameter). The
    # next *real* (non-preview) rebuild restores this and clears it, so an
    # edge hidden while previewing a reduced port count comes back exactly
    # as it was if the user never clicks Apply. None when no preview is
    # currently in progress.
    edge_preview_backup: dict = field(default=None)

    # ── Config / edit panel ──────────────────────────────────────────────────
    # The whole config panel column (name, type, position, options,
    # connections, symmetry, Apply button) — toggled visible when a node is
    # selected on the canvas, hidden otherwise.
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
    # sync with those widgets so their current values can be read back in bulk.
    options_table: object = field(default=None)
    options_rows_column: object = field(default=None)
    options_source: bkmodel.ColumnDataSource = field(default=None)

    # Connections sub-section: connections_table_widget is the static
    # wrapper column; connections_rows_column is the dynamic Column of
    # per-connection rows rebuilt by
    # ConfigPanelMixin._refresh_connections_table() for the selected node.
    # There is no backing CDS here — rows are read directly from edge_source.
    connections_table_widget: bkmodel.ColumnDataSource = field(default=None)
    connections_rows_column: object = field(default=None)

    # Dashed preview edges drawn while a connection is queued in the config
    # panel but not yet committed via Apply (distinct from the two-click
    # canvas workflow above, which writes straight to edge_source).
    panel_preview_edge_source: bkmodel.ColumnDataSource = field(default=None)
    # (starting_port_dict, ending_port_dict) pairs queued in the config
    # panel; flushed into edge_source, and this list cleared, by
    # ConfigPanelMixin.apply_node_configurations().
    panel_pending_connections: list = field(default_factory=list)

    # ── Per-component-type metadata (built once at startup) ─────────────────
    # All four dicts below are keyed by components_type string (see class
    # docstring), built once by ComponentPaletteBuilder.build() from
    # KNOWN_COMPONENTS / each component's constants.py, and treated as
    # read-only lookup tables afterwards.
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
    # from the four dicts above — read the name carefully.
    icon_to_component_types: dict = field(default_factory=dict)

    # ── Save / load ──────────────────────────────────────────────────────────
    # "Is this the final version?" overlay — shown by the Save button (and by
    # Save & Exit). "Final Version" saves without the GUI-only canvas_layout
    # section; "Edit Later" keeps it (see IOMixin._on_browse_save).
    save_version_overlay: object = field(default=None)
    # watcher_file_path found in the last loaded YAML (empty if none). There
    # is no UI for it; it is only carried over so re-saving a loaded design
    # does not silently drop a path the user set by hand in the file.
    watcher_file_path: str = field(default="")
    # Toggle triggers: JS changes the value of these hidden TextInputs to fire
    # the corresponding Python on_change callback (used to bridge JS-driven
    # UI actions, like a native OS file dialog, back into Python).
    browse_load_trigger: bkmodel.TextInput = field(default=None)
    browse_save_trigger: bkmodel.TextInput = field(default=None)

    # ── Startup overlay ──────────────────────────────────────────────────────
    new_design_button: bkmodel.Button = field(default=None)
    load_design_button: bkmodel.Button = field(default=None)
    # The whole startup overlay column — hidden (not just its two buttons) on dismiss.
    startup_overlay: object = field(default=None)

    # ── End-session flow ─────────────────────────────────────────────────────
    # "Unsaved changes" confirmation overlay — shown by End Session when
    # save_button is yellow (i.e. there are unsaved edits).
    unsaved_exit_overlay: object = field(default=None)
    # Toggle trigger used by the overlay's "Save & Exit" button to open the
    # save flow before exiting.
    end_session_save_trigger: bkmodel.TextInput = field(default=None)
    # Toggle trigger fired by JS when it is safe to actually stop the server.
    # Using a trigger (instead of on_click) means Python only runs
    # _end_session when JS has already decided it is appropriate — the
    # Python on_click callback on end_session_button is never registered,
    # avoiding unconditional firing.
    end_session_trigger: bkmodel.TextInput = field(default=None)
    # Trigger flipped by Python to make the browser call window.close().
    # Uses js_on_change so the JS callback runs when the server pushes the value.
    close_window_trigger: bkmodel.TextInput = field(default=None)

    # ── Undo / redo ──────────────────────────────────────────────────────────
    # EditHistoryStack instance (set by BuilderHandler.__init__).
    edit_history_stack: object = field(default=None)
    # Bokeh Button widgets for Undo and Redo, placed above the canvas.
    undo_button: bkmodel.Button = field(default=None)
    redo_button: bkmodel.Button = field(default=None)
