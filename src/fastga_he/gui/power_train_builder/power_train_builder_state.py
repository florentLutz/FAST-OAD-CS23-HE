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
    Holds references to all Bokeh widgets of the powertrain builder.

    Passed by reference between :class:`ComponentPaletteConfigurationTableBuilder`,
    :class:`BuilderHandler`, and :class:`PowertrainBuilderLauncher` so that
    all parties share the same widget instances.
    """

    buttons: list = field(default_factory=list)
    placed_nodes_source: bkmodel.ColumnDataSource = field(default=None)
    status_div: bkmodel.Div = field(default=None)
    selected_component: str = field(default=None)
    placed_counter: dict = field(default_factory=dict)
    delete_button: bkmodel.Button = field(default=None)
    save_button: bkmodel.Button = field(default=None)
    end_session_button: bkmodel.Button = field(default=None)
    delete_mode: bool = field(default=False)
    hover_source: bkmodel.ColumnDataSource = field(default=None)
    # Text input for Component ID (shown in the configurator panel).
    name_input: bkmodel.TextInput = field(default=None)
    # Select widget for Component Type (shown in the configurator panel).
    type_select: bkmodel.Select = field(default=None)
    # Select widget for Position (shown in the configurator panel).
    position_select: bkmodel.Select = field(default=None)
    # Column widget holding per-option rows (TextInput label + Select value).
    options_table: object = field(default=None)
    # Dynamic column of option rows – children rebuilt by _refresh_options_table.
    options_rows_column: object = field(default=None)
    options_source: bkmodel.ColumnDataSource = field(default=None)
    apply_button: bkmodel.Button = field(default=None)
    # Index of the currently selected canvas node (None = nothing selected).
    selected_node_index: int = field(default=None)
    # The whole configurator panel column – toggled visible / invisible.
    table_panel: object = field(default=None)
    # Source and target port data sources for each component, keyed by node
    # index in placed_nodes_source.
    source_port_source: bkmodel.ColumnDataSource = field(default=None)
    target_port_source: bkmodel.ColumnDataSource = field(default=None)
    # Spinners for editable port counts (only visible for components whose default count is 3).
    source_count_spinner: bkmodel.Spinner = field(default=None)
    target_count_spinner: bkmodel.Spinner = field(default=None)
    # Column section wrapping the spinners – toggled visible when ports are editable.
    port_count_section: object = field(default=None)
    # Selected-node alpha overlay source.
    selected_node_overlay_source: bkmodel.ColumnDataSource = field(default=None)
    # Edge line connection source.
    edge_source: bkmodel.ColumnDataSource = field(default=None)
    pending_port: dict = field(default=None)
    pending_port_source: bkmodel.ColumnDataSource = field(default=None)
    connections_table_widget: bkmodel.DataTable = field(default=None)
    # Dynamic column of connection rows – children rebuilt by _refresh_connections_table.
    connections_rows_column: object = field(default=None)
    # Temporary dashed preview edges shown before Apply is clicked.
    temp_edge_source: bkmodel.ColumnDataSource = field(default=None)
    # Pending port-pair connections waiting for Apply: list of (starting_port_dict, ending_port_dict).
    pending_connections: list = field(default_factory=list)
    # Select widget for Symmetry component (shown in the configurator panel).
    symmetry_select: bkmodel.Select = field(default=None)
    # Per-component-type default port counts, built once during ComponentPaletteBuilder.build().
    default_source_count: dict = field(default_factory=dict)
    default_target_count: dict = field(default_factory=dict)
    component_type_to_icon: dict = field(default_factory=dict)
    possible_position: dict = field(default_factory=dict)  # component_type -> list of positions
    possible_options: dict = field(default_factory=dict)
    # Watcher file path: visible TextInput in the save overlay where the user
    # types or browses to a watcher file path; its value is read by Python at save time.
    watcher_path_input: bkmodel.TextInput = field(default=None)
    # The save-options overlay column – shown by the Save button before the file dialogs open.
    save_overlay: object = field(default=None)
    # Startup overlay buttons shown on the canvas before the first action.
    new_design_button: bkmodel.Button = field(default=None)
    load_design_button: bkmodel.Button = field(default=None)
    # The whole startup overlay column – hidden (not just its buttons) on dismiss.
    startup_overlay: object = field(default=None)
    # Toggle triggers: JS flips '0' / '1' to fire Python on_change callbacks.
    browse_load_trigger: bkmodel.TextInput = field(default=None)
    browse_save_trigger: bkmodel.TextInput = field(default=None)
    browse_watcher_trigger: bkmodel.TextInput = field(default=None)
    # "Unsaved changes" confirmation overlay – shown by End Session when save_button is yellow.
    unsaved_exit_overlay: object = field(default=None)
    # Toggle trigger used by the overlay's "Save & Exit" button to open the save flow.
    end_session_save_trigger: bkmodel.TextInput = field(default=None)
    # Toggle trigger fired by JS when it is safe to actually stop the server.
    # Using a trigger (instead of on_click) means Python only runs _end_session
    # when JS has already decided it is appropriate – the Python on_click callback
    # on end_session_button is never registered, avoiding unconditional firing.
    end_session_trigger: bkmodel.TextInput = field(default=None)
    # Trigger flipped by Python to make the browser call window.close().
    # Uses js_on_change so the JS callback runs when the server pushes the value.
    close_window_trigger: bkmodel.TextInput = field(default=None)

    # ── Undo / redo ───────────────────────────────────────────────────────────
    # EditHistoryStack instance (set by BuilderHandler.__init__).
    edit_history_stack: object = field(default=None)
    # Bokeh Button widgets for Undo and Redo, placed above the canvas.
    undo_button: bkmodel.Button = field(default=None)
    redo_button: bkmodel.Button = field(default=None)
