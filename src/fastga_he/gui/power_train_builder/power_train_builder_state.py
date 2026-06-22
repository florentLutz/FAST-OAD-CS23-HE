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
"""

import math
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


# ============================================================================
# Port placement
# ============================================================================


def compute_ports(
    center_x: float,
    center_y: float,
    node_radius: float = NODE_RADIUS,
    port_radius: float = PORT_RADIUS,
    number_of_outputs: int = 0,
    number_of_inputs: int = 0,
    spread_angle_degrees: float = 120.0,
    gap_pixels: float = 20.0,
) -> dict:
    """
    Place port balls evenly around a circular node.

    Output ports fan around the top half (centred at −90°);
    input ports fan around the bottom half (centred at +90°).

    :param center_x: Node centre x in canvas coordinates.
    :param center_y: Node centre y in canvas coordinates.
    :param node_radius: Radius of the node circle in data-units / pixels.
    :param port_radius: Radius of each port ball in data-units / pixels.
    :param number_of_outputs: Number of output (source) ports.
    :param number_of_inputs: Number of input (target) ports.
    :param spread_angle_degrees: Angular fan for each half in degrees (default 120).
    :param gap_pixels: Clearance between the node edge and each port centre in pixels.
    :return: A dictionary ``{"outputs": [Port, …], "inputs": [Port, …]}`` where
        each Port is a dict with keys ``index``, ``kind``, ``angle_deg``, ``x``,
        and ``y``.
    """
    orbit_radius = node_radius + port_radius + gap_pixels

    def _angle_for_port(port_index: int, total_port_count: int, centre_angle: float) -> float:
        if total_port_count == 1:
            return centre_angle
        start_angle = centre_angle - spread_angle_degrees / 2
        angle_step = spread_angle_degrees / (total_port_count - 1)
        return start_angle + port_index * angle_step

    def _coordinates_from_angle(angle_degrees: float):
        angle_radians = math.radians(angle_degrees)
        return (
            center_x + orbit_radius * math.cos(angle_radians),
            center_y + orbit_radius * math.sin(angle_radians),
        )

    output_ports = []
    for port_index in range(number_of_outputs):
        angle_degrees = _angle_for_port(port_index, number_of_outputs, -90.0)
        port_x, port_y = _coordinates_from_angle(angle_degrees)
        output_ports.append(
            {
                "index": port_index,
                "kind": "output",
                "angle_deg": angle_degrees,
                "x": port_x,
                "y": port_y,
            }
        )

    input_ports = []
    for port_index in range(number_of_inputs):
        angle_degrees = _angle_for_port(port_index, number_of_inputs, 90.0)
        port_x, port_y = _coordinates_from_angle(angle_degrees)
        input_ports.append(
            {
                "index": port_index,
                "kind": "input",
                "angle_deg": angle_degrees,
                "x": port_x,
                "y": port_y,
            }
        )

    return {"outputs": output_ports, "inputs": input_ports}


# ============================================================================
# Shared mutable state
# ============================================================================


@dataclass
class BuilderState:
    """
    Holds references to all Bokeh widgets of the powertrain builder.

    Passed by reference between :class:`ComponentPaletteConfigurationTableBuilder`,
    :class:`PlacementHandler`, and :class:`PowertrainBuilderLauncher` so that
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
    connections_source: bkmodel.ColumnDataSource = field(default=None)
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
    possible_position: dict = field(default_factory=dict)  # component_type → list of positions
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
