# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Shared state, layout constants, and port-geometry utilities for the powertrain builder.

``BuilderState`` is the single source of truth for every Bokeh widget and data
source used by the builder.  All other modules receive it by reference so they
all operate on the same objects.

``compute_ports`` is the geometric helper that places port balls evenly around
a circular node icon.
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
PORT_RADIUS = 8  # radius of each port ball in data-units (≈ pixels)
NODE_RADIUS = ICON_SIZE / 2  # half icon size = orbit base

# Button appearance
BUTTON_DEFAULT_COLOR_TYPE = "light"  # unselected state
BUTTON_SELECTED_COLOR_TYPE = "primary"  # selected state (blue highlight)

NODE_SELECT_COLOR = "#FFD700"  # gold colour for selected-node overlay (semi-transparent)
_EMPTY = ""


# ============================================================================
# Port placement
# ============================================================================


def compute_ports(
    cx: float,
    cy: float,
    node_radius: float = NODE_RADIUS,
    port_radius: float = PORT_RADIUS,
    n_outputs: int = 0,
    n_inputs: int = 0,
    spread: float = 120.0,
    gap: float = 20.0,
) -> dict:
    """
    Place port balls evenly around a circular node.

    Output ports fan around the top half (centred at −90°);
    input ports fan around the bottom half (centred at +90°).

    :param cx: Node centre x in canvas coordinates.
    :param cy: Node centre y in canvas coordinates.
    :param node_radius: Radius of the node circle (data-units / pixels).
    :param port_radius: Radius of each port ball (data-units / pixels).
    :param n_outputs: Number of output (source) ports.
    :param n_inputs: Number of input (target) ports.
    :param spread: Angular fan for each half in degrees (default 120).
    :param gap: Clearance between node edge and port centre (pixels).

    :return: ``{"outputs": [Port, …], "inputs": [Port, …]}`` where each Port
             is a dict with keys ``index``, ``kind``, ``angle_deg``, ``x``, ``y``.
    """
    orbit = node_radius + port_radius + gap

    def _angle(index: int, total: int, centre: float) -> float:
        if total == 1:
            return centre
        start = centre - spread / 2
        step = spread / (total - 1)
        return start + index * step

    def _xy(angle_deg: float):
        a = math.radians(angle_deg)
        return cx + orbit * math.cos(a), cy + orbit * math.sin(a)

    outputs = []
    for i in range(n_outputs):
        d = _angle(i, n_outputs, -90.0)
        px, py = _xy(d)
        outputs.append({"index": i, "kind": "output", "angle_deg": d, "x": px, "y": py})

    inputs = []
    for i in range(n_inputs):
        d = _angle(i, n_inputs, 90.0)
        px, py = _xy(d)
        inputs.append({"index": i, "kind": "input", "angle_deg": d, "x": px, "y": py})

    return {"outputs": outputs, "inputs": inputs}


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
    # Text input for Component ID (shown in config panel)
    name_input: bkmodel.TextInput = field(default=None)
    # Select widget for Component Type (shown in config panel)
    type_select: bkmodel.Select = field(default=None)
    # Select widget for Position (shown in config panel)
    position_select: bkmodel.Select = field(default=None)
    # Column widget holding per-option rows (TextInput label + Select value)
    options_table: object = field(default=None)
    # Dynamic column of option rows – children rebuilt by _refresh_options_table
    options_rows_column: object = field(default=None)
    options_source: bkmodel.ColumnDataSource = field(default=None)
    apply_button: bkmodel.Button = field(default=None)
    # Index of the currently selected canvas node (None = nothing selected)
    selected_node_index: int = field(default=None)
    # The whole config panel column – toggled visible/invisible
    table_panel: object = field(default=None)
    # Source / Target port data sources for each component, keyed by node index in
    # placed_nodes_source
    source_port_source: bkmodel.ColumnDataSource = field(default=None)
    target_port_source: bkmodel.ColumnDataSource = field(default=None)
    # Spinners for editable port counts (only visible for components whose default count == 3)
    source_count_spinner: bkmodel.Spinner = field(default=None)
    target_count_spinner: bkmodel.Spinner = field(default=None)
    # Column section wrapping the spinners – toggled visible when ports are editable
    port_count_section: object = field(default=None)
    # Select node alpha change
    selected_node_overlay_source: bkmodel.ColumnDataSource = field(default=None)
    # Edge line connection source
    edge_source: bkmodel.ColumnDataSource = field(default=None)
    pending_port: dict = field(default=None)
    pending_port_source: bkmodel.ColumnDataSource = field(default=None)
    connections_source: bkmodel.ColumnDataSource = field(default=None)
    connections_table_widget: bkmodel.DataTable = field(default=None)
    # Dynamic column of connection rows – children rebuilt by _refresh_connections_table
    connections_rows_column: object = field(default=None)
    # Temporary (dashed) preview edges shown before Apply is clicked
    temp_edge_source: bkmodel.ColumnDataSource = field(default=None)
    # Pending port-pair connections waiting for Apply: list of (port_a_dict, port_b_dict)
    pending_connections: list = field(default_factory=list)
    # Select widget for Symmetry component (shown in config panel)
    symmetry_select: bkmodel.Select = field(default=None)
    # Per-component-type default port counts, built once during ComponentPaletteBuilder.build()
    default_source_count: dict = field(default_factory=dict)
    default_target_count: dict = field(default_factory=dict)
    component_type_to_icon: dict = field(default_factory=dict)
    possible_position: dict = field(default_factory=dict)  # component_type -> list of positions
    possible_options: dict = field(default_factory=dict)
    # component_type -> option name -> list of option values
    # Hidden TextInput widgets used to relay file paths chosen via browser prompt() back to Python
    json_path_input: bkmodel.TextInput = field(default=None)
    yaml_path_input: bkmodel.TextInput = field(default=None)
    # Hidden TextInput used to relay the JSON path chosen in the Load Design dialog
    load_path_input: bkmodel.TextInput = field(default=None)
    # Hidden TextInput used to push serialised YAML/JSON content from Python to the browser
    # for writing via the FileSystem Access API (showSaveFilePicker / createWritable).
    save_content_output: bkmodel.TextInput = field(default=None)
    # Startup overlay buttons shown on the canvas before the first action
    new_design_button: bkmodel.Button = field(default=None)
    load_design_button: bkmodel.Button = field(default=None)
    # The whole startup overlay column – hidden (not just its buttons) on dismiss
    startup_overlay: object = field(default=None)
