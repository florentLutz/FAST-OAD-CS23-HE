# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Component palette sidebar for the power-train network viewer.

Provides a clickable button palette (Bokeh server mode only) that lets the user
select a component type and place instances on the main canvas by clicking.

Typical usage
-------------
::

    from fastga_he.gui.component_palette import ComponentPaletteBuilder, PowertrainBuilderLauncher

    # --- standalone demo ---
    PowertrainBuilderLauncher.launch(port=5007)

    # --- embedded in an existing Bokeh document ---
    palette_layout, table_panel, state = ComponentPaletteBuilder.build()

    def make_doc(doc):
        from bokeh.layouts import row
        from bokeh.events import Tap
        from fastga_he.gui.component_palette import PlacementHandler

        handler = PlacementHandler(state, main_plot)
        main_plot.on_event(Tap, handler.on_canvas_tap)

        doc.add_root(row(palette_layout, main_plot))

"""

import sys
import math
import ast
import importlib
from pathlib import Path
import logging
from dataclasses import dataclass, field
import json
from datetime import datetime

import bokeh.models as bkmodel
import bokeh.plotting as bkplot
from bokeh.events import Tap
from bokeh.layouts import column, row
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
import webbrowser

from fastga_he.gui.constants import (
    POSSIBLE_OPTIONS,
)
from fastga_he.gui.power_train_network_writer import PowerTrainYAML
from fastga_he.gui.power_train_network_viewer import (
    BACKGROUND_COLOR_CODE,
    DEFAULT_COLOR,
    ICONS_CONFIG,
    _string_cleanup,
    _url_to_base64,
)
from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

_LOGGER = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
COMPONENTS_PATH = Path(__file__).resolve().parents[2] / "fastga_he/models/propulsion/components"

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

NODE_SELECT_COLOR = "#FFD700"  # gold color for selected node overlay (semi-transparent)


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

    Passed by reference between :class:`ComponentPaletteBuilder`,
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
    # Dynamic column of connection rows – children rebuilt by _refresh_connections_table_new
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


# ============================================================================
# Possible component types mapper
# ============================================================================


def _map_possible_component_types_to_ions() -> dict:
    """
    Build a dict mapping each component_type to its icon key.

    :return: ``{component_type: icon_key}``
    """
    type_to_icon = {}
    for component in KNOWN_COMPONENTS:
        component_type = component["components_type"]
        icon_key = component["icon_for_network_graph"]
        if icon_key not in type_to_icon.keys():
            type_to_icon[icon_key] = [component_type]
        else:
            type_to_icon[icon_key].append(component_type)

    return type_to_icon


# ============================================================================
# Port-count defaults builder
# ============================================================================


def _build_port_count_defaults() -> tuple[dict, dict]:
    """
    Derive default source/target port counts for every known component type.

    Iterates over :data:`KNOWN_COMPONENTS` and applies the same rules that
    were previously encoded as static imports from ``constants``:

    * If any attribute name contains ``"number_of_"`` → 3 sources, 3 targets
      (variable-count component).
    * If any attribute name contains ``"_mode"`` → 2 sources, 1 target.
    * ``"gearbox"`` → 1 source, 2 targets.
    * ``"propeller"`` or ``"aux_load"`` → 1 source, 0 targets.
    * ``"fuel_tank"`` or ``"gaseous_hydrogen_tank"`` → 0 sources, 1 target.
    * Everything else → 1 source, 1 target.

    :return: ``(default_source_count, default_target_count)`` – both are dicts
             keyed by ``components_type`` string.
    """
    default_source_count: dict = {}
    default_target_count: dict = {}

    for component in KNOWN_COMPONENTS:
        component_type = component["components_type"]
        attribute = component["attributes"]

        if component_type == "gearbox":
            default_source_count[component_type] = 1
            default_target_count[component_type] = 2
        elif component_type in ("propeller", "aux_load"):
            default_source_count[component_type] = 1
            default_target_count[component_type] = 0
        elif component_type in ("fuel_tank", "gaseous_hydrogen_tank", "battery_pack"):
            default_source_count[component_type] = 0
            default_target_count[component_type] = 1
        elif isinstance(attribute, list):
            # Set up default counts for components with attributes, then check for special cases
            default_source_count[component_type] = 1
            default_target_count[component_type] = 1

            for attr in attribute:
                if "number_of_" in attr:
                    default_source_count[component_type] = 3
                    default_target_count[component_type] = 3
                    break
                elif "_mode" in attr:
                    default_source_count[component_type] = 2
                    default_target_count[component_type] = 1
                    break

        else:
            default_source_count[component_type] = 1
            default_target_count[component_type] = 1

    return default_source_count, default_target_count


# ============================================================================
# Get Possible Positions
# ============================================================================


def _get_possible_position(constants_path: Path) -> list | None:
    """
    Parse constants.py with ast and extract the POSSIBLE_POSITION value
    without importing the module.
    """
    if not constants_path.exists():
        return None

    source = constants_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        # Look for: POSSIBLE_POSITION = [...]  or  POSSIBLE_POSITION = (...)
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "POSSIBLE_POSITION" for t in node.targets
        ):
            # Safely evaluate the assigned value (handles lists, tuples, strings, enums)
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                # Value contains non-literals (e.g. enum references) — fall back to importlib
                return None

    return None


def _get_possible_position_via_import(
    constants_path: Path, components_path: Path, base_package: str
) -> list | None:
    """
    Fallback: dynamically import constants.py and read POSSIBLE_POSITION directly.
    Used when ast.literal_eval fails (e.g. enum values).
    """
    try:
        # Build dotted module path from file path
        relative = constants_path.with_suffix("").relative_to(components_path.parent.parent.parent)
        module_path = ".".join(relative.parts)
        module = importlib.import_module(module_path)
        return getattr(module, "POSSIBLE_POSITION", None)
    except Exception as e:
        print(f"  Warning: could not import {constants_path}: {e}")
        return None


def _get_performance_component_names(
    components_path: str | Path,
    base_package: str = "fastga_he.models.propulsion.components",
) -> dict:
    components_path = Path(components_path)
    results = {}

    # Build a reverse map: OM_components_name -> components_type
    om_name_to_type = {
        component["OM_components_name"]: component["components_type"]
        for component in KNOWN_COMPONENTS
    }

    for init_file in sorted(components_path.rglob("__init__.py")):
        source = init_file.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name.startswith("Performances"):
                        stripped_name = alias.name.removeprefix("Performances")
                        component_type = om_name_to_type.get(stripped_name)
                        if component_type is None:
                            continue  # no matching component, skip

                        constants_file = init_file.parent / "constants.py"
                        possible_positions = _get_possible_position(constants_file)
                        if possible_positions is None and constants_file.exists():
                            possible_positions = _get_possible_position_via_import(
                                constants_file, components_path, base_package
                            )

                        results[component_type] = possible_positions or []

    # Fill in any known components not found during the scan
    for component in KNOWN_COMPONENTS:
        results.setdefault(component["components_type"], [])

    return results


# ============================================================================
# Palette builder
# ============================================================================


class ComponentPaletteBuilder:
    """
    Build the palette sidebar as a column of Bokeh ``Button`` widgets.

    This class is **pure** – it only constructs Bokeh objects and returns them;
    callbacks are wired by :class:`PlacementHandler`.
    """

    @staticmethod
    def build() -> tuple:
        """
        Construct the button palette and initialise a :class:`BuilderState`.

        :return: ``(palette_column_layout, table_layout, BuilderState)``
        """
        component_icon_keys = list(ICONS_CONFIG.keys())
        category_keys = {}
        for component in KNOWN_COMPONENTS:
            icon = component["icon_for_network_graph"]
            component_type_class = component["components_type_class"]
            if isinstance(component_type_class, list):
                # check if any of the types in the list are already in category_keys
                for type in component_type_class:
                    if type == "propulsive_load":
                        continue
                    elif type not in category_keys:
                        category_keys[type] = [icon]
                    elif icon not in category_keys[type]:
                        category_keys[type].append(icon)
            elif component_type_class == "propulsive_load":
                if "load" not in category_keys:
                    category_keys["load"] = [icon]
                else:
                    category_keys["load"].append(icon)
            elif component_type_class not in category_keys:
                category_keys[component_type_class] = [icon]
            elif icon not in component_type_class:
                category_keys[component_type_class].append(icon)

        # Title div for the palette sidebar
        title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Components</b>",
            width=PALETTE_WIDTH,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )

        # One button per icon – callbacks wired later by PlacementHandler.
        # Buttons are kept in ICONS_CONFIG order for index-based selection.
        buttons = []
        button_by_key: dict = {}
        for key in component_icon_keys:
            label = _string_cleanup(key)

            # Base64-encode the icon so it renders correctly inside the server
            icon_path = ICONS_CONFIG[key]["icon_path"]
            file_url = "file://" + str(Path(icon_path).resolve())
            b64_url = _url_to_base64(file_url)

            # Wrap image in an SVG to control size and avoid blurry raster scaling in Bokeh
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
                    <image href="{b64_url}" width="120" height="120"/></svg>"""

            button = bkmodel.Button(
                label=label,
                icon=bkmodel.SVGIcon(svg=svg, size="2.75em"),
                button_type=BUTTON_DEFAULT_COLOR_TYPE,
                width=PALETTE_WIDTH - 10,
                height=ROW_HEIGHT - 6,
                stylesheets=[
                    """:host .bk-btn {
                        font-size: 12pt;
                        white-space: normal;
                        padding-left: 6px;
                        display: flex !important;
                        flex-direction: row !important;
                        align-items: center !important;
                        justify-content: space-between !important;
                    }
                    :host .bk-btn .bk-btn-text {
                        order: 0 !important;
                        text-align: left;
                    }
                    :host .bk-btn .bk-icon {
                        order: 1 !important;
                        flex-shrink: 0;
                    }
                    """
                ],
            )
            buttons.append(button)
            button_by_key[key] = button

        # Status div – updated by PlacementHandler when a component is selected
        status_div = bkmodel.Div(
            text="<i style='color:#aaa;font-size:14pt'>Select a component</i>",
            width=PALETTE_WIDTH,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "14px"},
        )

        # Shared stylesheet for action buttons (Delete, Save, End Session)
        _action_stylesheet = [":host button { font-size: 1.4em; }"]

        delete_button = bkmodel.Button(
            label="Delete",
            icon=bkmodel.TablerIcon(icon_name="trash"),
            button_type=BUTTON_DEFAULT_COLOR_TYPE,
            width=PALETTE_WIDTH - 10,
            height=ROW_HEIGHT - 6,
            stylesheets=_action_stylesheet,
        )

        save_button = bkmodel.Button(
            label="Save",
            icon=bkmodel.TablerIcon(icon_name="device-floppy"),
            button_type="success",
            width=PALETTE_WIDTH - 10,
            height=ROW_HEIGHT - 6,
            stylesheets=_action_stylesheet,
        )

        end_session_button = bkmodel.Button(
            label="End Session",
            icon=bkmodel.TablerIcon(icon_name="power"),
            button_type="warning",
            width=PALETTE_WIDTH - 10,
            height=ROW_HEIGHT - 6,
            stylesheets=_action_stylesheet,
        )
        end_session_button.js_on_click(bkmodel.CustomJS(code="window.close();"))

        # ColumnDataSource for placed nodes – consumed by the main canvas image_url renderer
        placed_nodes_source = bkmodel.ColumnDataSource(
            data=dict(
                x=[],
                y=[],
                url=[],
                w=[],
                h=[],
                name=[],
                node_type=[],
                icon_type=[],
                position=[],
                options=[],  # JSON-encoded dict of option_name → value
                n_sources=[],  # current source-port count for this node
                n_targets=[],  # current target-port count for this node
                symmetry_name=[],  # name of the symmetry node
                symmetry_node_index=[],
            )
        )

        # Hover source – mirrors placed_nodes_source positions for the scatter hover tool
        hover_source = bkmodel.ColumnDataSource(data=dict(x=[], y=[], name=[], node_type=[]))

        # Port data sources – one row per port ball on the canvas
        # Columns: x, y (centre), color (hex), label (str port index), node_index (owner)
        source_port_source = bkmodel.ColumnDataSource(
            data=dict(
                x=[],
                y=[],
                color=[],
                label=[],
                node_index=[],
                node_name=[],
                node_type=[],
                fill_alpha=[],
                line_alpha=[],
                connected=[],
            )
        )
        target_port_source = bkmodel.ColumnDataSource(
            data=dict(
                x=[],
                y=[],
                color=[],
                label=[],
                node_index=[],
                node_name=[],
                node_type=[],
                fill_alpha=[],
                line_alpha=[],
                connected=[],
            )
        )
        selected_node_overlay_source = bkmodel.ColumnDataSource(data=dict(x=[], y=[]))

        # Options source – still used to track option names/values as plain lists
        options_table_source = bkmodel.ColumnDataSource(data=dict(options=[], value=[]))

        # Dynamic column that will hold one row per option (TextInput label + Select value)
        options_rows_column = column(
            [],
            styles={"background": BACKGROUND_COLOR_CODE},
        )

        # Configurator panel inputs
        name_input = bkmodel.TextInput(
            title="Component ID:",
            value="",
            width=380,
            styles={"color": "white", "font-size": "18px"},
        )

        # Select widget for Component Type – options are filtered to valid types
        # for the selected node whenever a canvas node is clicked
        type_select = bkmodel.Select(
            title="Component Type:",
            value="",
            options=[],
            width=380,
            styles={"color": "white", "font-size": "18px"},
        )

        # Select widget for Position – options come from POSSIBLE_POSITIONS keyed by node_type
        position_select = bkmodel.Select(
            title="Position:",
            value="",
            options=[],
            width=380,
            styles={"color": "white", "font-size": "18px"},
        )

        options_list = options_rows_column

        apply_button = bkmodel.Button(
            label="Apply",
            icon=bkmodel.TablerIcon(icon_name="check"),
            button_type="primary",
            width=380,
            height=ROW_HEIGHT - 6,
            stylesheets=_action_stylesheet,
        )

        # Section heading for the configurator panel
        table_title_div = bkmodel.Div(
            text="<b style='color:white;font-size:18pt'>Component Configurator</b>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE, "text-align": "center"},
        )

        component_id_type_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Component ID &amp; Type</b>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )

        component_option_title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Position &amp; Options</b>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )

        options_table_title = bkmodel.Div(
            text="<span style='color:white;font-size:14pt'>Options:</span>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE},
        )

        options_table = column(
            options_table_title,
            options_list,
            visible=False,
            styles={"background": BACKGROUND_COLOR_CODE},
        )

        # Port count spinners – only visible for components whose default count equals 3
        port_count_title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Port Counts</b>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )
        source_count_spinner = bkmodel.Spinner(
            title="Source Ports:",
            value=1,
            low=1,
            high=20,
            step=1,
            width=180,
            visible=False,
            styles={"color": "white", "font-size": "14px"},
        )
        target_count_spinner = bkmodel.Spinner(
            title="Target Ports:",
            value=1,
            low=1,
            high=20,
            step=1,
            width=180,
            visible=False,
            styles={"color": "white", "font-size": "14px"},
        )
        port_count_section = column(
            port_count_title_div,
            row(source_count_spinner, target_count_spinner),
            visible=False,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "4px"},
        )
        connections_source = bkmodel.ColumnDataSource(
            data=dict(my_port=[], connected_to=[], edge_idx=[])
        )
        # Dynamic column that will hold one row per connection (label + select)
        connections_rows_column = column(
            [],
            styles={"background": BACKGROUND_COLOR_CODE},
        )
        connections_table_widget = column(
            connections_rows_column,
            styles={"background": BACKGROUND_COLOR_CODE},
        )
        connections_title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Connections</b>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )
        symmetry_title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Symmetry & Distributed Load</b>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )
        symmetry_select = bkmodel.Select(
            value="",
            options=[],
            width=380,
            styles={"color": "white", "font-size": "18px"},
        )

        # Config panel column – hidden until a canvas node is selected
        table_panel = column(
            table_title_div,
            component_id_type_div,
            name_input,
            type_select,
            port_count_section,
            component_option_title_div,
            position_select,
            options_table,
            connections_title_div,
            connections_table_widget,
            symmetry_title_div,
            symmetry_select,
            apply_button,
            spacing=4,
            visible=False,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "10px"},
        )

        edge_source = bkmodel.ColumnDataSource(
            data=dict(
                xs=[],
                ys=[],
                color=[],
                node_a_idx=[],
                a_label=[],
                a_kind=[],
                node_b_idx=[],
                b_label=[],
                b_kind=[],
            )
        )
        pending_port_source = bkmodel.ColumnDataSource(data=dict(x=[], y=[], color=[]))
        temp_edge_source = bkmodel.ColumnDataSource(data=dict(xs=[], ys=[], color=[]))

        default_source_count, default_target_count = _build_port_count_defaults()

        component_type_to_icon = _map_possible_component_types_to_ions()

        possible_position = _get_performance_component_names(COMPONENTS_PATH)

        state = BuilderState(
            buttons=buttons,
            placed_nodes_source=placed_nodes_source,
            status_div=status_div,
            delete_button=delete_button,
            save_button=save_button,
            end_session_button=end_session_button,
            hover_source=hover_source,
            source_port_source=source_port_source,
            target_port_source=target_port_source,
            source_count_spinner=source_count_spinner,
            target_count_spinner=target_count_spinner,
            port_count_section=port_count_section,
            options_table=options_table,
            options_rows_column=options_rows_column,
            options_source=options_table_source,
            name_input=name_input,
            type_select=type_select,
            position_select=position_select,
            apply_button=apply_button,
            table_panel=table_panel,
            selected_node_overlay_source=selected_node_overlay_source,
            edge_source=edge_source,
            pending_port_source=pending_port_source,
            connections_source=connections_source,
            connections_table_widget=connections_table_widget,
            connections_rows_column=connections_rows_column,
            temp_edge_source=temp_edge_source,
            pending_connections=[],
            symmetry_select=symmetry_select,
            default_source_count=default_source_count,
            default_target_count=default_target_count,
            component_type_to_icon=component_type_to_icon,
            possible_position=possible_position,
        )

        # Build one TabPanel per category defined in ICON_TYPE
        tab_panels = []
        # Create tab panels by grouping buttons according to their category in ICON_TYPE
        for category, keys_in_category in category_keys.items():
            # extract buttons for this category, preserving the order in component_icon_keys
            category_buttons = [
                button_by_key[key] for key in keys_in_category if key in button_by_key
            ]
            if not category_buttons:
                continue
            tab_column = column(
                *category_buttons,
                spacing=2,
                styles={"background": BACKGROUND_COLOR_CODE, "padding": "10px"},
            )
            # Add the tab panel for this category, with the title capitalised
            tab_panels.append(bkmodel.TabPanel(child=tab_column, title=category.capitalize()))

        tabs = bkmodel.Tabs(tabs=tab_panels, width=PALETTE_WIDTH)

        palette_layout = column(
            title_div,
            tabs,
            status_div,
            delete_button,
            save_button,
            end_session_button,
            spacing=2,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "10px"},
        )

        return palette_layout, table_panel, state


# ============================================================================
# Placement handler
# ============================================================================


class PlacementHandler:
    """
    Wires palette button events to canvas placement events.

    Instantiate **after** building the palette; the constructor automatically
    wires all button ``on_click`` callbacks::

        palette_layout, table_panel, state = ComponentPaletteBuilder.build()
        handler = PlacementHandler(state, main_plot)
        main_plot.on_event(Tap, handler.on_canvas_tap)
        main_plot.image_url(
            url="url", x="x", y="y", w="w", h="h",
            anchor="center", source=state.placed_nodes_source,
        )
    """

    def __init__(self, state: BuilderState, main_plot, icon_size: int = 50):
        """
        :param state: Shared :class:`BuilderState` instance.
        :param main_plot: The Bokeh ``figure`` that acts as the placement canvas.
        :param icon_size: Pixel size (width = height) used for placed icons.
        """
        self.state = state
        self.main_plot = main_plot
        self.icon_size = icon_size
        self._wire_buttons()

    # -----------------------------------------------------------------------
    # Edge connection methods
    # -----------------------------------------------------------------------
    def _find_nearest_port(self, x: float, y: float) -> dict | None:
        """
        Return metadata for the nearest port ball within snap distance, or ``None``.

        Searches both ``source_port_source`` and ``target_port_source`` for the port
        whose centre is closest to ``(x, y)``. The snap radius is ``PORT_RADIUS * 2.5``
        data-units, giving a comfortable click target around each port ball.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.

        :return: A dict with keys ``kind``, ``x``, ``y``, ``color``, ``label``,
            and ``node_index`` for the nearest port, or ``None`` if no port is
            within snap distance.
        """
        snap = PORT_RADIUS * 2.5
        best, best_distance = None, snap
        for kind, port_source in [
            ("source", self.state.source_port_source),
            ("target", self.state.target_port_source),
        ]:
            if port_source is None:
                continue
            data = port_source.data
            for i, (port_x, port_y) in enumerate(
                zip(list(data.get("x", [])), list(data.get("y", [])))
            ):
                distance = ((x - port_x) ** 2 + (y - port_y) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best = {
                        "kind": kind,
                        "x": port_x,
                        "y": port_y,
                        "color": list(data["color"])[i],
                        "label": list(data["label"])[i],
                        "node_index": list(data["node_index"])[i],
                    }
        return best

    def _cancel_pending_connection(self):
        """
        Cancel any port selection that is waiting for a second click.

        Clears ``state.pending_port`` and removes the highlight ring rendered
        through ``pending_port_source``, returning the canvas to its idle state.
        """
        self.state.pending_port = None
        if self.state.pending_port_source is not None:
            self.state.pending_port_source.data = dict(x=[], y=[], color=[])

    def _handle_port_tap(self, port: dict):
        """
        Manage the two-click port-connection workflow.

        On the **first click** the port is stored as the pending endpoint and
        a highlight ring is drawn around it. On the **second click**:

        * If the same port is tapped again, the selection is cancelled.
        * If a port of the **opposite kind** (source ↔ target) with the **same
          energy colour** is tapped, a permanent edge is drawn and both ports
          are marked connected.
        * Any other combination is silently ignored and the pending selection
          is cleared.

        :param port: Port metadata dict with keys ``kind``, ``x``, ``y``,
            ``color``, ``label``, and ``node_index``.
        """
        self._clear_temp_edges()
        pending = self.state.pending_port
        if pending is None:
            self.state.pending_port = port
            if self.state.pending_port_source is not None:
                self.state.pending_port_source.data = dict(
                    x=[port["x"]], y=[port["y"]], color=[port["color"]]
                )
            return
        # Same port tapped again → cancel
        if (
            port["kind"] == pending["kind"]
            and port["node_index"] == pending["node_index"]
            and port["label"] == pending["label"]
        ):
            self._cancel_pending_connection()
            return
        # connects source↔target of the same energy type (color)
        if port["kind"] != pending["kind"] and port["color"] == pending["color"]:
            self._add_edge(pending, port)
            self._rebuild_all_ports()
        self._cancel_pending_connection()

    def _cursor_to_segment_distance(
        self, px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """
        Return the minimum Euclidean distance from cursor point ``(px, py)`` to the
        line segment from ``(x1, y1)`` to ``(x2, y2)``.

        Used by :meth:`_find_nearest_edge` to decide whether a tap falls close
        enough to an existing edge to delete it.

        :param px: X coordinate of the cursor point.
        :param py: Y coordinate of the cursor point.
        :param x1: X coordinate of the segment start.
        :param y1: Y coordinate of the segment start.
        :param x2: X coordinate of the segment end.
        :param y2: Y coordinate of the segment end.

        :return: Shortest distance (in data-units / pixels) from the point to
            the segment.
        """
        distance_x, distance_y = x2 - x1, y2 - y1
        if distance_x == 0 and distance_y == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        # Projection parameter for the cursor point onto the line defined by the segment,
        # clamped to [0, 1] to stay within the segment.
        projection_parameter = max(
            0.0,
            min(
                1.0,
                ((px - x1) * distance_x + (py - y1) * distance_y) / (distance_x**2 + distance_y**2),
            ),
        )
        return (
            (px - (x1 + projection_parameter * distance_x)) ** 2
            + (py - (y1 + projection_parameter * distance_y)) ** 2
        ) ** 0.5

    def _find_nearest_edge(self, x: float, y: float, snap: float = 12.0) -> int | None:
        """
        Return the index of the edge closest to ``(x, y)`` within *snap* pixels,
        or ``None`` if no edge is close enough.

        Iterates over every two-point polyline stored in ``edge_source`` and
        calls :meth:`_cursor_to_segment_distance` for each. Used in delete mode to
        let the user click near a connection line to remove it.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.
        :param snap: Maximum allowed distance (data-units) for a hit.

        :return: Zero-based row index into ``edge_source``, or ``None``.
        """
        # No edges → no hit
        if self.state.edge_source is None:
            return None

        edge_data = self.state.edge_source.data
        best_index, best_distance = None, snap
        for i, (xs, ys) in enumerate(zip(edge_data.get("xs", []), edge_data.get("ys", []))):
            if len(xs) < 2:
                continue
            distance = self._cursor_to_segment_distance(x, y, xs[0], ys[0], xs[1], ys[1])
            if distance < best_distance:
                best_distance = distance
                best_index = i
        return best_index

    def _add_edge(self, port_a: dict, port_b: dict):
        """
        Append a permanent edge between two ports to ``edge_source``.

        Before adding, the method guards against two invalid cases:

        * **Already-occupied port** – either port is already the endpoint of an
          existing edge.
        * **Loop connection** – the two nodes are already connected in either
          direction.

        If either check fails the edge is rejected and a warning is logged.
        Otherwise the edge row is appended and port ``connected`` flags are
        updated in-place so the Connections panel stays accurate.

        :param port_a: Source-side port metadata dict (keys: ``kind``, ``x``,
            ``y``, ``color``, ``label``, ``node_index``).
        :param port_b: Target-side port metadata dict (same keys).
        """
        if self.state.edge_source is None:
            return
        edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}

        for i in range(len(edge_data.get("node_a_idx", []))):
            port_a_used = (
                edge_data["node_a_idx"][i] == port_a["node_index"]
                and edge_data["a_label"][i] == port_a["label"]
                and edge_data["a_kind"][i] == port_a["kind"]
            ) or (
                edge_data["node_b_idx"][i] == port_a["node_index"]
                and edge_data["b_label"][i] == port_a["label"]
                and edge_data["b_kind"][i] == port_a["kind"]
            )
            port_b_used = (
                edge_data["node_a_idx"][i] == port_b["node_index"]
                and edge_data["a_label"][i] == port_b["label"]
                and edge_data["a_kind"][i] == port_b["kind"]
            ) or (
                edge_data["node_b_idx"][i] == port_b["node_index"]
                and edge_data["b_label"][i] == port_b["label"]
                and edge_data["b_kind"][i] == port_b["kind"]
            )

            if port_a_used or port_b_used:
                _LOGGER.info("Port already connected; edge rejected.")
                return

        # Check reverse connection: port_a's node → port_b's node already exists in reverse
        loop_connection = any(
            (
                edge_data["node_a_idx"][i] == port_b["node_index"]
                and edge_data["node_b_idx"][i] == port_a["node_index"]
            )
            or (
                edge_data["node_a_idx"][i] == port_a["node_index"]
                and edge_data["node_b_idx"][i] == port_b["node_index"]
            )
            for i in range(len(edge_data.get("node_a_idx", [])))
        )
        if loop_connection:
            _LOGGER.info("Loop connection detected; edge rejected.")
            return

        edge_data["xs"].append([port_a["x"], port_b["x"]])
        edge_data["ys"].append([port_a["y"], port_b["y"]])
        edge_data["color"].append(port_a["color"])
        edge_data["node_a_idx"].append(port_a["node_index"])
        edge_data["a_label"].append(port_a["label"])
        edge_data["a_kind"].append(port_a["kind"])
        edge_data["node_b_idx"].append(port_b["node_index"])
        edge_data["b_label"].append(port_b["label"])
        edge_data["b_kind"].append(port_b["kind"])
        self.state.edge_source.data = edge_data

        if self.state.selected_node_index is not None:
            if (
                port_a["node_index"] == self.state.selected_node_index
                or port_b["node_index"] == self.state.selected_node_index
            ):
                src_data = {k: list(v) for k, v in self.state.source_port_source.data.items()}
                tgt_data = {k: list(v) for k, v in self.state.target_port_source.data.items()}

                def _mark_connected(data, port):
                    data["connected"] = [
                        "True"
                        if (
                            data["node_index"][i] == port["node_index"]
                            and data["label"][i] == port["label"]
                        )
                        else data["connected"][i]  # preserve existing "True"
                        for i in range(len(data["node_index"]))
                    ]

                if port_a["kind"] == "source":
                    _mark_connected(src_data, port_a)
                    _mark_connected(tgt_data, port_b)
                else:
                    _mark_connected(tgt_data, port_a)
                    _mark_connected(src_data, port_b)

                # Full reassignment so Bokeh detects the change
                self.state.source_port_source.data = src_data
                self.state.target_port_source.data = tgt_data

                self._refresh_connections_table(self.state.selected_node_index)

        _LOGGER.info(
            "Edge: %s port %s (node %d) ↔ %s port %s (node %d)",
            port_a["kind"],
            port_a["label"],
            port_a["node_index"],
            port_b["kind"],
            port_b["label"],
            port_b["node_index"],
        )

    def _add_edge_temp(self, port_a: dict, port_b: dict):
        """
        Draw a dashed preview line between two ports in ``temp_edge_source``.

        The line is **not** registered in ``edge_source``; it is replaced by a
        permanent edge only when :meth:`apply_node_configurations` is called.

        :param port_a: Port metadata dict (keys: kind, x, y, color, label, node_index).
        :param port_b: Port metadata dict (same keys).
        """
        if self.state.temp_edge_source is None:
            return
        tdata = {k: list(v) for k, v in self.state.temp_edge_source.data.items()}
        tdata["xs"].append([port_a["x"], port_b["x"]])
        tdata["ys"].append([port_a["y"], port_b["y"]])
        tdata["color"].append(port_a["color"])
        self.state.temp_edge_source.data = tdata
        _LOGGER.info(
            "Temp edge: %s:%s → %s:%s",
            port_a["kind"],
            port_a["label"],
            port_b["kind"],
            port_b["label"],
        )

    def _clear_temp_edges(self):
        """Wipe all dashed preview edges and the pending-connection list."""
        self.state.pending_connections.clear()
        self._clear_temp_edge_visuals()

    def _clear_temp_edge_visuals(self):
        """Clear only the canvas dashed lines, leave pending_connections intact."""
        if self.state.temp_edge_source is not None:
            self.state.temp_edge_source.data = dict(xs=[], ys=[], color=[])

    def _rebuild_edges(self):
        """
        Recompute xs/ys in edge_source from stored port-identity columns.

        Called at the end of _rebuild_all_ports so edges track port positions
        after port-count spinner changes.  Edges whose ports no longer exist
        (e.g. port count reduced) are silently dropped.
        """
        if self.state.edge_source is None:
            return
        edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
        if not edge_data.get("node_a_idx"):
            return
        # Build (kind, node_index, label) → (x, y) from current port sources
        pos: dict = {}
        for kind, src in [
            ("source", self.state.source_port_source),
            ("target", self.state.target_port_source),
        ]:
            if src is None:
                continue
            d = src.data
            for i, (px, py) in enumerate(zip(list(d.get("x", [])), list(d.get("y", [])))):
                key = (kind, list(d["node_index"])[i], list(d["label"])[i])
                pos[key] = (px, py)

        new_xs, new_ys, valid = [], [], []
        for i in range(len(edge_data["node_a_idx"])):
            ka = (edge_data["a_kind"][i], edge_data["node_a_idx"][i], edge_data["a_label"][i])
            kb = (edge_data["b_kind"][i], edge_data["node_b_idx"][i], edge_data["b_label"][i])
            if ka in pos and kb in pos:
                ax, ay = pos[ka]
                bx, by = pos[kb]
                new_xs.append([ax, bx])
                new_ys.append([ay, by])
                valid.append(i)

        new_edge_data = {k: [edge_data[k][j] for j in valid] for k in edge_data}
        new_edge_data["xs"] = new_xs
        new_edge_data["ys"] = new_ys
        self.state.edge_source.data = new_edge_data

    # -----------------------------------------------------------------------
    # Internal wiring
    # -----------------------------------------------------------------------

    def _wire_buttons(self):
        """
        Attach ``on_click`` callbacks to every palette and action button.

        Called once from :meth:`__init__`. Connects:

        * Each component palette button → :meth:`on_palette_select` (via
          :meth:`_make_select_callback`).
        * The Save button → :meth:`_save_canvas_state`.
        * The End Session button → :meth:`_end_session`.
        * The Delete button → :meth:`_toggle_delete_mode`.
        """
        self.state.save_button.on_click(self._save_canvas_state)
        self.state.end_session_button.on_click(self._end_session)
        for index, button in enumerate(self.state.buttons):
            button.on_click(self._make_select_callback(index))
        if self.state.delete_button is not None:
            self.state.delete_button.on_click(self._toggle_delete_mode)

    def _make_select_callback(self, idx: int):
        """
        Return a zero-argument closure that selects the component at *idx*.

        :param idx: Zero-based index into ``list(ICONS_CONFIG.keys())``.

        :return: Callback function for a palette button.
        """
        return lambda: self.on_palette_select(idx)

    # -----------------------------------------------------------------------
    # Delete mode
    # -----------------------------------------------------------------------

    def _toggle_delete_mode(self):
        """
        Toggle delete mode on / off, updating button styling and status text.

        When entering delete mode:

        * Any active component selection and pending port connection are cleared.
        * The config panel is hidden and all palette buttons are reset to their
          default (unselected) style.
        * The Delete button turns red (``"danger"``) as a visual indicator.
        * The status label prompts the user to click a node or edge to remove it.

        When leaving delete mode the status label and Delete button style are
        both restored to their defaults.
        """
        self._cancel_pending_connection()
        self.state.delete_mode = not self.state.delete_mode
        if self.state.delete_mode:
            # Enter delete mode: deselect any active component and clear config panel
            self.state.selected_component = None
            self.state.selected_node_index = None  # ← clear selection
            self._clear_temp_edges()  # ← discard any pending connections
            self._clear_node_table()  # ← hides panel + full reset
            for btn in self.state.buttons:
                btn.button_type = BUTTON_DEFAULT_COLOR_TYPE
            self.state.status_div.text = (
                "<b style='color:#FF4444;font-size:14pt'>Delete mode: "
                "click an icon / a connection to remove it</b>"
            )
            self.state.delete_button.button_type = "danger"
        else:
            # Exit delete mode: restore default status text
            self.state.status_div.text = (
                "<i style='color:#aaa;font-size:14pt'>Select a component</i>"
            )
            self.state.delete_button.button_type = BUTTON_DEFAULT_COLOR_TYPE

    # -----------------------------------------------------------------------
    # Save canvas state
    # -----------------------------------------------------------------------

    def _save_canvas_state(self):
        """
        Serialise the current placed-nodes data to a timestamped JSON backup file
        and a YAML powertrain configuration file.

        Both files are written to the current working directory and share the same
        timestamp suffix.  The JSON file is kept as a full-fidelity backup of the
        canvas state, while the YAML file is the powertrain config consumed by
        FAST-OAD_CS23-HE.  The save button is reset to ``"success"`` after a
        1-second delay.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── JSON backup (full canvas state) ──────────────────────────────────
        json_filename = f"canvas_state_{timestamp}.json"

        nodes_data = {k: list(v) for k, v in self.state.placed_nodes_source.data.items()}
        edges_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
        source_data = {k: list(v) for k, v in self.state.source_port_source.data.items()}
        target_data = {k: list(v) for k, v in self.state.target_port_source.data.items()}

        canvas_state = {
            "components": nodes_data,
            "connections": edges_data,
            "source_ports": source_data,
            "target_ports": target_data,
        }

        with open(json_filename, "w") as f:
            json.dump(canvas_state, f, indent=2)

        _LOGGER.info("Canvas state (JSON backup) saved to %s", json_filename)

        # ── YAML powertrain configuration ─────────────────────────────────────
        yaml_filename = f"powertrain_config_{timestamp}.yml"

        try:
            pt_yaml = PowerTrainYAML(self.state)
            pt_yaml.set_title(f"powertrain_config_{timestamp}")

            n_nodes = len(nodes_data.get("name", []))
            for node_index in range(n_nodes):
                pt_yaml.add_component(node_index)

            pt_yaml.add_connection()

            pt_yaml.write(yaml_filename)
            _LOGGER.info("Powertrain YAML config saved to %s", yaml_filename)
        except Exception:
            _LOGGER.exception(
                "Failed to write YAML config; JSON backup at %s is still valid.", json_filename
            )

        IOLoop.current().call_later(
            1.0, lambda: setattr(self.state.save_button, "button_type", "success")
        )

    # -----------------------------------------------------------------------
    # End session
    # -----------------------------------------------------------------------

    def _end_session(self):
        """
        Stop the Bokeh IO loop, terminating the server session.

        Bound to the End Session button. After this call the browser tab will
        lose its WebSocket connection and the Python process will exit.
        """
        _LOGGER.info("Ending session and stopping server")
        IOLoop.current().stop()

    # -----------------------------------------------------------------------
    # Palette selection
    # -----------------------------------------------------------------------

    def on_palette_select(self, idx: int):
        """
        Select the component at position *idx* in :data:`ICONS_CONFIG`.

        A second click on the same button deselects it.  Updates button
        styling and the status label accordingly.  Can be called
        programmatically in tests without a running server.

        :param idx: Zero-based index into ``list(ICONS_CONFIG.keys())``.
        """
        component_icon_keys = list(ICONS_CONFIG.keys())
        if idx < 0 or idx >= len(component_icon_keys):
            return

        # Second click on the same button → deselect
        if self.state.selected_component == component_icon_keys[idx]:
            self.state.selected_component = None
            self.state.buttons[idx].button_type = BUTTON_DEFAULT_COLOR_TYPE
            self.state.status_div.text = (
                "<i style='color:#aaa;font-size:14pt'>Select a component</i>"
            )
            return

        self._cancel_pending_connection()
        self.state.selected_component = component_icon_keys[idx]

        # Exit delete mode if it was active
        if self.state.delete_mode:
            self.state.delete_mode = False
            if self.state.delete_button is not None:
                self.state.delete_button.button_type = BUTTON_DEFAULT_COLOR_TYPE

        # Highlight selected button, reset all others
        for j, btn in enumerate(self.state.buttons):
            btn.button_type = BUTTON_SELECTED_COLOR_TYPE if j == idx else BUTTON_DEFAULT_COLOR_TYPE

        label = _string_cleanup(self.state.selected_component)
        self.state.status_div.text = f"<b style='color:#FFD700;font-size:14pt'>Placing: {label}</b>"

    # -----------------------------------------------------------------------
    # Config panel helpers
    # -----------------------------------------------------------------------

    def _populate_node_table(self, idx: int):
        """
        Fill the config panel widgets with the stored values of node *idx*.

        Populates ``name_input``, ``type_select``, ``position_select``, and
        ``options_table`` from ``placed_nodes_source``.

        :param idx: Zero-based index into ``placed_nodes_source`` data arrays.
        """
        pdata = self.state.placed_nodes_source.data
        om_name = list(pdata.get("name", []))[idx]
        node_type = list(pdata.get("node_type", []))[idx]
        comp_key = list(pdata.get("icon_type", []))[idx]
        position = list(pdata.get("position", []))[idx] if pdata.get("position") else ""
        saved_opts_json = list(pdata.get("options", []))[idx] if pdata.get("options") else "{}"

        if saved_opts_json != "{}":
            self.state.options_table.visible = True
        else:
            self.state.options_table.visible = False

        self.state.name_input.value = om_name

        # Populate type_select with valid choices for this component key
        choices = self.state.component_type_to_icon.get(comp_key, comp_key)
        self.state.type_select.options = choices
        self.state.type_select.value = node_type if node_type in choices else choices[0]

        # Populate position_select with valid positions for this node_type
        pos_choices = self.state.possible_position.get(node_type, [])
        self.state.position_select.options = pos_choices
        if pos_choices:
            self.state.position_select.value = (
                position if position in pos_choices else pos_choices[0]
            )
        else:
            self.state.position_select.value = ""

        # Restore saved option values; fall back to defaults from POSSIBLE_OPTIONS
        saved_opts: dict = {}
        try:
            saved_opts = json.loads(saved_opts_json) if saved_opts_json else {}
        except (json.JSONDecodeError, TypeError):
            pass

        self._refresh_options_table(node_type, saved_opts)

        # Show / update port-count spinners for editable components (default count == 3)
        n_src_default = self.state.default_source_count.get(node_type, 0)
        n_tgt_default = self.state.default_target_count.get(node_type, 0)
        src_editable = n_src_default == 3
        tgt_editable = n_tgt_default == 3

        current_n_src = (
            list(pdata.get("n_sources", []))[idx] if pdata.get("n_sources") else n_src_default
        )
        current_n_tgt = (
            list(pdata.get("n_targets", []))[idx] if pdata.get("n_targets") else n_tgt_default
        )

        if self.state.source_count_spinner is not None:
            self.state.source_count_spinner.visible = src_editable
            if src_editable:
                self.state.source_count_spinner.value = int(current_n_src)
        if self.state.target_count_spinner is not None:
            self.state.target_count_spinner.visible = tgt_editable
            if tgt_editable:
                self.state.target_count_spinner.value = int(current_n_tgt)
        if self.state.port_count_section is not None:
            self.state.port_count_section.visible = src_editable or tgt_editable

        if self.state.selected_node_overlay_source is not None:
            pdata = self.state.placed_nodes_source.data
            self.state.selected_node_overlay_source.data = dict(
                x=[list(pdata["x"])[idx]],
                y=[list(pdata["y"])[idx]],
            )
        self._rebuild_all_ports()
        self._refresh_connections_table(idx)
        self._refresh_symmetry_select(idx, comp_key)

    @staticmethod
    def _option_val_to_str(v) -> str:
        """
        Convert an option value to its display string.

        :param v: Raw option value (``True``, ``False``, or a scalar).

        :return: Display string (``True`` → ``"on"``, ``False`` → ``"off"``).
        """
        if v is True:
            return "on"
        if v is False:
            return "off"
        return str(v)

    @staticmethod
    def _str_to_option_val(s: str):
        """
        Parse a display string back to a Python value.

        :param s: Display string from the options table.

        :return: ``True`` for ``"on"``, ``False`` for ``"off"``, otherwise tries
                 ``int`` then ``float`` before returning the raw string.
        """
        if s == "on":
            return True
        if s == "off":
            return False
        try:
            return int(s)
        except (ValueError, TypeError):
            pass
        try:
            return float(s)
        except (ValueError, TypeError):
            pass
        return s

    def _refresh_options_table(self, node_type: str, overrides: dict = None):
        """
        Rebuild the Options section of the config panel for a given node type.

        Reads the allowed option names and their possible values from
        ``POSSIBLE_OPTIONS[node_type]``, creates one row per option (a disabled
        ``TextInput`` label + a ``Select`` widget for the value), and wires a
        shared ``on_change`` callback so that ``options_source`` is always in
        sync with the user's current selection.

        :param node_type: The component type key used to look up ``POSSIBLE_OPTIONS``.
        :param overrides: Optional mapping of ``{option_name: current_value}`` used
            to pre-populate widgets with previously saved values instead of defaults.
        """
        if overrides is None:
            overrides = {}
        opts_def = POSSIBLE_OPTIONS.get(node_type, {})
        opt_names = list(opts_def.keys())
        opt_values = []
        new_rows = []
        value_selects = []  # keep references for the sync callback

        for k, v_list in opts_def.items():
            current = overrides[k] if k in overrides else (v_list[0] if v_list else "")
            current_str = self._option_val_to_str(current)

            label_input = bkmodel.TextInput(
                value=k,
                width=180,
                disabled=True,
                styles={"color": "white", "font-size": "14px"},
            )

            choices = [self._option_val_to_str(c) for c in v_list] if v_list else [current_str]
            if current_str not in choices:
                choices = [current_str] + choices

            value_select = bkmodel.Select(
                value=current_str,
                options=choices,
                width=180,
                styles={"color": "white", "font-size": "14px"},
            )

            new_rows.append(row(label_input, value_select, spacing=4))
            opt_values.append(current_str)
            value_selects.append(value_select)

        # Sync every Select change back to options_source so apply_node_configurations reads
        # the current user-chosen values and not the stale initial ones.
        def _make_sync_callback(selects, names):
            def _on_change(attr, old, new):
                self.state.options_source.data = dict(
                    options=names,
                    value=[s.value for s in selects],
                )

            return _on_change

        sync_cb = _make_sync_callback(value_selects, opt_names)
        for vs in value_selects:
            vs.on_change("value", sync_cb)

        if self.state.options_rows_column is not None:
            self.state.options_rows_column.children = new_rows

        self.state.options_source.data = dict(options=opt_names, value=opt_values)

    def _refresh_symmetry_select(self, current_node_idx: int, icon_type: str):
        """
        Populate ``symmetry_select`` with the names of all placed nodes that
        share *icon_type* with the currently selected node, excluding the node
        itself.  Prepend an empty sentinel so the user can clear the selection.

        Also restores the previously saved symmetry value for *current_node_idx*
        when it exists in ``placed_nodes_source``.

        :param current_node_idx: Index of the node being edited.
        :param icon_type: The ``icon_type`` key of the selected node.
        """
        if self.state.symmetry_select is None:
            return

        pdata = self.state.placed_nodes_source.data
        names = list(pdata.get("name", []))
        icon_types = list(pdata.get("icon_type", []))
        saved_symmetry = list(pdata.get("symmetry_name", []))

        peers = [
            names[i]
            for i in range(len(names))
            if i != current_node_idx and icon_types[i] == icon_type
        ]
        choices = [""] + peers
        self.state.symmetry_select.options = choices

        # Restore previously saved symmetry name (if any)
        current_sym = (
            saved_symmetry[current_node_idx] if current_node_idx < len(saved_symmetry) else ""
        )
        self.state.symmetry_select.value = current_sym if current_sym in choices else ""

    def _clear_node_table(self):
        """
        Reset all node configuration panel inputs to their empty defaults and hide the panel.

        Clears ``name_input``, ``type_select``, ``position_select``,
        ``options_source``, ``connections_source``, and ``symmetry_select``;
        hides the port-count spinner section; removes the selected-node overlay;
        empties the dynamic connection rows; and calls :meth:`_rebuild_all_ports`
        so port ball alphas are reset (no node highlighted).
        """
        self.state.name_input.value = ""
        self.state.type_select.options = []
        self.state.type_select.value = ""
        self.state.position_select.options = []
        self.state.position_select.value = ""
        self.state.options_source.data = dict(options=[], value=[])
        if self.state.port_count_section is not None:
            self.state.port_count_section.visible = False
            if self.state.source_count_spinner is not None:
                self.state.source_count_spinner.visible = False
            if self.state.target_count_spinner is not None:
                self.state.target_count_spinner.visible = False
        if self.state.table_panel is not None:
            self.state.table_panel.visible = False

        if self.state.selected_node_overlay_source is not None:
            self.state.selected_node_overlay_source.data = dict(x=[], y=[])

        if self.state.connections_source is not None:
            self.state.connections_source.data = dict(my_port=[], connected_to=[], edge_idx=[])

        if self.state.connections_rows_column is not None:
            self.state.connections_rows_column.children = []

        if self.state.symmetry_select is not None:
            self.state.symmetry_select.options = [""]
            self.state.symmetry_select.value = ""

        self._rebuild_all_ports()

    def _refresh_connections_table(self, node_idx: int):
        """
        Populate the Connections panel for node *node_idx* using dynamic rows.

        For every source port of the selected node a row is created with:
          - a disabled TextInput showing the port label (e.g. ``"source:1"``)
          - a Select whose choices are all currently-unconnected target ports;
            if the port is already connected the current peer is prepended so
            the existing selection is preserved.

        For every target port of the selected node a symmetric row is created
        showing which source port it is connected to (or the available
        unconnected source ports as choices).

        Selecting a new value in any Select immediately draws (or re-draws)
        the corresponding edge.
        """
        if (
            self.state.connections_rows_column is None
            or self.state.edge_source is None
            or self.state.source_port_source is None
            or self.state.target_port_source is None
        ):
            return

        source_data = {k: list(v) for k, v in self.state.source_port_source.data.items()}
        target_data = {k: list(v) for k, v in self.state.target_port_source.data.items()}
        node_data = {k: list(v) for k, v in self.state.placed_nodes_source.data.items()}
        edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}

        # ------------------------------------------------------------------ #
        # Build lookup: (kind, port_label) → (peer_node_idx, peer_label)     #
        # for every edge that touches node_idx.                              #
        # ------------------------------------------------------------------ #
        connected_src: dict[str, tuple[int, str]] = {}  # src_label → (peer_node, tgt_label)
        connected_tgt: dict[str, tuple[int, str]] = {}  # tgt_label → (peer_node, src_label)

        for i in range(len(edge_data.get("node_a_idx", []))):
            na, nb = edge_data["node_a_idx"][i], edge_data["node_b_idx"][i]
            al, bl = edge_data["a_label"][i], edge_data["b_label"][i]
            ak, bk = edge_data["a_kind"][i], edge_data["b_kind"][i]

            if na == node_idx and ak == "source":
                connected_src[al] = (nb, bl)
            elif nb == node_idx and bk == "source":
                connected_src[bl] = (na, al)
            if na == node_idx and ak == "target":
                connected_tgt[al] = (nb, bl)
            elif nb == node_idx and bk == "target":
                connected_tgt[bl] = (na, al)

        # ------------------------------------------------------------------ #
        # Helper: build a display string for a peer port                     #
        # ------------------------------------------------------------------ #
        def _peer_str(peer_node_idx: int, peer_label: str, peer_kind: str) -> str:
            name = (
                node_data["name"][peer_node_idx]
                if peer_node_idx < len(node_data.get("name", []))
                else f"node_{peer_node_idx}"
            )
            return f"{name} ({peer_kind}:{peer_label})"

        # ------------------------------------------------------------------ #
        # Helper: given a new Select value string find its (node_idx, label) #
        # ------------------------------------------------------------------ #
        def _parse_choice(choice_str: str, candidates: list[tuple]) -> tuple | None:
            """Return (node_idx, label, kind) matching the display string, or None."""
            for node_i, lbl, kind, disp in candidates:
                if disp == choice_str:
                    return node_i, lbl, kind
            return None

        # ------------------------------------------------------------------ #
        # Collect all unconnected target ports (for source-port selects)     #
        # ------------------------------------------------------------------ #
        free_targets: list[tuple[int, str, str, str]] = []  # (node_i, label, kind, display)
        _selected_source_color = ICONS_CONFIG.get(node_data["icon_type"][node_idx], {}).get(
            "source_color", ""
        )
        for j in range(len(target_data.get("node_index", []))):
            if (
                target_data["connected"][j] == "False"
                and target_data["node_index"][j] != node_idx
                and target_data["color"][j] == _selected_source_color
            ):
                free_targets.append(
                    (
                        target_data["node_index"][j],
                        target_data["label"][j],
                        "target",
                        _peer_str(target_data["node_index"][j], target_data["label"][j], "target"),
                    )
                )

        # Collect all unconnected source ports (for target-port selects)
        _selected_target_color = ICONS_CONFIG.get(node_data["icon_type"][node_idx], {}).get(
            "target_color", ""
        )
        free_sources: list[tuple[int, str, str, str]] = []
        for j in range(len(source_data.get("node_index", []))):
            if (
                source_data["connected"][j] == "False"
                and source_data["node_index"][j] != node_idx
                and source_data["color"][j] == _selected_target_color
            ):
                free_sources.append(
                    (
                        source_data["node_index"][j],
                        source_data["label"][j],
                        "source",
                        _peer_str(source_data["node_index"][j], source_data["label"][j], "source"),
                    )
                )

        _EMPTY = ""  # sentinel shown when port has no connection and no free peers

        new_rows = []

        # ------------------------------------------------------------------ #
        # Source-port rows                                                   #
        # ------------------------------------------------------------------ #
        for i in range(len(source_data.get("node_index", []))):
            if source_data["node_index"][i] != node_idx:
                continue

            src_label = source_data["label"][i]
            is_connected = source_data["connected"][i] == "True"

            label_input = bkmodel.TextInput(
                value=f"source:{src_label}",
                width=180,
                disabled=True,
                styles={"color": "white", "font-size": "14px"},
            )

            candidates = list(free_targets)
            current_str = _EMPTY
            if is_connected and src_label in connected_src:
                peer_node_i, peer_lbl = connected_src[src_label]
                current_str = _peer_str(peer_node_i, peer_lbl, "target")
                candidates = [
                    c for c in candidates if not (c[0] == peer_node_i and c[1] == peer_lbl)
                ] + [(peer_node_i, peer_lbl, "target", _peer_str(peer_node_i, peer_lbl, "target"))]

            choices = [_EMPTY] + [c[3] for c in candidates if c[3] != current_str]
            if current_str and current_str not in choices:
                choices = [current_str] + choices

            value_select = bkmodel.Select(
                value=current_str if current_str else _EMPTY,
                options=choices,
                width=180,
                styles={"color": "white", "font-size": "12px"},
            )

            def _make_src_callback(
                _src_label=src_label,
                _src_node_idx=node_idx,
                _src_color=source_data["color"][i],
                _src_x=source_data["x"][i],
                _src_y=source_data["y"][i],
                _candidates=candidates,
            ):
                def _on_change(attr, old, new_val):
                    # 1. Remove any pending connection for this source port
                    self.state.pending_connections = [
                        (pa, pb)
                        for pa, pb in self.state.pending_connections
                        if not (
                            pa["kind"] == "source"
                            and pa["node_index"] == _src_node_idx
                            and pa["label"] == _src_label
                        )
                    ]
                    # Redraw all temp edges from remaining pending list
                    self._clear_temp_edge_visuals()
                    for pa, pb in list(self.state.pending_connections):
                        self._add_edge_temp(pa, pb)

                    if new_val == _EMPTY:
                        # Delete the committed edge for this source port, if any
                        if self.state.edge_source is not None:
                            edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
                            keep = [
                                i
                                for i in range(len(edge_data.get("node_a_idx", [])))
                                if not (
                                    (
                                        edge_data["node_a_idx"][i] == _src_node_idx
                                        and edge_data["a_label"][i] == _src_label
                                        and edge_data["a_kind"][i] == "source"
                                    )
                                    or (
                                        edge_data["node_b_idx"][i] == _src_node_idx
                                        and edge_data["b_label"][i] == _src_label
                                        and edge_data["b_kind"][i] == "source"
                                    )
                                )
                            ]
                            self.state.edge_source.data = {
                                k: [edge_data[k][j] for j in keep] for k in edge_data
                            }
                            self._rebuild_all_ports()
                            self._refresh_connections_table(_src_node_idx)
                        return

                    parsed = _parse_choice(new_val, _candidates)
                    if parsed is None:
                        return
                    tgt_node_i, tgt_lbl, tgt_kind = parsed

                    # Fetch live target port position
                    tgt_data = self.state.target_port_source.data
                    tgt_x, tgt_y, tgt_color = None, None, None
                    for j in range(len(tgt_data.get("node_index", []))):
                        if (
                            tgt_data["node_index"][j] == tgt_node_i
                            and tgt_data["label"][j] == tgt_lbl
                        ):
                            tgt_x = tgt_data["x"][j]
                            tgt_y = tgt_data["y"][j]
                            tgt_color = tgt_data["color"][j]
                            break
                    if tgt_x is None:
                        return

                    live_src_x, live_src_y = _src_x, _src_y  # fallback
                    src_data_live = self.state.source_port_source.data
                    for j in range(len(src_data_live.get("node_index", []))):
                        if (
                            src_data_live["node_index"][j] == _src_node_idx
                            and src_data_live["label"][j] == _src_label
                        ):
                            live_src_x = src_data_live["x"][j]
                            live_src_y = src_data_live["y"][j]
                            break

                    port_a = {
                        "kind": "source",
                        "node_index": _src_node_idx,
                        "label": _src_label,
                        "x": live_src_x,
                        "y": live_src_y,
                        "color": _src_color,
                    }
                    port_b = {
                        "kind": "target",
                        "node_index": tgt_node_i,
                        "label": tgt_lbl,
                        "x": tgt_x,
                        "y": tgt_y,
                        "color": tgt_color,
                    }
                    self.state.pending_connections.append((port_a, port_b))
                    self._add_edge_temp(port_a, port_b)

                return _on_change

            value_select.on_change(
                "value",
                _make_src_callback(
                    _src_label=src_label,
                    _src_node_idx=node_idx,
                    _src_color=source_data["color"][i],
                    _src_x=source_data["x"][i],
                    _src_y=source_data["y"][i],
                    _candidates=candidates,
                ),
            )
            new_rows.append(row(label_input, value_select, spacing=4))

        # ------------------------------------------------------------------ #
        # Target-port rows                                                     #
        # ------------------------------------------------------------------ #
        for i in range(len(target_data.get("node_index", []))):
            if target_data["node_index"][i] != node_idx:
                continue

            tgt_label = target_data["label"][i]
            is_connected = target_data["connected"][i] == "True"

            label_input = bkmodel.TextInput(
                value=f"target:{tgt_label}",
                width=180,
                disabled=True,
                styles={"color": "white", "font-size": "14px"},
            )

            candidates = list(free_sources)
            current_str = _EMPTY
            if is_connected and tgt_label in connected_tgt:
                peer_node_i, peer_lbl = connected_tgt[tgt_label]
                current_str = _peer_str(peer_node_i, peer_lbl, "source")
                candidates = [
                    c for c in candidates if not (c[0] == peer_node_i and c[1] == peer_lbl)
                ] + [(peer_node_i, peer_lbl, "source", _peer_str(peer_node_i, peer_lbl, "source"))]

            choices = [_EMPTY] + [c[3] for c in candidates if c[3] != current_str]
            if current_str and current_str not in choices:
                choices = [current_str] + choices

            value_select = bkmodel.Select(
                value=current_str if current_str else _EMPTY,
                options=choices,
                width=180,
                styles={"color": "white", "font-size": "12px"},
            )

            def _make_tgt_callback(
                _tgt_label=tgt_label,
                _tgt_node_idx=node_idx,
                _tgt_color=target_data["color"][i],
                _tgt_x=target_data["x"][i],
                _tgt_y=target_data["y"][i],
                _candidates=candidates,
            ):
                def _on_change(attr, old, new_val):
                    # 1. Remove any pending connection for this target port
                    self.state.pending_connections = [
                        (pa, pb)
                        for pa, pb in self.state.pending_connections
                        if not (
                            pb["kind"] == "target"
                            and pb["node_index"] == _tgt_node_idx
                            and pb["label"] == _tgt_label
                        )
                    ]
                    self._clear_temp_edge_visuals()
                    for pa, pb in list(self.state.pending_connections):
                        self._add_edge_temp(pa, pb)

                    if new_val == _EMPTY:
                        # Delete the committed edge for this target port, if any
                        if self.state.edge_source is not None:
                            edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
                            keep = [
                                i
                                for i in range(len(edge_data.get("node_a_idx", [])))
                                if not (
                                    (
                                        edge_data["node_a_idx"][i] == _tgt_node_idx
                                        and edge_data["a_label"][i] == _tgt_label
                                        and edge_data["a_kind"][i] == "target"
                                    )
                                    or (
                                        edge_data["node_b_idx"][i] == _tgt_node_idx
                                        and edge_data["b_label"][i] == _tgt_label
                                        and edge_data["b_kind"][i] == "target"
                                    )
                                )
                            ]
                            self.state.edge_source.data = {
                                k: [edge_data[k][j] for j in keep] for k in edge_data
                            }
                            self._rebuild_all_ports()
                            self._refresh_connections_table(_tgt_node_idx)
                        return

                    parsed = _parse_choice(new_val, _candidates)
                    if parsed is None:
                        return
                    src_node_i, src_lbl, src_kind = parsed

                    src_data = self.state.source_port_source.data
                    src_x, src_y, src_color = None, None, None
                    for j in range(len(src_data.get("node_index", []))):
                        if (
                            src_data["node_index"][j] == src_node_i
                            and src_data["label"][j] == src_lbl
                        ):
                            src_x = src_data["x"][j]
                            src_y = src_data["y"][j]
                            src_color = src_data["color"][j]
                            break
                    if src_x is None:
                        return

                    live_tgt_x, live_tgt_y = _tgt_x, _tgt_y  # fallback
                    tgt_data_live = self.state.target_port_source.data
                    for j in range(len(tgt_data_live.get("node_index", []))):
                        if (
                            tgt_data_live["node_index"][j] == _tgt_node_idx
                            and tgt_data_live["label"][j] == _tgt_label
                        ):
                            live_tgt_x = tgt_data_live["x"][j]
                            live_tgt_y = tgt_data_live["y"][j]
                            break

                    port_a = {
                        "kind": "source",
                        "node_index": src_node_i,
                        "label": src_lbl,
                        "x": src_x,
                        "y": src_y,
                        "color": src_color,
                    }
                    port_b = {
                        "kind": "target",
                        "node_index": _tgt_node_idx,
                        "label": _tgt_label,
                        "x": live_tgt_x,
                        "y": live_tgt_y,
                        "color": _tgt_color,
                    }
                    self.state.pending_connections.append((port_a, port_b))
                    self._add_edge_temp(port_a, port_b)

                return _on_change

            value_select.on_change(
                "value",
                _make_tgt_callback(
                    _tgt_label=tgt_label,
                    _tgt_node_idx=node_idx,
                    _tgt_color=target_data["color"][i],
                    _tgt_x=target_data["x"][i],
                    _tgt_y=target_data["y"][i],
                    _candidates=candidates,
                ),
            )
            new_rows.append(row(label_input, value_select, spacing=4))

        self.state.connections_rows_column.children = new_rows

        # ------------------------------------------------------------------ #
        # Keep connections_source in sync so _delete_selected_connection can  #
        # resolve the correct edge_idx for any selected row.                  #
        # ------------------------------------------------------------------ #
        if self.state.connections_source is not None:
            cs_my_port: list = []
            cs_connected_to: list = []
            cs_edge_idx: list = []

            for i in range(len(edge_data.get("node_a_idx", []))):
                na = edge_data["node_a_idx"][i]
                nb = edge_data["node_b_idx"][i]
                al = edge_data["a_label"][i]
                bl = edge_data["b_label"][i]
                ak = edge_data["a_kind"][i]
                bk = edge_data["b_kind"][i]

                if na == node_idx:
                    peer_name = (
                        node_data["name"][nb]
                        if nb < len(node_data.get("name", []))
                        else f"node_{nb}"
                    )
                    cs_my_port.append(f"{ak}:{al}")
                    cs_connected_to.append(f"{peer_name} ({bk}:{bl})")
                    cs_edge_idx.append(i)
                elif nb == node_idx:
                    peer_name = (
                        node_data["name"][na]
                        if na < len(node_data.get("name", []))
                        else f"node_{na}"
                    )
                    cs_my_port.append(f"{bk}:{bl}")
                    cs_connected_to.append(f"{peer_name} ({ak}:{al})")
                    cs_edge_idx.append(i)

            self.state.connections_source.data = dict(
                my_port=cs_my_port,
                connected_to=cs_connected_to,
                edge_idx=cs_edge_idx,
            )

    def _delete_selected_connection(self):
        """
        Delete the edge(s) currently selected in the Connections DataTable.

        Reads the highlighted row indices from ``connections_source.selected``,
        resolves them to actual edge indices via the ``edge_idx`` column, removes
        those rows from ``edge_source``, and refreshes the Connections panel so
        the table reflects the updated state.
        """
        if self.state.connections_source is None or self.state.edge_source is None:
            return
        selected = list(self.state.connections_source.selected.indices)
        if not selected:
            return
        edge_idx_col = list(self.state.connections_source.data.get("edge_idx", []))
        to_delete = {edge_idx_col[i] for i in selected if i < len(edge_idx_col)}
        edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
        keep = [i for i in range(len(edge_data.get("xs", []))) if i not in to_delete]
        self.state.edge_source.data = {k: [edge_data[k][j] for j in keep] for k in edge_data}
        _LOGGER.info("Deleted connection(s) at edge indices %s", sorted(to_delete))
        if self.state.selected_node_index is not None:
            self._refresh_connections_table(self.state.selected_node_index)

    # -----------------------------------------------------------------------
    # Port management
    # -----------------------------------------------------------------------

    def _rebuild_all_ports(self):
        """
        Recompute every port ball position from the current placed-nodes data
        and push the result into ``source_port_source`` / ``target_port_source``.

        Called whenever nodes are added, moved, deleted, or their port counts
        are changed via the spinner widgets.
        """
        if self.state.source_port_source is None or self.state.target_port_source is None:
            return

        _SEL_FILL, _SEL_LINE = 0.3, 0.5
        _DIM_FILL, _DIM_LINE = 0.0, 0.0

        pdata = self.state.placed_nodes_source.data
        xs = list(pdata.get("x", []))
        ys = list(pdata.get("y", []))
        icon_types = list(pdata.get("icon_type", []))
        node_types = list(pdata.get("node_type", []))
        node_names = list(pdata.get("name", []))
        n_sources_list = list(pdata.get("n_sources", []))
        n_targets_list = list(pdata.get("n_targets", []))

        src_x, src_y, src_color, src_label, src_node_idx, src_node_name, src_node_type = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        src_fill_alpha, src_line_alpha = [], []

        tgt_x, tgt_y, tgt_color, tgt_label, tgt_node_idx, tgt_node_name, tgt_node_type = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        tgt_fill_alpha, tgt_line_alpha = [], []

        # Build a set of connected (kind, node_index, label) tuples from edge_source
        connected_ports: set = set()
        if self.state.edge_source is not None:
            edge_data = self.state.edge_source.data
            for i in range(len(edge_data.get("node_a_idx", []))):
                connected_ports.add(
                    (edge_data["a_kind"][i], edge_data["node_a_idx"][i], edge_data["a_label"][i])
                )
                connected_ports.add(
                    (edge_data["b_kind"][i], edge_data["node_b_idx"][i], edge_data["b_label"][i])
                )

        has_sel = self.state.selected_node_index is not None
        selected_idx = self.state.selected_node_index

        for i, (cx, cy, icon_type, node_type, node_name) in enumerate(
            zip(xs, ys, icon_types, node_types, node_names)
        ):
            n_src = (
                int(n_sources_list[i])
                if i < len(n_sources_list)
                else self.state.default_source_count.get(node_type, 0)
            )
            n_tgt = (
                int(n_targets_list[i])
                if i < len(n_targets_list)
                else self.state.default_target_count.get(node_type, 0)
            )

            cfg = ICONS_CONFIG.get(icon_type, {})
            raw_src_color = cfg.get("source_color") or DEFAULT_COLOR
            raw_tgt_color = cfg.get("target_color") or DEFAULT_COLOR

            ports = compute_ports(cx, cy, NODE_RADIUS, PORT_RADIUS, n_src, n_tgt)

            is_sel = has_sel and i == selected_idx
            f_a = _SEL_FILL if is_sel else _DIM_FILL
            l_a = _SEL_LINE if is_sel else _DIM_LINE

            for p in ports["outputs"]:
                src_x.append(p["x"])
                src_y.append(p["y"])
                src_color.append(raw_src_color)
                src_label.append(str(p["index"] + 1))
                src_node_idx.append(i)
                src_node_name.append([node_name])
                src_node_type.append([node_type])
                src_fill_alpha.append(f_a)
                src_line_alpha.append(l_a)

            for p in ports["inputs"]:
                tgt_x.append(p["x"])
                tgt_y.append(p["y"])
                tgt_color.append(raw_tgt_color)
                tgt_label.append(str(p["index"] + 1))
                tgt_node_idx.append(i)
                tgt_node_name.append([node_name])
                tgt_node_type.append([node_type])
                tgt_fill_alpha.append(f_a)
                tgt_line_alpha.append(l_a)

        self.state.source_port_source.data = dict(
            x=src_x,
            y=src_y,
            color=src_color,
            label=src_label,
            node_index=src_node_idx,
            node_name=src_node_name,
            node_type=src_node_type,
            fill_alpha=src_fill_alpha,
            line_alpha=src_line_alpha,
            kind=["source"] * len(src_x),
            connected=[
                "True" if ("source", src_node_idx[i], src_label[i]) in connected_ports else "False"
                for i in range(len(src_x))
            ],
        )
        self.state.target_port_source.data = dict(
            x=tgt_x,
            y=tgt_y,
            color=tgt_color,
            label=tgt_label,
            node_index=tgt_node_idx,
            node_name=tgt_node_name,
            node_type=tgt_node_type,
            fill_alpha=tgt_fill_alpha,
            line_alpha=tgt_line_alpha,
            kind=["target"] * len(tgt_x),
            connected=[
                "True" if ("target", tgt_node_idx[i], tgt_label[i]) in connected_ports else "False"
                for i in range(len(tgt_x))
            ],
        )
        self._rebuild_edges()

        if self.state.selected_node_index is not None:
            self._refresh_connections_table(self.state.selected_node_index)

    def _best_possible_node(self, x: float, y: float):
        """
        Find the nearest placed node to canvas coordinates ``(x, y)``.

        Searches all nodes in ``placed_nodes_source`` and returns the one
        closest to the tap point, provided it falls within ``icon_size`` pixels.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.

        :return: ``(best_idx, best_dist, current_data)`` where ``best_idx`` is the
            zero-based index of the closest node (``None`` if nothing is within snap
            distance), ``best_dist`` is the Euclidean distance to that node
            (``None`` if the source is empty), and ``current_data`` is the raw
            ``placed_nodes_source.data`` dict.
        """
        current = self.state.placed_nodes_source.data
        xs = list(current.get("x", []))
        ys = list(current.get("y", []))
        if not xs:
            return None, None, current

        # Find the nearest icon within snap distance
        snap = self.icon_size
        best_idx = None
        best_dist = float("inf")
        for i, (ix, iy) in enumerate(zip(xs, ys)):
            dist = ((x - ix) ** 2 + (y - iy) ** 2) ** 0.5
            if dist < snap and dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx, best_dist, current

    # -----------------------------------------------------------------------
    # Canvas tap handler
    # -----------------------------------------------------------------------

    def on_canvas_tap(self, event):
        """
        Handle a tap event on the main canvas.

        Behaviour depends on the current interaction mode:

        * **Delete mode** – remove the nearest placed icon within snap distance.
        * **Component selected** – place a new icon at the tap coordinates.
        * **Neither** – select or deselect the nearest existing node for editing.

        :param event: Bokeh ``Tap`` event carrying ``x`` and ``y`` coordinates.
        """
        x, y = event.x, event.y

        # ── Port connection (highest priority, only in idle mode) ─────────────
        if not self.state.delete_mode and self.state.selected_component is None:
            nearest_port = self._find_nearest_port(x, y)
            if nearest_port is not None:
                self._handle_port_tap(nearest_port)
                return

        if self.state.delete_mode:
            best_idx, best_dist, current = self._best_possible_node(x, y)

            if best_idx is not None:
                new_data = {k: list(v) for k, v in current.items()}
                for col in new_data:
                    new_data[col].pop(best_idx)
                self.state.placed_nodes_source.data = new_data

                # Keep hover_source in sync
                if self.state.hover_source is not None:
                    hdata = {k: list(v) for k, v in self.state.hover_source.data.items()}
                    for col in hdata:
                        if best_idx < len(hdata[col]):
                            hdata[col].pop(best_idx)
                    self.state.hover_source.data = hdata

                # Prune edges that referenced the deleted node; shift remaining indices
                if self.state.edge_source is not None:
                    edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
                    keep = [
                        i
                        for i, (na, nb) in enumerate(
                            zip(edge_data.get("node_a_idx", []), edge_data.get("node_b_idx", []))
                        )
                        if na != best_idx and nb != best_idx
                    ]
                    new_edge_data = {}
                    for k, vals in edge_data.items():
                        kept = [vals[i] for i in keep]
                        if k == "node_a_idx":
                            kept = [v - 1 if v > best_idx else v for v in kept]
                        elif k == "node_b_idx":
                            kept = [v - 1 if v > best_idx else v for v in kept]
                        new_edge_data[k] = kept
                    self.state.edge_source.data = new_edge_data

                # Rebuild port balls after deletion
                self._rebuild_all_ports()

                # Update selected index after deletion
                if self.state.selected_node_index == best_idx:
                    self.state.selected_node_index = None
                    self._clear_node_table()
                elif (
                    self.state.selected_node_index is not None
                    and self.state.selected_node_index > best_idx
                ):
                    self.state.selected_node_index -= 1

                _LOGGER.info("Deleted node at index %d", best_idx)

            else:
                # No nearest node – attempt to delete an edge if within snap distance
                edge_idx = self._find_nearest_edge(x, y)
                if edge_idx is not None and self.state.edge_source is not None:
                    edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
                    for k in edge_data:
                        edge_data[k].pop(edge_idx)
                    self.state.edge_source.data = edge_data
                    _LOGGER.info("Delete edge at index %d", edge_idx)

                if self.state.selected_node_index is not None:
                    self._refresh_connections_table(self.state.selected_node_index)

            return

        if self.state.selected_component is None:
            best_idx, best_dist, current = self._best_possible_node(x, y)

            # Cancel any pending port if tapping a node or empty space
            self._cancel_pending_connection()

            if best_idx is None and best_dist is None:
                return

            elif best_idx is None or self.state.selected_node_index == best_idx:
                # Tapped on empty space / Second tap on same node → deselect current node
                self.state.selected_node_index = None
                self._clear_temp_edges()
                self._clear_node_table()
                return

            else:
                # Select this node and show the configurator panel
                self._clear_temp_edges()
                self.state.selected_node_index = best_idx
                self._populate_node_table(best_idx)
                if self.state.table_panel is not None:
                    self.state.table_panel.visible = True
            return

        # Deselect any previously selected node before placing a new component
        if self.state.selected_node_index is not None:
            self.state.selected_node_index = None
            self._clear_temp_edges()
            self._clear_node_table()

        comp_key = self.state.selected_component

        # Generate a unique node name (e.g. "battery_1", "battery_2", …)
        count = self.state.placed_counter.get(comp_key, 0) + 1
        self.state.placed_counter[comp_key] = count
        node_name = f"{comp_key}_{count}"

        # Base64-encode the icon for reliable rendering inside the server
        icon_path = ICONS_CONFIG[comp_key]["icon_path"]
        file_url = "file://" + str(Path(icon_path).resolve())
        b64_url = _url_to_base64(file_url)

        # Resolve default node_type and position from lookup tables
        default_type = self.state.component_type_to_icon.get(comp_key, comp_key)[0]
        position_choices = self.state.possible_position.get(default_type, [])
        default_position = position_choices[0] if position_choices else ""

        # Build default options JSON from POSSIBLE_OPTIONS
        opts_def = POSSIBLE_OPTIONS.get(default_type, {})
        default_opts = {
            k: (True if v_list[0] is True else (False if v_list[0] is False else v_list[0]))
            for k, v_list in opts_def.items()
            if v_list
        }
        default_opts_json = json.dumps(default_opts)

        # Default port counts from constants
        default_n_src = self.state.default_source_count.get(default_type, 0)
        default_n_tgt = self.state.default_target_count.get(default_type, 0)

        # Append new node to the placed-nodes source
        size = self.icon_size
        current = self.state.placed_nodes_source.data
        self.state.placed_nodes_source.data = {
            "x": list(current["x"]) + [x],
            "y": list(current["y"]) + [y],
            "url": list(current["url"]) + [b64_url],
            "w": list(current["w"]) + [size],
            "h": list(current["h"]) + [size],
            "name": list(current["name"]) + [node_name],
            "icon_type": list(current.get("icon_type", [])) + [comp_key],
            "node_type": list(current.get("node_type", [])) + [default_type],
            "position": list(current.get("position", [])) + [default_position],
            "options": list(current.get("options", [])) + [default_opts_json],
            "n_sources": list(current.get("n_sources", [])) + [default_n_src],
            "n_targets": list(current.get("n_targets", [])) + [default_n_tgt],
        }

        # Keep hover_source in sync (position + metadata for the scatter hover tool)
        if self.state.hover_source is not None:
            hdata = self.state.hover_source.data
            self.state.hover_source.data = {
                "x": list(hdata["x"]) + [x],
                "y": list(hdata["y"]) + [y],
                "name": list(hdata["name"]) + [node_name],
                "node_type": list(hdata.get("node_type", [])) + [default_type],
            }

        # Rebuild port balls for the new node
        self._rebuild_all_ports()

        _LOGGER.info(
            "Placed %s (node_type=%s, position=%s) at (%.1f, %.1f)",
            node_name,
            default_type,
            default_position,
            x,
            y,
        )


# ============================================================================
# Standalone launcher
# ============================================================================


class PowertrainBuilderLauncher:
    """
    Launch a self-contained Bokeh server that demonstrates the powertrain builder.

    A blank canvas is placed at the center next to the component button palette on the left and
    the component configuration panel on the right.
    """

    @staticmethod
    def launch(port: int = 5007, address: str = "localhost"):
        """
        Start the palette demo server and open it in the default browser.

        :param port: TCP port for the Bokeh server.
        :param address: Server bind address.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logging.getLogger("bokeh").setLevel(logging.WARNING)
        logging.getLogger("tornado").setLevel(logging.WARNING)

        def make_document(doc):
            """Build the Bokeh document with palette, canvas, and configration panel."""
            palette_layout, table_panel, state = ComponentPaletteBuilder.build()

            # Blank canvas sized to accommodate all component rows
            canvas = bkplot.figure(
                width=800,
                height=950,
                x_range=(0, 800),
                y_range=(0, 950),
                toolbar_location="above",
                background_fill_color=BACKGROUND_COLOR_CODE,
                title="Powertrain Builder – click to place components",
            )
            canvas.xgrid.visible = False
            canvas.ygrid.visible = False
            canvas.xaxis.visible = False
            canvas.yaxis.visible = False
            canvas.title.text_color = BACKGROUND_COLOR_CODE

            # Edge lines
            canvas.multi_line(
                xs="xs",
                ys="ys",
                line_color="color",
                line_width=4,
                line_alpha=0.85,
                source=state.edge_source,
            )
            # Dashed preview edges – shown while connections are pending Apply
            canvas.multi_line(
                xs="xs",
                ys="ys",
                line_color="color",
                line_width=3,
                line_alpha=0.65,
                line_dash="dashed",
                source=state.temp_edge_source,
            )
            # Select indicator ring around selected nodes
            canvas.scatter(
                x="x",
                y="y",
                size=ICON_SIZE + 14,
                source=state.selected_node_overlay_source,
                fill_color=NODE_SELECT_COLOR,
                fill_alpha=0.3,
                line_color=NODE_SELECT_COLOR,
                line_alpha=0.7,
                line_width=3,
            )
            # Render placed node icons on the canvas
            canvas.image_url(
                url="url",
                x="x",
                y="y",
                w="w",
                h="h",
                anchor="center",
                source=state.placed_nodes_source,
            )
            # Pending port highlight ring
            canvas.scatter(
                x="x",
                y="y",
                size=PORT_RADIUS * 2 + 12,
                source=state.pending_port_source,
                fill_color="color",
                fill_alpha=0.35,
                line_color="white",
                line_width=2,
                line_dash="dashed",
            )

            # Transparent scatter glyph for hover interaction
            scatter_glyph = canvas.scatter(
                x="x",
                y="y",
                size=ICON_SIZE + 10,
                source=state.hover_source,
                fill_alpha=0,
                line_alpha=0,
                hover_fill_alpha=0.1,
                hover_line_alpha=0.3,
            )

            # Source port balls (button half of each node)
            canvas.scatter(
                x="x",
                y="y",
                size=PORT_RADIUS * 2,
                marker="circle",
                fill_color=BACKGROUND_COLOR_CODE,
                line_color="color",
                line_width=2.0,
                fill_alpha=0.9,
                source=state.source_port_source,
            )
            source_glyph = canvas.scatter(
                x="x",
                y="y",
                size=PORT_RADIUS * 2 + 5,
                source=state.source_port_source,
                fill_color="color",
                line_color="color",
                fill_alpha="fill_alpha",
                line_alpha="line_alpha",
                hover_fill_alpha=0.3,
                hover_line_alpha=0.5,
            )
            canvas.text(
                x="x",
                y="y",
                text="label",
                source=state.source_port_source,
                text_align="center",
                text_baseline="middle",
                text_font_size="9px",
                text_font_style="bold",
                text_color="color",
            )

            # Target port balls (top half of each node)
            canvas.scatter(
                x="x",
                y="y",
                size=PORT_RADIUS * 2,
                marker="circle",
                fill_color="color",
                line_color="white",
                line_width=2.0,
                fill_alpha=0.9,
                source=state.target_port_source,
            )
            target_glyph = canvas.scatter(
                x="x",
                y="y",
                size=PORT_RADIUS * 2 + 5,
                source=state.target_port_source,
                fill_color="color",
                line_color="color",
                fill_alpha="fill_alpha",
                line_alpha="line_alpha",
                hover_fill_alpha=0.3,
                hover_line_alpha=0.5,
            )
            canvas.text(
                x="x",
                y="y",
                text="label",
                source=state.target_port_source,
                text_align="center",
                text_baseline="middle",
                text_font_size="9px",
                text_font_style="bold",
                text_color="white",
            )
            # node hover tooltips
            hover_tool_component = bkmodel.HoverTool(
                renderers=[scatter_glyph],
                tooltips=[("Component id", "@name"), ("Component type", "@node_type")],
            )
            # port hover tooltips – show different port type in tooltip and relevant metadata
            hover_tool_source = bkmodel.HoverTool(
                renderers=[source_glyph],
                tooltips=[
                    ("Port type", "Source"),
                    ("Port number", "@label"),
                    ("Component id", "@node_name"),
                    ("Component type", "@node_type"),
                ],
            )
            hover_tool_target = bkmodel.HoverTool(
                renderers=[target_glyph],
                tooltips=[
                    ("Port type", "Target"),
                    ("Port number", "@label"),
                    ("Component id", "@node_name"),
                    ("Component type", "@node_type"),
                ],
            )
            canvas.add_tools(hover_tool_component, hover_tool_source, hover_tool_target)

            # Label set for placed node names
            placed_label_source = bkmodel.ColumnDataSource(data=dict(x=[], y=[], text=[]))
            canvas.add_layout(
                bkmodel.LabelSet(
                    x="x",
                    y="y",
                    text="text",
                    source=placed_label_source,
                    text_color="white",
                    text_font_size="8pt",
                    text_align="center",
                    text_baseline="top",
                    y_offset=-ICON_SIZE // 2 - 2,
                )
            )

            # Keep labels in sync with the placed-nodes source
            def _sync_labels(attr, old, new_data):
                placed_label_source.data = dict(
                    x=list(new_data.get("x", [])),
                    y=list(new_data.get("y", [])),
                    text=list(new_data.get("name", [])),
                )

            state.placed_nodes_source.on_change("data", _sync_labels)

            # Refresh position_select and options table when component type changes
            def _on_type_select_change(attr, old, new):
                pos_choices = state.possible_position.get(new, [])
                state.position_select.options = pos_choices
                state.position_select.value = pos_choices[0] if pos_choices else ""
                idx = state.selected_node_index
                saved_opts = {}
                if idx is not None:
                    saved_opts_json = list(state.placed_nodes_source.data.get("options", []))
                    if idx < len(saved_opts_json):
                        try:
                            saved_opts = json.loads(saved_opts_json[idx]) or {}
                        except (json.JSONDecodeError, TypeError):
                            pass
                handler._refresh_options_table(new, saved_opts)

                # Update options table visibility based on whether the new type has any options
                opts_def = POSSIBLE_OPTIONS.get(new, {})
                state.options_table.visible = bool(opts_def)

                # Refresh symmetry_select: peer candidates depend on icon_type, not node_type,
                # so look up the icon_type of the currently selected node.
                if idx is not None:
                    icon_types = list(state.placed_nodes_source.data.get("icon_type", []))
                    icon_type = icon_types[idx] if idx < len(icon_types) else ""
                    handler._refresh_symmetry_select(idx, icon_type)

            state.type_select.on_change("value", _on_type_select_change)

            # Write config panel values back to placed_nodes_source on Apply
            def apply_node_configurations():
                idx = state.selected_node_index
                if idx is None:
                    return
                new_om_name = state.name_input.value
                new_node_type = state.type_select.value
                new_position = state.position_select.value

                # Collect and encode options as JSON
                opt_names = list(state.options_source.data.get("options", []))
                opt_vals = list(state.options_source.data.get("value", []))
                opts_dict = {
                    k: PlacementHandler._str_to_option_val(v) for k, v in zip(opt_names, opt_vals)
                }
                opts_json = json.dumps(opts_dict)

                pdata = {k: list(v) for k, v in state.placed_nodes_source.data.items()}
                if idx < len(pdata.get("name", [])):
                    pdata["name"][idx] = new_om_name
                if idx < len(pdata.get("node_type", [])):
                    pdata["node_type"][idx] = new_node_type
                    # If the node_type changed, we may need to reset port counts to defaults for the new type
                    if not state.source_count_spinner.visible:
                        pdata["n_sources"][idx] = int(
                            state.default_source_count.get(new_node_type, 0)
                        )
                        pdata["n_targets"][idx] = int(
                            state.default_target_count.get(new_node_type, 0)
                        )
                if idx < len(pdata.get("position", [])):
                    pdata["position"][idx] = new_position
                if "options" not in pdata:
                    pdata["options"] = ["{}"] * len(pdata.get("name", []))
                if idx < len(pdata["options"]):
                    pdata["options"][idx] = opts_json
                # Update port counts if the spinners are visible (i.e. editable for this component)
                if state.source_count_spinner is not None and state.source_count_spinner.visible:
                    new_n_src = int(state.source_count_spinner.value)
                    if "n_sources" in pdata and idx < len(pdata["n_sources"]):
                        pdata["n_sources"][idx] = new_n_src

                if state.target_count_spinner is not None and state.target_count_spinner.visible:
                    new_n_tgt = int(state.target_count_spinner.value)
                    if "n_targets" in pdata and idx < len(pdata["n_targets"]):
                        pdata["n_targets"][idx] = new_n_tgt
                state.placed_nodes_source.data = pdata

                hdata = {k: list(v) for k, v in state.hover_source.data.items()}
                if idx < len(hdata.get("name", [])):
                    hdata["name"][idx] = new_om_name
                if idx < len(hdata.get("node_type", [])):
                    hdata["node_type"][idx] = new_node_type
                state.hover_source.data = hdata

                # ── Symmetry: persist selected symmetry peer ─────────────────
                new_sym_name = (
                    state.symmetry_select.value if state.symmetry_select is not None else ""
                )
                pdata2 = {k: list(v) for k, v in state.placed_nodes_source.data.items()}
                names_list = pdata2.get("name", [])

                # Find the node index of the chosen symmetry peer (or -1 if none)
                sym_peer_idx = -1
                if new_sym_name:
                    for _j, _n in enumerate(names_list):
                        if _n == new_sym_name:
                            sym_peer_idx = _j
                            break

                # Write this node's symmetry columns
                if "symmetry_name" not in pdata2:
                    pdata2["symmetry_name"] = [""] * len(names_list)
                if "symmetry_node_index" not in pdata2:
                    pdata2["symmetry_node_index"] = [-1] * len(names_list)
                if idx < len(pdata2["symmetry_name"]):
                    pdata2["symmetry_name"][idx] = new_sym_name
                if idx < len(pdata2["symmetry_node_index"]):
                    pdata2["symmetry_node_index"][idx] = sym_peer_idx

                # Sync back: if the peer exists, point it to this node so that
                # when the peer is selected its symmetry_select default is correct.
                current_node_name = names_list[idx] if idx < len(names_list) else ""
                if sym_peer_idx >= 0:
                    if sym_peer_idx < len(pdata2["symmetry_name"]):
                        pdata2["symmetry_name"][sym_peer_idx] = current_node_name
                    if sym_peer_idx < len(pdata2["symmetry_node_index"]):
                        pdata2["symmetry_node_index"][sym_peer_idx] = idx
                # If symmetry was cleared, also clear the old peer's back-reference
                elif not new_sym_name:
                    for _j in range(len(names_list)):
                        if (
                            _j != idx
                            and _j < len(pdata2["symmetry_name"])
                            and pdata2["symmetry_name"][_j] == current_node_name
                        ):
                            pdata2["symmetry_name"][_j] = ""
                            if _j < len(pdata2["symmetry_node_index"]):
                                pdata2["symmetry_node_index"][_j] = -1

                state.placed_nodes_source.data = pdata2
                # ─────────────────────────────────────────────────────────────

                for port_a, port_b in list(state.pending_connections):
                    handler._add_edge(port_a, port_b)
                handler._clear_temp_edges()  # removes dashed lines + clears list
                handler._rebuild_all_ports()
                # Refresh the connections panel so it reflects the newly committed
                # edges and keeps connections_source in sync with edge_source.
                handler._refresh_connections_table(idx)

                _cs = state.connections_source.data if state.connections_source is not None else {}
                _ed = state.edge_source.data if state.edge_source is not None else {}
                _names = list(state.placed_nodes_source.data.get("name", []))
                _conn_entries = []
                for _i in range(len(_ed.get("node_a_idx", []))):
                    _na = _ed["node_a_idx"][_i]
                    _nb = _ed["node_b_idx"][_i]
                    if _na == idx or _nb == idx:
                        _peer = _nb if _na == idx else _na
                        _peer_name = _names[_peer] if _peer < len(_names) else f"node_{_peer}"
                        _my_kind = _ed["a_kind"][_i] if _na == idx else _ed["b_kind"][_i]
                        _my_lbl = _ed["a_label"][_i] if _na == idx else _ed["b_label"][_i]
                        _pr_kind = _ed["b_kind"][_i] if _na == idx else _ed["a_kind"][_i]
                        _pr_lbl = _ed["b_label"][_i] if _na == idx else _ed["a_label"][_i]
                        _conn_entries.append(
                            {
                                "my_port": f"{_my_kind}:{_my_lbl}",
                                "connected_to_node": _peer_name,
                                "connected_to_port": f"{_pr_kind}:{_pr_lbl}",
                            }
                        )
                connections_json = json.dumps(_conn_entries)
                _LOGGER.info(
                    "Applied: node[%d] name=%s type=%s position=%s options=%s connections=%s",
                    idx,
                    new_om_name,
                    new_node_type,
                    new_position,
                    opts_json,
                    connections_json,
                )

            state.apply_button.on_click(apply_node_configurations)

            # Wire palette buttons and canvas tap
            handler = PlacementHandler(state, canvas)
            canvas.on_event(Tap, handler.on_canvas_tap)

            doc.add_root(row(palette_layout, canvas, table_panel))
            doc.title = "Powertrain Builder"

        def make_document_with_tracking(doc):
            """Wrap ``make_document`` to stop the IO loop when the session ends."""
            make_document(doc)

            def on_destroy(session_context):
                IOLoop.current().stop()

            doc.on_session_destroyed(on_destroy)

        server = Server(
            {"/": make_document_with_tracking},
            port=port,
            address=address,
            num_procs=1,
        )
        server.start()

        IOLoop.current().call_later(0.1, lambda: webbrowser.open(f"http://{address}:{port}/"))
        server.io_loop.start()
