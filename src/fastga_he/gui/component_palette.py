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

    from fastga_he.gui.component_palette import ComponentPaletteBuilder, ComponentPaletteLauncher

    # --- standalone demo ---
    ComponentPaletteLauncher.launch(port=5007)

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
    POSSIBLE_POSITIONS,
    ICON_TYPE,
    POSSIBLE_COMPONENT_TYPES,
    POSSIBLE_OPTIONS,
    DEFAULT_SOURCE_COUNT,
    DEFAULT_TARGET_NUMBER,
)
from fastga_he.gui.power_train_network_viewer import (
    BACKGROUND_COLOR_CODE,
    DEFAULT_COLOR,
    ICONS_CONFIG,
    _string_cleanup,
    _url_to_base64,
)

_LOGGER = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
class PaletteState:
    """
    Holds references to all Bokeh widgets managed by the palette.

    Passed by reference between :class:`ComponentPaletteBuilder`,
    :class:`PlacementHandler`, and :class:`ComponentPaletteLauncher` so that
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
    # DataTable for editing component options (shown in config panel)
    options_table: bkmodel.DataTable = field(default=None)
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
        Construct the button palette and initialise a :class:`PaletteState`.

        :return: ``(palette_column_layout, table_layout, PaletteState)``
        """
        component_keys = list(ICONS_CONFIG.keys())

        # Title div for the palette sidebar
        title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Components</b>",
            width=PALETTE_WIDTH,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )

        # One button per component – callbacks wired later by PlacementHandler.
        # Buttons are kept in ICONS_CONFIG order for index-based selection.
        buttons = []
        button_by_key: dict = {}
        for key in component_keys:
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
            )
        )
        selected_node_overlay_source = bkmodel.ColumnDataSource(data=dict(x=[], y=[]))

        # Options table source for the component configurator panel
        options_table_source = bkmodel.ColumnDataSource(data=dict(options=[], value=[]))
        option_column = bkmodel.TableColumn(field="options", title="Options")
        value_column = bkmodel.TableColumn(
            field="value",
            title="Value",
            editor=bkmodel.StringEditor(),
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

        options_list = bkmodel.DataTable(
            columns=[option_column, value_column],
            source=options_table_source,
            width=380,
            height=200,
            editable=True,
            styles={"color": "black", "font-size": "18px"},
        )

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
            apply_button,
            spacing=4,
            visible=False,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "10px"},
        )

        state = PaletteState(
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
            options_source=options_table_source,
            name_input=name_input,
            type_select=type_select,
            position_select=position_select,
            apply_button=apply_button,
            table_panel=table_panel,
            selected_node_overlay_source=selected_node_overlay_source,
        )

        # Build one TabPanel per category defined in ICON_TYPE
        tab_panels = []
        for category, keys_in_category in ICON_TYPE.items():
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

    def __init__(self, state: PaletteState, main_plot, icon_size: int = 50):
        """
        :param state: Shared :class:`PaletteState` instance.
        :param main_plot: The Bokeh ``figure`` that acts as the placement canvas.
        :param icon_size: Pixel size (width = height) used for placed icons.
        """
        self.state = state
        self.main_plot = main_plot
        self.icon_size = icon_size
        self._wire_buttons()

    # -----------------------------------------------------------------------
    # Internal wiring
    # -----------------------------------------------------------------------

    def _wire_buttons(self):
        """Attach ``on_click`` callbacks to every palette and action button."""
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

        def _callback():
            self.on_palette_select(idx)

        return _callback

    # -----------------------------------------------------------------------
    # Delete mode
    # -----------------------------------------------------------------------

    def _toggle_delete_mode(self):
        """Toggle delete mode on / off, updating button styling and status text."""
        self.state.delete_mode = not self.state.delete_mode
        if self.state.delete_mode:
            # Enter delete mode: deselect any active component
            self.state.selected_component = None
            for btn in self.state.buttons:
                btn.button_type = BUTTON_DEFAULT_COLOR_TYPE
            self.state.status_div.text = "<b style='color:#FF4444;font-size:14pt'>Delete mode: click an icon to remove it</b>"
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
        Serialise the current placed-nodes data to a timestamped JSON file.

        The file is written to the current working directory and the button
        type is reset to ``"success"`` after a 1-second delay.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"canvas_state_{timestamp}.json"
        data_to_save = self.state.placed_nodes_source.data
        with open(filename, "w") as f:
            json.dump(data_to_save, f, indent=2)
        _LOGGER.info("Canvas state saved to %s", filename)
        IOLoop.current().call_later(
            1.0, lambda: setattr(self.state.save_button, "button_type", "success")
        )

    # -----------------------------------------------------------------------
    # End session
    # -----------------------------------------------------------------------

    def _end_session(self):
        """Stop the Bokeh IO loop, terminating the server session."""
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
        component_keys = list(ICONS_CONFIG.keys())
        if idx < 0 or idx >= len(component_keys):
            return

        # Second click on the same button → deselect
        if self.state.selected_component == component_keys[idx]:
            self.state.selected_component = None
            self.state.buttons[idx].button_type = BUTTON_DEFAULT_COLOR_TYPE
            self.state.status_div.text = (
                "<i style='color:#aaa;font-size:14pt'>Select a component</i>"
            )
            return

        self.state.selected_component = component_keys[idx]

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
        possible = POSSIBLE_COMPONENT_TYPES.get(comp_key, comp_key)
        choices = possible if isinstance(possible, list) else [possible]
        self.state.type_select.options = choices
        self.state.type_select.value = node_type if node_type in choices else choices[0]

        # Populate position_select with valid positions for this node_type
        pos_choices = POSSIBLE_POSITIONS.get(node_type, [])
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
        n_src_default = DEFAULT_SOURCE_COUNT.get(node_type, 0)
        n_tgt_default = DEFAULT_TARGET_NUMBER.get(node_type, 0)
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
        Rebuild the options DataTable for *node_type*, applying any *overrides*.

        :param node_type: Component type string used as key into ``POSSIBLE_OPTIONS``.
        :param overrides: Previously saved option values that take precedence over defaults.
        """
        if overrides is None:
            overrides = {}
        opts_def = POSSIBLE_OPTIONS.get(node_type, {})
        opt_names = list(opts_def.keys())
        opt_values = []
        for k, v_list in opts_def.items():
            if k in overrides:
                opt_values.append(self._option_val_to_str(overrides[k]))
            else:
                default = v_list[0] if v_list else ""
                opt_values.append(self._option_val_to_str(default))
        self.state.options_source.data = dict(options=opt_names, value=opt_values)

    def _clear_node_table(self):
        """Reset all config panel inputs and hide the panel."""
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
        self._rebuild_all_ports()

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

        has_sel = self.state.selected_node_index is not None
        selected_idx = self.state.selected_node_index

        for i, (cx, cy, icon_type, node_type, node_name) in enumerate(
            zip(xs, ys, icon_types, node_types, node_names)
        ):
            n_src = (
                int(n_sources_list[i])
                if i < len(n_sources_list)
                else DEFAULT_SOURCE_COUNT.get(node_type, 0)
            )
            n_tgt = (
                int(n_targets_list[i])
                if i < len(n_targets_list)
                else DEFAULT_TARGET_NUMBER.get(node_type, 0)
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
        )

    def _best_possible_node(self, x: float, y: float):
        """Return the best possible node index for the tap action."""
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
            return

        if self.state.selected_component is None:
            best_idx, best_dist, current = self._best_possible_node(x, y)

            if best_idx is None and best_dist is None:
                return

            elif best_idx is None or self.state.selected_node_index == best_idx:
                # Tapped on empty space / Second tap on same node → deselect current node
                self.state.selected_node_index = None
                self._clear_node_table()
                return

            else:
                # Select this node and show the configurator panel
                self.state.selected_node_index = best_idx
                self._populate_node_table(best_idx)
                if self.state.table_panel is not None:
                    self.state.table_panel.visible = True
            return

        # Deselect any previously selected node before placing a new component
        if self.state.selected_node_index is not None:
            self.state.selected_node_index = None
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
        possible_type = POSSIBLE_COMPONENT_TYPES.get(comp_key, comp_key)
        default_type = possible_type[0] if isinstance(possible_type, list) else possible_type
        position_choices = POSSIBLE_POSITIONS.get(default_type, [])
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
        default_n_src = DEFAULT_SOURCE_COUNT.get(default_type, 0)
        default_n_tgt = DEFAULT_TARGET_NUMBER.get(default_type, 0)

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


class ComponentPaletteLauncher:
    """
    Launch a self-contained Bokeh server that demonstrates the palette.

    A blank canvas is placed next to the button palette so you can click
    components and see them appear on the canvas.
    """

    @staticmethod
    def launch(port: int = 5007, address: str = "localhost"):
        """
        Start the palette demo server and open it in the default browser.

        :param port: TCP port for the Bokeh server.
        :param address: Server bind address.
        """
        logging.getLogger("bokeh").setLevel(logging.WARNING)
        logging.getLogger("tornado").setLevel(logging.WARNING)

        def make_document(doc):
            """Build the Bokeh document with palette, canvas, and config panel."""
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
            hover_tool_component = bkmodel.HoverTool(
                renderers=[scatter_glyph],
                tooltips=[("Component id", "@name"), ("Component type", "@node_type")],
            )
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
                pos_choices = POSSIBLE_POSITIONS.get(new, [])
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

            state.type_select.on_change("value", _on_type_select_change)

            # Write config panel values back to placed_nodes_source on Apply
            def _apply_node_config():
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
                        pdata["n_sources"][idx] = int(DEFAULT_SOURCE_COUNT.get(new_node_type, 0))
                        pdata["n_targets"][idx] = int(DEFAULT_TARGET_NUMBER.get(new_node_type, 0))
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

                handler._rebuild_all_ports()

                _LOGGER.info(
                    "Applied: node[%d] name=%s type=%s position=%s options=%s",
                    idx,
                    new_om_name,
                    new_node_type,
                    new_position,
                    opts_json,
                )

            state.apply_button.on_click(_apply_node_config)

            # Wire palette buttons and canvas tap
            handler = PlacementHandler(state, canvas)
            canvas.on_event(Tap, handler.on_canvas_tap)

            doc.add_root(row(palette_layout, canvas, table_panel))
            doc.title = "Component Palette Demo"

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
