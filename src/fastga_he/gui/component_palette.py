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
    palette_layout, state = ComponentPaletteBuilder.build()

    def make_doc(doc):
        from bokeh.layouts import row
        from bokeh.events import Tap
        from fastga_he.gui.component_palette import PlacementHandler

        handler = PlacementHandler(state, main_plot)
        main_plot.on_event(Tap, handler.on_canvas_tap)

        doc.add_root(row(palette_layout, main_plot))

"""

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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastga_he.gui.power_train_network_viewer import (
    BACKGROUND_COLOR_CODE,
    ICONS_CONFIG,
    _string_cleanup,
    _url_to_base64,
)

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

PALETTE_WIDTH = 300
ROW_HEIGHT = 52
ICON_SIZE = 50

# Button appearance
BTN_TYPE_DEFAULT = "light"  # unselected
BTN_TYPE_SELECTED = "primary"  # selected (blue highlight)

ICON_TYPE = {
    "connector": [
        "bus_bar",
        "cable",
        "switch",
        "splitter",
        "rectifier",
        "dc_converter",
        "inverter",
        "gearbox",
        "fuel_system",
    ],
    "load": ["e_motor", "dc_load"],
    "source": [
        "battery",
        "generator",
        "ice",
        "turbine",
        "fuel_cell",
    ],
    "propulsor": ["propeller"],
    "tank": ["fuel_tank"],
}
POSSIBLE_TYPES = {
    "bus_bar": "DC_bus",
    "cable": "DC_cable_harness",
    "switch": "DC_SSPC",
    "splitter": "DC_splitter",
    "rectifier": "rectifier",
    "dc_converter": "DC_DC_converter",
    "inverter": "inverter",
    "gearbox": ["speed_reducer", "planetary_gear", "gearbox"],
    "e_motor": ["PMSM", "SM_PMSM"],
    "dc_load": "aux_load",
    "battery": "battery_pack",
    "generator": ["generator", "turbo_generator"],
    "ice": ["ICE", "high_rpm_ICE"],
    "turbine": "turboshaft",
    "fuel_cell": "PEMFC_stack",
    "propeller": "propeller",
    "fuel_tank": ["fuel_tank", "gaseous_hydrogen_tank"],
    "fuel_system": ["fuel_system", "H2_fuel_system"],
}
# Dict of component_type -> list of component_keys (e.g. "source" -> ["battery", "generator", …])

# ---------------------------------------------------------------------------
# Shared mutable state between palette and placement handler
# ---------------------------------------------------------------------------


@dataclass
class PaletteState:
    """Holds references to all Bokeh widgets managed by the palette."""

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
    # property/value table source – populated when a canvas node is selected
    table_source: bkmodel.ColumnDataSource = field(default=None)
    data_table: bkmodel.DataTable = field(default=None)
    # Select widget for Component Type options (shown in table panel)
    type_select: bkmodel.Select = field(default=None)
    apply_button: bkmodel.Button = field(default=None)
    # Index of the currently selected canvas node (None = nothing selected)
    selected_node_idx: int = field(default=None)
    # The whole config panel column – toggled visible/invisible
    table_panel: object = field(default=None)


# ---------------------------------------------------------------------------
# Palette builder
# ---------------------------------------------------------------------------


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

        # ------------------------------------------------------------------
        # Title div
        # ------------------------------------------------------------------
        title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Components</b>",
            width=PALETTE_WIDTH,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )

        # ------------------------------------------------------------------
        # One button per component (callbacks wired later by PlacementHandler)
        # Buttons are kept in ICONS_CONFIG order for index-based selection.
        # ------------------------------------------------------------------
        buttons = []
        btn_by_key: dict = {}
        for key in component_keys:
            # Add label with cleaned-up component name
            label = _string_cleanup(key)
            # Add icon (base64 so it renders inside the server)
            icon_path = ICONS_CONFIG[key]["icon_path"]
            file_url = "file://" + str(Path(icon_path).resolve())
            b64_url = _url_to_base64(file_url)

            # SVG conversion is a bit hacky but it allows us to control the icon size and avoid
            # blurry rendering that happens when resizing raster images in Bokeh
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 
            120 120">
              <image href="{b64_url}" width="120" height="120"/>
            </svg>"""

            btn = bkmodel.Button(
                label=label,
                icon=bkmodel.SVGIcon(svg=svg, size="2.75em"),
                button_type=BTN_TYPE_DEFAULT,
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
            buttons.append(btn)
            btn_by_key[key] = btn

        # ------------------------------------------------------------------
        # Status div (updated by PlacementHandler on selection)
        # ------------------------------------------------------------------
        status_div = bkmodel.Div(
            text="<i style='color:#aaa;font-size:14pt'>Select a component</i>",
            width=PALETTE_WIDTH,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "14px"},
        )

        delete_button = bkmodel.Button(
            label="Delete",
            icon=bkmodel.TablerIcon(icon_name="trash"),
            button_type=BTN_TYPE_DEFAULT,
            width=PALETTE_WIDTH - 10,
            height=ROW_HEIGHT - 6,
            stylesheets=[
                """
                :host button {
                    font-size: 1.4em;
                }
            """
            ],
        )

        save_button = bkmodel.Button(
            label="Save",
            icon=bkmodel.TablerIcon(icon_name="device-floppy"),
            button_type="success",
            width=PALETTE_WIDTH - 10,
            height=ROW_HEIGHT - 6,
            stylesheets=[
                """
                :host button {
                    font-size: 1.4em;
                }
            """
            ],
        )

        end_session_button = bkmodel.Button(
            label="End Session",
            icon=bkmodel.TablerIcon(icon_name="power"),
            button_type="warning",
            width=PALETTE_WIDTH - 10,
            height=ROW_HEIGHT - 6,
            stylesheets=[
                """
                :host button {
                    font-size: 1.4em;
                }
            """
            ],
        )
        end_session_button.js_on_click(bkmodel.CustomJS(code="window.close();"))

        # ------------------------------------------------------------------
        # Placed-nodes source used by the *main* canvas
        # ------------------------------------------------------------------
        placed_nodes_source = bkmodel.ColumnDataSource(
            data=dict(x=[], y=[], url=[], w=[], h=[], name=[], node_type=[], icon_type=[])
        )

        # ------------------------------------------------------------------
        # Hover source – mirrors placed_nodes_source positions for scatter
        # ------------------------------------------------------------------
        hover_source = bkmodel.ColumnDataSource(data=dict(x=[], y=[], name=[], node_type=[]))

        # ------------------------------------------------------------------
        # Table source – property/value rows for the selected canvas node.
        # property: ["OM Name", "Component Type"]
        # value:    [<om_name>, <node_type>]
        # Populated only when a canvas node is selected.
        # ------------------------------------------------------------------
        table_source = bkmodel.ColumnDataSource(data=dict(property=[], value=[]))

        # All possible node_type strings (used as SelectEditor options)
        _all_types: list = []
        for _v in POSSIBLE_TYPES.values():
            if isinstance(_v, list):
                _all_types.extend(_v)
            else:
                _all_types.append(_v)
        _all_types = sorted(set(_all_types))

        # SelectEditor for the value column – options set to valid types for
        # the selected node when a node is selected.  The "OM Name" row value
        # is also editable as free text via the same StringEditor fallback.
        type_select_editor = bkmodel.SelectEditor(options=_all_types)

        table_columns = [
            bkmodel.TableColumn(
                field="property",
                title="Property",
                width=140,
            ),
            bkmodel.TableColumn(
                field="value",
                title="Value",
                editor=bkmodel.StringEditor(),
                width=220,
            ),
        ]

        # Extra Select widget that shows valid choices and updates the
        # "Component Type" value row when the user picks from it.
        type_select = bkmodel.Select(
            title="Component Type options:",
            value="",
            options=[],
            width=380,
            styles={"color": "white"},
        )

        data_table = bkmodel.DataTable(
            source=table_source,
            columns=table_columns,
            width=380,
            height=80,  # exactly 2 data rows
            editable=True,
            index_position=None,
            styles={"color": "black"},
        )

        apply_button = bkmodel.Button(
            label="Apply",
            icon=bkmodel.TablerIcon(icon_name="check"),
            button_type="primary",
            width=380,
            height=ROW_HEIGHT - 6,
            stylesheets=[":host button { font-size: 1.2em; }"],
        )

        table_title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Component ID & Type</b>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )

        component_option_title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Position & Options</b>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )

        table_panel = column(
            table_title_div,
            data_table,
            type_select,
            component_option_title_div,
            apply_button,
            spacing=4,
            visible=False,  # hidden until a canvas node is selected
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
            table_source=table_source,
            data_table=data_table,
            type_select=type_select,
            apply_button=apply_button,
            table_panel=table_panel,
        )

        # ------------------------------------------------------------------
        # Build one TabPanel per category defined in ICON_TYPE
        # ------------------------------------------------------------------
        tab_panels = []
        for category, keys_in_cat in ICON_TYPE.items():
            cat_buttons = [btn_by_key[k] for k in keys_in_cat if k in btn_by_key]
            if not cat_buttons:
                continue
            tab_col = column(
                *cat_buttons,
                spacing=2,
                styles={"background": BACKGROUND_COLOR_CODE, "padding": "10px"},
            )
            tab_panels.append(bkmodel.TabPanel(child=tab_col, title=category.capitalize()))

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


# ---------------------------------------------------------------------------
# Placement handler
# ---------------------------------------------------------------------------


class PlacementHandler:
    """
    Wires palette button events to canvas placement events.

    Instantiate **after** building the palette; the constructor automatically
    wires all button ``on_click`` callbacks::

        palette_layout, state = ComponentPaletteBuilder.build()
        handler = PlacementHandler(state, main_plot)
        main_plot.on_event(Tap, handler.on_canvas_tap)
        main_plot.image_url(
            url="url", x="x", y="y", w="w", h="h",
            anchor="center", source=state.placed_nodes_source,
        )
    """

    def __init__(self, state: PaletteState, main_plot, icon_size: int = 50):
        """
        :param state: Shared :class:`PaletteState` instance
        :param main_plot: The Bokeh ``figure`` that acts as the canvas
        :param icon_size: Pixel size of placed icons (width = height)
        """
        self.state = state
        self.main_plot = main_plot
        self.icon_size = icon_size
        self._wire_buttons()

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    def _wire_buttons(self):
        """Attach an on_click callback to every palette button."""
        self.state.save_button.on_click(self._save_canvas_state)
        self.state.end_session_button.on_click(self._end_session)
        for idx, btn in enumerate(self.state.buttons):
            btn.on_click(self._make_select_cb(idx))
        if self.state.delete_button is not None:
            self.state.delete_button.on_click(self._toggle_delete_mode)

    def _make_select_cb(self, idx: int):
        """Return a zero-argument closure that selects component at *idx*."""

        def _cb():
            self.on_palette_select(idx)

        return _cb

    # ------------------------------------------------------------------
    # Delete mode toggle
    # ------------------------------------------------------------------

    def _toggle_delete_mode(self):
        """Toggle delete mode on/off."""
        self.state.delete_mode = not self.state.delete_mode
        if self.state.delete_mode:
            # Enter delete mode: deselect any component
            self.state.selected_component = None
            for btn in self.state.buttons:
                btn.button_type = BTN_TYPE_DEFAULT
            self.state.status_div.text = "<b style='color:#FF4444;font-size:14pt'>Delete mode: click an icon to remove it</b>"
            self.state.delete_button.button_type = "danger"
        else:
            # Exit delete mode
            self.state.status_div.text = (
                "<i style='color:#aaa;font-size:14pt'>Select a component</i>"
            )
            self.state.delete_button.button_type = BTN_TYPE_DEFAULT

    # ------------------------------------------------------------------
    # Delete mode toggle
    # ------------------------------------------------------------------

    def _save_canvas_state(self):
        """Save the current canvas state (placed nodes) into a JSON file."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"canvas_state_{timestamp}.json"
        data_to_save = self.state.placed_nodes_source.data
        with open(filename, "w") as f:
            json.dump(data_to_save, f, indent=2)
        _LOGGER.info("Canvas state saved to %s", filename)
        # Reset button state after a short delay
        IOLoop.current().call_later(
            1.0, lambda: setattr(self.state.save_button, "button_type", "success")
        )

    # ------------------------------------------------------------------
    # End session toggle
    # ------------------------------------------------------------------

    def _end_session(self):
        """End the current session by stopping the server."""
        _LOGGER.info("Ending session and stopping server")
        IOLoop.current().stop()

    # ------------------------------------------------------------------
    # Palette selection
    # ------------------------------------------------------------------

    def on_palette_select(self, idx: int):
        """
        Select the component at position *idx* in :data:`ICONS_CONFIG`.

        Updates button styling and the status label.
        Can be called programmatically in tests without a running server.

        :param idx: Zero-based index into ``list(ICONS_CONFIG.keys())``
        """
        component_keys = list(ICONS_CONFIG.keys())
        if idx < 0 or idx >= len(component_keys):
            return

        # Second click on the same button → unselect
        if self.state.selected_component == component_keys[idx]:
            self.state.selected_component = None
            self.state.buttons[idx].button_type = BTN_TYPE_DEFAULT
            self.state.status_div.text = (
                "<i style='color:#aaa;font-size:14pt'>Select a component</i>"
            )
            return

        self.state.selected_component = component_keys[idx]

        # Exit delete mode if active
        if self.state.delete_mode:
            self.state.delete_mode = False
            if self.state.delete_button is not None:
                self.state.delete_button.button_type = BTN_TYPE_DEFAULT

        # Highlight the selected button, reset all others
        for j, btn in enumerate(self.state.buttons):
            btn.button_type = BTN_TYPE_SELECTED if j == idx else BTN_TYPE_DEFAULT

        # Update status label
        label = _string_cleanup(self.state.selected_component)
        self.state.status_div.text = f"<b style='color:#FFD700;font-size:14pt'>Placing: {label}</b>"

    def _populate_node_table(self, idx: int):
        """Fill table_source with the property/value rows for node *idx*."""
        pdata = self.state.placed_nodes_source.data
        om_name = list(pdata.get("name", []))[idx]
        node_type = list(pdata.get("node_type", []))[idx]
        comp_key = list(pdata.get("icon_type", []))[idx]

        self.state.table_source.data = dict(
            property=["Component ID", "Component Type"],
            value=[om_name, node_type],
        )

        # Update type_select options to valid choices for this comp_key
        possible = POSSIBLE_TYPES.get(comp_key, comp_key)
        choices = possible if isinstance(possible, list) else [possible]
        self.state.type_select.options = choices
        self.state.type_select.value = node_type if node_type in choices else choices[0]

    def _clear_node_table(self):
        """Clear table_source and hide the config panel."""
        self.state.table_source.data = dict(property=[], value=[])
        self.state.type_select.options = []
        self.state.type_select.value = ""
        if self.state.table_panel is not None:
            self.state.table_panel.visible = False

    # ------------------------------------------------------------------
    # Canvas tap
    # ------------------------------------------------------------------

    def on_canvas_tap(self, event):
        """
        Called when the user taps the main canvas.

        * Delete mode  → remove nearest icon.
        * Component selected → place a new icon.
        * Neither  → select/deselect existing node for editing.
        """
        x, y = event.x, event.y

        if self.state.delete_mode:
            current = self.state.placed_nodes_source.data
            xs = list(current.get("x", []))
            ys = list(current.get("y", []))
            if not xs:
                return
            # Find nearest icon
            snap = self.icon_size
            best_idx = None
            best_dist = float("inf")
            for i, (ix, iy) in enumerate(zip(xs, ys)):
                dist = ((x - ix) ** 2 + (y - iy) ** 2) ** 0.5
                if dist < snap and dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                new_data = {k: list(v) for k, v in current.items()}
                for col in new_data:
                    new_data[col].pop(best_idx)
                self.state.placed_nodes_source.data = new_data

                # Sync hover_source
                if self.state.hover_source is not None:
                    hdata = {k: list(v) for k, v in self.state.hover_source.data.items()}
                    for col in hdata:
                        if best_idx < len(hdata[col]):
                            hdata[col].pop(best_idx)
                    self.state.hover_source.data = hdata

                # If the deleted node was selected, clear the config panel
                if self.state.selected_node_idx == best_idx:
                    self.state.selected_node_idx = None
                    self._clear_node_table()
                elif (
                    self.state.selected_node_idx is not None
                    and self.state.selected_node_idx > best_idx
                ):
                    self.state.selected_node_idx -= 1

                _LOGGER.info("Deleted node at index %d", best_idx)
            return

        if self.state.selected_component is None:
            # --- Node selection / deselection mode ---
            current = self.state.placed_nodes_source.data
            xs = list(current.get("x", []))
            ys = list(current.get("y", []))
            if not xs:
                return
            snap = self.icon_size
            best_idx = None
            best_dist = float("inf")
            for i, (ix, iy) in enumerate(zip(xs, ys)):
                dist = ((x - ix) ** 2 + (y - iy) ** 2) ** 0.5
                if dist < snap and dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is None:
                # Clicked on empty space → deselect
                self.state.selected_node_idx = None
                self._clear_node_table()
                return
            if self.state.selected_node_idx == best_idx:
                # Second click on same node → deselect
                self.state.selected_node_idx = None
                self._clear_node_table()
            else:
                # Select this node
                self.state.selected_node_idx = best_idx
                self._populate_node_table(best_idx)
                if self.state.table_panel is not None:
                    self.state.table_panel.visible = True
            return

        # Deselect any previously selected node when placing a new component
        if self.state.selected_node_idx is not None:
            self.state.selected_node_idx = None
            self._clear_node_table()

        comp_key = self.state.selected_component

        # Unique node name (e.g. "battery_1", "battery_2", …)
        count = self.state.placed_counter.get(comp_key, 0) + 1
        self.state.placed_counter[comp_key] = count
        node_name = f"{comp_key}_{count}"

        # Icon URL (base64 so it renders inside the server)
        icon_path = ICONS_CONFIG[comp_key]["icon_path"]
        file_url = "file://" + str(Path(icon_path).resolve())
        b64_url = _url_to_base64(file_url)

        # Determine default node_type from POSSIBLE_TYPES
        possible = POSSIBLE_TYPES.get(comp_key, comp_key)
        default_type = possible[0] if isinstance(possible, list) else possible

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
        }

        # Sync hover_source (position + name for scatter hover)
        if self.state.hover_source is not None:
            hdata = self.state.hover_source.data
            self.state.hover_source.data = {
                "x": list(hdata["x"]) + [x],
                "y": list(hdata["y"]) + [y],
                "name": list(hdata["name"]) + [node_name],
                "node_type": list(hdata.get("node_type", [])) + [default_type],
            }

        _LOGGER.info("Placed %s (node_type=%s) at (%.1f, %.1f)", node_name, default_type, x, y)


# ---------------------------------------------------------------------------
# Standalone launcher (demo / development helper)
# ---------------------------------------------------------------------------


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

        :param port: TCP port for the Bokeh server
        :param address: Server address
        """
        logging.getLogger("bokeh").setLevel(logging.WARNING)
        logging.getLogger("tornado").setLevel(logging.WARNING)

        def make_document(doc):
            palette_layout, table_panel, state = ComponentPaletteBuilder.build()

            # Blank canvas
            canvas = bkplot.figure(
                width=800,
                height=len(ICONS_CONFIG) * ROW_HEIGHT + 24,
                x_range=(0, 800),
                y_range=(0, len(ICONS_CONFIG) * ROW_HEIGHT),
                toolbar_location="above",
                background_fill_color=BACKGROUND_COLOR_CODE,
                title="Powertrain Builder – click to place components",
            )
            canvas.xgrid.visible = False
            canvas.ygrid.visible = False
            canvas.xaxis.visible = False
            canvas.yaxis.visible = False
            canvas.title.text_color = BACKGROUND_COLOR_CODE

            # Render placed nodes on the canvas
            canvas.image_url(
                url="url",
                x="x",
                y="y",
                w="w",
                h="h",
                anchor="center",
                source=state.placed_nodes_source,
            )

            # Scatter glyph for hover interaction (partially transparent circle)
            scatter_glyph = canvas.scatter(
                x="x",
                y="y",
                size=55,
                source=state.hover_source,
                fill_alpha=0,
                line_alpha=0,
                hover_fill_alpha=0.1,
                hover_line_alpha=0.3,
            )
            hover_tool = bkmodel.HoverTool(
                renderers=[scatter_glyph],
                tooltips=[("Component id", "@name"), ("Component type", "@node_type")],
            )
            canvas.add_tools(hover_tool)

            # Labels for placed nodes
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

            # Keep labels in sync with placed nodes
            def _sync_labels(attr, old, new_data):
                placed_label_source.data = dict(
                    x=list(new_data.get("x", [])),
                    y=list(new_data.get("y", [])),
                    text=list(new_data.get("name", [])),
                )

            state.placed_nodes_source.on_change("data", _sync_labels)

            # type_select → update "Component Type" value row in table
            def _on_type_select_change(attr, old, new):
                tdata = {k: list(v) for k, v in state.table_source.data.items()}
                values = tdata.get("value", [])
                if len(values) >= 2:
                    values[1] = new
                    tdata["value"] = values
                    state.table_source.data = tdata

            state.type_select.on_change("value", _on_type_select_change)

            # Apply button – write property/value table back to placed_nodes_source
            def _apply_node_config():
                idx = state.selected_node_idx
                if idx is None:
                    return
                tdata = state.table_source.data
                values = list(tdata.get("value", []))
                if len(values) < 2:
                    return
                new_om_name = values[0]
                new_node_type = values[1]

                pdata = {k: list(v) for k, v in state.placed_nodes_source.data.items()}
                if idx < len(pdata.get("name", [])):
                    pdata["name"][idx] = new_om_name
                if idx < len(pdata.get("node_type", [])):
                    pdata["node_type"][idx] = new_node_type
                state.placed_nodes_source.data = pdata

                hdata = {k: list(v) for k, v in state.hover_source.data.items()}
                if idx < len(hdata.get("name", [])):
                    hdata["name"][idx] = new_om_name
                if idx < len(hdata.get("node_type", [])):
                    hdata["node_type"][idx] = new_node_type
                state.hover_source.data = hdata

                _LOGGER.info("Applied: node[%d] name=%s type=%s", idx, new_om_name, new_node_type)

            state.apply_button.on_click(_apply_node_config)

            # Wire buttons and canvas tap
            handler = PlacementHandler(state, canvas)
            canvas.on_event(Tap, handler.on_canvas_tap)

            doc.add_root(row(palette_layout, canvas, table_panel))
            doc.title = "Component Palette Demo"

        def make_document_with_tracking(doc):
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


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ComponentPaletteLauncher.launch()
