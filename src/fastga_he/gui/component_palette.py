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
from pathlib import Path
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
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
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
# Dict of component_type -> list of component_keys (e.g. "source" -> ["battery", "generator", …])

# ---------------------------------------------------------------------------
# Shared mutable state between palette and placement handler
# ---------------------------------------------------------------------------


@dataclass
class PaletteState:
    """Holds references to all Bokeh widgets managed by the palette."""

    # List of Button widgets, one per component (in ICONS_CONFIG order)
    buttons: list = field(default_factory=list)
    # ColumnDataSource that accumulates icons placed on the main canvas
    placed_nodes_source: bkmodel.ColumnDataSource = field(default=None)
    # Div widget showing the currently selected component
    status_div: bkmodel.Div = field(default=None)
    # Currently selected component key (e.g. "battery")
    selected_component: str = field(default=None)
    # Deduplication counters: component_key -> int
    placed_counter: dict = field(default_factory=dict)
    # Delete button
    delete_button: bkmodel.Button = field(default=None)
    # Whether delete mode is active (click on canvas removes nearest icon)
    save_button: bkmodel.Button = field(default=None)
    # Save the current canvas state into a JSON file
    end_session_button: bkmodel.Button = field(default=None)
    # button to end the current session (stop the server)
    delete_mode: bool = field(default=False)


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

        :return: ``(palette_column_layout, PaletteState)``
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
        )

        save_button = bkmodel.Button(
            label="Save",
            button_type="success",
            width=PALETTE_WIDTH - 10,
            height=ROW_HEIGHT - 6,
        )

        end_session_button = bkmodel.Button(
            label="End Session",
            button_type="warning",
            width=PALETTE_WIDTH - 10,
            height=ROW_HEIGHT - 6,
        )
        end_session_button.js_on_click(bkmodel.CustomJS(code="window.close();"))

        # ------------------------------------------------------------------
        # Placed-nodes source used by the *main* canvas
        # ------------------------------------------------------------------
        placed_nodes_source = bkmodel.ColumnDataSource(
            data=dict(x=[], y=[], url=[], w=[], h=[], name=[])
        )

        state = PaletteState(
            buttons=buttons,
            placed_nodes_source=placed_nodes_source,
            status_div=status_div,
            delete_button=delete_button,
            save_button=save_button,
            end_session_button=end_session_button,
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

        return palette_layout, state


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
    # End session
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

    # ------------------------------------------------------------------
    # Canvas tap
    # ------------------------------------------------------------------

    def on_canvas_tap(self, event):
        """
        Called when the user taps the main canvas.
        In normal mode: places the currently selected component icon at the tapped position.
        In delete mode: removes the nearest placed icon within snap distance.
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
                _LOGGER.info("Deleted node at index %d", best_idx)
            return

        if self.state.selected_component is None:
            return

        comp_key = self.state.selected_component

        # Unique node name (e.g. "battery_1", "battery_2", …)
        count = self.state.placed_counter.get(comp_key, 0) + 1
        self.state.placed_counter[comp_key] = count
        node_name = f"{comp_key}_{count}"

        # Icon URL (base64 so it renders inside the server)
        icon_path = ICONS_CONFIG[comp_key]["icon_path"]
        file_url = "file://" + str(Path(icon_path).resolve())
        b64_url = _url_to_base64(file_url)

        size = self.icon_size
        current = self.state.placed_nodes_source.data
        self.state.placed_nodes_source.data = {
            "x": list(current["x"]) + [x],
            "y": list(current["y"]) + [y],
            "url": list(current["url"]) + [b64_url],
            "w": list(current["w"]) + [size],
            "h": list(current["h"]) + [size],
            "name": list(current["name"]) + [node_name],
        }

        _LOGGER.info("Placed %s at (%.1f, %.1f)", node_name, x, y)


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
            palette_layout, state = ComponentPaletteBuilder.build()

            # Blank canvas
            canvas = bkplot.figure(
                width=800,
                height=len(ICONS_CONFIG) * ROW_HEIGHT + 24,
                x_range=(0, 800),
                y_range=(0, len(ICONS_CONFIG) * ROW_HEIGHT),
                toolbar_location="above",
                background_fill_color=BACKGROUND_COLOR_CODE,
                title="Canvas – click to place components",
            )
            canvas.xgrid.visible = False
            canvas.ygrid.visible = False
            canvas.xaxis.visible = False
            canvas.yaxis.visible = False
            canvas.title.text_color = "white"

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

            # Wire buttons and canvas tap
            handler = PlacementHandler(state, canvas)
            canvas.on_event(Tap, handler.on_canvas_tap)

            doc.add_root(row(palette_layout, canvas))
            doc.title = "Component Palette Demo"

        def make_document_with_tracking(doc):
            make_document(doc)

            def on_destroy(_session_context):
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
