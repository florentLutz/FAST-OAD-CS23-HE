# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Component palette sidebar for the power-train network viewer.

Provides a clickable icon palette (Bokeh server mode only) that lets the user
select a component type and place instances on the main canvas by clicking.

Typical usage
-------------
::

    from fastga_he.gui.component_palette import ComponentPaletteBuilder, ComponentPaletteLauncher

    # --- standalone demo ---
    ComponentPaletteLauncher.launch(port=5007)

    # --- embedded in an existing Bokeh document ---
    palette_fig, state = ComponentPaletteBuilder.build()

    def make_doc(doc):
        from bokeh.layouts import row
        from bokeh.events import Tap
        from fastga_he.gui.component_palette import PlacementHandler

        handler = PlacementHandler(state, main_plot)
        state.tap_source.selected.on_change("indices", handler.on_palette_select)
        main_plot.on_event(Tap, handler.on_canvas_tap)

        doc.add_root(row(palette_fig, main_plot))

"""

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path

import bokeh.models as bkmodel
import bokeh.plotting as bkplot
from bokeh.events import Tap
from bokeh.layouts import column, row
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
import webbrowser

try:
    from . import icons
    from .power_train_network_viewer import (
        BACKGROUND_COLOR_CODE,
        ICONS_CONFIG,
        _string_cleanup,
        _url_to_base64,
    )
except ImportError:
    # Fallback when the file is executed directly as a script
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # src/
    from fastga_he.gui import icons  # noqa: F401 (unused here but kept for consistency)
    from fastga_he.gui.power_train_network_viewer import (
        BACKGROUND_COLOR_CODE,
        ICONS_CONFIG,
        _string_cleanup,
        _url_to_base64,
    )

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

PALETTE_WIDTH = 175
ROW_HEIGHT = 52
ICON_SIZE = 30


# ---------------------------------------------------------------------------
# Shared mutable state between palette and placement handler
# ---------------------------------------------------------------------------


@dataclass
class PaletteState:
    """Holds references to all Bokeh data-sources managed by the palette."""

    # ColumnDataSource for the clickable tap target (one row per component)
    tap_source: bkmodel.ColumnDataSource = field(default=None)
    # ColumnDataSource that drives the highlight rectangle
    highlight_source: bkmodel.ColumnDataSource = field(default=None)
    # ColumnDataSource that accumulates icons placed on the main canvas
    placed_nodes_source: bkmodel.ColumnDataSource = field(default=None)
    # ColumnDataSource for the status label (inside the palette figure)
    status_source: bkmodel.ColumnDataSource = field(default=None)
    # Currently selected component key (e.g. "battery")
    selected_component: str = field(default=None)
    # Deduplication counters: component_key -> int
    placed_counter: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Palette builder
# ---------------------------------------------------------------------------


class ComponentPaletteBuilder:
    """
    Build the palette sidebar figure containing all available component icons.

    This class is **pure** – it only constructs Bokeh objects and returns them;
    it does not register any callbacks.  Callbacks are handled by
    :class:`PlacementHandler`.
    """

    @staticmethod
    def build() -> tuple:
        """
        Construct the palette figure and initialise a :class:`PaletteState`.

        :return: ``(palette_figure, PaletteState)``
        """
        component_keys = list(ICONS_CONFIG.keys())
        n = len(component_keys)

        # Vertical centres for each row (top of figure = highest y)
        ys = [(n - i - 0.5) * ROW_HEIGHT for i in range(n)]

        # Convert all icons to base64 so they work inside a Bokeh server
        icon_urls = []
        for key in component_keys:
            path = ICONS_CONFIG[key]["icon_path"]
            file_url = "file://" + str(Path(path).resolve())
            icon_urls.append(_url_to_base64(file_url))

        # ------------------------------------------------------------------
        # Data sources
        # ------------------------------------------------------------------

        palette_source = bkmodel.ColumnDataSource(
            data=dict(
                x=[ICON_SIZE] * n,
                y=ys,
                url=icon_urls,
                w=[ICON_SIZE] * n,
                h=[ICON_SIZE] * n,
            )
        )

        # One invisible row-highlight rectangle; data is patched on selection
        highlight_source = bkmodel.ColumnDataSource(
            data=dict(x=[-9999.0], y=[-9999.0])  # off-screen until first selection
        )

        # Placed-nodes source used by the *main* canvas (populated on tap)
        placed_nodes_source = bkmodel.ColumnDataSource(
            data=dict(x=[], y=[], url=[], w=[], h=[], name=[])
        )

        # Status label at the bottom of the palette figure
        status_source = bkmodel.ColumnDataSource(
            data=dict(x=[PALETTE_WIDTH / 2], y=[6], text=["Select a component"])
        )

        # Tap-target source – wider invisble scatter that covers the full row
        tap_source = bkmodel.ColumnDataSource(
            data=dict(
                x=[PALETTE_WIDTH / 2] * n,
                y=ys,
                component_key=component_keys,
                label=[_string_cleanup(k) for k in component_keys],
            )
        )

        # ------------------------------------------------------------------
        # Figure
        # ------------------------------------------------------------------

        fig = bkplot.figure(
            width=PALETTE_WIDTH,
            height=n * ROW_HEIGHT + 24,
            x_range=(0, PALETTE_WIDTH),
            y_range=(0, n * ROW_HEIGHT),
            toolbar_location=None,
            background_fill_color=BACKGROUND_COLOR_CODE,
            title="Components",
        )
        fig.xgrid.visible = False
        fig.ygrid.visible = False
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.title.text_color = "white"

        # Highlight rect (behind selected row)
        fig.rect(
            x="x",
            y="y",
            width=PALETTE_WIDTH - 4,
            height=ROW_HEIGHT - 4,
            source=highlight_source,
            fill_color="#FFD700",
            fill_alpha=0.20,
            line_color="#FFD700",
            line_width=1,
        )

        # Component icons
        fig.image_url(
            url="url",
            x="x",
            y="y",
            w="w",
            h="h",
            anchor="center",
            source=palette_source,
        )

        # Component labels to the right of the icons
        fig.add_layout(
            bkmodel.LabelSet(
                x="x",
                y="y",
                text="label",
                x_offset=ICON_SIZE // 2 + 4,
                source=tap_source,
                text_color="white",
                text_font_size="8pt",
                text_baseline="middle",
            )
        )

        # Horizontal dividers between rows
        for i in range(1, n):
            y_div = i * ROW_HEIGHT
            divider_source = bkmodel.ColumnDataSource(
                data=dict(x=[0, PALETTE_WIDTH], y=[y_div, y_div])
            )
            fig.line(
                x="x",
                y="y",
                source=divider_source,
                line_color="#808080",
                line_width=0.5,
                line_alpha=0.4,
            )

        # Invisible wide scatter used as tap target (covers the full row)
        tap_glyph = fig.scatter(
            x="x",
            y="y",
            size=PALETTE_WIDTH,
            source=tap_source,
            fill_alpha=0,
            line_alpha=0,
        )
        fig.add_tools(
            bkmodel.TapTool(renderers=[tap_glyph]),
            bkmodel.HoverTool(
                renderers=[tap_glyph],
                tooltips=[("Component", "@label")],
            ),
        )

        # Status label at the bottom
        fig.add_layout(
            bkmodel.LabelSet(
                x="x",
                y="y",
                text="text",
                source=status_source,
                text_color="#FFD700",
                text_font_size="7pt",
                text_align="center",
                text_baseline="bottom",
            )
        )

        state = PaletteState(
            tap_source=tap_source,
            highlight_source=highlight_source,
            placed_nodes_source=placed_nodes_source,
            status_source=status_source,
        )

        return fig, state


# ---------------------------------------------------------------------------
# Placement handler
# ---------------------------------------------------------------------------


class PlacementHandler:
    """
    Wires palette selection events to canvas placement events.

    Register the two callbacks after building the palette::

        handler = PlacementHandler(state, main_plot)
        state.tap_source.selected.on_change("indices", handler.on_palette_select)
        main_plot.on_event(Tap, handler.on_canvas_tap)
        main_plot.image_url(
            url="url", x="x", y="y", w="w", h="h",
            anchor="center", source=state.placed_nodes_source,
        )
    """

    def __init__(self, state: PaletteState, main_plot, icon_size: int = 30):
        """
        :param state: Shared :class:`PaletteState` instance
        :param main_plot: The Bokeh ``figure`` that acts as the canvas
        :param icon_size: Pixel size of placed icons (width = height)
        """
        self.state = state
        self.main_plot = main_plot
        self.icon_size = icon_size

    # ------------------------------------------------------------------
    # Palette selection callback
    # ------------------------------------------------------------------

    def on_palette_select(self, attr, old, new):
        """
        Called when the user clicks a row in the palette.
        Highlights the selected row and updates the status label.
        """
        if not new:
            return

        idx = new[0]
        component_keys = list(ICONS_CONFIG.keys())
        if idx >= len(component_keys):
            return

        self.state.selected_component = component_keys[idx]

        # Move highlight rect to the selected row
        n = len(component_keys)
        y_center = (n - idx - 0.5) * ROW_HEIGHT
        self.state.highlight_source.data = dict(
            x=[PALETTE_WIDTH / 2],
            y=[y_center],
        )

        # Update status label
        label = _string_cleanup(self.state.selected_component)
        self.state.status_source.data = dict(
            x=[PALETTE_WIDTH / 2],
            y=[6],
            text=[f"Placing: {label}"],
        )

    # ------------------------------------------------------------------
    # Canvas tap callback
    # ------------------------------------------------------------------

    def on_canvas_tap(self, event):
        """
        Called when the user taps the main canvas.
        Places the currently selected component icon at the tapped position.
        """
        if self.state.selected_component is None:
            return

        comp_key = self.state.selected_component
        x, y = event.x, event.y

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

        _log.debug("Placed %s at (%.1f, %.1f)", node_name, x, y)


# ---------------------------------------------------------------------------
# Standalone launcher (demo / development helper)
# ---------------------------------------------------------------------------


class ComponentPaletteLauncher:
    """
    Launch a self-contained Bokeh server that demonstrates the palette.

    A blank canvas is placed next to the palette so you can click
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
            palette_fig, state = ComponentPaletteBuilder.build()

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
            placed_label_source = bkmodel.ColumnDataSource(
                data=dict(x=[], y=[], text=[])
            )
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

            handler = PlacementHandler(state, canvas)
            state.tap_source.selected.on_change("indices", handler.on_palette_select)
            canvas.on_event(Tap, handler.on_canvas_tap)

            doc.add_root(row(palette_fig, canvas))
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
