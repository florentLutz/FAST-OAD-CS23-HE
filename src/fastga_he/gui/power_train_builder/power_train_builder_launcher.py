# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Standalone Bokeh server launcher for the powertrain builder.

:class:`PowertrainBuilderLauncher` builds the full Bokeh document — canvas,
toolbox, configurator panel, undo / redo toolbar, and startup overlay — and
starts a local Bokeh server, opening the application in the default browser.

Typical usage::

    from fastga_he.gui.power_train_builder_launcher import PowertrainBuilderLauncher

    PowertrainBuilderLauncher.launch(port=5007)
"""

import logging
import webbrowser

import bokeh.models as bkmodel
import bokeh.plotting as bkplot
from bokeh.events import Tap
from bokeh.layouts import column, row
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

from fastga_he.gui.power_train_network_viewer import (
    BACKGROUND_COLOR_CODE,
    COLOR_ICON_CONFIG,
    ELECTRICITY_CURRENT_COLOR_CODE,
    FUEL_FLOW_COLOR_CODE,
    MECHANICAL_POWER_COLOR_CODE,
    _url_to_base64,
)

from .power_train_builder_state import (
    ICON_SIZE,
    NODE_SELECT_COLOR,
    PALETTE_WIDTH,
    PORT_RADIUS,
)
from .power_train_builder_toolbox import ComponentToolboxConfigurationTableBuilder
from .power_train_builder_handler import BuilderHandler

_LOGGER = logging.getLogger(__name__)

# ── Canvas legend ─────────────────────────────────────────────────────────────
# Mirrors the viewer's LegendBuilder._add_legend layout:
#   [icon PNG]  [colored label text]
# Each row: icon anchored at (x_icon, y), label left-aligned at x_label
# and drawn in the edge color of that connection type.
#
# All coordinates are in canvas data space (x_range 0–800, y_range 0–950).
_LEGEND_ENTRIES = [
    ("fuel", "Fuel Flow", FUEL_FLOW_COLOR_CODE),
    ("mechanical", "Mechanical Power", MECHANICAL_POWER_COLOR_CODE),
    ("electricity", "Electrical Current", ELECTRICITY_CURRENT_COLOR_CODE),
]

_LEGEND_X_ICON = 680  # center-x of the PNG icon
_LEGEND_X_LABEL = 691  # left edge of the text label (icon_center + ~11)
_LEGEND_Y_TOP = 940  # y of the first (topmost) row
_LEGEND_ROW_STEP = 22  # vertical spacing between rows
_LEGEND_ICON_W = 9  # icon width  in data units (matches viewer)
_LEGEND_ICON_H = 12  # icon height in data units (matches viewer)


def _add_canvas_legend(canvas) -> None:
    """
    Draw a static three-entry connection-type legend in the upper-right corner
    of the builder canvas, mirroring the layout of the network viewer's
    :meth:`LegendBuilder._add_legend`.

    Each row contains a small colour PNG icon (fuel, mechanical, or
    electricity) on the left — loaded via :func:`_url_to_base64` — and a
    text label drawn in the edge colour of that connection type on the right.

    :param canvas: The Bokeh ``Figure`` to annotate.
    """
    for entry_index, (icon_key, description, edge_color) in enumerate(_LEGEND_ENTRIES):
        entry_y = _LEGEND_Y_TOP - entry_index * _LEGEND_ROW_STEP

        # ── PNG icon (left) ───────────────────────────────────────────────────
        icon_file_path = COLOR_ICON_CONFIG[icon_key]
        icon_base64_url = _url_to_base64("file://" + str(icon_file_path.resolve()))
        canvas.image_url(
            url=[icon_base64_url],
            x=[_LEGEND_X_ICON],
            y=[entry_y],
            w=_LEGEND_ICON_W,
            h=_LEGEND_ICON_H,
            anchor="center",
        )

        # ── Colored text label (right) ────────────────────────────────────────
        canvas.add_layout(
            bkmodel.LabelSet(
                x="x",
                y="y",
                text="text",
                source=bkmodel.ColumnDataSource(
                    data=dict(x=[_LEGEND_X_LABEL], y=[entry_y], text=[description])
                ),
                text_align="left",
                text_baseline="middle",
                text_color=edge_color,
                text_font_size="9pt",
            )
        )


class PowertrainBuilderLauncher:
    """
    Launches a self-contained Bokeh server that runs the powertrain builder GUI.

    Assembles the full page layout around three pieces built by
    :class:`~.power_train_builder_toolbox.ComponentToolboxConfigurationTableBuilder`
    (itself composed of several mixins for the toolbox and configuration panel):

    * the **component toolbox** — on the left, one button per known component
      type, used to arm a component for placement;
    * the **canvas** — centred, the interactive Bokeh figure where components
      are placed, connected, selected, and edited;
    * the **configuration panel** — on the right, the node / options /
      connections tables for the currently selected component.

    An Undo / Redo button row sits directly above the canvas.

    :class:`~.power_train_builder_handler.BuilderHandler` wires all
    interaction logic onto this layout; this class is only responsible for
    layout assembly and server lifecycle. The Bokeh server it starts stops
    automatically once the browser session ends.
    """

    @staticmethod
    def launch(port: int = 5007, address: str = "localhost"):
        """
        Start the Bokeh server and open it in the default browser.

        Configures logging, builds the Bokeh document, starts the server on
        the given port, and blocks the calling thread until the session ends.

        :param port: TCP port for the Bokeh server (default 5007).
        :param address: Server bind address (default ``"localhost"``).
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logging.getLogger("bokeh").setLevel(logging.WARNING)
        logging.getLogger("tornado").setLevel(logging.WARNING)

        def make_document(doc):
            """Build the complete Bokeh document: toolbox, canvas, and configurator panel."""
            toolbox_layout, config_panel_column, state = (
                ComponentToolboxConfigurationTableBuilder.build()
            )

            # ── Undo / Redo buttons ───────────────────────────────────────────
            # Created here (before BuilderHandler) so they exist on the state
            # when _wire_buttons() runs and can be wired immediately.
            _undo_redo_stylesheet = [":host button { font-size: 1.25em; min-width: 90px; }"]

            undo_button = bkmodel.Button(
                label="Undo",
                icon=bkmodel.TablerIcon(icon_name="arrow-back-up"),
                button_type="default",
                width=100,
                height=36,
                stylesheets=_undo_redo_stylesheet,
            )
            redo_button = bkmodel.Button(
                label="Redo",
                icon=bkmodel.TablerIcon(icon_name="arrow-forward-up"),
                button_type="default",
                width=100,
                height=36,
                stylesheets=_undo_redo_stylesheet,
            )

            # Store on state so BuilderHandler._wire_buttons() can reach them.
            state.undo_button = undo_button
            state.redo_button = redo_button

            # ── Canvas ────────────────────────────────────────────────────────
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

            # ── Static connection-type legend (upper-right corner) ────────────
            _add_canvas_legend(canvas)

            # ── Permanent edge lines ──────────────────────────────────────────
            canvas.multi_line(
                xs="xs",
                ys="ys",
                line_color="color",
                line_width=4,
                line_alpha=0.85,
                source=state.edge_source,
            )
            # Dashed preview edges – visible while a connection is pending Apply
            canvas.multi_line(
                xs="xs",
                ys="ys",
                line_color="color",
                line_width=3,
                line_alpha=0.65,
                line_dash="dashed",
                source=state.panel_preview_edge_source,
            )
            # Gold ring around the currently selected node
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
            # Placed node icons
            canvas.image_url(
                url="url",
                x="x",
                y="y",
                w="w",
                h="h",
                anchor="center",
                source=state.placed_nodes_source,
            )
            # Pending-port highlight ring
            canvas.scatter(
                x="x",
                y="y",
                size=PORT_RADIUS * 2 + 12,
                source=state.awaiting_second_port_click_source,
                fill_color="color",
                fill_alpha=0.35,
                line_color="white",
                line_width=2,
                line_dash="dashed",
            )

            # ── Hover glyphs ──────────────────────────────────────────────────
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

            # ── Source port balls ─────────────────────────────────────────────
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

            # ── Target port balls ─────────────────────────────────────────────
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

            # ── Hover tooltips ────────────────────────────────────────────────
            canvas.add_tools(
                bkmodel.HoverTool(
                    renderers=[scatter_glyph],
                    tooltips=[("Component id", "@name"), ("Component type", "@node_type")],
                ),
                bkmodel.HoverTool(
                    renderers=[source_glyph],
                    tooltips=[
                        ("Port type", "Source"),
                        ("Port number", "@label"),
                        ("Component id", "@node_name"),
                        ("Component type", "@node_type"),
                    ],
                ),
                bkmodel.HoverTool(
                    renderers=[target_glyph],
                    tooltips=[
                        ("Port type", "Target"),
                        ("Port number", "@label"),
                        ("Component id", "@node_name"),
                        ("Component type", "@node_type"),
                    ],
                ),
            )

            # ── Node-name labels ──────────────────────────────────────────────
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

            def _sync_labels(attr, old, new_data):
                placed_label_source.data = dict(
                    x=list(new_data.get("x", [])),
                    y=list(new_data.get("y", [])),
                    text=list(new_data.get("name", [])),
                )

            state.placed_nodes_source.on_change("data", _sync_labels)

            # ── Wire canvas interaction ───────────────────────────────────────
            # Also wires the type_select live-refresh callback and the Apply
            # button (config panel commit) — see BuilderHandler._wire_buttons.
            handler = BuilderHandler(state, canvas)
            canvas.on_event(Tap, handler.on_canvas_tap)

            # ── Startup overlay ───────────────────────────────────────────────
            startup_overlay = column(
                bkmodel.Div(
                    text=(
                        "<div style='"
                        "color:white;font-size:22pt;font-weight:bold;"
                        "text-align:center;padding:24px 0 18px 0;letter-spacing:0.04em;"
                        "'>Powertrain Builder</div>"
                        "<div style='"
                        "color:#aaa;font-size:12pt;"
                        "text-align:center;padding-bottom:28px;"
                        "'>Start a new design or restore a previous session</div>"
                    ),
                    width=500,
                ),
                row(
                    state.new_design_button,
                    bkmodel.Div(text="", width=30),
                    state.load_design_button,
                    styles={"justify-content": "center"},
                ),
                styles={
                    "background": "rgba(30,30,40,0.97)",
                    "border": "2px solid #444",
                    "border-radius": "16px",
                    "padding": "10px 40px 30px 40px",
                    "position": "absolute",
                    "left": f"{PALETTE_WIDTH + 150}px",
                    "top": "340px",
                    "z-index": "100",
                    "box-shadow": "0 8px 32px rgba(0,0,0,0.6)",
                },
            )

            # ── Undo / Redo toolbar row (above the canvas) ────────────────────
            # Floated in the upper-left of the canvas area via absolute positioning
            # so it does not shift the canvas itself.  z-index keeps it above the
            # Bokeh toolbar but below the startup / save overlays (z-index 100).
            undo_redo_overlay = row(
                undo_button,
                bkmodel.Spacer(width=4),
                redo_button,
                styles={
                    "position": "absolute",
                    "left": f"{PALETTE_WIDTH + 38}px",  # increase this to move right
                    "top": "30px",  # increase this to move down
                    "z-index": "50",
                    "padding": "4px 8px",
                    "box-shadow": "0 2px 8px rgba(0,0,0,0.1)",
                },
            )
            doc.add_root(row(toolbox_layout, canvas, config_panel_column))
            state.startup_overlay = startup_overlay
            doc.add_root(startup_overlay)
            doc.add_root(undo_redo_overlay)

            # "Final version?" overlay – floats above the canvas; created hidden in
            # ComponentToolboxConfigurationTableBuilder and shown by the Save button's
            # (or Save & Exit's) CustomJS.
            if state.save_version_overlay is not None:
                doc.add_root(state.save_version_overlay)
            # Unsaved-exit overlay – floats above the canvas, shown when the user
            # clicks End Session with unsaved changes (save button is yellow).
            if state.unsaved_exit_overlay is not None:
                doc.add_root(state.unsaved_exit_overlay)
            doc.title = "Powertrain Builder"

        def make_document_with_tracking(doc):
            """Wrap :func:`make_document` and register a session-destroyed hook that stops the IO loop."""
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
