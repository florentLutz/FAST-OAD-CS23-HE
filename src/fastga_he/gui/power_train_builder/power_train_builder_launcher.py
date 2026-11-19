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
from bokeh.events import DocumentReady, Tap
from bokeh.layouts import column, row
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

from fastga_he.gui.power_train_network_viewer import (
    BACKGROUND_COLOR_CODE,
    COLOR_ICON_CONFIG,
    ELECTRICITY_CURRENT_COLOR_CODE,
    FUEL_FLOW_COLOR_CODE,
    MECHANICAL_POWER_COLOR_CODE,
)

from .power_train_builder_state import (
    CANVAS_FIT_NODE_MARGIN,
    CANVAS_MIN_HEIGHT,
    CANVAS_MIN_WIDTH,
    CANVAS_WORLD_X_RANGE,
    CANVAS_WORLD_Y_RANGE,
    CONFIG_PANEL_RESERVED_WIDTH,
    DEFAULT_NODE_ICON_SIZE,
    DIALOG_STYLES,
    HOVER_RING_BASE_SIZE,
    NODE_LABEL_BASE_OFFSET,
    NODE_SELECT_COLOR,
    SELECTION_RING_BASE_SIZE,
    PALETTE_WIDTH,
    PORT_RADIUS,
)
from .power_train_builder_metadata import icon_data_url
from .power_train_builder_toolbox_config_table import ComponentToolboxConfigurationTableBuilder
from .power_train_builder_handler import BuilderHandler

_LOGGER = logging.getLogger(__name__)

# Canvas legend
# Mirrors the viewer's LegendBuilder._add_legend layout:
#   [icon PNG]  [colored label text]
# drawn in the edge color of each connection type.
#
# The legend is pinned to the top-right corner of the canvas *in screen
# pixels*: whenever the view changes (window resize, pan, zoom, reset),
# _JS_PIN_LEGEND re-maps the pixel layout below into the current data ranges,
# so the legend never moves or changes size on screen
_LEGEND_ENTRIES = [
    ("fuel", "Fuel Flow", FUEL_FLOW_COLOR_CODE),
    ("mechanical", "Mechanical Power", MECHANICAL_POWER_COLOR_CODE),
    ("electricity", "Electrical Current", ELECTRICITY_CURRENT_COLOR_CODE),
]

_LEGEND_ICON_FROM_RIGHT_PX = 118  # icon centre, measured from the frame's right edge
_LEGEND_LABEL_FROM_RIGHT_PX = 107  # label left edge, measured from the frame's right edge
_LEGEND_TOP_PX = 10  # first row centre, measured from the frame's top edge
_LEGEND_ROW_STEP_PX = 22  # vertical spacing between rows
_LEGEND_ICON_W_PX = 9
_LEGEND_ICON_H_PX = 12

# Client-side only. Converts the pixel layout into data coordinates of the
# current view (Bokeh image glyphs can only be placed in data space)
_JS_PIN_LEGEND = """
const fw = plot.inner_width, fh = plot.inner_height;
if (!(fw > 0 && fh > 0)) { return; }
const per_px_x = (x_range.end - x_range.start) / fw;
const per_px_y = (y_range.end - y_range.start) / fh;
const row_y = px.rows.map((i) => y_range.end - (px.top + i * px.row_step) * per_px_y);

icon_source.data = {
    url: icon_source.data.url,
    x: row_y.map(() => x_range.end - px.icon_from_right * per_px_x),
    y: row_y,
    w: row_y.map(() => px.icon_w * per_px_x),
    h: row_y.map(() => px.icon_h * per_px_y),
};
label_source.data = {
    text: label_source.data.text,
    color: label_source.data.color,
    x: row_y.map(() => x_range.end - px.label_from_right * per_px_x),
    y: row_y,
};
"""


def _add_canvas_legend(canvas) -> None:
    """
    Draw a three-entry connection-type legend in the upper-right corner
    of the builder canvas, mirroring the layout of the network viewer's
    :meth:`LegendBuilder._add_legend`.

    Each row contains a small colour PNG icon (fuel, mechanical, or
    electricity) on the left — loaded via :func:`icon_data_url` — and a
    text label drawn in the edge colour of that connection type on the right.

    The legend is pinned to the screen: resizing the window, panning,
    zooming and resetting the canvas move the components but not the legend
    (see ``_JS_PIN_LEGEND``).

    :param canvas: The Bokeh ``Figure`` to annotate.
    """
    num_entries = len(_LEGEND_ENTRIES)
    icon_urls = [icon_data_url(COLOR_ICON_CONFIG[icon_key]) for icon_key, _, _ in _LEGEND_ENTRIES]
    # Initial positions assume the original 800 x 950 canvas (about 1 data
    # unit per pixel); _JS_PIN_LEGEND corrects them as soon as the page lays out
    x_end, y_end = CANVAS_WORLD_X_RANGE[1], CANVAS_WORLD_Y_RANGE[1]
    row_y = [
        y_end - _LEGEND_TOP_PX - row_index * _LEGEND_ROW_STEP_PX for row_index in range(num_entries)
    ]

    # PNG icons (left)
    icon_source = bkmodel.ColumnDataSource(
        data=dict(
            url=icon_urls,
            x=[x_end - _LEGEND_ICON_FROM_RIGHT_PX] * num_entries,
            y=row_y,
            w=[_LEGEND_ICON_W_PX] * num_entries,
            h=[_LEGEND_ICON_H_PX] * num_entries,
        )
    )
    canvas.image_url(url="url", x="x", y="y", w="w", h="h", anchor="center", source=icon_source)

    # Colored text labels (right)
    label_source = bkmodel.ColumnDataSource(
        data=dict(
            x=[x_end - _LEGEND_LABEL_FROM_RIGHT_PX] * num_entries,
            y=row_y,
            text=[description for _, description, _ in _LEGEND_ENTRIES],
            color=[edge_color for _, _, edge_color in _LEGEND_ENTRIES],
        )
    )
    canvas.add_layout(
        bkmodel.LabelSet(
            x="x",
            y="y",
            text="text",
            source=label_source,
            text_align="left",
            text_baseline="middle",
            text_color="color",
            text_font_size="9pt",
        )
    )

    # Pin to the screen
    pin_legend = bkmodel.CustomJS(
        args=dict(
            plot=canvas,
            x_range=canvas.x_range,
            y_range=canvas.y_range,
            icon_source=icon_source,
            label_source=label_source,
            px=dict(
                rows=list(range(num_entries)),
                top=_LEGEND_TOP_PX,
                row_step=_LEGEND_ROW_STEP_PX,
                icon_from_right=_LEGEND_ICON_FROM_RIGHT_PX,
                label_from_right=_LEGEND_LABEL_FROM_RIGHT_PX,
                icon_w=_LEGEND_ICON_W_PX,
                icon_h=_LEGEND_ICON_H_PX,
            ),
        ),
        code=_JS_PIN_LEGEND,
    )
    for canvas_range in (canvas.x_range, canvas.y_range):
        canvas_range.js_on_change("start", pin_legend)
        canvas_range.js_on_change("end", pin_legend)
    canvas.js_on_change("inner_width", pin_legend)
    canvas.js_on_change("inner_height", pin_legend)


# Window-fitted canvas
# The canvas takes the space between the two side panels, whatever the
# screen / window aspect ratio (16:9, 16:10, ...), and is refitted on resize.
# Two client-side steps:
#   1. _JS_FIT_CANVAS_SIZE (document ready + window resize) sets the figure's
#      pixel width / height from the window size.
#   2. _JS_FIT_VIEW (whenever the frame size changes, or Python flips
#      fit_view_trigger after a load / New Design) sets the x / y ranges so that
#      both axes use the SAME data-units-per-pixel (no stretched icons) and the
#      design world (CANVAS_WORLD_*) plus every placed component is visible.
#      Extra room goes to the right / bottom. The new view is also what the
#      Reset tool returns to. The data-per-pixel scale is sent to Python
#      (view_scale_input) to keep screen-pixel glyphs in step with the icons,
#      together with the area an *empty* design shows in this window (used to
#      spread automatically generated layouts over the whole canvas)
_CANVAS_BOTTOM_MARGIN_PX = 8

_JS_FIT_CANVAS_SIZE = """
const fit_size = () => {
    const view = Bokeh.index.find_one(plot);
    if (view == null) { return; }
    const rect = view.el.getBoundingClientRect();
    const left = rect.left + window.scrollX, top = rect.top + window.scrollY;
    const width = Math.max(min_width, Math.floor(window.innerWidth - left - right_reserved));
    const height = Math.max(min_height, Math.floor(window.innerHeight - top - bottom_margin));
    if (plot.width != width || plot.height != height) {
        plot.setv({width: width, height: height});
    }
};
fit_size();
if (!window.__powertrain_builder_resize_hooked) {
    window.__powertrain_builder_resize_hooked = true;
    let pending = false;
    window.addEventListener("resize", () => {
        if (pending) { return; }
        pending = true;
        requestAnimationFrame(() => { pending = false; fit_size(); });
    });
}
"""

_JS_FIT_VIEW = """
const fw = plot.inner_width, fh = plot.inner_height;
if (!(fw > 0 && fh > 0)) { return; }
let x0 = world.x0, x1 = world.x1, y0 = world.y0, y1 = world.y1;
const xs = nodes.data.x, ys = nodes.data.y;
for (let i = 0; i < xs.length; i++) {
    x0 = Math.min(x0, xs[i] - margin); x1 = Math.max(x1, xs[i] + margin);
    y0 = Math.min(y0, ys[i] - margin); y1 = Math.max(y1, ys[i] + margin);
}
// Same scale on both axes: the axis that needs more data per pixel wins.
const k = Math.max((x1 - x0) / fw, (y1 - y0) / fh);
const x_end = x0 + fw * k, y_start = y1 - fh * k;
x_range.setv({start: x0, end: x_end, reset_start: x0, reset_end: x_end});
y_range.setv({start: y_start, end: y1, reset_start: y_start, reset_end: y1});
// Canvas area of this window for an empty design (world only), for Python.
const k0 = Math.max((world.x1 - world.x0) / fw, (world.y1 - world.y0) / fh);
const report = JSON.stringify({
    k: +k.toFixed(4),
    area: [world.x0, +(world.x0 + fw * k0).toFixed(1), +(world.y1 - fh * k0).toFixed(1), world.y1],
    area_k: +k0.toFixed(4),
});
if (view_scale_input.value != report) { view_scale_input.value = report; }
"""


def _make_canvas_fit_window(doc, canvas, state) -> None:
    """
    Wire the client-side callbacks that size the canvas to the browser window
    and fit its view (see the comment above ``_JS_FIT_CANVAS_SIZE``).

    :param doc: The Bokeh document (for the ``DocumentReady`` event).
    :param canvas: The builder ``Figure``.
    :param state: :class:`BuilderState` providing ``placed_nodes_source``,
        ``view_scale_input`` and ``fit_view_trigger``.
    """
    fit_view = bkmodel.CustomJS(
        args=dict(
            plot=canvas,
            x_range=canvas.x_range,
            y_range=canvas.y_range,
            nodes=state.placed_nodes_source,
            view_scale_input=state.view_scale_input,
            margin=CANVAS_FIT_NODE_MARGIN,
            world=dict(
                x0=CANVAS_WORLD_X_RANGE[0],
                x1=CANVAS_WORLD_X_RANGE[1],
                y0=CANVAS_WORLD_Y_RANGE[0],
                y1=CANVAS_WORLD_Y_RANGE[1],
            ),
        ),
        code=_JS_FIT_VIEW,
    )
    canvas.js_on_change("inner_width", fit_view)
    canvas.js_on_change("inner_height", fit_view)
    state.fit_view_trigger.js_on_change("value", fit_view)

    doc.js_on_event(
        DocumentReady,
        bkmodel.CustomJS(
            args=dict(
                plot=canvas,
                right_reserved=CONFIG_PANEL_RESERVED_WIDTH,
                bottom_margin=_CANVAS_BOTTOM_MARGIN_PX,
                min_width=CANVAS_MIN_WIDTH,
                min_height=CANVAS_MIN_HEIGHT,
            ),
            code=_JS_FIT_CANVAS_SIZE,
        ),
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

            # Undo / Redo buttons
            # Created here (before BuilderHandler) so they exist on the state
            # when _wire_buttons() runs and can be wired immediately
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

            # Store on state so BuilderHandler._wire_buttons() can reach them
            state.undo_button = undo_button
            state.redo_button = redo_button

            # Canvas
            # Initial size only: resized to the browser window on load / resize
            # (see _make_canvas_fit_window)
            canvas = bkplot.figure(
                width=800,
                height=950,
                x_range=CANVAS_WORLD_X_RANGE,
                y_range=CANVAS_WORLD_Y_RANGE,
                name="builder_canvas",
                toolbar_location="above",
                background_fill_color=BACKGROUND_COLOR_CODE,
                title="Powertrain Builder – click to place components",
            )
            canvas.xgrid.visible = False
            canvas.ygrid.visible = False
            canvas.xaxis.visible = False
            canvas.yaxis.visible = False
            canvas.title.text_color = BACKGROUND_COLOR_CODE

            # Static connection-type legend (upper-right corner)
            _add_canvas_legend(canvas)

            # Permanent edge lines
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
            # Gold ring around the currently selected node (resized by the icon-size slider)
            state.selection_ring_renderer = canvas.scatter(
                x="x",
                y="y",
                size=SELECTION_RING_BASE_SIZE,
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

            # Hover ring around each icon; its tooltip reads name / node_type
            # straight from the node data
            scatter_glyph = canvas.scatter(
                x="x",
                y="y",
                size=HOVER_RING_BASE_SIZE,
                source=state.placed_nodes_source,
                fill_alpha=0,
                line_alpha=0,
                hover_fill_alpha=0.1,
                hover_line_alpha=0.3,
            )

            # Source port balls
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

            # Target port balls
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

            # Hover tooltips
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

            # Node-name labels, read straight from the node data
            state.node_label_set = bkmodel.LabelSet(
                x="x",
                y="y",
                text="name",
                source=state.placed_nodes_source,
                text_color="white",
                text_font_size="8pt",
                text_align="center",
                text_baseline="top",
                y_offset=NODE_LABEL_BASE_OFFSET,
            )
            canvas.add_layout(state.node_label_set)

            # Hover ring, resized together with the icons by the icon-size slider
            state.hover_ring_renderer = scatter_glyph

            # Wire canvas interaction
            # Also wires the type_select live-refresh callback and the Apply
            # button (config panel commit) — see BuilderHandler._wire_buttons
            handler = BuilderHandler(state, canvas, icon_size=DEFAULT_NODE_ICON_SIZE)
            canvas.on_event(Tap, handler.on_canvas_tap)

            # Size the canvas to the browser window (any aspect ratio)
            _make_canvas_fit_window(doc, canvas, state)

            # Startup overlay
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
                styles=dict(DIALOG_STYLES),
            )

            # Undo / Redo toolbar row (above the canvas)
            # Floated in the upper-left of the canvas area via absolute positioning
            # so it does not shift the canvas itself.  z-index keeps it above the
            # Bokeh toolbar but below the startup / save overlays (z-index 100)
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
            # (or Save & Exit's) CustomJS
            doc.add_root(state.save_version_overlay)
            # Unsaved-exit overlay – floats above the canvas, shown when the user
            # clicks End Session with unsaved changes (save button is yellow)
            doc.add_root(state.unsaved_exit_overlay)
            doc.title = "Powertrain Builder"

        def make_document_with_tracking(doc):
            """Wrap :func:`make_document` and register a session-destroyed hook that stops the IO loop."""
            make_document(doc)

            def on_destroy(session_context):
                """Stop the server once the browser session is closed."""
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
