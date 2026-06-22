# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Standalone Bokeh server launcher for the powertrain builder.

:class:`PowertrainBuilderLauncher` builds the full Bokeh document — canvas,
palette, configurator panel, and startup overlay — and starts a local Bokeh
server, opening the application in the default browser.

Typical usage::

    from fastga_he.gui.power_train_builder_launcher import PowertrainBuilderLauncher

    PowertrainBuilderLauncher.launch(port=5007)
"""

import json
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
    _EMPTY,
)
from .power_train_builder_palette import ComponentPaletteConfigurationTableBuilder
from .power_train_builder_handler import PlacementHandler

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
    Launch a self-contained Bokeh server running the powertrain builder.

    A blank canvas is placed in the centre, flanked by the component palette
    on the left and the component configurator panel on the right. The server
    stops automatically when the browser session ends.
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
            """Build the complete Bokeh document: palette, canvas, and configurator panel."""
            palette_layout, table_panel, state = ComponentPaletteConfigurationTableBuilder.build()

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
                source=state.temp_edge_source,
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
                source=state.pending_port_source,
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
            handler = PlacementHandler(state, canvas)
            canvas.on_event(Tap, handler.on_canvas_tap)

            # ── type_select callback: refresh position + options on type change
            def _on_type_select_change(attr, old, new):
                pos_choices = state.possible_position.get(new, [])
                state.position_select.options = pos_choices
                state.position_select.value = pos_choices[0] if pos_choices else _EMPTY

                current_selected_node_index_for_type = state.selected_node_index
                saved_options = {}
                if current_selected_node_index_for_type is not None:
                    saved_options_json_list = list(
                        state.placed_nodes_source.data.get("options", [])
                    )
                    if current_selected_node_index_for_type < len(saved_options_json_list):
                        try:
                            saved_options = (
                                json.loads(
                                    saved_options_json_list[current_selected_node_index_for_type]
                                )
                                or {}
                            )
                        except (json.JSONDecodeError, TypeError):
                            pass
                handler._refresh_options_table(new, saved_options)
                state.options_table.visible = bool(state.possible_options.get(new, {}))

                if current_selected_node_index_for_type is not None:
                    icon_type_list = list(state.placed_nodes_source.data.get("icon_type", []))
                    icon_type = (
                        icon_type_list[current_selected_node_index_for_type]
                        if current_selected_node_index_for_type < len(icon_type_list)
                        else _EMPTY
                    )
                    handler._refresh_symmetry_select(
                        current_selected_node_index_for_type, icon_type
                    )

            state.type_select.on_change("value", _on_type_select_change)

            # ── Apply button: commit config panel edits to placed_nodes_source ─
            def apply_node_configurations():
                current_selected_node_index = state.selected_node_index
                if current_selected_node_index is None:
                    return

                # Get node properties from the configurator panel
                new_node_name_id = state.name_input.value
                new_node_type = state.type_select.value
                new_position = state.position_select.value

                option_names = list(state.options_source.data.get("options", []))
                option_values = list(state.options_source.data.get("value", []))
                option_values_dict = {
                    option_name: PlacementHandler._strings_to_option_values(option_value)
                    for option_name, option_value in zip(option_names, option_values)
                }
                # convert the options dict to a JSON string for storage in the ColumnDataSource
                options_json_string = json.dumps(option_values_dict)

                placed_nodes_data = {
                    key: list(values) for key, values in state.placed_nodes_source.data.items()
                }

                # Check first if the current_selected_node_index is within bounds for each property
                # before updating, to avoid IndexErrors.
                if current_selected_node_index < len(placed_nodes_data.get("name", [])):
                    placed_nodes_data["name"][current_selected_node_index] = new_node_name_id
                if current_selected_node_index < len(placed_nodes_data.get("node_type", [])):
                    placed_nodes_data["node_type"][current_selected_node_index] = new_node_type
                    # Reset port counts to defaults when the type changes and spinners are hidden.
                    if not state.source_count_spinner.visible:
                        placed_nodes_data["n_sources"][current_selected_node_index] = int(
                            state.default_source_count.get(new_node_type, 0)
                        )
                        placed_nodes_data["n_targets"][current_selected_node_index] = int(
                            state.default_target_count.get(new_node_type, 0)
                        )
                if current_selected_node_index < len(placed_nodes_data.get("position", [])):
                    placed_nodes_data["position"][current_selected_node_index] = new_position
                # Place empty dict if there is no options entry for this node yet
                if "options" not in placed_nodes_data:
                    placed_nodes_data["options"] = ["{}"] * len(placed_nodes_data.get("name", []))
                # Update options JSON string for this node
                if current_selected_node_index < len(placed_nodes_data["options"]):
                    placed_nodes_data["options"][current_selected_node_index] = options_json_string

                # These two spinners shows only if the node type supports multiple ports
                if state.source_count_spinner is not None and state.source_count_spinner.visible:
                    new_source_port_count = int(state.source_count_spinner.value)
                    if "n_sources" in placed_nodes_data and current_selected_node_index < len(
                        placed_nodes_data["n_sources"]
                    ):
                        placed_nodes_data["n_sources"][current_selected_node_index] = (
                            new_source_port_count
                        )
                if state.target_count_spinner is not None and state.target_count_spinner.visible:
                    new_target_port_count = int(state.target_count_spinner.value)
                    if "n_targets" in placed_nodes_data and current_selected_node_index < len(
                        placed_nodes_data["n_targets"]
                    ):
                        placed_nodes_data["n_targets"][current_selected_node_index] = (
                            new_target_port_count
                        )

                # Update hover_source name and type
                hover_data = {key: list(values) for key, values in state.hover_source.data.items()}
                if current_selected_node_index < len(hover_data.get("name", [])):
                    hover_data["name"][current_selected_node_index] = new_node_name_id
                if current_selected_node_index < len(hover_data.get("node_type", [])):
                    hover_data["node_type"][current_selected_node_index] = new_node_type
                state.hover_source.data = hover_data

                # ── Persist symmetry selection ────────────────────────────────
                _NO_SYMMETRY_PEER = -1
                all_node_names = placed_nodes_data.get("name", [])
                node_count = len(all_node_names)

                # Guard: current index must be valid before we touch names.
                if current_selected_node_index >= node_count:
                    return

                current_node_name = all_node_names[current_selected_node_index]
                new_symmetry_name = (
                    state.symmetry_select.value if state.symmetry_select is not None else _EMPTY
                )

                # Ensure symmetry columns exist and are fully sized (handles newly added nodes).
                symmetry_names = list(placed_nodes_data.get("symmetry_name", []))
                symmetry_indices = list(placed_nodes_data.get("symmetry_node_index", []))
                if len(symmetry_names) < node_count:
                    symmetry_names += [_EMPTY] * (node_count - len(symmetry_names))
                if len(symmetry_indices) < node_count:
                    symmetry_indices += [_NO_SYMMETRY_PEER] * (node_count - len(symmetry_indices))

                # Resolve the newly chosen peer index.
                new_peer_index = _NO_SYMMETRY_PEER
                if new_symmetry_name and new_symmetry_name != _EMPTY:
                    for node_position, node_name in enumerate(all_node_names):
                        if node_name == new_symmetry_name:
                            new_peer_index = node_position
                            break

                # ── Clear stale references on both sides before writing ────────
                #
                # Two nodes may be displaced by this Apply:
                #   (a) current node's *old* peer  — it pointed back at current_node;
                #       now current_node is leaving, so clear it.
                #   (b) new peer's *old* partner   — new_peer previously pointed at
                #       some other node D; D still points at new_peer, so clear D too.
                #
                # We handle both by scanning the full table and wiping every node
                # whose symmetry_name matches current_node_name OR new_symmetry_name,
                # excluding the two nodes that will get fresh values written below.

                nodes_getting_fresh_write = {current_selected_node_index}
                if new_peer_index != _NO_SYMMETRY_PEER:
                    nodes_getting_fresh_write.add(new_peer_index)

                for position in range(node_count):
                    if position in nodes_getting_fresh_write:
                        continue
                    sym_name = symmetry_names[position]
                    if sym_name == current_node_name or (
                        new_symmetry_name
                        and new_symmetry_name != _EMPTY
                        and sym_name == new_symmetry_name
                    ):
                        symmetry_names[position] = _EMPTY
                        symmetry_indices[position] = _NO_SYMMETRY_PEER

                # Write the current node's new symmetry entry.
                symmetry_names[current_selected_node_index] = new_symmetry_name
                symmetry_indices[current_selected_node_index] = new_peer_index

                # Write the new peer's back-reference (if a peer was chosen).
                if new_peer_index != _NO_SYMMETRY_PEER:
                    symmetry_names[new_peer_index] = current_node_name
                    symmetry_indices[new_peer_index] = current_selected_node_index

                placed_nodes_data["symmetry_name"] = symmetry_names
                placed_nodes_data["symmetry_node_index"] = symmetry_indices

                # Update the placed_nodes_source with the modified data
                state.placed_nodes_source.data = placed_nodes_data

                # ── Commit pending connections ─────────────────────────────────
                for starting_port, ending_port in list(state.pending_connections):
                    handler._add_edge(starting_port, ending_port)
                handler._clear_temp_edges()
                handler._rebuild_all_ports()
                handler._refresh_connections_table(current_selected_node_index)

                # Log applied state
                edge_data_snapshot = state.edge_source.data if state.edge_source is not None else {}
                all_placed_names = list(state.placed_nodes_source.data.get("name", []))
                connection_log_entries = []
                for edge_index in range(len(edge_data_snapshot.get("starting_node_index", []))):
                    starting_node_index = edge_data_snapshot["starting_node_index"][edge_index]
                    ending_node_index = edge_data_snapshot["ending_node_index"][edge_index]
                    if (
                        starting_node_index == current_selected_node_index
                        or ending_node_index == current_selected_node_index
                    ):
                        connected_node_index = (
                            ending_node_index
                            if starting_node_index == current_selected_node_index
                            else starting_node_index
                        )
                        connected_node_name = (
                            all_placed_names[connected_node_index]
                            if connected_node_index < len(all_placed_names)
                            else f"node_{connected_node_index}"
                        )
                        my_port_kind = (
                            edge_data_snapshot["starting_port_kind"][edge_index]
                            if starting_node_index == current_selected_node_index
                            else edge_data_snapshot["ending_port_kind"][edge_index]
                        )
                        my_port_label = (
                            edge_data_snapshot["starting_port_label"][edge_index]
                            if starting_node_index == current_selected_node_index
                            else edge_data_snapshot["ending_port_label"][edge_index]
                        )
                        connected_port_kind = (
                            edge_data_snapshot["ending_port_kind"][edge_index]
                            if starting_node_index == current_selected_node_index
                            else edge_data_snapshot["starting_port_kind"][edge_index]
                        )
                        connected_port_label = (
                            edge_data_snapshot["ending_port_label"][edge_index]
                            if starting_node_index == current_selected_node_index
                            else edge_data_snapshot["starting_port_label"][edge_index]
                        )
                        connection_log_entries.append(
                            {
                                "my_port": f"{my_port_kind}:{my_port_label}",
                                "connected_to_node": connected_node_name,
                                "connected_to_port": f"{connected_port_kind}:{connected_port_label}",
                            }
                        )

                _LOGGER.info(
                    "Applied: node[%d] name=%s type=%s position=%s options=%s connections=%s",
                    current_selected_node_index,
                    new_node_name_id,
                    new_node_type,
                    new_position,
                    options_json_string,
                    json.dumps(connection_log_entries),
                )
                handler._mark_unsaved()

            state.apply_button.on_click(apply_node_configurations)

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

            doc.add_root(row(palette_layout, canvas, table_panel))
            state.startup_overlay = startup_overlay
            doc.add_root(startup_overlay)
            # Save-options overlay (watcher path dialog) – floats above the canvas.
            # state.save_overlay was created and wired in ComponentPaletteConfigurationTableBuilder;
            # it is already visible=False and will be shown by the Save button's CustomJS.
            if state.save_overlay is not None:
                doc.add_root(state.save_overlay)
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
