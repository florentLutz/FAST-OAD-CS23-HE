# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Standalone Bokeh server launcher for the powertrain builder.

:class:`PowertrainBuilderLauncher` builds the full Bokeh document (canvas,
palette, configurator panel, startup overlay) and starts a local server,
opening the application in the default browser.

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

from fastga_he.gui.power_train_network_viewer import BACKGROUND_COLOR_CODE

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


class PowertrainBuilderLauncher:
    """
    Launch a self-contained Bokeh server running the powertrain builder.

    A blank canvas is placed in the centre, flanked by the component palette
    on the left and the configurator panel on the right.
    """

    @staticmethod
    def launch(port: int = 5007, address: str = "localhost"):
        """
        Start the Bokeh server and open it in the default browser.

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
            """Build the Bokeh document with palette, canvas, and configurator panel."""
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

                idx = state.selected_node_index
                saved_options = {}
                if idx is not None:
                    saved_options_json = list(state.placed_nodes_source.data.get("options", []))
                    if idx < len(saved_options_json):
                        try:
                            saved_options = json.loads(saved_options_json[idx]) or {}
                        except (json.JSONDecodeError, TypeError):
                            pass
                handler._refresh_options_table(new, saved_options)
                state.options_table.visible = bool(state.possible_options.get(new, {}))

                if idx is not None:
                    icon_types = list(state.placed_nodes_source.data.get("icon_type", []))
                    icon_type = icon_types[idx] if idx < len(icon_types) else _EMPTY
                    handler._refresh_symmetry_select(idx, icon_type)

            state.type_select.on_change("value", _on_type_select_change)

            # ── Apply button: commit config panel edits to placed_nodes_source ─
            def apply_node_configurations():
                selected_node_index = state.selected_node_index
                if selected_node_index is None:
                    return

                new_om_name = state.name_input.value
                new_node_type = state.type_select.value
                new_position = state.position_select.value

                option_names = list(state.options_source.data.get("options", []))
                option_values = list(state.options_source.data.get("value", []))
                options_dict = {
                    name: PlacementHandler._strings_to_option_values(value)
                    for name, value in zip(option_names, option_values)
                }
                opts_json = json.dumps(options_dict)

                pdata = {k: list(v) for k, v in state.placed_nodes_source.data.items()}

                if selected_node_index < len(pdata.get("name", [])):
                    pdata["name"][selected_node_index] = new_om_name
                if selected_node_index < len(pdata.get("node_type", [])):
                    pdata["node_type"][selected_node_index] = new_node_type
                    # Reset port counts to defaults when the type changes and spinners are hidden
                    if not state.source_count_spinner.visible:
                        pdata["n_sources"][selected_node_index] = int(
                            state.default_source_count.get(new_node_type, 0)
                        )
                        pdata["n_targets"][selected_node_index] = int(
                            state.default_target_count.get(new_node_type, 0)
                        )
                if selected_node_index < len(pdata.get("position", [])):
                    pdata["position"][selected_node_index] = new_position
                if "options" not in pdata:
                    pdata["options"] = ["{}"] * len(pdata.get("name", []))
                if selected_node_index < len(pdata["options"]):
                    pdata["options"][selected_node_index] = opts_json

                if state.source_count_spinner is not None and state.source_count_spinner.visible:
                    new_n_src = int(state.source_count_spinner.value)
                    if "n_sources" in pdata and selected_node_index < len(pdata["n_sources"]):
                        pdata["n_sources"][selected_node_index] = new_n_src
                if state.target_count_spinner is not None and state.target_count_spinner.visible:
                    new_n_tgt = int(state.target_count_spinner.value)
                    if "n_targets" in pdata and selected_node_index < len(pdata["n_targets"]):
                        pdata["n_targets"][selected_node_index] = new_n_tgt

                state.placed_nodes_source.data = pdata

                # Update hover_source name and type
                hdata = {k: list(v) for k, v in state.hover_source.data.items()}
                if selected_node_index < len(hdata.get("name", [])):
                    hdata["name"][selected_node_index] = new_om_name
                if selected_node_index < len(hdata.get("node_type", [])):
                    hdata["node_type"][selected_node_index] = new_node_type
                state.hover_source.data = hdata

                # ── Persist symmetry selection ────────────────────────────────
                new_sym_name = (
                    state.symmetry_select.value if state.symmetry_select is not None else _EMPTY
                )
                pdata2 = {k: list(v) for k, v in state.placed_nodes_source.data.items()}
                names_list = pdata2.get("name", [])

                sym_peer_idx = -1
                if new_sym_name:
                    for _j, _n in enumerate(names_list):
                        if _n == new_sym_name:
                            sym_peer_idx = _j
                            break

                pdata2.setdefault("symmetry_name", [_EMPTY] * len(names_list))
                pdata2.setdefault("symmetry_node_index", [-1] * len(names_list))

                if selected_node_index < len(pdata2["symmetry_name"]):
                    pdata2["symmetry_name"][selected_node_index] = new_sym_name
                if selected_node_index < len(pdata2["symmetry_node_index"]):
                    pdata2["symmetry_node_index"][selected_node_index] = sym_peer_idx

                current_node_name = (
                    names_list[selected_node_index]
                    if selected_node_index < len(names_list)
                    else _EMPTY
                )
                if sym_peer_idx >= 0:
                    if sym_peer_idx < len(pdata2["symmetry_name"]):
                        pdata2["symmetry_name"][sym_peer_idx] = current_node_name
                    if sym_peer_idx < len(pdata2["symmetry_node_index"]):
                        pdata2["symmetry_node_index"][sym_peer_idx] = selected_node_index
                elif not new_sym_name:
                    # Clear the old peer's back-reference if symmetry was removed
                    for _j in range(len(names_list)):
                        if (
                            _j != selected_node_index
                            and _j < len(pdata2["symmetry_name"])
                            and pdata2["symmetry_name"][_j] == current_node_name
                        ):
                            pdata2["symmetry_name"][_j] = _EMPTY
                            if _j < len(pdata2["symmetry_node_index"]):
                                pdata2["symmetry_node_index"][_j] = -1

                state.placed_nodes_source.data = pdata2

                # ── Commit pending connections ─────────────────────────────────
                for port_a, port_b in list(state.pending_connections):
                    handler._add_edge(port_a, port_b)
                handler._clear_temp_edges()
                handler._rebuild_all_ports()
                handler._refresh_connections_table(selected_node_index)

                # Log applied state
                _ed = state.edge_source.data if state.edge_source is not None else {}
                _names = list(state.placed_nodes_source.data.get("name", []))
                _conn_entries = []
                for _i in range(len(_ed.get("node_a_idx", []))):
                    _na = _ed["node_a_idx"][_i]
                    _nb = _ed["node_b_idx"][_i]
                    if _na == selected_node_index or _nb == selected_node_index:
                        _peer = _nb if _na == selected_node_index else _na
                        _peer_name = _names[_peer] if _peer < len(_names) else f"node_{_peer}"
                        _my_kind = (
                            _ed["a_kind"][_i] if _na == selected_node_index else _ed["b_kind"][_i]
                        )
                        _my_lbl = (
                            _ed["a_label"][_i] if _na == selected_node_index else _ed["b_label"][_i]
                        )
                        _pr_kind = (
                            _ed["b_kind"][_i] if _na == selected_node_index else _ed["a_kind"][_i]
                        )
                        _pr_lbl = (
                            _ed["b_label"][_i] if _na == selected_node_index else _ed["a_label"][_i]
                        )
                        _conn_entries.append(
                            {
                                "my_port": f"{_my_kind}:{_my_lbl}",
                                "connected_to_node": _peer_name,
                                "connected_to_port": f"{_pr_kind}:{_pr_lbl}",
                            }
                        )

                _LOGGER.info(
                    "Applied: node[%d] name=%s type=%s position=%s options=%s connections=%s",
                    selected_node_index,
                    new_om_name,
                    new_node_type,
                    new_position,
                    opts_json,
                    json.dumps(_conn_entries),
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
