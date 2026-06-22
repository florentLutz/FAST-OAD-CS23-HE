# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Palette sidebar and component-configurator panel builder.

:class:`ComponentPaletteConfigurationTableBuilder` is a pure factory that
constructs every Bokeh widget and data source needed by the builder UI and
packages them into a :class:`BuilderState`. No event callbacks are wired
here; that responsibility belongs to :class:`PlacementHandler`.

Typical usage::

    from fastga_he.gui.power_train_builder_palette import (
        ComponentPaletteConfigurationTableBuilder,
    )

    palette_layout, table_panel, state = ComponentPaletteConfigurationTableBuilder.build()
"""

import sys
from pathlib import Path

import bokeh.models as bkmodel
from bokeh.layouts import column, row

from fastga_he.gui.power_train_network_viewer import (
    BACKGROUND_COLOR_CODE,
    ICONS_CONFIG,
    _string_cleanup,
    _url_to_base64,
)
from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from .power_train_builder_state import (
    BuilderState,
    PALETTE_WIDTH,
    ROW_HEIGHT,
    BUTTON_DEFAULT_COLOR_TYPE,
    _EMPTY,
)
from .power_train_builder_metadata import (
    _build_port_count_defaults,
    _map_possible_component_types_to_icons,
    _get_performance_component_names,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
COMPONENTS_PATH = Path(__file__).resolve().parents[3] / "fastga_he/models/propulsion/components"

# ── Load dialog ───────────────────────────────────────────────────────────────
# Toggles browse_load_trigger to signal Python to open a tkinter open-file dialog.
_JS_LOAD_DIALOG = """
browse_load_trigger.value = browse_load_trigger.value === '1' ? '0' : '1';
"""

# ── Save click: show watcher-path overlay ─────────────────────────────────────
_JS_SAVE_CLICK = """
save_overlay_widget.visible = true;
"""

# ── End Session click: show unsaved-exit overlay if save_button is yellow,
#    otherwise flip end_session_trigger so Python stops the server cleanly. ──
_JS_END_SESSION_CLICK = """
if (save_button_widget.button_type === 'warning') {
    unsaved_exit_overlay_widget.visible = true;
} else {
    end_session_trigger.value = end_session_trigger.value === '1' ? '0' : '1';
    window.close();
}
"""


class ComponentPaletteConfigurationTableBuilder:
    """
    Build the palette sidebar and component-configurator panel as Bokeh widgets.

    This class is **pure**: it only constructs Bokeh objects and returns them.
    All event callbacks are wired by :class:`PlacementHandler`.
    """

    @staticmethod
    def build() -> tuple:
        """
        Construct the button palette, configurator panel, and shared builder state.

        :return: A three-element tuple
            ``(palette_column_layout, table_panel, BuilderState)`` where

            * ``palette_column_layout`` is the left-side Bokeh column widget
              containing the tabbed component buttons and action buttons.
            * ``table_panel`` is the right-side Bokeh column widget for the
              component configurator.
            * :class:`BuilderState` holds references to every shared widget
              and data source.
        """
        # ── Categorise icons into tab groups ──────────────────────────────────
        component_icon_keys = list(ICONS_CONFIG.keys())
        category_to_icon_keys: dict = {}
        for component in KNOWN_COMPONENTS:
            icon_key = component["icon_for_network_graph"]
            component_type_class = component["components_type_class"]
            if isinstance(component_type_class, list):
                for type_class in component_type_class:
                    if type_class == "propulsive_load":
                        continue
                    elif type_class not in category_to_icon_keys:
                        category_to_icon_keys[type_class] = [icon_key]
                    elif icon_key not in category_to_icon_keys[type_class]:
                        category_to_icon_keys[type_class].append(icon_key)
            elif component_type_class == "propulsive_load":
                category_to_icon_keys.setdefault("load", [])
                if icon_key not in category_to_icon_keys["load"]:
                    category_to_icon_keys["load"].append(icon_key)
            elif component_type_class not in category_to_icon_keys:
                category_to_icon_keys[component_type_class] = [icon_key]
            elif icon_key not in category_to_icon_keys[component_type_class]:
                category_to_icon_keys[component_type_class].append(icon_key)

        # ── Shared stylesheet for action buttons (Delete, Save, End Session) ──
        _action_stylesheet = [":host button { font-size: 1.4em; }"]

        # ── Palette buttons (one per icon, kept in ICONS_CONFIG order) ────────
        title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Components</b>",
            width=PALETTE_WIDTH,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )

        _btn_stylesheet = [
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
        ]

        buttons = []
        button_by_icon_key: dict = {}
        for icon_key in component_icon_keys:
            label = _string_cleanup(icon_key)
            icon_path = ICONS_CONFIG[icon_key]["icon_path"]
            file_url = "file://" + str(Path(icon_path).resolve())
            base64_url = _url_to_base64(file_url)
            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"'
                f' viewBox="0 0 120 120">'
                f'<image href="{base64_url}" width="120" height="120"/></svg>'
            )
            button = bkmodel.Button(
                label=label,
                icon=bkmodel.SVGIcon(svg=svg, size="2.75em"),
                button_type=BUTTON_DEFAULT_COLOR_TYPE,
                width=PALETTE_WIDTH - 10,
                height=ROW_HEIGHT - 6,
                stylesheets=_btn_stylesheet,
            )
            buttons.append(button)
            button_by_icon_key[icon_key] = button

        status_div = bkmodel.Div(
            text="<i style='color:#aaa;font-size:14pt'>Select a component</i>",
            width=PALETTE_WIDTH,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "14px"},
        )

        # ── Action buttons ────────────────────────────────────────────────────
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
        # js_on_click for End Session is wired later, after unsaved_exit_overlay is built,
        # so that it can reference both save_button and unsaved_exit_overlay_widget.

        # ── Hidden TextInput relay widgets ────────────────────────────────────
        # Three toggle triggers: JS flips '0'/'1' to fire Python on_change callbacks.
        # browse_load_trigger    → _on_browse_load    (opens tkinter open-file dialog)
        # browse_save_trigger    → _on_browse_save    (opens tkinter save-file dialogs)
        # browse_watcher_trigger → _on_browse_watcher (opens tkinter save-file dialog for CSV)

        browse_load_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        browse_save_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        browse_watcher_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        end_session_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        # Flipped by Python after a Save & Exit save completes; JS responds with window.close().
        close_window_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        close_window_trigger.js_on_change(
            "value",
            bkmodel.CustomJS(code="window.close();"),
        )

        # ── Wire Load Design buttons → signal Python to open tkinter dialog ───
        _load_js = bkmodel.CustomJS(
            args=dict(browse_load_trigger=browse_load_trigger),
            code=_JS_LOAD_DIALOG,
        )

        # Temporary load button (kept for any direct palette-bar placement)
        load_design_button_tmp = bkmodel.Button(
            label="Load Design",
            icon=bkmodel.TablerIcon(icon_name="folder-open"),
            button_type="light",
            width=220,
            height=70,
        )
        load_design_button_tmp.js_on_click(_load_js)

        # ── Watcher-path overlay widgets ──────────────────────────────────────
        # Built here (palette) so they can be wired to JS args; registered in
        # BuilderState so the launcher can add the overlay to the document and
        # the handler can read watcher_path_input.value at save time.
        watcher_path_input = bkmodel.TextInput(
            title="Watcher file path (optional):",
            value=_EMPTY,
            width=340,
            placeholder="Leave blank to skip",
            styles={"color": "white", "font-size": "14px"},
            # visible=False
        )
        _browse_watcher_btn = bkmodel.Button(
            label="...",
            button_type="light",
            width=40,
            height=31,
        )
        _browse_watcher_btn.js_on_click(
            bkmodel.CustomJS(
                args=dict(browse_trigger=browse_watcher_trigger),
                code="browse_trigger.value = browse_trigger.value === '1' ? '0' : '1';",
            )
        )
        _continue_save_btn = bkmodel.Button(
            label="Continue to Save",
            icon=bkmodel.TablerIcon(icon_name="device-floppy"),
            button_type="primary",
            width=200,
            height=40,
        )
        _cancel_watcher_btn = bkmodel.Button(
            label="Cancel",
            button_type="light",
            width=100,
            height=40,
        )

        # The overlay column – hidden until the Save button is clicked
        save_overlay = column(
            bkmodel.Div(
                text=(
                    "<div style='"
                    "color:white;font-size:22pt;font-weight:bold;"
                    "text-align:center;padding:24px 0 18px 0;letter-spacing:0.04em;"
                    "'>Save Design</div>"
                    "<div style='"
                    "color:#aaa;font-size:12pt;"
                    "text-align:center;padding-bottom:28px;"
                    "'>Optionally specify a watcher file path, then click "
                    "<b>Continue</b> to choose output locations.</div>"
                ),
                width=500,
            ),
            row(
                watcher_path_input,
                bkmodel.Spacer(width=6),
                _browse_watcher_btn,
                styles={"align-items": "flex-end"},
            ),
            bkmodel.Spacer(height=14),
            row(
                _cancel_watcher_btn,
                bkmodel.Div(text="", width=30),
                _continue_save_btn,
                styles={"justify-content": "center"},
            ),
            visible=False,
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

        # ── Wire Save button: click → show watcher overlay ────────────────────
        save_button.js_on_click(
            bkmodel.CustomJS(
                args=dict(save_overlay_widget=save_overlay),
                code=_JS_SAVE_CLICK,
            )
        )

        # ── Wire Cancel button: hide overlay without saving ───────────────────
        _cancel_watcher_btn.js_on_click(
            bkmodel.CustomJS(
                args=dict(save_overlay_widget=save_overlay),
                code="save_overlay_widget.visible = false;",
            )
        )

        # ── Wire Continue button → signal Python to open tkinter save dialogs ─
        _continue_save_btn.js_on_click(
            bkmodel.CustomJS(
                args=dict(
                    save_overlay_widget=save_overlay,
                    browse_save_trigger=browse_save_trigger,
                ),
                code="save_overlay_widget.visible = false; "
                "browse_save_trigger.value = browse_save_trigger.value === '1' ? '0' : '1';",
            )
        )

        # ── Unsaved-exit overlay ──────────────────────────────────────────────
        # Shown when the user clicks End Session while save_button is yellow.
        # Hidden trigger used by "Save & Exit" to open the watcher-path overlay,
        # then chain straight into the save flow and end the session on completion.
        end_session_save_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)

        _end_anyway_btn = bkmodel.Button(
            label="End Anyway",
            icon=bkmodel.TablerIcon(icon_name="power"),
            button_type="danger",
            width=200,
            height=40,
        )
        _save_and_exit_btn = bkmodel.Button(
            label="Save & Exit",
            icon=bkmodel.TablerIcon(icon_name="device-floppy"),
            button_type="primary",
            width=200,
            height=40,
        )
        _cancel_exit_btn = bkmodel.Button(
            label="Cancel",
            button_type="light",
            width=100,
            height=40,
        )

        unsaved_exit_overlay = column(
            bkmodel.Div(
                text=(
                    "<div style='"
                    "color:white;font-size:22pt;font-weight:bold;"
                    "text-align:center;padding:24px 0 18px 0;letter-spacing:0.04em;"
                    "'>Unsaved Changes</div>"
                    "<div style='"
                    "color:#aaa;font-size:12pt;"
                    "text-align:center;padding-bottom:28px;"
                    "'>End session without saving changes?</div>"
                ),
                width=500,
            ),
            row(
                _cancel_exit_btn,
                bkmodel.Div(text="", width=20),
                _end_anyway_btn,
                bkmodel.Div(text="", width=20),
                _save_and_exit_btn,
                styles={"justify-content": "center"},
            ),
            end_session_save_trigger,
            visible=False,
            styles={
                "background": "rgba(30,30,40,0.97)",
                "border": "2px solid #444",
                "border-radius": "16px",
                "padding": "10px 40px 30px 40px",
                "position": "absolute",
                "left": f"{PALETTE_WIDTH + 100}px",
                "top": "340px",
                "z-index": "100",
                "box-shadow": "0 8px 32px rgba(0,0,0,0.6)",
            },
        )

        # Cancel: just hide the overlay
        _cancel_exit_btn.js_on_click(
            bkmodel.CustomJS(
                args=dict(unsaved_exit_overlay_widget=unsaved_exit_overlay),
                code="unsaved_exit_overlay_widget.visible = false;",
            )
        )

        # End Anyway: hide the overlay, flip end_session_trigger so Python stops the server,
        # then close the browser tab.
        _end_anyway_btn.js_on_click(
            bkmodel.CustomJS(
                args=dict(
                    unsaved_exit_overlay_widget=unsaved_exit_overlay,
                    end_session_trigger=end_session_trigger,
                ),
                code=(
                    "unsaved_exit_overlay_widget.visible = false; "
                    "end_session_trigger.value = end_session_trigger.value === '1' ? '0' : '1'; "
                    "window.close();"
                ),
            )
        )

        # Save & Exit: open the watcher-path / save overlay, then Python handler
        # ends the session after the files are written (via end_session_save_trigger).
        _save_and_exit_btn.js_on_click(
            bkmodel.CustomJS(
                args=dict(
                    unsaved_exit_overlay_widget=unsaved_exit_overlay,
                    save_overlay_widget=save_overlay,
                    end_session_save_trigger=end_session_save_trigger,
                ),
                code=(
                    "unsaved_exit_overlay_widget.visible = false; "
                    "save_overlay_widget.visible = true; "
                    "end_session_save_trigger.value = "
                    "    end_session_save_trigger.value === '1' ? '0' : '1';"
                ),
            )
        )

        # ── Wire End Session button: check for unsaved state before closing ───
        end_session_button.js_on_click(
            bkmodel.CustomJS(
                args=dict(
                    save_button_widget=save_button,
                    unsaved_exit_overlay_widget=unsaved_exit_overlay,
                    end_session_trigger=end_session_trigger,
                ),
                code=_JS_END_SESSION_CLICK,
            )
        )

        # ── Startup overlay buttons ───────────────────────────────────────────
        _startup_button_css = [
            """:host button {
                font-size: 1.6em;
                padding: 18px 40px;
                border-radius: 10px;
                font-weight: bold;
                letter-spacing: 0.04em;
            }"""
        ]
        new_design_button = bkmodel.Button(
            label="New Design",
            icon=bkmodel.TablerIcon(icon_name="file-plus"),
            button_type="primary",
            width=220,
            height=70,
            stylesheets=_startup_button_css,
        )
        load_design_button = bkmodel.Button(
            label="Load Design",
            icon=bkmodel.TablerIcon(icon_name="folder-open"),
            button_type="light",
            width=220,
            height=70,
            stylesheets=_startup_button_css,
        )
        load_design_button.js_on_click(_load_js)

        # ── Canvas data sources ───────────────────────────────────────────────
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
                symmetry_name=[],  # name of the symmetry peer node
                symmetry_node_index=[],
            )
        )
        hover_source = bkmodel.ColumnDataSource(data=dict(x=[], y=[], name=[], node_type=[]))
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
        options_table_source = bkmodel.ColumnDataSource(data=dict(options=[], value=[]))
        connections_source = bkmodel.ColumnDataSource(
            data=dict(my_port=[], connected_to=[], edge_idx=[])
        )
        edge_source = bkmodel.ColumnDataSource(
            data=dict(
                xs=[],
                ys=[],
                color=[],
                starting_node_index=[],
                starting_port_label=[],
                starting_port_kind=[],
                ending_node_index=[],
                ending_port_label=[],
                ending_port_kind=[],
            )
        )
        pending_port_source = bkmodel.ColumnDataSource(data=dict(x=[], y=[], color=[]))
        temp_edge_source = bkmodel.ColumnDataSource(data=dict(xs=[], ys=[], color=[]))

        # ── Configurator panel widgets ────────────────────────────────────────
        options_rows_column = column([], styles={"background": BACKGROUND_COLOR_CODE})
        connections_rows_column = column([], styles={"background": BACKGROUND_COLOR_CODE})
        connections_table_widget = column(
            connections_rows_column, styles={"background": BACKGROUND_COLOR_CODE}
        )

        name_input = bkmodel.TextInput(
            title="Component ID:",
            value=_EMPTY,
            width=380,
            styles={"color": "white", "font-size": "18px"},
        )
        type_select = bkmodel.Select(
            title="Component Type:",
            value=_EMPTY,
            options=[],
            width=380,
            styles={"color": "white", "font-size": "18px"},
        )
        position_select = bkmodel.Select(
            title="Position:",
            value=_EMPTY,
            options=[],
            width=380,
            styles={"color": "white", "font-size": "18px"},
        )
        apply_button = bkmodel.Button(
            label="Apply",
            icon=bkmodel.TablerIcon(icon_name="check"),
            button_type="primary",
            width=380,
            height=ROW_HEIGHT - 6,
            stylesheets=_action_stylesheet,
        )
        symmetry_select = bkmodel.Select(
            value="",
            options=[],
            width=380,
            styles={"color": "white", "font-size": "18px"},
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
            bkmodel.Div(
                text="<b style='color:white;font-size:16pt'>Port Counts</b>",
                width=380,
                styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
            ),
            row(source_count_spinner, target_count_spinner),
            visible=False,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "4px"},
        )

        options_table = column(
            bkmodel.Div(
                text="<span style='color:white;font-size:14pt'>Options:</span>",
                width=380,
                styles={"background": BACKGROUND_COLOR_CODE},
            ),
            options_rows_column,
            visible=False,
            styles={"background": BACKGROUND_COLOR_CODE},
        )

        table_panel = column(
            bkmodel.Div(
                text="<b style='color:white;font-size:18pt'>Component Configurator</b>",
                width=380,
                styles={"background": BACKGROUND_COLOR_CODE, "text-align": "center"},
            ),
            bkmodel.Div(
                text="<b style='color:white;font-size:16pt'>Component ID &amp; Type</b>",
                width=380,
                styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
            ),
            name_input,
            type_select,
            port_count_section,
            bkmodel.Div(
                text="<b style='color:white;font-size:16pt'>Position &amp; Options</b>",
                width=380,
                styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
            ),
            position_select,
            options_table,
            bkmodel.Div(
                text="<b style='color:white;font-size:16pt'>Connections</b>",
                width=380,
                styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
            ),
            connections_table_widget,
            bkmodel.Div(
                text="<b style='color:white;font-size:16pt'>Symmetry &amp; Distributed Load</b>",
                width=380,
                styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
            ),
            symmetry_select,
            apply_button,
            spacing=4,
            visible=False,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "10px"},
        )

        # ── Metadata lookup tables ────────────────────────────────────────────
        default_source_port_count, default_target_port_count = _build_port_count_defaults()
        component_type_to_icon_map = _map_possible_component_types_to_icons()
        possible_position_map, possible_options_map = _get_performance_component_names(
            COMPONENTS_PATH
        )

        # ── Assemble BuilderState ─────────────────────────────────────────────
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
            default_source_count=default_source_port_count,
            default_target_count=default_target_port_count,
            component_type_to_icon=component_type_to_icon_map,
            possible_position=possible_position_map,
            possible_options=possible_options_map,
            browse_load_trigger=browse_load_trigger,
            browse_save_trigger=browse_save_trigger,
            browse_watcher_trigger=browse_watcher_trigger,
            watcher_path_input=watcher_path_input,
            save_overlay=save_overlay,
            new_design_button=new_design_button,
            load_design_button=load_design_button,
            unsaved_exit_overlay=unsaved_exit_overlay,
            end_session_save_trigger=end_session_save_trigger,
            end_session_trigger=end_session_trigger,
            close_window_trigger=close_window_trigger,
        )

        # ── Build tabbed palette sidebar ──────────────────────────────────────
        tab_panels = []
        for category, keys_in_category in category_to_icon_keys.items():
            category_buttons = [
                button_by_icon_key[icon_key]
                for icon_key in keys_in_category
                if icon_key in button_by_icon_key
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
            browse_load_trigger,
            browse_save_trigger,
            browse_watcher_trigger,
            end_session_save_trigger,
            end_session_trigger,
            close_window_trigger,
            spacing=2,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "10px"},
        )

        return palette_layout, table_panel, state
