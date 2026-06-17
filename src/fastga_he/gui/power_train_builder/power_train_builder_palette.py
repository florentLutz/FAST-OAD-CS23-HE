# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Palette sidebar and component-configurator panel builder.

:class:`ComponentPaletteConfigurationTableBuilder` is a pure factory — it
constructs every Bokeh widget and data source needed by the builder UI and
packages them into a :class:`BuilderState`.  No callbacks are wired here;
that responsibility belongs to :class:`PlacementHandler`.

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
# When showOpenFilePicker is available (Chrome/Edge/Opera) the file's full text
# content is read in JS and written into load_path_input so Python never needs
# an OS path.  The prompt() fallback still sends a raw filesystem path (the
# server must be on the same machine, as before).
_JS_LOAD_DIALOG = """
(async () => {
    if (typeof window.showOpenFilePicker === 'function') {
        // ── Native OS "Open" dialog (Chrome / Edge / Opera) ─────────────────
        try {
            const [fileHandle] = await window.showOpenFilePicker({
                types: [{
                    description: 'JSON canvas state backup',
                    accept: { 'application/json': ['.json'] },
                }],
                multiple: false,
            });
            // Read the actual file content – no OS path needed
            const file = await fileHandle.getFile();
            const content = await file.text();
            load_input.value = content;
        } catch (e) {
            if (e.name !== 'AbortError') { throw e; }
            // User cancelled – do nothing
        }
    } else {
        // ── Fallback: window.prompt() ────────────────────────────────────────
        const raw = window.prompt('Load JSON canvas state – enter the full file path:');
        if (raw === null || raw.trim() === '') { return; }
        load_input.value = raw.trim();
    }
})();
"""

# ── Save click dialog ─────────────────────────────────────────────────────────
# Fired by the Save button click.  Opens the OS file-picker dialogs to let the
# user choose filenames/locations, then signals Python by writing a bundle into
# yaml_path_input.  Python serialises the canvas and pushes the content strings
# back through save_content_output (see _JS_WRITE_CONTENT below).
# The prompt() fallback sends bare filenames for Python to write server-side.
_JS_SAVE_CLICK = """
(async () => {
    const now = new Date();
    const pad = n => String(n).padStart(2, '0');
    const ts = `${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}_` +
               `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;

    let yamlName = null;
    let jsonName = null;
    let useBrowserWrite = false;

    if (typeof window.showSaveFilePicker === 'function') {
        useBrowserWrite = true;

        // ── YAML dialog first (primary file) ────────────────────────────────
        try {
            const yamlHandle = await window.showSaveFilePicker({
                suggestedName: `powertrain_config_${ts}.yml`,
                types: [{
                    description: 'YAML powertrain configuration',
                    accept: { 'text/yaml': ['.yml', '.yaml'] },
                }],
            });
            // Store handle on window so _JS_WRITE_CONTENT can access it
            window._ptb_yaml_handle = yamlHandle;
            yamlName = yamlHandle.name;
        } catch (e) {
            if (e.name !== 'AbortError') { throw e; }
            window._ptb_yaml_handle = null;
        }

        // ── JSON dialog second (backup file) ────────────────────────────────
        try {
            const jsonHandle = await window.showSaveFilePicker({
                suggestedName: `canvas_state_${ts}.json`,
                types: [{
                    description: 'JSON canvas state backup',
                    accept: { 'application/json': ['.json'] },
                }],
            });
            window._ptb_json_handle = jsonHandle;
            jsonName = jsonHandle.name;
        } catch (e) {
            if (e.name !== 'AbortError') { throw e; }
            window._ptb_json_handle = null;
        }

    } else {
        // ── Fallback: window.prompt() ────────────────────────────────────────
        const rawYaml = window.prompt(
            'YAML powertrain config – enter a file name (Cancel to skip):',
            `powertrain_config_${ts}.yml`
        );
        if (rawYaml !== null && rawYaml.trim() !== '') {
            yamlName = rawYaml.trim();
        }

        const rawJson = window.prompt(
            'JSON backup file – enter a file name (Cancel to skip):',
            `canvas_state_${ts}.json`
        );
        if (rawJson !== null && rawJson.trim() !== '') {
            jsonName = rawJson.trim();
        }
    }

    // Both dialogs cancelled → do nothing
    if (yamlName === null && jsonName === null) { return; }

    // Signal Python: send chosen names + flag so it knows which save path to use
    save_btn.button_type = 'warning';
    yaml_input.value = JSON.stringify({
        yaml:             yamlName  || '',
        json:             jsonName  || '',
        ts:               ts,
        use_browser_write: useBrowserWrite,
    });
})();
"""

# ── Write content callback ────────────────────────────────────────────────────
# Fired by an on_change on save_content_output when Python pushes the serialised
# YAML/JSON strings back to the browser.  Uses the FileSystemFileHandle objects
# stored by _JS_SAVE_CLICK to write directly, or falls back to blob downloads.
_JS_WRITE_CONTENT = """
(async () => {
    if (!cb_obj.value || cb_obj.value === '') { return; }

    let bundle;
    try {
        bundle = JSON.parse(cb_obj.value);
    } catch (e) {
        console.error('PTB: could not parse save bundle', e);
        return;
    }

    if (typeof window.showSaveFilePicker === 'function') {
        // Write via the stored FileSystemFileHandle objects
        if (bundle.yaml && window._ptb_yaml_handle) {
            try {
                const writable = await window._ptb_yaml_handle.createWritable();
                await writable.write(bundle.yaml);
                await writable.close();
                window._ptb_yaml_handle = null;
            } catch (e) { console.error('PTB: YAML write failed', e); }
        }
        if (bundle.json && window._ptb_json_handle) {
            try {
                const writable = await window._ptb_json_handle.createWritable();
                await writable.write(bundle.json);
                await writable.close();
                window._ptb_json_handle = null;
            } catch (e) { console.error('PTB: JSON write failed', e); }
        }
    } else {
        // Fallback: trigger browser blob downloads
        ['yaml', 'json'].forEach(type => {
            if (!bundle[type]) { return; }
            const mime = type === 'yaml' ? 'text/yaml' : 'application/json';
            const blob = new Blob([bundle[type]], { type: mime });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = type === 'yaml'
                ? `powertrain_config_${bundle.ts}.yml`
                : `canvas_state_${bundle.ts}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(a.href);
        });
    }
})();
"""


class ComponentPaletteConfigurationTableBuilder:
    """
    Build the palette sidebar and component-configurator panel as Bokeh widgets.

    This class is **pure** – it only constructs Bokeh objects and returns them;
    all callbacks are wired by :class:`PlacementHandler`.
    """

    @staticmethod
    def build() -> tuple:
        """
        Construct the button palette, configurator panel, and shared state.

        :return: ``(palette_column_layout, table_panel, BuilderState)``
        """
        # ── Categorise icons into tab groups ──────────────────────────────────
        component_icon_keys = list(ICONS_CONFIG.keys())
        category_keys: dict = {}
        for component in KNOWN_COMPONENTS:
            icon = component["icon_for_network_graph"]
            component_type_class = component["components_type_class"]
            if isinstance(component_type_class, list):
                for type_class in component_type_class:
                    if type_class == "propulsive_load":
                        continue
                    elif type_class not in category_keys:
                        category_keys[type_class] = [icon]
                    elif icon not in category_keys[type_class]:
                        category_keys[type_class].append(icon)
            elif component_type_class == "propulsive_load":
                category_keys.setdefault("load", [])
                if icon not in category_keys["load"]:
                    category_keys["load"].append(icon)
            elif component_type_class not in category_keys:
                category_keys[component_type_class] = [icon]
            elif icon not in category_keys[component_type_class]:
                category_keys[component_type_class].append(icon)

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
        button_by_key: dict = {}
        for key in component_icon_keys:
            label = _string_cleanup(key)
            icon_path = ICONS_CONFIG[key]["icon_path"]
            file_url = "file://" + str(Path(icon_path).resolve())
            b64_url = _url_to_base64(file_url)
            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"'
                f' viewBox="0 0 120 120">'
                f'<image href="{b64_url}" width="120" height="120"/></svg>'
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
            button_by_key[key] = button

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
        end_session_button.js_on_click(bkmodel.CustomJS(code="window.close();"))

        # ── Hidden TextInput relay widgets ────────────────────────────────────
        # load_path_input  : JS → Python  (file content or fallback path)
        # yaml_path_input  : JS → Python  (save-dialog signal bundle)
        # save_content_output : Python → JS  (serialised YAML+JSON content)
        # json_path_input  : retained for BuilderState compat (unused in new flow)
        json_path_input = bkmodel.TextInput(value=_EMPTY, width=0, height=0, visible=False)
        yaml_path_input = bkmodel.TextInput(value=_EMPTY, width=0, height=0, visible=False)
        load_path_input = bkmodel.TextInput(value=_EMPTY, width=0, height=0, visible=False)
        save_content_output = bkmodel.TextInput(value=_EMPTY, width=0, height=0, visible=False)

        # ── Wire load dialog to both Load Design buttons (tmp + overlay) ──────
        _load_js = bkmodel.CustomJS(
            args=dict(load_input=load_path_input),
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

        # ── Wire Save button: click → open OS dialogs → signal Python ─────────
        save_button.js_on_click(
            bkmodel.CustomJS(
                args=dict(yaml_input=yaml_path_input, save_btn=save_button),
                code=_JS_SAVE_CLICK,
            )
        )

        # ── Wire save_content_output: Python push → JS writes files ──────────
        # on_change fires when Python sets save_content_output.value with the
        # serialised canvas strings; _JS_WRITE_CONTENT uses the stored handles
        # (or blob downloads) to complete the write.
        save_content_output.js_on_change(
            "value",
            bkmodel.CustomJS(
                args=dict(save_output=save_content_output),
                code=_JS_WRITE_CONTENT,
            ),
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
        default_source_count, default_target_count = _build_port_count_defaults()
        component_type_to_icon = _map_possible_component_types_to_icons()
        possible_position, possible_options = _get_performance_component_names(COMPONENTS_PATH)

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
            default_source_count=default_source_count,
            default_target_count=default_target_count,
            component_type_to_icon=component_type_to_icon,
            possible_position=possible_position,
            possible_options=possible_options,
            json_path_input=json_path_input,
            yaml_path_input=yaml_path_input,
            load_path_input=load_path_input,
            save_content_output=save_content_output,
            new_design_button=new_design_button,
            load_design_button=load_design_button,
        )

        # ── Build tabbed palette sidebar ──────────────────────────────────────
        tab_panels = []
        for category, keys_in_category in category_keys.items():
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
            json_path_input,
            yaml_path_input,
            load_path_input,
            save_content_output,  # must be in the DOM for js_on_change to fire
            spacing=2,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "10px"},
        )

        return palette_layout, table_panel, state
