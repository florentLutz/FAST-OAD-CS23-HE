# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Toolbox sidebar and component-configurator panel builder.

:class:`ComponentToolboxConfigurationTableBuilder` is a pure factory that
constructs every Bokeh widget and data source needed by the builder UI and
packages them into a :class:`BuilderState`. Each group of fields comes from
its own ``_build_*`` function, returning ``{BuilderState field name: object}``
in the order of the class sections. The only callbacks wired here are the
browser-side (``CustomJS``) ones; every Python callback is wired by
:class:`BuilderHandler`.

Typical usage::

    from fastga_he.gui.power_train_builder.power_train_builder_toolbox import (
        ComponentToolboxConfigurationTableBuilder,
    )

    toolbox_layout, config_panel_column, state = ComponentToolboxConfigurationTableBuilder.build()
"""

from pathlib import Path

import bokeh.models as bkmodel
from bokeh.layouts import column, row

from fastga_he.gui.power_train_network_viewer import (
    BACKGROUND_COLOR_CODE,
    _string_cleanup,
)
from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from .power_train_builder_state import (
    BuilderState,
    PALETTE_WIDTH,
    ROW_HEIGHT,
    BUTTON_DEFAULT_COLOR_TYPE,
    DEFAULT_NODE_ICON_SIZE,
    DIALOG_STYLES,
    NODE_ICON_SIZE_MAX,
    NODE_ICON_SIZE_MIN,
    EDGE_COLUMNS,
    NODE_COLUMNS,
    PORT_COLUMNS,
    SAVE_MODE_CANCEL,
    SAVE_MODE_EDITABLE,
    SAVE_MODE_FINAL,
    STATUS_IDLE_HTML,
    BUILDER_ICONS_CONFIG,
    PORT_TYPE_CHOICES,
    UNDEFINED_PORT_TYPE,
    UNKNOWN_COMPONENT_TABS,
    UNKNOWN_ICON_KEY,
    UNKNOWN_NO_PORT_WARNING_HTML,
    _EMPTY,
    empty_columns,
)
from .power_train_builder_metadata import (
    _build_port_count_defaults,
    _map_possible_component_types_to_icons,
    _get_performance_component_names,
    component_type_classes,
    icon_data_url,
)

COMPONENTS_PATH = Path(__file__).resolve().parents[2] / "models" / "propulsion" / "components"


def _js_flip(trigger_name: str) -> str:
    """
    JS statement flipping a hidden trigger TextInput between ``'0'`` and ``'1'``.

    Every flip is a value change, so the Python ``on_change`` callback of the
    trigger fires on every click (see :func:`_build_hidden_triggers`).

    :param trigger_name: Name the trigger has in the ``CustomJS`` args.
    :return: One line of JS.
    """
    return f"{trigger_name}.value = {trigger_name}.value === '1' ? '0' : '1';"


# Load dialog
# Toggles browse_load_trigger to signal Python to open a tkinter open-file dialog
_JS_LOAD_DIALOG = _js_flip("browse_load_trigger")

# Watcher CSV dialog
# Toggles browse_watcher_trigger to signal Python to open a tkinter save-file dialog
_JS_WATCHER_DIALOG = _js_flip("browse_watcher_trigger")

# "Final version?" overlay buttons: hide it and send the mode to Python
#    (__MODE__ is replaced by final / editable / cancel). The timestamp makes
#    every click a new value, so on_change always fires. Final Version and Edit
#    Later first commit a watcher path still being typed (value_input holds the
#    keystrokes, value only changes on Enter / blur), so Python stores it
#    before the save starts
_JS_SAVE_MODE_CLICK = """
save_version_overlay_widget.visible = false;
watcher_path_input.value = watcher_path_input.value_input;
browse_save_trigger.value = '__MODE__:' + Date.now();
"""
_JS_SAVE_CANCEL_CLICK = """
save_version_overlay_widget.visible = false;
browse_save_trigger.value = '__MODE__:' + Date.now();
"""

# Toolbox action buttons (Delete, Save, End Session) and the panel's Apply button
_ACTION_BUTTON_CSS = [":host button { font-size: 1.4em; }"]

# Toolbox component buttons: label on the left, icon on the right
_TOOLBOX_BUTTON_CSS = [
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

# Width of the "Components" title; with the icon-size slider next to it, the
# header row adds up to PALETTE_WIDTH so it never grows wider than the panel
_TITLE_DIV_WIDTH = 140

# Style of the config panel's section headings
_SECTION_HEADING_STYLES = {"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"}

# Same scroll behaviour for both side panels: scroll inside the panel instead
# of growing the page on short screens
_SIDE_PANEL_STYLES = {
    "background": BACKGROUND_COLOR_CODE,
    "padding": "10px",
    "max-height": "calc(100vh - 8px)",
    "overflow-y": "auto",
}

# Shared by the startup overlay and the "final version?" overlay buttons
_STARTUP_BUTTON_CSS = [
    """:host button {
        font-size: 1.6em;
        padding: 18px 40px;
        border-radius: 10px;
        font-weight: bold;
        letter-spacing: 0.04em;
    }"""
]

# End Session click: show unsaved-exit overlay if save_button is yellow,
#    otherwise flip end_session_trigger so Python stops the server cleanly. ──
_JS_END_SESSION_CLICK = f"""
if (save_button_widget.button_type === 'warning') {{
    unsaved_exit_overlay_widget.visible = true;
}} else {{
    {_js_flip("end_session_trigger")}
    window.close();
}}
"""


def _group_icons_by_tab(known_components: list) -> dict:
    """
    Toolbox tab of every component icon, from the components' type classes.

    Each type class is a tab, in the order the classes are first met. A
    component that is only a ``"propulsive_load"`` goes to the ``"load"`` tab;
    in a list of classes, ``"propulsive_load"`` adds no tab of its own. An icon
    is listed once per tab even if several components share it.

    :param known_components: Entries of ``KNOWN_COMPONENTS``.
    :return: ``{tab_name: [icon_key, …]}``.
    """
    icons_by_tab: dict = {}
    for component in known_components:
        icon_key = component["icon_for_network_graph"]
        if component["components_type_class"] == "propulsive_load":
            tab_names = ["load"]
        else:
            tab_names = [
                type_class
                for type_class in component_type_classes(component)
                if type_class != "propulsive_load"
            ]
        for tab_name in tab_names:
            tab_icon_keys = icons_by_tab.setdefault(tab_name, [])
            if icon_key not in tab_icon_keys:
                tab_icon_keys.append(icon_key)
    return icons_by_tab


def _with_unknown_component(icons_by_tab: dict) -> dict:
    """
    Add the unknown component, last, to each of the :data:`UNKNOWN_COMPONENT_TABS`.

    Only tabs that exist get it: a tab is never created for it alone.

    :param icons_by_tab: ``{tab_name: [icon_key, …]}`` from :func:`_group_icons_by_tab`.
    :return: The same mapping, with the unknown icon appended where it belongs.
    """
    return {
        tab_name: (
            icon_keys + [UNKNOWN_ICON_KEY] if tab_name in UNKNOWN_COMPONENT_TABS else icon_keys
        )
        for tab_name, icon_keys in icons_by_tab.items()
    }


def _build_unknown_component_widgets() -> dict:
    """
    Config-panel widgets used only for an unknown component.

    The text inputs replace the type, position and symmetry selects (an
    unknown component has no list of possible values), the "+" button adds an
    option row, the two port-type selects set the energy type of its source
    and target ports, and the warning tells the user it has no port at all.
    All start hidden.

    :return: ``{BuilderState field name: widget}``.
    """
    text_style = {"color": "white", "font-size": "18px"}

    def port_type_select(title: str):
        return bkmodel.Select(
            title=title,
            value=UNDEFINED_PORT_TYPE,
            options=PORT_TYPE_CHOICES,
            width=180,
            visible=False,
            styles={"color": "white", "font-size": "14px"},
        )

    return {
        "unknown_type_input": bkmodel.TextInput(
            title="Component Type:", width=380, visible=False, styles=text_style
        ),
        "unknown_position_input": bkmodel.TextInput(
            title="Position:", width=380, visible=False, styles=text_style
        ),
        "unknown_symmetry_input": bkmodel.TextInput(
            placeholder="Name of the symmetrical component",
            width=380,
            visible=False,
            styles=text_style,
        ),
        "add_option_button": bkmodel.Button(
            label="Add option",
            icon=bkmodel.TablerIcon(icon_name="plus"),
            button_type=BUTTON_DEFAULT_COLOR_TYPE,
            width=180,
            visible=False,
        ),
        "source_port_type_select": port_type_select("Source port type:"),
        "target_port_type_select": port_type_select("Target port type:"),
        "unknown_no_port_warning_div": bkmodel.Div(
            text=UNKNOWN_NO_PORT_WARNING_HTML,
            width=370,
            visible=False,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "4px 2px"},
        ),
    }


def _build_toolbox_tabs(icons_by_tab: dict, button_by_icon_key: dict):
    """
    One toolbox tab per group of :func:`_group_icons_by_tab`, holding its buttons.

    Icons without a button (not in ``BUILDER_ICONS_CONFIG``) are left out, and a tab
    left empty is not shown.

    :param icons_by_tab: ``{tab_name: [icon_key, …]}``.
    :param button_by_icon_key: Toolbox button of every icon key.
    :return: The Bokeh ``Tabs`` widget.
    """
    tab_panels = []
    for tab_name, icon_keys in icons_by_tab.items():
        tab_buttons = [
            button_by_icon_key[icon_key] for icon_key in icon_keys if icon_key in button_by_icon_key
        ]
        if not tab_buttons:
            continue
        tab_column = column(
            *tab_buttons,
            spacing=2,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "10px"},
        )
        tab_panels.append(bkmodel.TabPanel(child=tab_column, title=tab_name.capitalize()))
    return bkmodel.Tabs(tabs=tab_panels, width=PALETTE_WIDTH)


# ============================================================================
# One builder per BuilderState section, each returning {field name: object}
# ============================================================================


def _build_canvas_sources() -> dict:
    """
    Empty data sources of the canvas: design data, then derived and temporary data.

    :return: ``{BuilderState field name: ColumnDataSource}``.
    """
    return {
        # Design data
        "placed_nodes_source": bkmodel.ColumnDataSource(data=empty_columns(NODE_COLUMNS)),
        "edge_source": bkmodel.ColumnDataSource(data=empty_columns(EDGE_COLUMNS)),
        # Derived and temporary canvas data
        "source_port_source": bkmodel.ColumnDataSource(data=empty_columns(PORT_COLUMNS)),
        "target_port_source": bkmodel.ColumnDataSource(data=empty_columns(PORT_COLUMNS)),
        "selected_node_overlay_source": bkmodel.ColumnDataSource(data=dict(x=[], y=[])),
        "awaiting_second_port_click_source": bkmodel.ColumnDataSource(
            data=dict(x=[], y=[], color=[])
        ),
        "panel_preview_edge_source": bkmodel.ColumnDataSource(data=dict(xs=[], ys=[], color=[])),
    }


def _build_toolbox_widgets() -> dict:
    """
    Widgets of the left panel: component buttons, icon-size slider, status and actions.

    :return: ``{BuilderState field name: widget}``; ``buttons`` holds one button
        per icon, in ``BUILDER_ICONS_CONFIG`` order.
    """
    buttons = []
    for icon_key, icon_config in BUILDER_ICONS_CONFIG.items():
        base64_url = icon_data_url(icon_config["icon_path"])
        icon_svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"'
            f' viewBox="0 0 120 120">'
            f'<image href="{base64_url}" width="120" height="120"/></svg>'
        )
        buttons.append(
            bkmodel.Button(
                label=_string_cleanup(icon_key),
                icon=bkmodel.SVGIcon(svg=icon_svg, size="2.75em"),
                button_type=BUTTON_DEFAULT_COLOR_TYPE,
                width=PALETTE_WIDTH - 10,
                height=ROW_HEIGHT - 6,
                stylesheets=_TOOLBOX_BUTTON_CSS,
            )
        )

    def action_button(label: str, icon_name: str, button_type: str):
        return bkmodel.Button(
            label=label,
            icon=bkmodel.TablerIcon(icon_name=icon_name),
            button_type=button_type,
            width=PALETTE_WIDTH - 10,
            height=ROW_HEIGHT - 6,
            stylesheets=_ACTION_BUTTON_CSS,
        )

    return {
        "buttons": buttons,
        "node_size_slider": bkmodel.Slider(
            title=f"Icon size (default {DEFAULT_NODE_ICON_SIZE})",
            start=NODE_ICON_SIZE_MIN,
            end=NODE_ICON_SIZE_MAX,
            step=1,
            value=DEFAULT_NODE_ICON_SIZE,
            width=PALETTE_WIDTH - _TITLE_DIV_WIDTH - 20,
            styles={"color": "white"},
            stylesheets=[":host .bk-slider-title { color: white; font-size: 10pt; }"],
        ),
        "delete_button": action_button("Delete", "trash", BUTTON_DEFAULT_COLOR_TYPE),
        "save_button": action_button("Save", "device-floppy", "success"),
        "end_session_button": action_button("End Session", "power", "warning"),
        "status_div": bkmodel.Div(
            text=STATUS_IDLE_HTML,
            width=PALETTE_WIDTH,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "14px"},
        ),
    }


def _section_heading(text: str) -> bkmodel.Div:
    """Bold heading of one section of the config panel."""
    return bkmodel.Div(
        text=f"<b style='color:white;font-size:16pt'>{text}</b>",
        width=380,
        styles=_SECTION_HEADING_STYLES,
    )


def _build_config_panel(unknown_widgets: dict) -> dict:
    """
    Widgets of the right panel (component configurator), laid out in its column.

    :param unknown_widgets: From :func:`_build_unknown_component_widgets`; they
        are placed next to the fields they replace, but returned by that function.
    :return: ``{BuilderState field name: widget}``, ``config_panel_column`` first.
    """
    field_styles = {"color": "white", "font-size": "18px"}
    name_input = bkmodel.TextInput(
        title="Component ID:", value=_EMPTY, width=380, styles=field_styles
    )
    type_select = bkmodel.Select(
        title="Component Type:", value=_EMPTY, options=[], width=380, styles=field_styles
    )
    position_select = bkmodel.Select(
        title="Position:", value=_EMPTY, options=[], width=380, styles=field_styles
    )
    symmetry_select = bkmodel.Select(value="", options=[], width=380, styles=field_styles)
    apply_button = bkmodel.Button(
        label="Apply",
        icon=bkmodel.TablerIcon(icon_name="check"),
        button_type="primary",
        width=380,
        height=ROW_HEIGHT - 6,
        stylesheets=_ACTION_BUTTON_CSS,
    )

    def port_count_spinner(title: str):
        return bkmodel.Spinner(
            title=title,
            value=1,
            low=1,
            high=20,
            step=1,
            width=180,
            visible=False,
            styles={"color": "white", "font-size": "14px"},
        )

    source_count_spinner = port_count_spinner("Source Ports:")
    target_count_spinner = port_count_spinner("Target Ports:")
    port_count_section = column(
        _section_heading("Ports"),
        row(source_count_spinner, target_count_spinner),
        row(
            unknown_widgets["source_port_type_select"],
            unknown_widgets["target_port_type_select"],
        ),
        unknown_widgets["unknown_no_port_warning_div"],
        visible=False,
        styles={"background": BACKGROUND_COLOR_CODE, "padding": "4px"},
    )

    options_rows_column = column([], styles={"background": BACKGROUND_COLOR_CODE})
    options_table = column(
        bkmodel.Div(
            text="<span style='color:white;font-size:14pt'>Options:</span>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE},
        ),
        options_rows_column,
        unknown_widgets["add_option_button"],
        visible=False,
        styles={"background": BACKGROUND_COLOR_CODE},
    )
    connections_rows_column = column([], styles={"background": BACKGROUND_COLOR_CODE})

    config_panel_column = column(
        bkmodel.Div(
            text="<b style='color:white;font-size:18pt'>Component Configurator</b>",
            width=380,
            styles={"background": BACKGROUND_COLOR_CODE, "text-align": "center"},
        ),
        _section_heading("Component ID &amp; Type"),
        name_input,
        type_select,
        unknown_widgets["unknown_type_input"],
        port_count_section,
        _section_heading("Position &amp; Options"),
        position_select,
        unknown_widgets["unknown_position_input"],
        options_table,
        _section_heading("Connections"),
        column(connections_rows_column, styles={"background": BACKGROUND_COLOR_CODE}),
        _section_heading("Symmetry &amp; Distributed Load"),
        symmetry_select,
        unknown_widgets["unknown_symmetry_input"],
        apply_button,
        spacing=4,
        visible=False,
        styles=dict(_SIDE_PANEL_STYLES),
    )
    return {
        "config_panel_column": config_panel_column,
        "name_input": name_input,
        "type_select": type_select,
        "position_select": position_select,
        "symmetry_select": symmetry_select,
        "apply_button": apply_button,
        "source_count_spinner": source_count_spinner,
        "target_count_spinner": target_count_spinner,
        "port_count_section": port_count_section,
        "options_table": options_table,
        "options_rows_column": options_rows_column,
        "connections_rows_column": connections_rows_column,
    }


def _hidden_input(value: str = "0", **kwargs) -> bkmodel.TextInput:
    """Invisible, zero-size TextInput used to pass a value between browser and Python."""
    return bkmodel.TextInput(value=value, width=0, height=0, visible=False, **kwargs)


def _build_hidden_triggers() -> dict:
    """
    Hidden TextInputs relaying actions between the browser and Python.

    Bokeh's CustomJS cannot call Python directly, so each action that needs
    server-side logic (file dialogs, session shutdown) is routed through a
    hidden TextInput whose value is toggled between '0' and '1' by JS (see
    :func:`_js_flip`). Python registers an ``on_change("value", …)`` callback
    on each trigger; the toggle guarantees the callback fires even when the
    value was already '0' or '1' from a previous click.

    Trigger routing:

    * ``browse_load_trigger`` -> ``IOMixin._on_browse_load``: opens a tkinter
      open-file dialog and restores the canvas from the chosen powertrain YAML.
    * ``browse_save_trigger`` -> ``IOMixin._on_browse_save``: set by the
      "final version?" overlay to ``"<final|editable|cancel>:<timestamp>"``.
      Opens a tkinter save-file dialog and writes the YAML config, with the
      canvas_layout section only if "editable".
    * ``browse_watcher_trigger`` -> ``IOMixin._on_browse_watcher``: set by the
      Browse button of the "final version?" overlay. Opens a tkinter save-file
      dialog and writes the chosen watcher CSV path into watcher_path_input.
    * ``end_session_save_trigger``: flipped by "Save & Exit" so Python ends the
      session once the save (started from the "final version?" overlay) completes.
    * ``end_session_trigger`` -> ``IOMixin._on_end_session_trigger``: stops the
      Bokeh IOLoop. Flipped by JS only when it is safe to do so (design already
      saved, or "End Anyway" confirmed).
    * ``close_window_trigger`` -> browser ``window.close()``: flipped by Python
      after a Save & Exit save completes, so the tab closes before the IOLoop stops.
    * ``view_scale_input`` / ``fit_view_trigger``: window-fitted canvas (see the
      launcher). The browser reports its view scale to Python, and Python asks
      the browser to re-fit the view after a load / New Design.

    :return: ``{BuilderState field name: TextInput}``.
    """
    return {
        "browse_load_trigger": _hidden_input(),
        "browse_save_trigger": _hidden_input(),
        "browse_watcher_trigger": _hidden_input(),
        "end_session_save_trigger": _hidden_input(),
        "end_session_trigger": _hidden_input(),
        "close_window_trigger": _hidden_input(),
        "view_scale_input": _hidden_input("1", name="view_scale_input"),
        "fit_view_trigger": _hidden_input(),
    }


def _build_startup_buttons() -> dict:
    """
    New Design / Load Design buttons of the startup overlay (laid out by the launcher).

    :return: ``{BuilderState field name: Button}``.
    """

    def startup_button(label: str, icon_name: str, button_type: str):
        return bkmodel.Button(
            label=label,
            icon=bkmodel.TablerIcon(icon_name=icon_name),
            button_type=button_type,
            width=220,
            height=70,
            stylesheets=_STARTUP_BUTTON_CSS,
        )

    return {
        "new_design_button": startup_button("New Design", "file-plus", "primary"),
        "load_design_button": startup_button("Load Design", "folder-open", "light"),
    }


def _dialog_title(title: str, subtitle: str, width: int) -> bkmodel.Div:
    """Large centred title and grey subtitle at the top of an overlay dialog."""
    return bkmodel.Div(
        text=(
            "<div style='"
            "color:white;font-size:22pt;font-weight:bold;"
            "text-align:center;padding:24px 0 18px 0;letter-spacing:0.04em;"
            f"'>{title}</div>"
            "<div style='"
            "color:#aaa;font-size:12pt;"
            "text-align:center;padding-bottom:28px;"
            f"'>{subtitle}</div>"
        ),
        width=width,
    )


def _build_save_version_overlay(triggers: dict) -> dict:
    """
    "Is this the final version?" overlay, with its buttons wired.

    Shown when the Save button (or Save & Exit) is clicked. The choice decides
    whether the GUI-only canvas_layout section is written; the watcher CSV path
    is set here too. Each button sends its mode to Python inside the
    browse_save_trigger value itself ("<mode>:<timestamp>"), so the mode and
    the "go" signal arrive as one change.

    :param triggers: From :func:`_build_hidden_triggers`.
    :return: ``{BuilderState field name: widget}``.
    """
    final_version_button = bkmodel.Button(
        label="Final Version",
        icon=bkmodel.TablerIcon(icon_name="circle-check"),
        button_type="success",
        width=240,
        height=70,
        stylesheets=_STARTUP_BUTTON_CSS,
    )
    editable_version_button = bkmodel.Button(
        label="Edit Later",
        icon=bkmodel.TablerIcon(icon_name="pencil"),
        button_type="primary",
        width=240,
        height=70,
        stylesheets=_STARTUP_BUTTON_CSS,
    )
    cancel_button = bkmodel.Button(label="Cancel", button_type="light", width=100, height=40)

    # Watcher CSV: the path written as watcher_file_path in the saved YAML,
    # typed in directly or picked with the Browse button. Optional: an empty
    # field leaves the key out of the file
    watcher_title_div = bkmodel.Div(
        text=(
            "<div style='color:white;font-size:13pt;font-weight:bold;'>"
            "Watcher CSV <span style='color:#aaa;font-weight:normal;'>(optional)</span>"
            "</div>"
        ),
        width=560,
    )
    watcher_path_input = bkmodel.TextInput(
        value="",
        placeholder="No watcher file (not written to the YAML)",
        width=440,
        height=34,
        align="center",
        styles={"font-size": "11pt"},
    )
    watcher_browse_button = bkmodel.Button(
        label="Browse",
        icon=bkmodel.TablerIcon(icon_name="folder-open"),
        button_type=BUTTON_DEFAULT_COLOR_TYPE,
        width=110,
        height=34,
        align="center",
    )

    save_version_overlay = column(
        _dialog_title(
            "Is this the final version?",
            "<b>Final Version</b> writes only the FAST-OAD configuration.<br>"
            "<b>Edit Later</b> also stores the canvas layout so the design "
            "re-opens exactly as it is now.",
            width=560,
        ),
        watcher_title_div,
        row(watcher_path_input, watcher_browse_button, spacing=8),
        bkmodel.Spacer(height=22),
        row(
            final_version_button,
            bkmodel.Div(text="", width=30),
            editable_version_button,
            styles={"justify-content": "center"},
        ),
        bkmodel.Spacer(height=14),
        row(cancel_button, styles={"justify-content": "center"}),
        visible=False,
        styles=dict(DIALOG_STYLES),
    )

    # Browser-side wiring of the overlay's own buttons
    watcher_browse_button.js_on_click(
        bkmodel.CustomJS(
            args=dict(browse_watcher_trigger=triggers["browse_watcher_trigger"]),
            code=_JS_WATCHER_DIALOG,
        )
    )
    for mode_button, save_mode, mode_click_code in (
        (final_version_button, SAVE_MODE_FINAL, _JS_SAVE_MODE_CLICK),
        (editable_version_button, SAVE_MODE_EDITABLE, _JS_SAVE_MODE_CLICK),
        (cancel_button, SAVE_MODE_CANCEL, _JS_SAVE_CANCEL_CLICK),
    ):
        mode_button.js_on_click(
            bkmodel.CustomJS(
                args=dict(
                    save_version_overlay_widget=save_version_overlay,
                    browse_save_trigger=triggers["browse_save_trigger"],
                    watcher_path_input=watcher_path_input,
                ),
                code=mode_click_code.replace("__MODE__", save_mode),
            )
        )

    return {
        "save_version_overlay": save_version_overlay,
        "watcher_path_input": watcher_path_input,
    }


def _build_unsaved_exit_overlay(triggers: dict, save_version_overlay) -> dict:
    """
    "Unsaved Changes" overlay, with its buttons wired.

    Shown when the user clicks End Session while save_button is yellow:

    * Cancel hides it;
    * End Anyway hides it, flips end_session_trigger so Python stops the
      server, then closes the browser tab;
    * Save & Exit opens the "final version?" overlay; the Python handler ends
      the session after the file is written (flagged via end_session_save_trigger).

    :param triggers: From :func:`_build_hidden_triggers`.
    :param save_version_overlay: The "final version?" overlay, opened by Save & Exit.
    :return: ``{BuilderState field name: widget}``.
    """
    end_anyway_button = bkmodel.Button(
        label="End Anyway",
        icon=bkmodel.TablerIcon(icon_name="power"),
        button_type="danger",
        width=200,
        height=40,
    )
    save_and_exit_button = bkmodel.Button(
        label="Save & Exit",
        icon=bkmodel.TablerIcon(icon_name="device-floppy"),
        button_type="primary",
        width=200,
        height=40,
    )
    cancel_button = bkmodel.Button(label="Cancel", button_type="light", width=100, height=40)

    unsaved_exit_overlay = column(
        _dialog_title("Unsaved Changes", "End session without saving changes?", width=500),
        row(
            cancel_button,
            bkmodel.Div(text="", width=20),
            end_anyway_button,
            bkmodel.Div(text="", width=20),
            save_and_exit_button,
            styles={"justify-content": "center"},
        ),
        triggers["end_session_save_trigger"],
        visible=False,
        styles=dict(DIALOG_STYLES),
    )

    # Browser-side wiring of the overlay's own buttons
    cancel_button.js_on_click(
        bkmodel.CustomJS(
            args=dict(unsaved_exit_overlay_widget=unsaved_exit_overlay),
            code="unsaved_exit_overlay_widget.visible = false;",
        )
    )
    end_anyway_button.js_on_click(
        bkmodel.CustomJS(
            args=dict(
                unsaved_exit_overlay_widget=unsaved_exit_overlay,
                end_session_trigger=triggers["end_session_trigger"],
            ),
            code=(
                "unsaved_exit_overlay_widget.visible = false; "
                f"{_js_flip('end_session_trigger')} "
                "window.close();"
            ),
        )
    )
    save_and_exit_button.js_on_click(
        bkmodel.CustomJS(
            args=dict(
                unsaved_exit_overlay_widget=unsaved_exit_overlay,
                save_version_overlay_widget=save_version_overlay,
                end_session_save_trigger=triggers["end_session_save_trigger"],
            ),
            code=(
                "unsaved_exit_overlay_widget.visible = false; "
                "save_version_overlay_widget.visible = true; "
                f"{_js_flip('end_session_save_trigger')}"
            ),
        )
    )

    return {"unsaved_exit_overlay": unsaved_exit_overlay}


def _build_registry_lookups() -> dict:
    """
    Read-only per-component-type tables, from the registry and the component sources.

    :return: ``{BuilderState field name: dict}``.
    """
    default_source_count, default_target_count = _build_port_count_defaults()
    possible_position, possible_options = _get_performance_component_names(COMPONENTS_PATH)
    return {
        "default_source_count": default_source_count,
        "default_target_count": default_target_count,
        "possible_position": possible_position,
        "possible_options": possible_options,
        "icon_to_component_types": _map_possible_component_types_to_icons(),
    }


# ============================================================================
# Wiring and layout, once every widget exists
# ============================================================================


def _wire_toolbox_callbacks(state: BuilderState) -> None:
    """
    Browser-side (``CustomJS``) callbacks that link widgets of different groups.

    These are the only ones that need several groups to exist, which is why
    they run after :class:`BuilderState` is assembled. Callbacks between
    widgets of one overlay are wired by that overlay's builder.

    * Save -> show the "final version?" overlay.
    * End Session -> show the unsaved-exit overlay if save_button is yellow,
      otherwise flip end_session_trigger and close the tab.
    * Load Design -> flip browse_load_trigger (Python opens the file dialog).
    * close_window_trigger change -> close the tab.

    :param state: The assembled builder state.
    """
    state.save_button.js_on_click(
        bkmodel.CustomJS(
            args=dict(save_version_overlay_widget=state.save_version_overlay),
            code="save_version_overlay_widget.visible = true;",
        )
    )
    state.end_session_button.js_on_click(
        bkmodel.CustomJS(
            args=dict(
                save_button_widget=state.save_button,
                unsaved_exit_overlay_widget=state.unsaved_exit_overlay,
                end_session_trigger=state.end_session_trigger,
            ),
            code=_JS_END_SESSION_CLICK,
        )
    )
    state.load_design_button.js_on_click(
        bkmodel.CustomJS(
            args=dict(browse_load_trigger=state.browse_load_trigger),
            code=_JS_LOAD_DIALOG,
        )
    )
    state.close_window_trigger.js_on_change("value", bkmodel.CustomJS(code="window.close();"))


def _build_toolbox_layout(state: BuilderState):
    """
    Left panel: header, tabbed component buttons, status, actions and hidden triggers.

    :param state: The assembled builder state.
    :return: The toolbox column.
    """
    toolbox_header = row(
        bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Components</b>",
            width=_TITLE_DIV_WIDTH,
            styles=_SECTION_HEADING_STYLES,
        ),
        state.node_size_slider,
        width=PALETTE_WIDTH,
        styles={"background": BACKGROUND_COLOR_CODE},
    )
    # Tabs by component type class; the unknown component closes several tabs
    button_by_icon_key = dict(zip(BUILDER_ICONS_CONFIG, state.buttons, strict=True))
    tabs = _build_toolbox_tabs(
        _with_unknown_component(_group_icons_by_tab(KNOWN_COMPONENTS)), button_by_icon_key
    )
    return column(
        toolbox_header,
        tabs,
        state.status_div,
        state.delete_button,
        state.save_button,
        state.end_session_button,
        state.browse_load_trigger,
        state.browse_save_trigger,
        state.browse_watcher_trigger,
        state.end_session_save_trigger,
        state.end_session_trigger,
        state.close_window_trigger,
        state.view_scale_input,
        state.fit_view_trigger,
        spacing=2,
        styles=dict(_SIDE_PANEL_STYLES),
    )


class ComponentToolboxConfigurationTableBuilder:
    """
    Build the toolbox sidebar and component-configurator panel as Bokeh widgets.

    This class is **pure**: it only constructs Bokeh objects and returns them.
    All Python event callbacks are wired by :class:`BuilderHandler`.
    """

    @staticmethod
    def build() -> tuple:
        """
        Construct the button toolbox, configurator panel, and shared builder state.

        Every group of :class:`BuilderState` fields is built first, in the
        order of its sections; the state is assembled from them; then the
        browser-side callbacks between groups are wired and the toolbox laid
        out. Fields not passed here keep their default and are set later: the
        undo / redo buttons, canvas renderers and startup overlay by the
        launcher, the undo stack by BuilderHandler, and the interaction /
        session values at run time.

        :return: A three-element tuple
            ``(toolbox_column_layout, config_panel_column, BuilderState)`` where

            * ``toolbox_column_layout`` is the left-side Bokeh column widget
              containing the tabbed component buttons and action buttons.
            * ``config_panel_column`` is the right-side Bokeh column widget for the
              component configurator.
            * :class:`BuilderState` holds references to every shared widget
              and data source.
        """
        # Groups that other groups are laid out with or wired to
        unknown_widgets = _build_unknown_component_widgets()
        triggers = _build_hidden_triggers()
        save_version_overlay = _build_save_version_overlay(triggers)
        unsaved_exit_overlay = _build_unsaved_exit_overlay(
            triggers, save_version_overlay["save_version_overlay"]
        )

        state = BuilderState(
            **_build_canvas_sources(),  # design data, derived and temporary data
            **_build_toolbox_widgets(),
            **_build_config_panel(unknown_widgets),
            **unknown_widgets,
            **_build_startup_buttons(),
            **save_version_overlay,
            **unsaved_exit_overlay,
            **triggers,
            **_build_registry_lookups(),
        )
        _wire_toolbox_callbacks(state)
        return _build_toolbox_layout(state), state.config_panel_column, state
