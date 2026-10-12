# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Palette sidebar and component-configurator panel builder.

:class:`ComponentPaletteConfigurationTableBuilder` is a pure factory that
constructs every Bokeh widget and data source needed by the builder UI and
packages them into a :class:`BuilderState`. No event callbacks are wired
here; that responsibility belongs to :class:`BuilderHandler`.

Typical usage::

    from fastga_he.gui.power_train_builder_toolbox import (
        ComponentPaletteConfigurationTableBuilder,
    )

    toolbox_layout, config_panel_column, state = ComponentPaletteConfigurationTableBuilder.build()
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
    trigger fires on every click (see the "Hidden TextInput relay widgets"
    comment in :meth:`ComponentToolboxConfigurationTableBuilder.build`).

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
    option row, and the two port-type selects set the energy type of its
    source and target ports. All start hidden.

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


class ComponentToolboxConfigurationTableBuilder:
    """
    Build the toolbox sidebar and component-configurator panel as Bokeh widgets.

    This class is **pure**: it only constructs Bokeh objects and returns them.
    All event callbacks are wired by :class:`BuilderHandler`.
    """

    @staticmethod
    def build() -> tuple:
        """
        Construct the button toolbox, configurator panel, and shared builder state.

        :return: A three-element tuple
            ``(toolbox_column_layout, config_panel_column, BuilderState)`` where

            * ``toolbox_column_layout`` is the left-side Bokeh column widget
              containing the tabbed component buttons and action buttons.
            * ``config_panel_column`` is the right-side Bokeh column widget for the
              component configurator.
            * :class:`BuilderState` holds references to every shared widget
              and data source.
        """
        # Categorise icons into tab groups; the unknown component closes several tabs
        component_icon_keys = list(BUILDER_ICONS_CONFIG.keys())
        category_to_icon_keys = _with_unknown_component(_group_icons_by_tab(KNOWN_COMPONENTS))

        # Shared stylesheet for action buttons (Delete, Save, End Session)
        _action_stylesheet = [":host button { font-size: 1.4em; }"]

        # Palette buttons (one per icon, kept in BUILDER_ICONS_CONFIG order)
        # Header row: "Components" title + node-size slider. Both widths add up to
        # PALETTE_WIDTH so the row never grows wider than the left panel
        _title_div_width = 140
        title_div = bkmodel.Div(
            text="<b style='color:white;font-size:16pt'>Components</b>",
            width=_title_div_width,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
        )
        node_size_slider = bkmodel.Slider(
            title=f"Icon size (default {DEFAULT_NODE_ICON_SIZE})",
            start=NODE_ICON_SIZE_MIN,
            end=NODE_ICON_SIZE_MAX,
            step=1,
            value=DEFAULT_NODE_ICON_SIZE,
            width=PALETTE_WIDTH - _title_div_width - 20,
            styles={"color": "white"},
            stylesheets=[":host .bk-slider-title { color: white; font-size: 10pt; }"],
        )
        toolbox_header = row(
            title_div,
            node_size_slider,
            width=PALETTE_WIDTH,
            styles={"background": BACKGROUND_COLOR_CODE},
        )

        _toolbox_button_stylesheet = [
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
            base64_url = icon_data_url(BUILDER_ICONS_CONFIG[icon_key]["icon_path"])
            icon_svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"'
                f' viewBox="0 0 120 120">'
                f'<image href="{base64_url}" width="120" height="120"/></svg>'
            )
            button = bkmodel.Button(
                label=label,
                icon=bkmodel.SVGIcon(svg=icon_svg, size="2.75em"),
                button_type=BUTTON_DEFAULT_COLOR_TYPE,
                width=PALETTE_WIDTH - 10,
                height=ROW_HEIGHT - 6,
                stylesheets=_toolbox_button_stylesheet,
            )
            buttons.append(button)
            button_by_icon_key[icon_key] = button

        status_div = bkmodel.Div(
            text=STATUS_IDLE_HTML,
            width=PALETTE_WIDTH,
            styles={"background": BACKGROUND_COLOR_CODE, "padding": "14px"},
        )

        # Action buttons
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
        # The End Session button's js_on_click is wired further below, after both
        # ``unsaved_exit_overlay`` and ``end_session_trigger`` have been constructed.
        # The CustomJS callback needs to reference all three objects (``save_button``,
        # ``unsaved_exit_overlay_widget``, and ``end_session_trigger``) as args, so the
        # wiring must be deferred until all of them exist

        # Hidden TextInput relay widgets
        # Bokeh's CustomJS cannot call Python directly, so each action that needs
        # server-side logic (file dialogs, session shutdown) is routed through a
        # hidden TextInput whose value is toggled between '0' and '1' by JS.
        # Python registers an on_change("value", …) callback on each trigger;
        # the toggle guarantees the callback fires even when the value was already
        # '0' or '1' from a previous click.
        #
        # Trigger routing:
        #   browse_load_trigger    -> IOMixin._on_browse_load
        #                            Opens a tkinter open-file dialog; restores the
        #                            canvas from the chosen powertrain YAML.
        #   browse_watcher_trigger -> IOMixin._on_browse_watcher
        #                            Set by the Browse button of the "final version?"
        #                            overlay. Opens a tkinter save-file dialog; writes
        #                            the chosen watcher CSV path into watcher_path_input.
        #   browse_save_trigger    -> IOMixin._on_browse_save
        #                            Set by the "final version?" overlay to
        #                            "<final|editable|cancel>:<timestamp>". Opens a
        #                            tkinter save-file dialog and writes the YAML config,
        #                            with the canvas_layout section only if "editable".
        #   end_session_trigger    -> IOMixin._on_end_session_trigger
        #                            Stops the Bokeh IOLoop, terminating the server.
        #                            Flipped by JS only when it is safe to do so (design
        #                            already saved, or "End Anyway" confirmed).
        #   close_window_trigger   -> js_on_change CustomJS (window.close())
        #                            Flipped by Python after a Save & Exit save completes
        #                            so the browser tab closes before the IOLoop stops

        browse_load_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        browse_save_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        browse_watcher_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        end_session_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        close_window_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        # Window-fitted canvas (see the launcher): JS -> Python view scale, and
        # Python -> JS "re-fit the view" request after a load / New Design
        view_scale_input = bkmodel.TextInput(
            value="1", width=0, height=0, visible=False, name="view_scale_input"
        )
        fit_view_trigger = bkmodel.TextInput(value="0", width=0, height=0, visible=False)
        close_window_trigger.js_on_change(
            "value",
            bkmodel.CustomJS(code="window.close();"),
        )

        # Wire Load Design buttons -> signal Python to open tkinter dialog
        _load_js = bkmodel.CustomJS(
            args=dict(browse_load_trigger=browse_load_trigger),
            code=_JS_LOAD_DIALOG,
        )

        # "Final version?" overlay
        # Shown when the Save button (or Save & Exit) is clicked. The choice decides whether the GUI-only
        # canvas_layout section is written; the watcher CSV path is set here too. Each button sends its mode to Python
        # inside the browse_save_trigger value itself ("<mode>:<timestamp>"), so
        # the mode and the "go" signal arrive as one change
        _final_version_btn = bkmodel.Button(
            label="Final Version",
            icon=bkmodel.TablerIcon(icon_name="circle-check"),
            button_type="success",
            width=240,
            height=70,
            stylesheets=_STARTUP_BUTTON_CSS,
        )
        _editable_version_btn = bkmodel.Button(
            label="Edit Later",
            icon=bkmodel.TablerIcon(icon_name="pencil"),
            button_type="primary",
            width=240,
            height=70,
            stylesheets=_STARTUP_BUTTON_CSS,
        )
        _cancel_version_btn = bkmodel.Button(
            label="Cancel",
            button_type="light",
            width=100,
            height=40,
        )

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
        watcher_browse_button.js_on_click(
            bkmodel.CustomJS(
                args=dict(browse_watcher_trigger=browse_watcher_trigger),
                code=_JS_WATCHER_DIALOG,
            )
        )
        save_version_overlay = column(
            bkmodel.Div(
                text=(
                    "<div style='"
                    "color:white;font-size:22pt;font-weight:bold;"
                    "text-align:center;padding:24px 0 18px 0;letter-spacing:0.04em;"
                    "'>Is this the final version?</div>"
                    "<div style='"
                    "color:#aaa;font-size:12pt;"
                    "text-align:center;padding-bottom:28px;"
                    "'><b>Final Version</b> writes only the FAST-OAD configuration.<br>"
                    "<b>Edit Later</b> also stores the canvas layout so the design "
                    "re-opens exactly as it is now.</div>"
                ),
                width=560,
            ),
            watcher_title_div,
            row(watcher_path_input, watcher_browse_button, spacing=8),
            bkmodel.Spacer(height=22),
            row(
                _final_version_btn,
                bkmodel.Div(text="", width=30),
                _editable_version_btn,
                styles={"justify-content": "center"},
            ),
            bkmodel.Spacer(height=14),
            row(_cancel_version_btn, styles={"justify-content": "center"}),
            visible=False,
            styles=dict(DIALOG_STYLES),
        )
        for mode_button, save_mode, mode_click_code in (
            (_final_version_btn, SAVE_MODE_FINAL, _JS_SAVE_MODE_CLICK),
            (_editable_version_btn, SAVE_MODE_EDITABLE, _JS_SAVE_MODE_CLICK),
            (_cancel_version_btn, SAVE_MODE_CANCEL, _JS_SAVE_CANCEL_CLICK),
        ):
            mode_button.js_on_click(
                bkmodel.CustomJS(
                    args=dict(
                        save_version_overlay_widget=save_version_overlay,
                        browse_save_trigger=browse_save_trigger,
                        watcher_path_input=watcher_path_input,
                    ),
                    code=mode_click_code.replace("__MODE__", save_mode),
                )
            )

        # Wire Save button: click -> ask "final version?"
        save_button.js_on_click(
            bkmodel.CustomJS(
                args=dict(save_version_overlay_widget=save_version_overlay),
                code="save_version_overlay_widget.visible = true;",
            )
        )

        # Unsaved-exit overlay
        # Shown when the user clicks End Session while save_button is yellow.
        # Hidden trigger used by "Save & Exit" to flag Python that the session must
        # end once the save (started from the "final version?" overlay) completes
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
            styles=dict(DIALOG_STYLES),
        )

        # Cancel: just hide the overlay
        _cancel_exit_btn.js_on_click(
            bkmodel.CustomJS(
                args=dict(unsaved_exit_overlay_widget=unsaved_exit_overlay),
                code="unsaved_exit_overlay_widget.visible = false;",
            )
        )

        # End Anyway: hide the overlay, flip end_session_trigger so Python stops the server,
        # then close the browser tab
        _end_anyway_btn.js_on_click(
            bkmodel.CustomJS(
                args=dict(
                    unsaved_exit_overlay_widget=unsaved_exit_overlay,
                    end_session_trigger=end_session_trigger,
                ),
                code=(
                    "unsaved_exit_overlay_widget.visible = false; "
                    f"{_js_flip('end_session_trigger')} "
                    "window.close();"
                ),
            )
        )

        # Save & Exit: open the "final version?" overlay; the Python handler ends the
        # session after the file is written (flagged via end_session_save_trigger)
        _save_and_exit_btn.js_on_click(
            bkmodel.CustomJS(
                args=dict(
                    unsaved_exit_overlay_widget=unsaved_exit_overlay,
                    save_version_overlay_widget=save_version_overlay,
                    end_session_save_trigger=end_session_save_trigger,
                ),
                code=(
                    "unsaved_exit_overlay_widget.visible = false; "
                    "save_version_overlay_widget.visible = true; "
                    f"{_js_flip('end_session_save_trigger')}"
                ),
            )
        )

        # Wire End Session button: check for unsaved state before closing
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

        # Startup overlay buttons
        new_design_button = bkmodel.Button(
            label="New Design",
            icon=bkmodel.TablerIcon(icon_name="file-plus"),
            button_type="primary",
            width=220,
            height=70,
            stylesheets=_STARTUP_BUTTON_CSS,
        )
        load_design_button = bkmodel.Button(
            label="Load Design",
            icon=bkmodel.TablerIcon(icon_name="folder-open"),
            button_type="light",
            width=220,
            height=70,
            stylesheets=_STARTUP_BUTTON_CSS,
        )
        load_design_button.js_on_click(_load_js)

        # Canvas data sources
        placed_nodes_source = bkmodel.ColumnDataSource(data=empty_columns(NODE_COLUMNS))
        source_port_source = bkmodel.ColumnDataSource(data=empty_columns(PORT_COLUMNS))
        target_port_source = bkmodel.ColumnDataSource(data=empty_columns(PORT_COLUMNS))
        selected_node_overlay_source = bkmodel.ColumnDataSource(data=dict(x=[], y=[]))
        edge_source = bkmodel.ColumnDataSource(data=empty_columns(EDGE_COLUMNS))
        awaiting_second_port_click_source = bkmodel.ColumnDataSource(
            data=dict(x=[], y=[], color=[])
        )
        panel_preview_edge_source = bkmodel.ColumnDataSource(data=dict(xs=[], ys=[], color=[]))

        # Configurator panel widgets
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
        unknown_widgets = _build_unknown_component_widgets()

        port_count_section = column(
            bkmodel.Div(
                text="<b style='color:white;font-size:16pt'>Ports</b>",
                width=380,
                styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
            ),
            row(source_count_spinner, target_count_spinner),
            row(
                unknown_widgets["source_port_type_select"],
                unknown_widgets["target_port_type_select"],
            ),
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
            unknown_widgets["add_option_button"],
            visible=False,
            styles={"background": BACKGROUND_COLOR_CODE},
        )

        config_panel_column = column(
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
            unknown_widgets["unknown_type_input"],
            port_count_section,
            bkmodel.Div(
                text="<b style='color:white;font-size:16pt'>Position &amp; Options</b>",
                width=380,
                styles={"background": BACKGROUND_COLOR_CODE, "padding": "6px 4px 2px 4px"},
            ),
            position_select,
            unknown_widgets["unknown_position_input"],
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
            unknown_widgets["unknown_symmetry_input"],
            apply_button,
            spacing=4,
            visible=False,
            # Scroll inside the panel instead of growing the page on short screens
            styles={
                "background": BACKGROUND_COLOR_CODE,
                "padding": "10px",
                "max-height": "calc(100vh - 8px)",
                "overflow-y": "auto",
            },
        )

        # Metadata lookup tables
        default_source_port_count, default_target_port_count = _build_port_count_defaults()
        icon_to_component_types_map = _map_possible_component_types_to_icons()
        possible_position_map, possible_options_map = _get_performance_component_names(
            COMPONENTS_PATH
        )

        # Assemble BuilderState
        state = BuilderState(
            buttons=buttons,
            placed_nodes_source=placed_nodes_source,
            status_div=status_div,
            node_size_slider=node_size_slider,
            delete_button=delete_button,
            save_button=save_button,
            end_session_button=end_session_button,
            source_port_source=source_port_source,
            target_port_source=target_port_source,
            source_count_spinner=source_count_spinner,
            target_count_spinner=target_count_spinner,
            port_count_section=port_count_section,
            options_table=options_table,
            options_rows_column=options_rows_column,
            name_input=name_input,
            type_select=type_select,
            position_select=position_select,
            apply_button=apply_button,
            config_panel_column=config_panel_column,
            selected_node_overlay_source=selected_node_overlay_source,
            edge_source=edge_source,
            awaiting_second_port_click_source=awaiting_second_port_click_source,
            connections_rows_column=connections_rows_column,
            panel_preview_edge_source=panel_preview_edge_source,
            panel_pending_connections=[],
            symmetry_select=symmetry_select,
            **unknown_widgets,
            default_source_count=default_source_port_count,
            default_target_count=default_target_port_count,
            icon_to_component_types=icon_to_component_types_map,
            possible_position=possible_position_map,
            possible_options=possible_options_map,
            browse_load_trigger=browse_load_trigger,
            browse_save_trigger=browse_save_trigger,
            browse_watcher_trigger=browse_watcher_trigger,
            watcher_path_input=watcher_path_input,
            save_version_overlay=save_version_overlay,
            new_design_button=new_design_button,
            load_design_button=load_design_button,
            unsaved_exit_overlay=unsaved_exit_overlay,
            end_session_save_trigger=end_session_save_trigger,
            end_session_trigger=end_session_trigger,
            close_window_trigger=close_window_trigger,
            view_scale_input=view_scale_input,
            fit_view_trigger=fit_view_trigger,
        )

        # Build tabbed toolbox sidebar
        tabs = _build_toolbox_tabs(category_to_icon_keys, button_by_icon_key)

        toolbox_layout = column(
            toolbox_header,
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
            view_scale_input,
            fit_view_trigger,
            spacing=2,
            # Scroll inside the panel instead of growing the page on short screens
            styles={
                "background": BACKGROUND_COLOR_CODE,
                "padding": "10px",
                "max-height": "calc(100vh - 8px)",
                "overflow-y": "auto",
            },
        )

        return toolbox_layout, config_panel_column, state
