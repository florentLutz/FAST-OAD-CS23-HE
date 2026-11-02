# Toolbox and panel layout

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_toolbox_config_table.py`. This module is a pure factory: it creates every Bokeh widget and data source of the builder and packs them into a [`BuilderState`](powertrain_builder_state#builderstate). The only callbacks wired here are browser-side (`CustomJS`) ones; every Python callback is wired by [`BuilderHandler`](powertrain_builder_handler).

Each `_build_*` function returns `{BuilderState field name: object}`, so the state is assembled by unpacking them.

---

## ComponentToolboxConfigurationTableBuilder.build

```python
ComponentToolboxConfigurationTableBuilder.build()
```

Construct the component toolbox, the configuration panel and the shared builder state.

---

### Parameters

None.

---

### Returns

- **toolbox_layout** : *bokeh.models.Column*

  Left panel: title, icon-size slider, tabbed component buttons, status line, action buttons and hidden triggers.

- **config_panel_column** : *bokeh.models.Column*

  Right panel (component configurator), hidden until a node is selected.

- **state** : *BuilderState*

  Shared state holding every widget and data source.

---

### See also

- [`PowertrainBuilderLauncher.launch`](powertrain_builder_launcher#powertrainbuilderlauncherlaunch): only caller.

---

### Notes

The build is done in four steps.

1. **Groups needed by other groups.** The unknown-component widgets (laid out inside the panel), the hidden triggers 
(wired to the overlays), the "final version?" overlay and the unsaved-exit overlay (which opens the former).
2. **State.** `BuilderState(**...)` from the canvas sources, toolbox widgets, panel widgets, unknown-component widgets, 
startup buttons, overlays, triggers and registry lookups.
3. **Cross-group wiring.** `_wire_toolbox_callbacks(state)`.
4. **Layout.** `_build_toolbox_layout(state)`.

Some fields are left at their default and set later: `undo_button`, `redo_button`, `startup_overlay` and the three 
canvas renderers by the launcher; `edit_history_stack` by `BuilderHandler`; and the interaction values at run time.

---

### Examples

```python
from fastga_he.gui.power_train_builder.power_train_builder_toolbox_config_table import (
    ComponentToolboxConfigurationTableBuilder,
)

toolbox_layout, config_panel_column, state = ComponentToolboxConfigurationTableBuilder.build()
print(len(state.buttons), "toolbox buttons")
```

---

## _js_flip

```python
_js_flip(trigger_name)
```

Return a JavaScript statement that flips a hidden trigger `TextInput` between `'0'` and `'1'`.

Every flip is a value change, so the Python `on_change` callback of the trigger runs on every click.

---

### Parameters

- **trigger_name** : *str*

  Name of the trigger in the `CustomJS` arguments.

---

### Returns

- **code** : *str*

  One line of JavaScript, e.g. `browse_load_trigger.value = browse_load_trigger.value === '1' ? '0' : '1';`.

---

## _group_icons_by_tab

```python
_group_icons_by_tab(known_components)
```

Assign every component icon to the toolbox tab(s) of its type classes.

---

### Parameters

- **known_components** : *list of dict*

  Entries of `KNOWN_COMPONENTS`.

---

### Returns

- **icons_by_tab** : *dict*

  `{tab_name: [icon_key, ...]}`, tabs in the order their class is first met.

---

### Notes

- Each value of `components_type_class` is a tab.
- A component whose only class is `"propulsive_load"` goes to the `"load"` tab.
- In a list of classes, `"propulsive_load"` does not create a tab of its own.
- An icon is listed once per tab, even if several components share it.

---

## _with_unknown_component

```python
_with_unknown_component(icons_by_tab)
```

Append the unknown component, last, to each tab of `UNKNOWN_COMPONENT_TABS` (`connector`, `propulsor`, `load`, `tank`, 
`source`) that exists.

No tab is created for the unknown component alone.

---

### Parameters

- **icons_by_tab** : *dict*

  `{tab_name: [icon_key, ...]}` from `_group_icons_by_tab`.

---

### Returns

- **icons_by_tab** : *dict*

  New mapping with `"unknown"` appended where it belongs.

---

## _build_toolbox_tabs

```python
_build_toolbox_tabs(icons_by_tab, button_by_icon_key)
```

Build one toolbox tab per group, holding the buttons of its icons.

Icons without a button are skipped, and a tab left empty is not shown. Tab titles are the class names capitalised.

---

### Parameters

- **icons_by_tab** : *dict*

  `{tab_name: [icon_key, ...]}`.

- **button_by_icon_key** : *dict*

  Toolbox button of every icon key.

---

### Returns

- **tabs** : *bokeh.models.Tabs*

  Tab widget of width `PALETTE_WIDTH`.

---

## _build_canvas_sources

```python
_build_canvas_sources()
```

Create the empty data sources of the canvas.

---

### Parameters

None.

---

### Returns

- **sources** : *dict*

  `{BuilderState field name: ColumnDataSource}`:

  | Field | Columns |
  |---|---|
  | `placed_nodes_source` | `NODE_COLUMNS` |
  | `edge_source` | `EDGE_COLUMNS` |
  | `source_port_source`, `target_port_source` | `PORT_COLUMNS` |
  | `selected_node_overlay_source` | `x`, `y` |
  | `awaiting_second_port_click_source` | `x`, `y`, `color` |
  | `panel_preview_edge_source` | `xs`, `ys`, `color` |

---

## _build_toolbox_widgets

```python
_build_toolbox_widgets()
```

Create the widgets of the left panel.

---

### Parameters

None.

---

### Returns

- **widgets** : *dict*

  - `buttons`: one button per entry of `BUILDER_ICONS_CONFIG`, in the same order. Label is the cleaned-up icon key; icon
    is the component image wrapped in an SVG, shown right of the label.
  - `node_size_slider`: icon size from `NODE_ICON_SIZE_MIN` to `NODE_ICON_SIZE_MAX` (20–80), starting at 
     `DEFAULT_NODE_ICON_SIZE` (50).
  - `delete_button`, `save_button` (green), `end_session_button` (yellow).
  - `status_div`: status line, initially `STATUS_IDLE_HTML`.

---

## _section_heading

```python
_section_heading(text)
```

Return the bold, white 16 pt heading of one section of the configuration panel.

---

### Parameters

- **text** : *str*

  Heading text (HTML allowed, e.g. `&amp;`).

---

### Returns

- **heading** : *bokeh.models.Div*

---

## _build_config_panel

```python
_build_config_panel(unknown_widgets)
```

Create the widgets of the configuration panel and lay them out in its column.

---

### Parameters

- **unknown_widgets** : *dict*

  Widgets from `_build_unknown_component_widgets`. They are placed next to the widget they replace, but are returned by 
  that function, not by this one.

---

### Returns

- **widgets** : *dict*

  `config_panel_column`, `name_input`, `type_select`, `position_select`, `symmetry_select`, `apply_button`, 
  `source_count_spinner`, `target_count_spinner`, `port_count_section`, `options_table`, `options_rows_column`, 
  `connections_rows_column`.

---

### Notes

Order of the panel, top to bottom:

1. *Component Configurator* title
2. *Component ID & Type*: name input, type select / unknown type input
3. *Ports* section (hidden by default): source / target count spinners (1–20), source / target port-type selects
4. *Position & Options*: position select / unknown position input, options table (rows + **Add option** button)
5. *Connections*: one row per port of the selected node
6. *Symmetry & Distributed Load*: symmetry select / unknown symmetry input
7. **Apply** button

The panel column is hidden until a node is selected, and scrolls inside itself on short screens.

---

## _build_unknown_component_widgets

```python
_build_unknown_component_widgets()
```

Create the configuration-panel widgets used only for an [unknown component](powertrain_builder_unknown). All start 
hidden.

---

### Parameters

None.

---

### Returns

- **widgets** : *dict*

  - `unknown_type_input`, `unknown_position_input`, `unknown_symmetry_input`: free-text counterparts of the three 
    selects.
  - `add_option_button`: adds an empty option row.
  - `source_port_type_select`, `target_port_type_select`: energy type of each port side, from `PORT_TYPE_CHOICES`, 
    starting at `"undefined"`.

---

## _hidden_input

```python
_hidden_input(value='0', **kwargs)
```

Return an invisible, zero-size `TextInput` used to pass a value between the browser and Python.

---

### Parameters

- **value** : *str, optional*

  Initial value (default: `'0'`).

- **kwargs**

  Forwarded to `bokeh.models.TextInput` (e.g. `name`).

---

### Returns

- **trigger** : *bokeh.models.TextInput*

---

## _build_hidden_triggers

```python
_build_hidden_triggers()
```

Create the hidden `TextInput` widgets that relay actions between the browser and Python.

A `CustomJS` callback cannot call Python. Each action needing server-side logic changes the value of one of these 
inputs, and Python reacts in its `on_change` callback.

---

### Parameters

None.

---

### Returns

- **triggers** : *dict*

  | Field | Direction | Effect |
  |---|---|---|
  | `browse_load_trigger` | JS -> Python | `_on_browse_load`: open-file dialog, then load |
  | `browse_save_trigger` | JS -> Python | `_on_browse_save`; value `"<final\|editable\|cancel>:<timestamp>"` |
  | `browse_watcher_trigger` | JS -> Python | `_on_browse_watcher`: watcher CSV dialog |
  | `end_session_save_trigger` | JS -> Python | Ends the session after the next save (Save & Exit) |
  | `end_session_trigger` | JS -> Python | `_on_end_session_trigger`: stops the server |
  | `close_window_trigger` | Python -> JS | Closes the browser tab |
  | `view_scale_input` | JS -> Python | View report of the fitted canvas (JSON) |
  | `fit_view_trigger` | Python -> JS | Re-fits the view |

---

### See also

- [I/O](powertrain_builder_io): Python side of every trigger.

---

## _build_startup_buttons

```python
_build_startup_buttons()
```

Create the **New Design** and **Load Design** buttons of the startup overlay. The overlay itself is laid out by the 
launcher.

---

### Parameters

None.

---

### Returns

- **buttons** : *dict*

  `new_design_button`, `load_design_button`.

---

## _dialog_title

```python
_dialog_title(title, subtitle, width)
```

Return the large centred title and grey subtitle shown at the top of an overlay dialog.

---

### Parameters

- **title** : *str*

  Title text.

- **subtitle** : *str*

  Subtitle text (HTML allowed).

- **width** : *int*

  Width of the `Div` in pixels.

---

### Returns

- **title_div** : *bokeh.models.Div*

---

## _build_save_version_overlay

```python
_build_save_version_overlay(triggers)
```

Create the "Is this the final version?" dialog, with its buttons wired.

The dialog is shown by **Save** and by **Save & Exit**. It holds the watcher CSV field and three buttons.

---

### Parameters

- **triggers** : *dict*

  Hidden triggers from `_build_hidden_triggers`.

---

### Returns

- **widgets** : *dict*

  `save_version_overlay`, `watcher_path_input`.

---

### Notes

| Button | JavaScript | Python effect |
|---|---|---|
| **Final Version** | Hide dialog, commit watcher field, set `browse_save_trigger` to `"final:<ms>"` | Save without `canvas_layout` |
| **Edit Later** | Same with `"editable:<ms>"` | Save with `canvas_layout` |
| **Cancel** | Hide dialog, set `"cancel:<ms>"` | Nothing saved, Save & Exit aborted |
| **Browse** | Flip `browse_watcher_trigger` | Watcher CSV dialog |

The mode is sent in the same value change as the "go" signal, so it cannot arrive out of order, and the timestamp makes 
every click a new value.

**Final Version** and **Edit Later** first copy `watcher_path_input.value_input` (updated on every keystroke) into 
`value` (normally only updated on Enter or blur). A path that is still being typed is therefore stored before the 
save starts.

---

## _build_unsaved_exit_overlay

```python
_build_unsaved_exit_overlay(triggers, save_version_overlay)
```

Create the "Unsaved Changes" dialog shown when **End Session** is clicked with unsaved changes, with its buttons wired.

---

### Parameters

- **triggers** : *dict*

  Hidden triggers from `_build_hidden_triggers`.

- **save_version_overlay** : *bokeh.models.Column*

  "Final version?" dialog, opened by **Save & Exit**.

---

### Returns

- **widgets** : *dict*

  `unsaved_exit_overlay`.

---

### Notes

- **Cancel** hides the dialog.
- **End Anyway** hides it, flips `end_session_trigger` (Python stops the server) and closes the tab.
- **Save & Exit** hides it, shows the "final version?" dialog and flips `end_session_save_trigger`, so Python ends the 
session once the file is written.

---

## _build_registry_lookups

```python
_build_registry_lookups()
```

Build the read-only per-component-type tables from the registry and the component sources.

---

### Parameters

None.

---

### Returns

- **lookups** : *dict*

  `default_source_count`, `default_target_count`, `possible_position`, `possible_options`, `icon_to_component_types`.

---

### See also

- [Metadata](powertrain_builder_metadata): functions that compute each table.

---

### Notes

The components tree scanned for positions and options is `COMPONENTS_PATH = fastga_he/models/propulsion/components`.

---

## _wire_toolbox_callbacks

```python
_wire_toolbox_callbacks(state)
```

Wire the browser-side callbacks that link widgets of different groups. These need the whole state, so they run after it 
is assembled.

---

### Parameters

- **state** : *BuilderState*

  Assembled builder state.

---

### Returns

None.

---

### Notes

- **Save** -> show the "final version?" dialog.
- **End Session** -> if the Save button is yellow (`'warning'`, unsaved changes), show the unsaved-exit dialog; 
otherwise flip `end_session_trigger` and close the tab.
- **Load Design** -> flip `browse_load_trigger`.
- `close_window_trigger` change -> close the tab.

---

## _build_toolbox_layout

```python
_build_toolbox_layout(state)
```

Lay out the left panel.

Top to bottom: *Components* title with the icon-size slider, component tabs, status line, **Delete**, **Save**, 
**End Session**, then the hidden triggers (they must be in the document to work).

---

### Parameters

- **state** : *BuilderState*

  Assembled builder state.

---

### Returns

- **toolbox_layout** : *bokeh.models.Column*