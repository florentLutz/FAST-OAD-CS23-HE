# State

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_state.py`. It defines `BuilderState`, the single shared object of the builder, together with the layout constants, the column layouts of the data sources and the small helpers used to edit them.

---

## Constants

#### Layout

| Name | Value | Meaning |
|---|---|---|
| `PALETTE_WIDTH` | 300 | Width of the toolbox (px) |
| `ROW_HEIGHT` | 52 | Height of a toolbox row (px) |
| `ICON_SIZE` | 45 | Reference icon size for rings and labels |
| `PORT_RADIUS` | 8 | Radius of a port ball |
| `NODE_RADIUS` | 22.5 | `ICON_SIZE / 2`, base of the port orbit |
| `CANVAS_WORLD_X_RANGE`, `CANVAS_WORLD_Y_RANGE` | (0, 800), (0, 950) | Design area always kept visible (data units) |
| `CONFIG_PANEL_RESERVED_WIDTH` | 425 | Space kept right of the canvas (px) |
| `CANVAS_LEFT_OFFSET` | 335 | Approximate left edge of the canvas (px) |
| `CANVAS_FIT_NODE_MARGIN` | 60 | Margin around the outermost components (data units) |
| `CANVAS_MIN_WIDTH`, `CANVAS_MIN_HEIGHT` | 480, 420 | Smallest canvas size (px) |
| `DEFAULT_NODE_ICON_SIZE` | 50 | Default icon size (data units) |
| `NODE_ICON_SIZE_MIN`, `NODE_ICON_SIZE_MAX` | 20, 80 | Range of the icon-size slider |
| `HOVER_RING_BASE_SIZE`, `SELECTION_RING_BASE_SIZE` | 55, 59 | Ring sizes at the default icon size (px) |
| `NODE_LABEL_BASE_OFFSET` | −25 | Name label offset at the default icon size (px) |
| `DIALOG_POSITION_STYLES`, `DIALOG_STYLES` | — | CSS shared by the overlays: centred over the canvas, z-index 100 |

#### Appearance and sentinels

| Name | Value | Meaning |
|---|---|---|
| `BUTTON_DEFAULT_COLOR_TYPE` | `"light"` | Unselected toolbox button |
| `BUTTON_SELECTED_COLOR_TYPE` | `"primary"` | Armed toolbox button |
| `NODE_SELECT_COLOR` | `"#FFD700"` | Gold selection ring |
| `STATUS_IDLE_HTML` | *Select a component* | Status line when idle |
| `_EMPTY` | `""` | Empty string sentinel |
| `_NO_SYMMETRY_PEER` | −1 | `symmetry_node_index` of a node without peer |
| `VARIABLE_PORT_DEFAULT` | 3 | Default port count of a variable-port component; also marks such components |
| `SAVE_MODE_FINAL`, `SAVE_MODE_EDITABLE`, `SAVE_MODE_CANCEL` | `"final"`, `"editable"`, `"cancel"` | Modes sent by the "final version?" dialog |

#### Unknown component

| Name | Value | Meaning |
|---|---|---|
| `UNKNOWN_ICON_KEY` | `"unknown"` | Icon key that identifies an unknown component |
| `UNKNOWN_COMPONENT_TYPE` | `"unknown"` | Default type of a new unknown component |
| `UNKNOWN_COMPONENT_TABS` | connector, propulsor, load, tank, source | Toolbox tabs listing it |
| `UNKNOWN_DEFAULT_PORT_COUNT` | 1 | Default port count on each side |
| `UNDEFINED_PORT_TYPE`, `UNDEFINED_PORT_COLOR` | `"undefined"`, `"#666666"` | Port type not chosen yet; cannot be connected |
| `PORT_TYPE_COLORS` | — | `{port_type: colour}` for undefined, electricity, mechanical, fuel |
| `PORT_TYPE_CHOICES` | — | `(value, label)` options of the port-type selects |
| `BUILDER_ICONS_CONFIG` | — | `ICONS_CONFIG` of the network viewer plus the unknown icon (no port colours) |

#### Column layouts

| Name | Columns |
|---|---|
| `NODE_COLUMNS` | `x`, `y`, `url`, `w`, `h`, `name`, `node_type`, `icon_type`, `position`, `options` (JSON string), `n_sources`, `n_targets`, `symmetry_name`, `symmetry_node_index`, `source_port_type`, `target_port_type` |
| `EDGE_COLUMNS` | `xs`, `ys`, `color`, `starting_node_index`, `starting_port_label`, `starting_port_kind`, `ending_node_index`, `ending_port_label`, `ending_port_kind` |
| `PORT_COLUMNS` | `x`, `y`, `color`, `label`, `node_index`, `node_name`, `node_type`, `fill_alpha`, `line_alpha`, `connected` (`"True"` / `"False"`) |

---

## BuilderState

```python
BuilderState(placed_nodes_source=None, edge_source=None, ...)
```

Dataclass holding every Bokeh widget, data source and interaction value of the builder.

It is created once by [`ComponentToolboxConfigurationTableBuilder.build`](powertrain_builder_toolbox#componenttoolboxconfigurationtablebuilderbuild) and passed by reference to the launcher and the handler, so that all of them work on the same objects.

---

### Fields

#### 1. Design data (saved and snapshotted for undo)

- **placed_nodes_source** : *ColumnDataSource* — one row per component (`NODE_COLUMNS`). Drives the icons, the hover ring and tooltip, and the name labels.
- **edge_source** : *ColumnDataSource* — one row per committed connection (`EDGE_COLUMNS`).
- **placed_counter** : *dict* — `{icon_key: count}` used to name new components (`battery_pack_1`, `battery_pack_2`, ...).

#### 2. Derived and temporary canvas data

- **source_port_source**, **target_port_source** : *ColumnDataSource* — port balls, rebuilt from the nodes by `_rebuild_all_ports`.
- **selected_node_overlay_source** : *ColumnDataSource* — position of the selection ring.
- **awaiting_second_port_click_source** : *ColumnDataSource* — highlight of the first port of a two-click connection.
- **panel_preview_edge_source** : *ColumnDataSource* — dashed previews of connections picked in the panel.

#### 3. Interaction and session values

- **selected_component** : *str or None* — icon key armed in the toolbox.
- **delete_mode** : *bool* — whether delete mode is on.
- **selected_node_index** : *int or None* — node open in the configuration panel.
- **awaiting_second_port_click** : *dict or None* — first port of a two-click connection.
- **panel_pending_connections** : *list* — `(source_port, target_port)` picks waiting for Apply.
- **edge_preview_backup** : *dict or None* — true edges while a port-count preview hides some (see [`_sync_edge_preview_backup`](powertrain_builder_ports_edges#_sync_edge_preview_backup)).
- **view_scale** : *float* — data units per screen pixel of the fitted view (default 1.0).
- **canvas_area** : *dict or None* — area an empty design shows in this window, `{"x0", "x1", "y0", "y1", "per_px"}`.
- **watcher_file_path** : *str* — watcher CSV path written to the YAML (empty: not written).
- **edit_history_stack** : *EditHistoryStack* — set by `BuilderHandler`.

#### 4. Widgets

- Toolbox and canvas: `buttons`, `node_size_slider`, `delete_button`, `save_button`, `end_session_button`, `status_div`, `undo_button`, `redo_button`, `hover_ring_renderer`, `selection_ring_renderer`, `node_label_set`.
- Configuration panel: `config_panel_column`, `name_input`, `type_select`, `position_select`, `symmetry_select`, `apply_button`, `source_count_spinner`, `target_count_spinner`, `port_count_section`, `options_table`, `options_rows_column`, `connections_rows_column`.
- Unknown component: `unknown_type_input`, `unknown_position_input`, `unknown_symmetry_input`, `source_port_type_select`, `target_port_type_select`, `add_option_button`.
- Overlays and triggers: `startup_overlay`, `new_design_button`, `load_design_button`, `save_version_overlay`, `watcher_path_input`, `browse_load_trigger`, `browse_save_trigger`, `browse_watcher_trigger`, `unsaved_exit_overlay`, `end_session_save_trigger`, `end_session_trigger`, `close_window_trigger`, `view_scale_input`, `fit_view_trigger`.

#### 5. Registry lookups (read-only, built at startup)

- **default_source_count**, **default_target_count** : *dict* — `components_type -> int`.
- **possible_position** : *dict* — `components_type -> [position, ...]`, first = default.
- **possible_options** : *dict* — `components_type -> {option: [values, ...]}`, first value = default.
- **icon_to_component_types** : *dict* — `icon_key -> [components_type, ...]`, first = default type of a new component. Note the reversed key compared with the four tables above.

---

### Notes

Two conventions apply to all fields:

- A field ending in `_source` is a `ColumnDataSource` and must be edited with the copy -> mutate -> reassign pattern (see `copy_column_data`).
- Tables keyed by component type use the `components_type` string of `KNOWN_COMPONENTS` (e.g. `"battery_pack"`), not the toolbox icon key.

---

## copy_column_data

```python
copy_column_data(data)
```

Return an independent copy of a `ColumnDataSource.data` dictionary, with a new list for each column.

---

### Parameters

- **data** : *dict*

  Live `.data` of a `ColumnDataSource`.

---

### Returns

- **copy** : *dict*

  Same keys, each mapped to a new list holding the same values (shallow copy per column).

---

### Notes

Bokeh only sends a change to the browser when a whole column or the whole `.data` is reassigned. Editing a column list in place (`append`, `pop`, index assignment) changes the Python data but leaves the browser showing the old data. Every edit in the builder therefore copies the data with this function, edits the copy, and reassigns `source.data` once.

---

### Examples

```python
new_data = copy_column_data(state.edge_source.data)
new_data["color"][0] = "#FF0000"
state.edge_source.data = new_data
```

---

## is_unknown_node

```python
is_unknown_node(nodes_data, node_index)
```

Return whether a node is an unknown component, i.e. whether its `icon_type` is `"unknown"`.

The test uses the icon and not the type, so typing a known type for an unknown component never makes it pick up that type's options or ports.

---

### Parameters

- **nodes_data** : *dict*

  `placed_nodes_source.data` or a copy.

- **node_index** : *int*

  Zero-based index of the node.

---

### Returns

- **is_unknown** : *bool*

---

## node_port_colors

```python
node_port_colors(nodes_data, node_index)
```

Return the energy-type colour of the source ports and of the target ports of a node.

A known component takes the colours of its icon in `BUILDER_ICONS_CONFIG`. An unknown component takes the colours of the port types chosen in its panel (gray `#666666` when undefined).

---

### Parameters

- **nodes_data** : *dict*

  `placed_nodes_source.data` or a copy.

- **node_index** : *int*

  Zero-based index of the node.

---

### Returns

- **colors** : *tuple of (str or None, str or None)*

  `(source_color, target_color)`. `None` for a side of a known component that has no ports.

---

## ports_compatible

```python
ports_compatible(first_color, second_color)
```

Return whether two ports of these colours may be connected: same colour, and neither is the undefined gray.

---

### Parameters

- **first_color**, **second_color** : *str*

  Colours of the two ports.

---

### Returns

- **compatible** : *bool*

---

## empty_columns

```python
empty_columns(column_names)
```

Return column data with every column of *column_names* empty.

---

### Parameters

- **column_names** : *tuple of str*

  `NODE_COLUMNS`, `EDGE_COLUMNS` or `PORT_COLUMNS`.

---

### Returns

- **data** : *dict*

  `{column_name: []}`.

---

## append_row

```python
append_row(column_data, column_names, row)
```

Return a copy of *column_data* with one more row at the end. The input is not changed.

---

### Parameters

- **column_data** : *dict*

  Column data or a copy.

- **column_names** : *tuple of str*

  Columns of the result.

- **row** : *dict*

  One value per column of *column_names*.

---

### Returns

- **data** : *dict*

  New column data.

---

## drop_row

```python
drop_row(column_data, row_index)
```

Return a copy of *column_data* without the row at *row_index*. The input is not changed.

---

### Parameters

- **column_data** : *dict*

  Column data or a copy.

- **row_index** : *int*

  Zero-based row to leave out.

---

### Returns

- **data** : *dict*

  New column data.

---

## default_options

```python
default_options(options_definition)
```

Return the default value of every option of a component type: the first possible value. Options without any possible value are left out.

---

### Parameters

- **options_definition** : *dict*

  `possible_options[component_type]`, i.e. `{option_name: [values, ...]}`.

---

### Returns

- **options** : *dict*

  `{option_name: first value}`.

---

### Examples

```python
default_options({"cell_chemistry": ["li_ion", "li_s"], "empty": []})
# {'cell_chemistry': 'li_ion'}
```

---

## default_position

```python
default_position(position_choices)
```

Return the default position of a component type: the first possible position, or `""`.

---

### Parameters

- **position_choices** : *list of str*

  `possible_position[component_type]`.

---

### Returns

- **position** : *str*

---

## select_edges

```python
select_edges(edge_data, kept_edge_indices)
```

Return a copy of *edge_data* holding only the edges at the given positions, in the given order.

---

### Parameters

- **edge_data** : *dict*

  `edge_source.data` or a copy.

- **kept_edge_indices** : *list of int*

  Positions of the edges to keep.

---

### Returns

- **data** : *dict*

  New edge data with the same columns.

---

## edge_end_ports

```python
edge_end_ports(edge_data)
```

Return the two ports of every edge as `(starting_port, ending_port)` pairs of port keys `(kind, node_index, label)`.

An edge starts at whichever port was clicked first, so the starting port may be a source or a target.

---

### Parameters

- **edge_data** : *dict*

  `edge_source.data` or a copy.

---

### Returns

- **pairs** : *list of tuple*

  One pair per edge, in edge order.

---

## edge_indices_not_touching_node

```python
edge_indices_not_touching_node(edge_data, node_index)
```

Return the positions of the edges not attached to a node.

---

### Parameters

- **edge_data** : *dict*

  `edge_source.data` or a copy.

- **node_index** : *int*

  Zero-based index of the node.

---

### Returns

- **positions** : *list of int*

---

## edge_indices_not_touching_port

```python
edge_indices_not_touching_port(edge_data, node_index, port_label, port_kind)
```

Return the positions of the edges not attached to one port.

---

### Parameters

- **edge_data** : *dict*

  `edge_source.data` or a copy.

- **node_index** : *int*

  Zero-based index of the node owning the port.

- **port_label** : *str*

  Port label (`"1"`, `"2"`, ...).

- **port_kind** : *str*

  `"source"` or `"target"`.

---

### Returns

- **positions** : *list of int*

---

## link_symmetry

```python
link_symmetry(symmetry_names, symmetry_indices, node_names, first, second)
```

Make two nodes each other's symmetry peer, in place: each one stores the name and index of the other.

---

### Parameters

- **symmetry_names** : *list of str*

  `symmetry_name` column, edited in place.

- **symmetry_indices** : *list of int*

  `symmetry_node_index` column, edited in place.

- **node_names** : *list of str*

  `name` column.

- **first**, **second** : *int*

  Zero-based indices of the two nodes.

---

### Returns

None.

---

## unlink_symmetry

```python
unlink_symmetry(symmetry_names, symmetry_indices, node_positions)
```

Reset the symmetry entries of the given nodes to `""` / `-1`, in place.

Only the listed nodes are reset; their peers keep pointing at them unless they are listed too.

---

### Parameters

- **symmetry_names** : *list of str*

  `symmetry_name` column, edited in place.

- **symmetry_indices** : *list of int*

  `symmetry_node_index` column, edited in place.

- **node_positions** : *iterable of int*

  Positions of the nodes to unlink.

---

### Returns

None.