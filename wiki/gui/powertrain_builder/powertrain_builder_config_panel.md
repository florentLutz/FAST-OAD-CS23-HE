# Config panel

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_handler_config_panel.py`, mixin `ConfigPanelMixin` of [`BuilderHandler`](powertrain_builder_handler).

It fills, refreshes and clears the right-hand **Component Configurator** of the selected component (name, type, position, options, port counts, connections, symmetry) and commits it with **Apply**.

Edits in the panel are **not** written to the design until Apply is clicked, with three exceptions that apply at once, each as an undoable step:

- clearing a port in the Connections panel removes its connection;
- switching a known component to another type removes its connections and symmetry link;
- changing a port type of an unknown component (see [Unknown components](powertrain_builder_unknown)).

**Class attributes.** `_is_populating_node_table` is `True` while the panel is being filled, so that callbacks can tell a programmatic change from a user change. `_options_names` and `_options_value_selects` hold the options of the selected known component, in the same order, for Apply to read.

**Module constants.** `PRIMARY_PORT_LABEL_COLOR` (`#ADD8E6`, light blue), `PRIMARY_PORT_LABEL` (`"1"`), `PRIMARY_SOURCE_PORT_TYPE_KEYWORDS` (`splitter`, `gearbox`), `PRIMARY_TARGET_PORT_TYPE_KEYWORDS` (`planetary_gear`): see `_port_label_stylesheets`.

---

## _populate_node_table

```python
_populate_node_table(index)
```

Fill the panel with the stored values of a component. Called when a component is selected.

---

### Parameters

- **index** : *int*

  Index of the component in `placed_nodes_source`.

---

### Returns

None.

---

### Notes

1. Set the name field.
2. Show the selects (known component) or the free-text fields (unknown component), and fill them with `_populate_known_fields` or `_populate_unknown_fields`.
3. Move the gold selection ring to the component.
4. Rebuild the ports, which shows this component's port balls.
5. Build the Connections panel and, for a known component, the symmetry select.

---

## _populate_known_fields

```python
_populate_known_fields(index)
```

Fill the type, position, options and port-count widgets of a known component.

---

### Parameters

- **index** : *int*

  Index of the component.

---

### Returns

None.

---

### Notes

- **Type:** the choices are the types sharing the component's icon (`icon_to_component_types`). Setting the value is flagged as programmatic, so selecting a component never clears its connections.
- **Position:** the possible positions of the stored type; the stored position, else the first one.
- **Options:** shown only if the component has stored options; built by `_refresh_options_table` with the stored values.
- **Port counts:** a spinner is shown only on a side whose default count is `VARIABLE_PORT_DEFAULT`, with its stored value and a minimum of 1. The *Ports* section is shown if either spinner is.

---

## _clear_node_table

```python
_clear_node_table()
```

Reset every panel widget to empty and hide the panel. Also clears the selection ring, the Connections rows and the symmetry select, and rebuilds the ports, which hides the port balls of the deselected component.

---

### Parameters

None.

---

### Returns

None.

---

## _on_port_count_spinner_change

```python
_on_port_count_spinner_change(attr, old, new)
```

Callback of both port-count spinners: preview the new port counts on the canvas without changing the design.

---

### Parameters

- **attr** : *str*

  Bokeh attribute name (unused).

- **old**, **new** : *int*

  Previous and new value (unused: both spinners are read again, since either may have changed).

---

### Returns

None.

---

### See also

- [`_sync_edge_preview_backup`](powertrain_builder_ports_edges#_sync_edge_preview_backup): how hidden connections come back.

---

### Notes

The selected component's ports are redrawn with `override_counts`, using the value of each **visible** spinner (a hidden spinner may hold a stale value, so the stored count is used for that side). Connections on ports the new count removes disappear from the canvas, but are kept in a backup.

The counts are committed only by Apply. Leaving the component any other way (selecting another one, deleting, placing, delete mode) triggers a normal rebuild that restores the stored counts and every hidden connection.

Does nothing if no component is selected.

---

## _refresh_options_table

```python
_refresh_options_table(node_type, overrides)
```

Rebuild the Options rows of a known component for a component type.

Each option is a row with a disabled text field (the name) and a select (the value). The selects are kept in `_options_value_selects`, in the order of `_options_names`, for Apply.

---

### Parameters

- **node_type** : *str*

  Component type whose `possible_options` are shown.

- **overrides** : *dict*

  `{option_name: value}` of the values to show instead of the defaults.

---

### Returns

None.

---

### Notes

- The shown value is the override if there is one, else the first possible value.
- The choices are the possible values as strings. An option without possible values offers its current value only. A current value that is not a possible value (e.g. edited in the file) is added first, so it is not lost.

---

## _refresh_symmetry_select

```python
_refresh_symmetry_select(current_node_index, node_type)
```

Fill the symmetry select with the names of the other components of type *node_type*.

---

### Parameters

- **current_node_index** : *int*

  Component being edited, excluded from the choices.

- **node_type** : *str*

  Type to filter the candidates by: the stored type, or the type currently picked in the type select before Apply.

---

### Returns

None.

---

### Notes

The choices are `""` (no peer) followed by the candidates. The current peer is added if it is not a candidate (e.g. a component of another type linked from an unknown component), so selecting this component and applying keeps the link. The current peer is pre-selected.

---

## _refresh_connections_table

```python
_refresh_connections_table(node_index)
```

Rebuild the Connections panel of a component: one row per port, sources first, then targets.

---

### Parameters

- **node_index** : *int*

  Index of the component.

---

### Returns

None.

---

### Notes

The rows are built from copies of the two port sources, the node data and the edge data:

1. `_connected_far_ports` gives the far end of each committed connection of the component, and `_pending_far_ports` the far end of each pick; a pick replaces the connection of the same port.
2. `node_port_colors` gives the colour of each side, and `_free_port_candidates` lists the free ports of the opposite side with that colour.
3. `_build_port_row` builds one row per port of the component, sources first, and the rows replace the children of `connections_rows_column`.

The result:

Each row has the port label (`source:1`, `target:2`, ...) and a dropdown of the ports it can be connected to:

1. **Shown value:** the pick waiting for Apply if there is one, else the committed connection, else empty.
2. **Choices:** the free ports of *other* components, on the opposite side, compatible in energy type (`_free_port_candidates`), plus the shown port.
3. Choosing a port draws a dashed preview; choosing the empty value removes the connection at once (`_on_port_select_change`).

The primary port of a power split (port 1 of the source side of splitters and gearboxes, of the target side of planetary gears) is shown in light blue.

---

## _free_port_candidates

```python
_free_port_candidates(port_data, kind, node_index, color, node_data)
```

Return the unconnected ports of other components that are compatible with an energy type.

---

### Parameters

- **port_data** : *dict*

  `source_port_source.data` or `target_port_source.data`.

- **kind** : *str*

  `"source"` or `"target"`, the side *port_data* holds.

- **node_index** : *int*

  Selected component; its own ports are excluded.

- **color** : *str*

  Energy-type colour the ports must be compatible with.

- **node_data** : *dict*

  Snapshot of `placed_nodes_source.data`.

---

### Returns

- **candidates** : *list of tuple*

  `(node_index, label, kind, display_string)` of each candidate.

---

## _build_port_row

```python
_build_port_row(port_data, port_position, own_kind, node_type, free_far_ports, shown_far_ports, node_data)
```

Build one row of the Connections panel: the port label and its connection dropdown, with its callback.

The *far* port is the port at the other end of the connection, on the opposite side.

---

### Parameters

- **port_data** : *dict*

  Port source holding the port.

- **port_position** : *int*

  Position of the port in *port_data*.

- **own_kind** : *str*

  `"source"` or `"target"`, the side of the port.

- **node_type** : *str*

  Type of the selected component, used to highlight its primary port.

- **free_far_ports** : *list of tuple*

  Free candidates on the opposite side.

- **shown_far_ports** : *dict*

  `{own_label: (far_node_index, far_label)}` of the far port each dropdown on this side shows.

- **node_data** : *dict*

  Snapshot of `placed_nodes_source.data`.

---

### Returns

- **row** : *bokeh.layouts.row*

---

### Notes

1. The label is a disabled `TextInput` showing `<kind>:<label>`, styled by `_port_label_stylesheets` (light blue for a primary port).
2. If *shown_far_ports* has an entry for this port, it becomes the shown candidate, with its text from `_far_string`.
3. `_dropdown_entries` merges the shown candidate with the free ones and orders the choices.
4. The `Select` shows the shown candidate, or empty, and its callback comes from `_make_port_callback`, which binds this port's kind, label, component, colour, position and candidates.

---

## _far_string

```python
_far_string(node_data, far_node_index, far_label, far_kind)
```

Format a port as shown in a dropdown: `"<component name> (<kind>:<label>)"`. Static method.

---

### Parameters

- **node_data** : *dict*

  Snapshot of `placed_nodes_source.data`.

- **far_node_index** : *int*

  Index of the component owning the port (`node_<index>` is used if it is out of range).

- **far_label** : *str*

  Port label.

- **far_kind** : *str*

  `"source"` or `"target"`.

---

### Returns

- **text** : *str*

  E.g. `"battery_pack_1 (target:1)"`.

---

## _parse_choice

```python
_parse_choice(choice_string, candidates)
```

Find the port of a dropdown choice in its candidates. Static method.

---

### Parameters

- **choice_string** : *str*

  Text selected in the dropdown.

- **candidates** : *list of tuple*

  `(node_index, label, kind, display_string)` tuples.

---

### Returns

- **port** : *tuple or None*

  `(node_index, label, kind)` of the match, or `None`.

---

## _make_port_callback

```python
_make_port_callback(_own_kind, _own_label, _own_node_index, _own_color, _own_x, _own_y, _candidates)
```

Return the `on_change` callback of one Connections dropdown: `_on_port_select_change` with every argument of the row already bound (`functools.partial`), so Bokeh receives a plain `(attr, old, new)` callable.

---

### Parameters

- **_own_kind** : *str*

  `"source"` or `"target"`.

- **_own_label** : *str*

  Label of the row's port.

- **_own_node_index** : *int*

  Component owning the port.

- **_own_color** : *str*

  Colour of the port.

- **_own_x**, **_own_y** : *float*

  Port position when the row was built.

- **_candidates** : *list of tuple*

  Candidates of the dropdown.

---

### Returns

- **callback** : *functools.partial*

---

## _on_port_select_change

```python
_on_port_select_change(attr, old, new_value, *, _own_kind, _own_label, _own_node_index,
                       _own_color, _own_x, _own_y, _candidates)
```

Handle a change of a Connections dropdown.

---

### Parameters

- **attr**, **old** : *str*

  Bokeh arguments (unused).

- **new_value** : *str*

  Chosen display string, or `""`.

- **keyword arguments**

  The row's port and candidates, bound by `_make_port_callback`.

---

### Returns

None.

---

### Notes

The change is handled in three phases.

1. **Prune.** Drop any pick waiting for Apply that already uses this port, and redraw the remaining previews.
2. **Disconnect.** If the empty value was chosen, remove every committed connection of this port at once (`_disconnect_port`).
3. **Connect.** Otherwise, find the chosen port, read the current position of both ends, order them as source -> target, store the pick in `panel_pending_connections` and draw its dashed preview.

---

## _disconnect_port

```python
_disconnect_port(port_kind, node_index, port_label)
```

Remove every connection of one port, as a single undoable step, and rebuild the ports.

---

### Parameters

- **port_kind** : *str*

  `"source"` or `"target"`.

- **node_index** : *int*

  Component owning the port.

- **port_label** : *str*

  Port label.

---

### Returns

None.

---

## _on_type_select_change

```python
_on_type_select_change(attr, old, new)
```

Callback of the type select: update the panel to the type picked, before Apply.

Ignored while an unknown component is selected, whose type is free text.

---

### Parameters

- **attr**, **old** : *str*

  Bokeh arguments (unused).

- **new** : *str*

  Picked component type.

---

### Returns

None.

---

### Notes

1. Position choices -> those of the new type, first one selected.
2. Options -> those of the new type, pre-filled with the component's stored values of the same name.
3. If the user switched to a type other than the stored one (not while the panel is being filled), remove the component's connections and symmetry link at once (`_clear_node_connections_and_symmetry`): its ports and peer belong to the old type.
4. Symmetry choices -> components of the new type.

The type, position and options themselves are written only by Apply.

---

## _clear_node_connections_and_symmetry

```python
_clear_node_connections_and_symmetry(node_index)
```

Remove every connection of a component and break its symmetry link, as a single undoable step.

---

### Parameters

- **node_index** : *int*

  Index of the component.

---

### Returns

None.

---

### Notes

1. Discard the pending panel picks and end any port-count preview, so the removal works on the real connections.
2. Find the connections to remove and the symmetry entries to reset: the component's own link and every component pointing at it, by index or by name.
3. If there is nothing to remove, only refresh the Connections panel; no undo step is recorded.
4. Otherwise push an undo snapshot, apply both removals, rebuild the ports and mark the design unsaved.

---

## apply_node_configurations

```python
apply_node_configurations()
```

Commit the panel to the selected component. Callback of the **Apply** button.

Does nothing if no component is selected.

---

### Parameters

None.

---

### Returns

None.

---

### Notes

The whole operation is one undo step:

1. Push an undo snapshot.
2. Write the name, type, position, options and port counts (`_write_node_fields`).
3. Write the symmetry link (`_write_symmetry`).
4. Commit every pick of the Connections panel as a connection, clear the previews and rebuild the ports.
5. Mark the design unsaved.

---

### Examples

```python
handler.state.selected_node_index = 0
handler.state.name_input.value = "battery_left"
handler.apply_node_configurations()
```

---

## _write_node_fields

```python
_write_node_fields(placed_nodes_data, node_index)
```

Write the panel's name, type, position, options and port counts into a node row.

---

### Parameters

- **placed_nodes_data** : *dict*

  Copy of `placed_nodes_source.data`, edited in place.

- **node_index** : *int*

  Index of the component.

---

### Returns

None.

---

### Notes

- Type, position and options come from `_known_panel_fields` or, for an unknown component, `_unknown_panel_fields`. Options are stored as a JSON string.
- If the source spinner is hidden (fixed-port type), the port counts are reset to the defaults of the written type, so changing the type also changes the ports.
- A visible spinner writes its value as the port count of its side.

---

## _known_panel_fields

```python
_known_panel_fields()
```

Return the type, position and options picked in the selects of a known component, as Apply writes them. Option values are converted back from strings with `_strings_to_option_values`.

---

### Parameters

None.

---

### Returns

- **fields** : *dict*

  `{"node_type": str, "position": str, "options": dict}`.

---

## _write_symmetry

```python
_write_symmetry(placed_nodes_data, node_index)
```

Link the component with the peer chosen in the panel; any older link to either of them is removed.

---

### Parameters

- **placed_nodes_data** : *dict*

  Copy of `placed_nodes_source.data` whose name column already holds the applied name; edited in place.

- **node_index** : *int*

  Index of the component.

---

### Returns

None.

---

### Notes

The peer comes from the symmetry select, or from the free-text field for an unknown component.

1. Find the new peer's index by name (`_NO_SYMMETRY_PEER` if none).
2. Clear stale links: every component pointing at this component or at the new peer, except these two. This covers this component's old peer and the new peer's old partner. Links are matched by index and by name, since the name may have just changed in this same Apply.
3. Write the new entry (`_write_symmetry_entry`):
   - peer on the canvas -> both components point at each other;
   - unknown component naming a component not on the canvas -> the name is kept as is, without index (written to the YAML);
   - no peer -> this component is unlinked.

---

## _node_position

```python
_node_position(node_names, node_name)
```

Return the index of the component called *node_name*.

---

### Parameters

- **node_names** : *list of str*

  `name` column.

- **node_name** : *str*

  Name to look for; empty for none.

---

### Returns

- **position** : *int*

  Index, or `_NO_SYMMETRY_PEER` (−1) if no component has that name.

---

## _write_symmetry_entry

```python
_write_symmetry_entry(symmetry_columns, node_names, node_index, peer_index, kept_name)
```

Write the new symmetry entry of a component, in place.

---

### Parameters

- **symmetry_columns** : *tuple of list*

  `(symmetry_names, symmetry_indices)`, edited in place.

- **node_names** : *list of str*

  `name` column.

- **node_index** : *int*

  Component being applied.

- **peer_index** : *int*

  Its new peer, or `_NO_SYMMETRY_PEER`.

- **kept_name** : *str*

  Peer name to keep without index when there is no peer on the canvas; empty to unlink.

---

### Returns

None.

---

## _set_cell

```python
_set_cell(column_data, column_name, row_index, value)
```

Set one cell of column data, if the column exists and the row is in range.

---

### Parameters

- **column_data** : *dict*

  Column data, edited in place.

- **column_name** : *str*

  Column to write.

- **row_index** : *int*

  Row to write.

- **value** : *object*

  New value.

---

### Returns

- **written** : *bool*

---

## _is_same_port

```python
_is_same_port(port, port_kind, node_index, port_label)
```

Return whether a port dict is the port `(port_kind, node_index, port_label)`.

---

### Parameters

- **port** : *dict*

  Port dict.

- **port_kind**, **node_index**, **port_label**

  Port to compare with.

---

### Returns

- **same** : *bool*

---

## _connected_far_ports

```python
_connected_far_ports(edge_data, node_index)
```

Return the far end of every committed connection of a component, split by the component's port side. Both ends of each edge are checked, since an edge may start at either side.

---

### Parameters

- **edge_data** : *dict*

  `edge_source.data` or a copy.

- **node_index** : *int*

  Index of the component.

---

### Returns

- **connected_source** : *dict*

  `{own_label: (far_node_index, far_label)}` of its source ports.

- **connected_target** : *dict*

  Same for its target ports.

---

## _pending_far_ports

```python
_pending_far_ports(pending_connections, node_index)
```

Return the far end of every pending panel pick of a component, split by the component's port side.

---

### Parameters

- **pending_connections** : *list of tuple*

  `state.panel_pending_connections`, as `(source_port, target_port)` pairs.

- **node_index** : *int*

  Index of the component.

---

### Returns

- **pending_source** : *dict*
- **pending_target** : *dict*

  Same shape as `_connected_far_ports`.

---

## _dropdown_entries

```python
_dropdown_entries(free_far_ports, shown_candidate)
```

Return the candidates and the displayed choices of one Connections dropdown.

---

### Parameters

- **free_far_ports** : *list of tuple*

  Free candidates.

- **shown_candidate** : *tuple or None*

  Candidate tuple of the port the dropdown shows, or `None` when it shows no connection.

---

### Returns

- **candidates** : *list of tuple*

  Tuples the dropdown can resolve, with the shown port added once.

- **choices** : *list of str*

  Displayed strings: the shown port first, then `""`, then the free ports. Without a shown port: `""` then the free ports.

---

## _symmetry_positions_to_clear

```python
_symmetry_positions_to_clear(symmetry_names, symmetry_indices, node_index, node_name)
```

Return the components whose symmetry entry must be reset to unlink a component: every component pointing at it (by index or name), plus the component itself if it has a link.

---

### Parameters

- **symmetry_names**, **symmetry_indices** : *list*

  Symmetry columns, padded to one entry per component.

- **node_index** : *int*

  Component to unlink.

- **node_name** : *str or None*

  Its name.

---

### Returns

- **positions** : *list of int*

  Without duplicates.

---

## _padded_symmetry_columns

```python
_padded_symmetry_columns(placed_nodes_data)
```

Return copies of the two symmetry columns padded to one entry per component (`""` / −1).

---

### Parameters

- **placed_nodes_data** : *dict*

  `placed_nodes_source.data` or a copy.

---

### Returns

- **symmetry_names** : *list of str*
- **symmetry_indices** : *list of int*

---

## _positions_linked_to

```python
_positions_linked_to(symmetry_names, symmetry_indices, node_indices, node_names)
```

Return the components whose symmetry peer is one of the given components. A link counts if either its index or its name matches, so a link is found even when one of the two is out of date.

---

### Parameters

- **symmetry_names**, **symmetry_indices** : *list*

  Symmetry columns.

- **node_indices** : *set of int*

  Indices of the components whose incoming links are wanted.

- **node_names** : *set of str*

  Their names (empty names are ignored).

---

### Returns

- **positions** : *list of int*

  In order.

---

## _port_label_stylesheets

```python
_port_label_stylesheets(node_type, port_label, primary_type_keywords)
```

Return the stylesheets of a port label in the Connections panel: light blue for the primary port of a power split, none otherwise.

---

### Parameters

- **node_type** : *str*

  Type of the component owning the port.

- **port_label** : *str*

  Port label.

- **primary_type_keywords** : *tuple of str*

  Type keywords whose port 1 on this side is the primary port.

---

### Returns

- **stylesheets** : *list of str*

---

### Notes

The colour cannot be set with `styles`, which only styles the widget's outer element: the text is in an `<input>` inside the widget's shadow DOM, coloured by Bokeh's own `.bk-input:disabled` rule. A `:host input:disabled` rule is specific enough to override it, and `-webkit-text-fill-color` is also needed because Chromium and Safari paint disabled text with it.

---

## _option_values_to_strings

```python
_option_values_to_strings(value)
```

Convert an option value to the string shown in the panel.

---

### Parameters

- **value** : *bool, int, float or str*

---

### Returns

- **text** : *str*

  `"True"` / `"False"` for booleans, `str(value)` otherwise.

---

## _strings_to_option_values

```python
_strings_to_option_values(string)
```

Convert a string of the panel back to a typed option value: boolean, then integer, then float, else the string itself.

---

### Parameters

- **string** : *str*

---

### Returns

- **value** : *bool, int, float or str*

---

### Examples

```python
_strings_to_option_values("True")   # True
_strings_to_option_values("3")      # 3
_strings_to_option_values("2.5")    # 2.5
_strings_to_option_values("li_ion") # 'li_ion'
```