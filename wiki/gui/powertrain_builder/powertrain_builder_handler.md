# Handler

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_handler.py`. It defines `BuilderHandler`, the object that wires every Python callback of 
the builder and dispatches canvas taps. Its behaviour comes from six mixins, one per responsibility, all working on the 
shared [`BuilderState`](powertrain_builder_state#builderstate).

| Mixin | Responsibility | Page |
|---|---|---|
| `CanvasModeMixin` | Startup overlay, New Design, delete mode, toolbox selection | [Canvas mode](powertrain_builder_canvas_mode) |
| `PlacementMixin` | New components, icon size, view scale | [Placement](powertrain_builder_placement) |
| `IOMixin` | Save, load, undo / redo, end session | [I/O](powertrain_builder_io) |
| `PortEdgeMixin` | Hit-testing, ports, connections | [Ports and edges](powertrain_builder_ports_edges) |
| `ConfigPanelMixin` | Configuration panel and Apply | [Config panel](powertrain_builder_config_panel) |
| `UnknownComponentMixin` | Panel of unknown components | [Unknown components](powertrain_builder_unknown) |

---

## BuilderHandler

```python
BuilderHandler(state, main_plot, icon_size=DEFAULT_NODE_ICON_SIZE)
```

Create the handler, its undo / redo history, and wire every button and widget callback.

It must be created after the toolbox and the undo / redo buttons exist on the state.

---

### Parameters

- **state** : *BuilderState*

  Shared builder state.

- **main_plot** : *bokeh.plotting.figure*

  Canvas on which components are placed.

- **icon_size** : *int, optional*

  Initial icon size in canvas data units (default: `50`). Changed afterwards by the icon-size slider.

---

### Attributes

- **state** : *BuilderState* — shared state, as given.
- **main_plot** : *bokeh.plotting.figure* — the canvas, as given.
- **icon_size** : *int* — current icon size, updated by the icon-size slider.

---

### Notes

The constructor:

1. Stores its arguments.
2. Creates an empty `EditHistoryStack` on `state.edit_history_stack`.
3. Wires every callback (`_wire_buttons_widgets`).
4. Sets the initial look of the Undo / Redo buttons (both dimmed).

The canvas `Tap` event is not wired here; the launcher connects it to `on_canvas_tap`.

---

### Examples

```python
from bokeh.events import Tap
from bokeh.models import Button

toolbox_layout, config_panel_column, state = ComponentToolboxConfigurationTableBuilder.build()
state.undo_button, state.redo_button = Button(label="Undo"), Button(label="Redo")
canvas = figure()   # the launcher builds the real canvas
handler = BuilderHandler(state, canvas)
canvas.on_event(Tap, handler.on_canvas_tap)
```

---

## _wire_buttons_widgets

```python
_wire_buttons_widgets()
```

Attach the Python callbacks of every static widget. Called once by the constructor.

---

### Parameters

None.

---

### Returns

None.

---

### Notes

| Widget | Event | Callback |
|---|---|---|
| `end_session_trigger` | value | `_on_end_session_trigger` |
| `new_design_button` | click | `_on_new_design` |
| each toolbox button *i* | click | `on_toolbox_select(i)` |
| `delete_button` | click | `_toggle_delete_mode` |
| `browse_load_trigger` | value | `_on_browse_load` |
| `browse_save_trigger` | value | `_on_browse_save` |
| `browse_watcher_trigger` | value | `_on_browse_watcher` |
| `watcher_path_input` | value | `_on_watcher_path_change` |
| `end_session_save_trigger` | value | `_on_end_session_save_trigger` |
| `node_size_slider` | value | `_on_node_size_slider_change` (also set to `icon_size`) |
| `view_scale_input` | value | `_on_view_scale_change` |
| `undo_button` / `redo_button` | click | `_on_undo` / `_on_redo` |
| `type_select` | value | `_on_type_select_change` |
| `apply_button` | click | `apply_node_configurations` |
| `source_count_spinner`, `target_count_spinner` | value | `_on_port_count_spinner_change` |
| `add_option_button` | click | `_on_add_option` |
| `source_port_type_select` / `target_port_type_select` | value | `_on_port_type_change(side="source" / "target")` |

The **End Session** button has no Python `on_click`. Its JavaScript callback decides whether it is safe to stop the 
server and only then flips `end_session_trigger`; a Python `on_click` would run in every case.

---

## _make_select_callback

```python
_make_select_callback(button_index)
```

Return a callback without arguments that selects the toolbox component at *button_index*.

---

### Parameters

- **button_index** : *int*

  Zero-based index into `list(BUILDER_ICONS_CONFIG)`.

---

### Returns

- **callback** : *callable*

  `lambda: self.on_toolbox_select(button_index)`.

---

## on_canvas_tap

```python
on_canvas_tap(event)
```

Handle a click on the canvas. What it does depends on the current mode.

---

### Parameters

- **event** : *bokeh.events.Tap*

  Tap event carrying the `x` and `y` data coordinates.

---

### Returns

None.

---

### Notes

| Mode | Action |
|---|---|
| Delete mode on | `_delete_at(x, y)`: delete the component there, else the connection there |
| A toolbox component is armed | `_place_new_node(x, y)` |
| Idle, a port is within reach | `_handle_port_tap(port)`: two-click connection |
| Idle, otherwise | `_select_at(x, y)`: select or deselect a component |

In idle mode, a port has priority over a component, because port balls overlap the icon edge.

Every branch that changes the design pushes an undo snapshot before the change.

---

## _delete_at

```python
_delete_at(x, y)
```

Delete mode: remove the component at `(x, y)` or, if there is none, the connection there.

---

### Parameters

- **x**, **y** : *float*

  Tap coordinates in canvas data units.

---

### Returns

None.

---

### Notes

If a component is still selected afterwards, its Connections panel is refreshed.

---

## _delete_node

```python
_delete_node(node_index, nodes_data)
```

Remove a component and its connections, as a single undoable step.

---

### Parameters

- **node_index** : *int*

  Zero-based index of the component.

- **nodes_data** : *dict*

  Current `placed_nodes_source.data`.

---

### Returns

None.

---

### Notes

Nodes are identified by their row, so every node after the deleted one moves up by one index. The deletion is done in 
five steps.

1. Push an undo snapshot.
2. Drop the node row and re-index the symmetry links (`_reindex_symmetry_after_deletion`).
3. Drop every edge attached to the node, and decrement the node indices of the remaining edges that pointed after it.
4. Rebuild the ports.
5. Update the selection: cleared (with the panel) if the deleted node was selected, decremented if the selected node 
was after it.

For example, deleting `B` from `[A, B, C, D]` gives `[A, C, D]`: `C` goes from index 2 to 1, `D` from 3 to 2.

---

## _delete_edge

```python
_delete_edge(edge_index)
```

Remove one connection, as a single undoable step.

---

### Parameters

- **edge_index** : *int*

  Zero-based position of the edge in `edge_source`.

---

### Returns

None.

---

## _select_at

```python
_select_at(x, y)
```

Idle mode: select the component at `(x, y)`, or deselect.

---

### Parameters

- **x**, **y** : *float*

  Tap coordinates in canvas data units.

---

### Returns

None.

---

### Notes

1. `_find_nearest_node` finds the component under the tap, if any.
2. `_cancel_pending_connection` and `_clear_temp_edges` drop the first click of a two-click connection and the panel 
picks waiting for Apply.
3. Then:
   - no component, or the one already selected: `selected_node_index` is cleared and `_clear_node_table` hides 
   the panel;
   - another component: `selected_node_index` is set, `_populate_node_table` fills the panel, and the panel is shown.

---

## _reindex_symmetry_after_deletion

```python
_reindex_symmetry_after_deletion(nodes_data, deleted_node_index)
```

Update the symmetry links, in place, after a node was removed.

The peer of the deleted node loses its link, and every peer index after the deleted node is decremented, like the 
node rows.

---

### Parameters

- **nodes_data** : *dict*

  Node columns with the deleted row already removed; edited in place.

- **deleted_node_index** : *int*

  Index the deleted node had before its removal.

---

### Returns

None.
