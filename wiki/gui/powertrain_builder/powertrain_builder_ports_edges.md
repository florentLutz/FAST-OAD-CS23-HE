# Ports and edges

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_handler_ports.py`, mixin `PortEdgeMixin` of [`BuilderHandler`](powertrain_builder_handler).

It handles the geometry of the canvas: finding the component, port or connection under a click; drawing the port balls around each component; adding connections; and keeping the connections and their previews attached to the ports.

**Vocabulary.** A *port* is identified by its key `(kind, node_index, label)`, where `kind` is `"source"` or `"target"` and `label` is a string `"1"`, `"2"`, .... A *port dict* adds its `x`, `y` and `color`. An *edge* is a committed connection in `edge_source`; a *preview* is a dashed line for a panel pick waiting for Apply.

---

## _find_nearest_node

```python
_find_nearest_node(x, y)
```

Find the component closest to `(x, y)`, within one icon size of its centre.

---

### Parameters

- **x**, **y** : *float*

  Tap coordinates in canvas data units.

---

### Returns

- **best_index** : *int or None*

  Index of the closest component, or `None` if none is within `icon_size`.

- **shortest_distance** : *float or None*

  Distance from the tap to that component, or `None`.

- **nodes_data** : *dict*

  The live `placed_nodes_source.data`.

---

### Notes

A linear scan of the `x` / `y` columns of `placed_nodes_source`: the distance from the tap to each component centre is computed, and the closest one under `icon_size` wins. The snap radius follows the icon-size slider, so larger icons are easier to hit.

The live data is returned with the result, so `_delete_at` can pass it to `_delete_node` without reading it again.

---

## _find_nearest_port

```python
_find_nearest_port(x, y)
```

Find the port ball closest to `(x, y)`, within `2.5 × PORT_RADIUS` (20 data units).

Hidden port balls (those of non-selected components) count too: they still exist in the port sources.

---

### Parameters

- **x**, **y** : *float*

  Tap coordinates in canvas data units.

---

### Returns

- **port** : *dict or None*

  Port dict with keys `kind`, `node_index`, `label`, `x`, `y`, `color`, or `None`.

---

### Notes

The two port sources are scanned (`_port_data_by_kind`), source ports first, and the closest ball under the snap distance is returned as a port dict (`_port_dict`). On a tie, the port met first wins, so a source port is preferred.

The snap distance is fixed in data units: unlike `_find_nearest_node`, it does not follow the icon-size slider.

---

## _live_port

```python
_live_port(port_kind, node_index, port_label)
```

Return the current port dict of one port, read from the port sources.

---

### Parameters

- **port_kind** : *str*

  `"source"` or `"target"`.

- **node_index** : *int*

  Index of the component owning the port.

- **port_label** : *str*

  Port label.

---

### Returns

- **port** : *dict or None*

  Port dict, or `None` if the port does not exist.

---

### Notes

A linear scan of the port source of *port_kind* for the row whose `node_index` and `label` match, returned with `_port_dict`. The Connections panel uses it because it keeps port keys, but needs the position and colour the port has now.

---

## _port_data_by_kind

```python
_port_data_by_kind()
```

Return the two port sources with their kind.

---

### Parameters

None.

---

### Returns

- **ports** : *list of tuple*

  `[("source", source_port_source.data), ("target", target_port_source.data)]`.

---

## _cancel_pending_connection

```python
_cancel_pending_connection()
```

Cancel the port waiting for a second click: forget it and remove its highlight.

---

### Parameters

None.

---

### Returns

None.

---

## _handle_port_tap

```python
_handle_port_tap(port)
```

Run the two-click connection workflow on the canvas.

---

### Parameters

- **port** : *dict*

  Port dict of the tapped port.

---

### Returns

None.

---

### Notes

Any dashed panel preview is cleared first.

| Situation | Effect |
|---|---|
| No port waiting | The tapped port waits for a second click and is highlighted |
| Same port tapped again | Cancelled |
| Port of the other kind, compatible colour, accepted by `_edge_rejection` | Undo snapshot, edge added, ports rebuilt, waiting port cleared |
| Anything else | Waiting port cleared, nothing added |

The rejection check runs before the snapshot, so a refused connection leaves no empty undo step.

The edge is stored from the port clicked first to the second one, so on the canvas an edge can start at a target port. The panel always stores source -> target; every reader of `edge_source` handles both directions.

---

## _cursor_to_segment_distance

```python
_cursor_to_segment_distance(px, py, x1, y1, x2, y2)
```

Return the shortest distance from a point to a segment. Static method.

---

### Parameters

- **px**, **py** : *float*

  Point (cursor) coordinates.

- **x1**, **y1**, **x2**, **y2** : *float*

  Segment end points.

---

### Returns

- **distance** : *float*

---

### Notes

The point is projected on the segment line, and the projection parameter is clamped to `[0, 1]` so the nearest point stays on the segment. A segment of zero length gives the distance to its end point.

---

## _find_nearest_edge

```python
_find_nearest_edge(x, y, snap=12.0)
```

Find the connection closest to `(x, y)`, within *snap* data units.

---

### Parameters

- **x**, **y** : *float*

  Tap coordinates in canvas data units.

- **snap** : *float, optional*

  Largest distance for a hit (default: `12.0`).

---

### Returns

- **edge_index** : *int or None*

  Position of the edge in `edge_source`, or `None`.

---

### Notes

Each edge is taken as the segment between the first two points of its `xs` / `ys` (a committed edge always has exactly two); an edge with fewer points is skipped. `_cursor_to_segment_distance` gives the distance from the tap to each segment, and the closest one under *snap* wins.

*snap* is in data units, so on screen the hit zone shrinks when the view is zoomed out (`k > 1`).

---

## _add_edge

```python
_add_edge(starting_port, ending_port)
```

Append a connection between two ports to `edge_source`, unless `_edge_rejection` refuses it (the refusal is logged).

---

### Parameters

- **starting_port**, **ending_port** : *dict*

  Port dicts of the two ends.

---

### Returns

None.

---

### Notes

1. Copy `edge_source.data` and check the connection with `_edge_rejection`. A refused connection is logged and nothing else happens.
2. Build the row: `xs` / `ys` from the two port positions, the colour of the starting port, and the node index, label and kind of each end.
3. Append it with `append_row`, reassign `edge_source.data`, and mark the design unsaved.

This method only writes `edge_source` and marks the design unsaved. The caller must:

- call `_push_undo()` **before**, so the connection is undoable;
- call `_rebuild_all_ports()` **after**, which updates the `connected` flags and the Connections panel. Apply adds several edges in a row and rebuilds once at the end.

The edge takes the colour of its starting port.

---

## _add_edge_temp

```python
_add_edge_temp(starting_port, ending_port)
```

Draw a dashed preview line between two ports in `panel_preview_edge_source`. Nothing is added to `edge_source`; the connection is committed only by Apply.

---

### Parameters

- **starting_port**, **ending_port** : *dict*

  Port dicts of the two ends.

---

### Returns

None.

---

## _clear_temp_edges

```python
_clear_temp_edges()
```

Remove every dashed preview and forget every panel pick waiting for Apply.

---

### Parameters

None.

---

### Returns

None.

---

## _clear_temp_edge_visuals

```python
_clear_temp_edge_visuals()
```

Remove the dashed preview lines only; the panel picks are kept.

---

### Parameters

None.

---

### Returns

None.

---

## _rebuild_edges

```python
_rebuild_edges()
```

Recompute the end points of every edge from the current port positions, and drop the edges whose ports no longer exist (e.g. a port count was reduced).

Colours are kept: both ends of an edge always carry its energy type.

---

### Parameters

None.

---

### Returns

None.

---

### Notes

1. Return at once if there is no edge.
2. `_port_positions` maps every port key drawn now to its `(x, y)`.
3. For each edge, `edge_end_ports` gives its two port keys. If both are drawn, the new end points are recorded and the edge is kept.
4. `select_edges` keeps those edges, in order, and the new `xs` / `ys` replace the old ones.

An edge is therefore dropped exactly when one of its ports is no longer drawn.

---

## _rebuild_all_ports

```python
_rebuild_all_ports(override_counts=None)
```

Redraw every port ball from the node data, then bring everything attached to the ports back in line with them.

Called after nearly every change: node placed, deleted or edited, Apply, undo, redo, load, icon resize, and on every tick of a port-count spinner.

---

### Parameters

- **override_counts** : *dict, optional*

  `{node_index: (n_sources, n_targets)}` used **only** for this drawing, never written to `placed_nodes_source`. This is how the port-count spinners show a live, reversible preview. Default: `None`.

---

### Returns

None.

---

### Notes

The rebuild is done in three steps.

1. `_sync_edge_preview_backup` keeps the true edge list safe while a preview hides edges.
2. `_draw_ports` recomputes the port sources.
3. `_sync_edges_to_ports` re-routes or drops the edges, redraws the dashed previews and rebuilds the Connections panel.

---

## _draw_ports

```python
_draw_ports(override_counts)
```

Recompute every port ball and write the two port sources.

---

### Parameters

- **override_counts** : *dict*

  `{node_index: (n_sources, n_targets)}` replacing the stored counts of those nodes for this drawing only.

---

### Returns

None.

---

### Notes

For each node:

1. Port counts come from `override_counts`, else from `n_sources` / `n_targets`, else from the defaults of its type.
2. Port positions come from `compute_ports`, with every length scaled by `size_factor = icon_size / DEFAULT_NODE_ICON_SIZE`.
3. Colours come from `node_port_colors`.
4. Only the selected node's balls are visible (fill alpha 0.3, line alpha 0.5); the others are transparent but still clickable.

Every port is then flagged `connected` (`"True"` / `"False"`) from `edge_source`.

---

## _sync_edges_to_ports

```python
_sync_edges_to_ports()
```

Bring the edges, the dashed previews and the Connections panel in line with the port balls just drawn: `_rebuild_edges`, `_redraw_panel_previews`, then `_refresh_connections_table` of the selected node.

During a preview, edges are only pruned from the canvas copy; the true edges stay in the backup.

---

### Parameters

None.

---

### Returns

None.

---

## _port_positions

```python
_port_positions()
```

Return the position of every port ball currently drawn.

---

### Parameters

None.

---

### Returns

- **positions** : *dict*

  `{(kind, node_index, label): (x, y)}` in data units.

---

## _redraw_panel_previews

```python
_redraw_panel_previews()
```

Redraw the dashed preview of every panel pick at the current port positions.

The coordinates stored in each pick are updated too, so Apply commits an edge drawn where the ports are now. A pick whose port is not drawn at the moment (hidden by a port-count preview) is kept but not drawn.

---

### Parameters

None.

---

### Returns

None.

---

## _sync_edge_preview_backup

```python
_sync_edge_preview_backup(is_preview)
```

Keep the true edge list safe while a port-count preview hides edges.

---

### Parameters

- **is_preview** : *bool*

  Whether the current rebuild is a port-count preview.

---

### Returns

None.

---

### Notes

A preview rebuild still prunes `edge_source`, so a connection on a port the preview removes disappears at once, exactly as Apply would do. This step makes that reversible:

| Rebuild | Effect |
|---|---|
| First preview tick | Copy `edge_source.data` into `state.edge_preview_backup` |
| Later preview ticks | Restore the backup before pruning, so each tick prunes the true list. Spinning the count back up brings a hidden edge back |
| Real rebuild | Restore the backup and clear it (`_end_edge_preview`) |

So leaving without Apply (selecting another node, deleting, placing, entering delete mode) leaves `edge_source` exactly as it was before the preview.

---

## _end_edge_preview

```python
_end_edge_preview()
```

End a port-count preview: put back the edges it hid, if any, and clear the backup.

Called by every real rebuild, and before every real edit records its undo step (`_push_undo`, undo, redo). Otherwise the edit and its snapshot would start from the pruned list, and the next real rebuild would overwrite the edit's own edges with the old backup.

---

### Parameters

None.

---

### Returns

None.

---

## compute_ports

```python
compute_ports(center_x, center_y, node_radius=NODE_RADIUS, port_radius=PORT_RADIUS,
              number_of_outputs=0, number_of_inputs=0, spread_angle_degrees=120.0, gap_pixels=20.0)
```

Place port balls evenly on a circle around a component. Static method.

---

### Parameters

- **center_x**, **center_y** : *float*

  Component centre.

- **node_radius** : *float, optional*

  Radius of the component (default: `NODE_RADIUS`, 22.5).

- **port_radius** : *float, optional*

  Radius of a port ball (default: `PORT_RADIUS`, 8).

- **number_of_outputs** : *int, optional*

  Number of source ports (default: `0`).

- **number_of_inputs** : *int, optional*

  Number of target ports (default: `0`).

- **spread_angle_degrees** : *float, optional*

  Angular fan of each side (default: `120.0`).

- **gap_pixels** : *float, optional*

  Clearance between the component edge and the port centres (default: `20.0`).

---

### Returns

- **ports** : *dict*

  `{"outputs": [port, ...], "inputs": [port, ...]}`, each port a dict with keys `index`, `kind` (`"output"` / `"input"`), `angle_deg`, `x`, `y`.

---

### Notes

All ports are on a circle of radius `node_radius + port_radius + gap_pixels` (50.5 at the default size).

- Source (output) ports fan around **−90°**, target (input) ports around **+90°**.
- A single port sits at the centre angle; `n` ports are spread evenly over `spread_angle_degrees`, from `centre − spread / 2` to `centre + spread / 2`.

The Bokeh canvas has `y` pointing up, so −90° is **below** the icon and +90° **above** it: source ports are drawn under the component, target ports over it. (The in-code docstrings say "top" and "bottom" the other way round, which holds for a `y`-down frame.) This matches the layout of the network viewer, with propulsors at the top and energy storage at the bottom.

---

### Examples

```python
ports = PortEdgeMixin.compute_ports(0.0, 0.0, number_of_outputs=1, number_of_inputs=3)
[round(p["angle_deg"]) for p in ports["inputs"]]    # [30, 90, 150]
ports["outputs"][0]["y"]                            # -50.5
```

---

## _port_dict

```python
_port_dict(kind, port_data, port_position)
```

Return one port ball of a port source as a port dict.

---

### Parameters

- **kind** : *str*

  `"source"` or `"target"`.

- **port_data** : *dict*

  `source_port_source.data` or `target_port_source.data`.

- **port_position** : *int*

  Row of the port.

---

### Returns

- **port** : *dict*

  Keys `kind`, `node_index`, `label`, `x`, `y`, `color`.

---

## _append_port_row

```python
_append_port_row(port_columns, port, color, node_row)
```

Append one port ball to port data being built, in place. The label is `index + 1` as a string. The `connected` column is filled afterwards.

---

### Parameters

- **port_columns** : *dict*

  Port data built with `PORT_COLUMNS`.

- **port** : *dict*

  Port from `compute_ports`.

- **color** : *str*

  Colour of the port.

- **node_row** : *dict*

  Values shared by every port of the node: `node_index`, `node_name`, `node_type`, `fill_alpha`, `line_alpha`.

---

### Returns

None.

---

## _set_connected_flags

```python
_set_connected_flags(port_columns, kind, connected_port_keys)
```

Fill the `connected` column (`"True"` / `"False"`) of port data being built, in place.

---

### Parameters

- **port_columns** : *dict*

  Port data built with `PORT_COLUMNS`.

- **kind** : *str*

  `"source"` or `"target"`.

- **connected_port_keys** : *set of tuple*

  Keys of every port with an edge.

---

### Returns

None.

---

## _edge_rejection

```python
_edge_rejection(starting_port, ending_port, edge_data)
```

Return why a connection between two ports may not be added, if it may not.

Shared by `_add_edge`, which refuses such a connection, and by the canvas workflow, which checks before pushing an undo step.

---

### Parameters

- **starting_port**, **ending_port** : *dict*

  Port dicts of the two ends.

- **edge_data** : *dict*

  `edge_source.data` or a copy.

---

### Returns

- **reason** : *str or None*

  `None` if the connection may be added, else one of:

  - `"A component cannot be connected to itself"`
  - `"Port already connected"`: either port already has an edge
  - `"These two components are already connected"`: in either direction

---

### Notes

The kind and colour checks are not done here: they are done by the callers (canvas workflow) or guaranteed by the choices offered (Connections panel).

---

## _connected_port_keys

```python
_connected_port_keys(edge_data)
```

Return the key of every port that has an edge.

---

### Parameters

- **edge_data** : *dict*

  `edge_source.data` or a copy.

---

### Returns

- **keys** : *set of tuple*

  `{(kind, node_index, label), ...}`.

---

## _stored_port_count

```python
_stored_port_count(port_counts, node_index, default_counts, node_type)
```

Return the port count of one side of a node, falling back to its type's default when the column has no value for it.

---

### Parameters

- **port_counts** : *list of int*

  `n_sources` or `n_targets` column.

- **node_index** : *int*

  Index of the node.

- **default_counts** : *dict*

  `default_source_count` or `default_target_count`.

- **node_type** : *str*

  Component type of the node.

---

### Returns

- **count** : *int*