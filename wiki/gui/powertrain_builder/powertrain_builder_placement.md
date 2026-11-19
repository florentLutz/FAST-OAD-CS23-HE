# Placement

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_handler_placement.py`, mixin `PlacementMixin` of [`BuilderHandler`](powertrain_builder_handler).

It creates new components on the canvas and keeps everything drawn around them in proportion to the icon size and to 
the window scale.

---

## _place_new_node

```python
_place_new_node(x, y)
```

Place a component of the armed toolbox type at `(x, y)`, as a single undoable step.

---

### Parameters

- **x**, **y** : *float*

  Canvas coordinates (data units) of the component centre.

---

### Returns

None.

---

### See also

- [`on_toolbox_select`](powertrain_builder_canvas_mode#on_toolbox_select): arms the component.
- [`_new_component_defaults`](#_new_component_defaults): type, position, options and ports of the new component.

---

### Notes

1. Deselect the selected node, if any, and clear its panel and pending picks.
2. Push an undo snapshot.
3. Name the component `<icon_key>_<n>`, where `n` is the next value of `placed_counter[icon_key]` (e.g. 
`battery_pack_1`, then `battery_pack_2`).
4. Build the node row: position, icon URL, icon size, name, icon key, no symmetry peer, plus the defaults of its type.
5. Append the row to `placed_nodes_source`, rebuild the ports so the new port balls appear, and mark the design unsaved.

The component stays armed, so several components of the same type can be placed with successive clicks.

---

## _new_component_defaults

```python
_new_component_defaults(icon_key)
```

Return the type, position, options and port settings of a new component.

---

### Parameters

- **icon_key** : *str*

  Toolbox icon of the new component.

---

### Returns

- **columns** : *dict*

  Values of the `node_type`, `position`, `options`, `n_sources`, `n_targets`, `source_port_type` and 
  `target_port_type` columns.

---

### Notes

| Column | Known component | Unknown component |
|---|---|---|
| `node_type` | First type of `icon_to_component_types[icon_key]` | `"unknown"` |
| `position` | First possible position of that type, or `""` | `""` |
| `options` | JSON of the first value of every possible option | `"{}"` |
| `n_sources`, `n_targets` | Default counts of the type | 1, 1 |
| `source_port_type`, `target_port_type` | `""` (colours come from the icon) | `"undefined"` |

---

## _on_node_size_slider_change

```python
_on_node_size_slider_change(attr, old, new)
```

Callback of the icon-size slider: resize every node, existing and future.

This is a display setting: it is not an undoable step and does not mark the design unsaved.

---

### Parameters

- **attr** : *str*

  Bokeh attribute name (unused).

- **old** : *int*

  Previous value (unused).

- **new** : *int*

  New icon size in canvas data units.

---

### Returns

None.

---

### Notes

1. Store the new size in `self.icon_size`.
2. Resize the icons, rings and labels (`_apply_node_icon_size`).
3. Rebuild the ports, whose orbit scales with the icon size; this also re-routes the edges.
4. Refresh the Connections panel of the selected node, if any.

---

## _on_view_scale_change

```python
_on_view_scale_change(attr, old, new)
```

Callback of the hidden `view_scale_input`: receive the view report the browser sends after fitting the canvas.

---

### Parameters

- **attr** : *str*

  Bokeh attribute name (unused).

- **old** : *str*

  Previous value (unused).

- **new** : *str*

  JSON report, e.g. `{"k": 1.27, "area": [x0, x1, y0, y1], "area_k": 1.27}`, or a bare number read as `k`.

---

### Returns

None.

---

### See also

- [`_make_canvas_fit_window`](powertrain_builder_launcher#_make_canvas_fit_window): browser side of the report.

---

### Notes

The text is decoded by `_parse_view_report`; nothing happens if it is not a JSON object or number. Each part is then 
read by its own helper, and a malformed part is ignored without affecting the other:

- `area` / `area_k` -> `_canvas_area_from_report` -> `state.canvas_area`, used to spread automatic layouts over the 
window when loading.
- `k` -> `_view_scale_from_report` -> `state.view_scale`. Icons are sized in data units, so they shrink on screen when 
`k > 1`; the rings and the label offset are in screen pixels and are rescaled to match (`_apply_node_icon_size`), but 
only when `k` has changed.

---

## _apply_node_icon_size

```python
_apply_node_icon_size()
```

Resize every icon, and the rings and name labels around them, to the current `self.icon_size`.

Also called after undo, redo and load, whose node data may carry icon sizes from before the slider moved. It does not 
rebuild the ports.

---

### Parameters

None.

---

### Returns

None.

---

### Notes

- **Icons:** `w` and `h` of every node are set to `icon_size` (data units). The source is only reassigned if a 
value differs.
- **Hover ring, selection ring, name-label offset** (screen pixels): base size × `size_factor`, with

  ```python
  size_factor = icon_size / DEFAULT_NODE_ICON_SIZE / view_scale
  ```

  so they keep hugging the icon whatever the window size. The hover ring's `hover_glyph` is resized too.

  The port balls use a different factor, without `view_scale` (see `_draw_ports`): they are placed in data units, like 
  the icons.

Renderers that do not exist (headless use, tests) are skipped.

---

## _parse_view_report

```python
_parse_view_report(report_text)
```

Decode the JSON written by the browser into `view_scale_input`.

---

### Parameters

- **report_text** : *str*

  New value of `view_scale_input`.

---

### Returns

- **report** : *dict or None*

  The report; a bare number becomes `{"k": number}`. `None` if the text is not valid JSON, or not an object or a number.

---

### Examples

```python
_parse_view_report('{"k": 1.2}')   # {'k': 1.2}
_parse_view_report('1.25')         # {'k': 1.25}
_parse_view_report('oops')         # None
```

---

## _canvas_area_from_report

```python
_canvas_area_from_report(report)
```

Return the canvas area that an empty design shows in the user's window, from a view report.

---

### Parameters

- **report** : *dict*

  Decoded report.

---

### Returns

- **area** : *dict or None*

  `{"x0", "x1", "y0", "y1", "per_px"}` in data units, with `per_px` from `area_k`, else `k`, else 1. `None` if `area` 
  is not four increasing numbers or the scale is not positive.

---

## _view_scale_from_report

```python
_view_scale_from_report(report)
```

Return the data units per pixel of the fitted view, from a view report.

---

### Parameters

- **report** : *dict*

  Decoded report.

---

### Returns

- **view_scale** : *float or None*

  `k` as a positive float, or `None` if missing or invalid.
