# Canvas mode

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_handler_canvas_mode.py`, mixin `CanvasModeMixin` of [`BuilderHandler`](powertrain_builder_handler).

These methods set the *mode* the canvas is in: which startup choice was made, whether delete mode is on, which toolbox 
component is armed. They prepare what the next canvas click will do, but never place, edit or delete anything 
themselves (except New Design, which clears the canvas).

---

## _dismiss_startup_overlay

```python
_dismiss_startup_overlay()
```

Hide the startup overlay and its two buttons, so the canvas is unobstructed. Called as soon as the user picks 
**New Design** or a file to load.

---

### Parameters

None.

---

### Returns

None.

---

## _on_new_design

```python
_on_new_design()
```

Start a blank design. Callback of the **New Design** button.

---

### Parameters

None.

---

### Returns

None.

---

### Notes

1. Hide the startup overlay.
2. Return the canvas to idle (`_reset_interaction_state`).
3. Empty the node and edge sources and the placement counter, and clear the watcher path.
4. Rebuild the ports, which empties the port sources.
5. Clear the undo / redo history: a new design has no previous edits.
6. Turn the Save button green and ask the browser to refit the view.

---

## _reset_interaction_state

```python
_reset_interaction_state()
```

Return the canvas to idle: no component selected or armed for placement, no connection waiting for a second click or 
for Apply, delete mode off.

Called when entering delete mode, and before the canvas content is replaced (New Design, load, undo, redo): after a 
replacement, a stored node index may point at another component. The design itself is not changed.

---

### Parameters

None.

---

### Returns

None.

---

### Notes

In order, it clears:

1. the port clicked first in a two-click connection;
2. the panel picks waiting for Apply and their dashed previews;
3. the node selection, hiding the panel;
4. the armed toolbox component, its button highlight and the status line;
5. delete mode.

---

## _disarm_toolbox

```python
_disarm_toolbox()
```

Deselect the armed toolbox component, if any: reset every toolbox button to the default style and the status line to 
*Select a component*.

---

### Parameters

None.

---

### Returns

None.

---

## _toggle_delete_mode

```python
_toggle_delete_mode()
```

Turn delete mode on or off. Callback of the **Delete** button.

Turning it on first returns the canvas to idle, then turns the Delete button red and shows *Delete mode: click an icon 
/ a connection to remove it* in red. Turning it off restores both.

---

### Parameters

None.

---

### Returns

None.

---

## _exit_delete_mode

```python
_exit_delete_mode()
```

Turn delete mode off and restore the Delete button and the status line. Does nothing if delete mode is already off.

---

### Parameters

None.

---

### Returns

None.

---

## on_toolbox_select

```python
on_toolbox_select(component_index)
```

Arm the toolbox component at *component_index* for placement, or disarm it if it is already armed. Callback of every 
toolbox button.

---

### Parameters

- **component_index** : *int*

  Zero-based index into `list(BUILDER_ICONS_CONFIG)`. An index out of range is ignored.

---

### Returns

None.

---

### Notes

- **Same button again:** disarm it, reset its style and the status line.
- **Another button:**
  1. cancel a pending two-click connection;
  2. arm the component;
  3. leave delete mode;
  4. reset the previously highlighted button and highlight the new one (`"primary"`);
  5. show *Placing: \<component\>* in gold.

The selected node, if any, stays selected; it is deselected when the component is actually placed.

---

### Examples

```python
icon_keys = list(BUILDER_ICONS_CONFIG)
handler.on_toolbox_select(icon_keys.index("battery_pack"))
print(handler.state.selected_component)   # 'battery_pack'
```
