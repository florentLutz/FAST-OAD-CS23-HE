# I/O

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_handler_io.py`, mixin `IOMixin` of [`BuilderHandler`](powertrain_builder_handler).

It saves and loads designs, manages the watcher CSV path, runs undo and redo, and ends the session.

A design is saved as one YAML file: the FAST-OAD_CS23-HE powertrain configuration, plus, unless saved as a 
final version, a GUI-only `canvas_layout` section with the canvas position of each component. See [YAML writer](powertrain_builder_yaml_writer) 
and [YAML reader](powertrain_builder_yaml_reader).

**Save button colour.** Green (`"success"`) means there is nothing to save: the canvas matches the last saved or loaded 
file, or is a new blank design; orange (`"warning"`) means there are unsaved changes. **End Session** reads this colour 
to decide whether to ask before quitting.

---

## Flows

#### Save

1. **Save** (JavaScript) shows the "Is this the final version?" dialog.
2. **Final Version** / **Edit Later** sets `browse_save_trigger` to `"final:<ms>"` / `"editable:<ms>"`.
3. `_on_browse_save` opens a save-file dialog and calls `_save_canvas_state`.

#### Load

1. **Load Design** (JavaScript) flips `browse_load_trigger`.
2. `_on_browse_load` opens an open-file dialog, hides the startup overlay and calls `_load_canvas_state`.

#### End session

| Situation | Path |
|---|---|
| Saved (green) | **End Session** -> `end_session_trigger` -> `_end_session`; the tab closes |
| Unsaved -> **End Anyway** | `end_session_trigger` -> `_end_session`; the tab closes |
| Unsaved -> **Save & Exit** | `end_session_save_trigger` -> `_on_end_session_save_trigger` sets a flag; the "final version?" dialog -> file dialog -> `_on_browse_save` saves, closes the tab, stops the server |
| Tab closed | Session destroyed -> server stopped by the launcher |

---

## _mark_unsaved

```python
_mark_unsaved()
```

Turn the Save button orange to signal unsaved changes, and refresh the look of the Undo and Redo buttons. Called by 
every method that changes the design.

---

### Parameters

None.

---

### Returns

None.

---

## _save_canvas_state

```python
_save_canvas_state(yaml_path='', include_canvas_layout=True)
```

Write the current design to a YAML file.

---

### Parameters

- **yaml_path** : *str, optional*

  Path of the file. An empty string (cancelled dialog) skips the save (default: `''`).

- **include_canvas_layout** : *bool, optional*

  `False` for a "final version" file without the `canvas_layout` section (default: `True`).

---

### Returns

- **saved** : *bool*

  Whether the file was written.

---

### See also

- [`PowerTrainYAMLWriter`](powertrain_builder_yaml_writer#powertrainyamlwriter): builds the file content.

---

### Notes

1. Copy the node and edge data.
2. Create the parent folder if needed.
3. Build the YAML: title = file name without extension, every component, every connection, and the watcher path if set.
4. Write the file.

On success, the Save button turns green. On any error, the exception is logged, an error is shown in red in the status 
line, and the button stays orange.

---

## _load_canvas_state

```python
_load_canvas_state(file_path)
```

Replace the canvas with the design of a YAML file.

---

### Parameters

- **file_path** : *str*

  Path chosen by the user.

---

### Returns

None.

---

### Notes

The file must exist and end in `.yml` or `.yaml`. It is read with [`PowerTrainYAMLReader`](powertrain_builder_yaml_reader#powertrainyamlreader) and applied with 
`_restore_canvas_from_dict`. Any failure (missing file, wrong extension, parse error) is logged and shown in the status 
line; the canvas is left unchanged.

---

## _show_status_error

```python
_show_status_error(message)
```

Show an error in red in the status line. It stays until the status line changes again, e.g. when a component is armed or 
delete mode is toggled.

---

### Parameters

- **message** : *str*

  Plain text; HTML special characters are escaped.

---

### Returns

None.

---

## _restore_canvas_from_dict

```python
_restore_canvas_from_dict(canvas_state)
```

Replace the canvas with a design read from a YAML file, as returned by `PowerTrainYAMLReader.read`.

---

### Parameters

- **canvas_state** : *dict*

  Result of `PowerTrainYAMLReader.read`: keys `"components"` (node columns), `"connections"` (edge columns) and 
  `"watcher_file_path"`.

---

### Returns

None.

---

### Notes

1. Return the canvas to idle and clear the placement counter.
2. Load the node and edge data and the watcher path.
3. Rebuild the placement counter from the names, so new components continue the numbering: for each name `<key>_<n>`, 
the counter of `<key>` is the largest `n`.
4. Resize the icons, rings and labels to the current icon size.
5. Ask the browser to refit the view so every component is visible.
6. Rebuild the ports, which also recomputes every edge position and drops the edges pointing at ports that do not exist.
7. Turn the Save button green.

No undo step is recorded for the load, and the history is not cleared: undoing right after a load restores the snapshot 
of the last edit made *before* the load, not the design that was on the canvas when the file was opened.

---

## _request_view_fit

```python
_request_view_fit()
```

Ask the browser to refit the canvas view, by flipping `fit_view_trigger`.

---

### Parameters

None.

---

### Returns

None.

---

## _on_browse_load

```python
_on_browse_load(attr, old, new)
```

Callback of `browse_load_trigger`: open an open-file dialog for a powertrain YAML and load the chosen file. If the 
dialog is cancelled, nothing happens and the startup overlay stays.

---

### Parameters

- **attr**, **old**, **new** : *str*

  Bokeh arguments (unused).

---

### Returns

None.

---

## _on_browse_watcher

```python
_on_browse_watcher(attr, old, new)
```

Callback of `browse_watcher_trigger`: open a save-file dialog for the watcher CSV and put the chosen path in the 
watcher field. The field's own callback then stores it.

---

### Parameters

- **attr**, **old**, **new** : *str*

  Bokeh arguments (unused).

---

### Returns

None.

---

## _on_watcher_path_change

```python
_on_watcher_path_change(attr, old, new)
```

Callback of the watcher field: store the path written as `watcher_file_path` in the YAML. An empty field leaves the key 
out of the file.

A real change marks the design unsaved. Values set by a load or New Design match the stored path, so they do not.

---

### Parameters

- **attr**, **old** : *str*

  Bokeh arguments (unused).

- **new** : *str*

  New field value (stripped).

---

### Returns

None.

---

## _set_watcher_path

```python
_set_watcher_path(watcher_csv_path)
```

Set the watcher path without marking the design unsaved. The stored path is updated first, so the field's callback 
sees no change.

---

### Parameters

- **watcher_csv_path** : *str*

  Path, or an empty string.

---

### Returns

None.

---

## _open_tkinter_dialog

```python
_open_tkinter_dialog(dialog_function, **kwargs)
```

Run a native file dialog with tkinter and return the chosen path. Static method.

A hidden, topmost tkinter root window is created for the dialog, so it opens in front of the browser, and 
destroyed afterwards.

---

### Parameters

- **dialog_function** : *callable*

  `filedialog` function, e.g. `filedialog.asksaveasfilename`.

- **kwargs**

  Forwarded to *dialog_function* (`title`, `filetypes`, ...).

---

### Returns

- **path** : *str*

  The chosen path, or `""` if cancelled.

---

## _push_undo

```python
_push_undo()
```

Record the current design on the undo stack. Must be called **before** every change of `placed_nodes_source` or 
`edge_source`.

---

### Parameters

None.

---

### Returns

None.

---

### See also

- [`make_snapshot`](powertrain_builder_history#make_snapshot): what is recorded.

---

### Notes

It does not mark the design unsaved: the caller does, once its change is made.

---

## _restore_from_snapshot

```python
_restore_from_snapshot(snapshot)
```

Replace the design with an undo / redo snapshot.

---

### Parameters

- **snapshot** : *dict*

  Snapshot from `make_snapshot`.

---

### Returns

None.

---

### Notes

1. `_reset_interaction_state` clears the selection, pending clicks and picks, the armed component and delete mode, 
since a node index of the snapshot may now be another node.
2. The node and edge data are copied into the two sources (`copy_column_data`). `placed_counter` is cleared and updated 
rather than replaced, so the state keeps the same dictionary.
3. `_apply_node_icon_size` sets the icons to the current slider value, as the snapshot may hold an older size.
4. `_rebuild_all_ports` redraws the port balls and re-routes the edges.

It neither marks the design unsaved nor touches the history: `_on_undo` and `_on_redo` do.

---

## _on_undo

```python
_on_undo()
```

Undo the last change. Callback of the **Undo** button. Does nothing if there is nothing to undo.

---

### Parameters

None.

---

### Returns

None.

---

### Notes

1. Return if `can_undo` is false.
2. `_end_edge_preview`, then `make_snapshot` of the current design, pushed on the redo stack with `push_redo`.
3. `undo()` pops the last snapshot, and `_restore_from_snapshot` applies it.
4. `_mark_unsaved`, which also refreshes the Undo and Redo buttons.

---

## _on_redo

```python
_on_redo()
```

Redo the last undone change. Callback of the **Redo** button. Does nothing if there is nothing to redo.

---

### Parameters

None.

---

### Returns

None.

---

### Notes

1. Return if `can_redo` is false.
2. `_end_edge_preview`, then `make_snapshot` of the current design, pushed on the undo stack with `push_undo_keep_redo`, 
which keeps the remaining redo steps.
3. `redo()` pops the last undone snapshot, and `_restore_from_snapshot` applies it.
4. `_mark_unsaved`, which also refreshes the Undo and Redo buttons.

---

## _refresh_undo_redo_buttons

```python
_refresh_undo_redo_buttons()
```

Show the Undo and Redo buttons as active (`"primary"`) when a step is available, dimmed (`"default"`) otherwise.

---

### Parameters

None.

---

### Returns

None.

---

## _end_session

```python
_end_session()
```

Stop the Bokeh IO loop, which ends the server.

Never called from a button directly, only through the triggers, so the JavaScript check on unsaved changes always 
runs first.

---

### Parameters

None.

---

### Returns

None.

---

## _on_end_session_trigger

```python
_on_end_session_trigger(attr, old, new)
```

Callback of `end_session_trigger`: end the session.

JavaScript flips this trigger in two cases only: **End Session** with a saved design, and **End Anyway** in the 
unsaved-changes dialog.

---

### Parameters

- **attr**, **old**, **new** : *str*

  Bokeh arguments (unused).

---

### Returns

None.

---

## _on_end_session_save_trigger

```python
_on_end_session_save_trigger(attr, old, new)
```

Callback of `end_session_save_trigger` (**Save & Exit**): set `_pending_exit_after_save`, so the session ends after the 
next successful save.

---

### Parameters

- **attr**, **old**, **new** : *str*

  Bokeh arguments (unused).

---

### Returns

None.

---

## _on_browse_save

```python
_on_browse_save(attr, old, new)
```

Callback of `browse_save_trigger`: open a save-file dialog and write the YAML in the chosen mode.

---

### Parameters

- **attr**, **old** : *str*

  Bokeh arguments (unused).

- **new** : *str*

  `"<mode>:<timestamp>"`, set by the "final version?" dialog.

---

### Returns

None.

---

### Notes

1. Read the mode: the text before `:` in *new*.
2. `cancel`: clear `_pending_exit_after_save` and return.
3. Otherwise open a save-file dialog (`_open_tkinter_dialog`, default extension `.yml`) and call `_save_canvas_state`, 
with `include_canvas_layout` false for `final` only. A cancelled dialog returns `""`, which `_save_canvas_state` skips.
4. If Save & Exit is pending, clear the flag and, if the file was written, end the session (see below).

The modes:

| Mode | Effect |
|---|---|
| `final` | Save without `canvas_layout` |
| `editable`, or any other value (e.g. a legacy `"0"` / `"1"`) | Save with `canvas_layout` |
| `cancel` | No dialog, nothing saved; a pending Save & Exit is aborted |

With a pending Save & Exit, a successful save:

1. flips `close_window_trigger`, so the browser closes the tab;
2. stops the IO loop 0.3 s later, so Bokeh has time to send that change before the server stops.

If the dialog is cancelled or the write fails, the session stays open so no work is lost.

---

## _flip_trigger

```python
_flip_trigger(trigger)
```

Flip a hidden trigger between `"0"` and `"1"`, so its JavaScript callback always runs. Python counterpart of the 
toolbox's `_js_flip`.

---

### Parameters

- **trigger** : *bokeh.models.TextInput*

---

### Returns

None.

---

## _set_text_input

```python
_set_text_input(text_input, text)
```

Set the text of a `TextInput` from Python, in both `value_input` and `value`.

Setting `value` alone would leave the old text in `value_input`, which the "final version?" buttons copy back into 
`value` before saving.

---

### Parameters

- **text_input** : *bokeh.models.TextInput*

- **text** : *str*

---

### Returns

None.

