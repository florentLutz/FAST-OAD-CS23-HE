# Powertrain builder

The powertrain builder is a graphical editor for FAST-OAD_CS23-HE powertrain configurations. Components are placed on a canvas, connected port to port, configured in a side panel, and saved as the powertrain YAML file read by the analysis. The same file can be re-opened in the builder for later edits.

The GUI runs as a local [Bokeh](https://docs.bokeh.org/) server opened in the default web browser. All its code is in the package `fastga_he.gui.power_train_builder`.

---

## Launching

From a terminal:

```bash
powertrain-builder
powertrain-builder --port 5008 --address localhost
```

See [Launcher](powertrain_builder_launcher#launching) for the options and the `python -m` alternative.

From Python or a notebook:

```python
from fastga_he.gui.power_train_builder.power_train_builder_launcher import PowertrainBuilderLauncher

PowertrainBuilderLauncher.launch(port=5007)
```

The call blocks until the session ends: **End Session**, **Save & Exit**, or closing the browser tab all stop the server.

---

## What the user can do

| Action | How | Where it is handled                                                                                    |
|---|---|--------------------------------------------------------------------------------------------------------|
| Start a design | **New Design** or **Load Design** on the startup overlay | [Canvas mode](powertrain_builder_canvas_mode), [I/O](powertrain_builder_io)                            |
| Place a component | Pick a toolbox button, then click on the canvas | [Placement](powertrain_builder_placement)                                                              |
| Select / configure | Click a component, edit the right-hand panel, click **Apply** | [Config panel](powertrain_builder_config_panel)                                                        |
| Connect two ports | Click a port, then a compatible port of another component; or pick a port in the **Connections** panel and **Apply** | [Ports and edges](powertrain_builder_ports_edges), [Config panel](powertrain_builder_config_panel) |
| Define a component the builder does not know | Place the **Unknown** component and type its type, options and port types | [Unknown components](powertrain_builder_unknown)                                            |
| Delete | **Delete** button, then click a component or a connection | [Handler](powertrain_builder_handler)                                                                  |
| Undo / redo | Buttons above the canvas | [History](powertrain_builder_history)                                                                  |
| Resize icons | Slider next to the *Components* title | [Placement](powertrain_builder_placement)                                                              |
| Save | **Save**, then **Final Version** (analysis input only) or **Edit Later** (with canvas layout) | [I/O](powertrain_builder_io), [YAML writer](powertrain_builder_yaml_writer)                            |

---

## Architecture

```
cli.main -> PowertrainBuilderLauncher.launch
                 |
                 |- ComponentToolboxConfigurationTableBuilder.build()  -> BuilderState (every widget + data source)
                 |- canvas figure + glyphs bound to the BuilderState data sources
                 `- BuilderHandler(state, canvas)  -> wires every Python callback
                        = CanvasModeMixin + PlacementMixin + IOMixin
                        + PortEdgeMixin + ConfigPanelMixin + UnknownComponentMixin
```

- **`BuilderState`** is the single shared object. It holds every Bokeh widget, every `ColumnDataSource`, the interaction state and the per-component-type lookup tables. Every module receives it by reference.
- **`BuilderHandler`** holds no data of its own except the icon size. It is assembled from six mixins, one per responsibility, which all work on `self.state`.
- **The design** is two data sources: `placed_nodes_source` (one row per component) and `edge_source` (one row per connection). Everything else drawn on the canvas (port balls, previews, rings) is derived from them and rebuilt by `_rebuild_all_ports`.
- **The file** is the FAST-OAD_CS23-HE powertrain YAML, optionally followed by a GUI-only `canvas_layout` section. FAST-OAD ignores that section.

---

## Conventions used throughout the code

- **Copy, mutate, reassign.** A `ColumnDataSource` is never edited in place: the data is copied with `copy_column_data`, the copy is edited, and the whole `.data` is reassigned. Otherwise the browser is not updated.
- **Undo before mutating.** Every action that changes the design calls `_push_undo()` first and `_mark_unsaved()` last.
- **Nodes are referred to by index.** A node is its row in `placed_nodes_source`. Deleting a node shifts every later index, so edges, symmetry links and the selection are re-indexed.
- **Port key.** A port is identified by `(kind, node_index, label)`, with `kind` in `"source"` / `"target"` and `label` a string `"1"`, `"2"`, ...
- **Energy type = colour.** Two ports may be connected only if they have the same, defined colour (fuel, mechanical, electricity).
- **Browser <-> Python triggers.** Actions that need Python after a JavaScript click (file dialogs, ending the session, refitting the view) go through hidden `TextInput` widgets whose value is flipped.
- **Edits are committed by Apply.** Config-panel fields only change the design when **Apply** is clicked, except the port-type selects of an unknown component, which apply at once.

---

## Pages

| Page | Module(s) | Content |
|---|---|---|
| [Launcher](powertrain_builder_launcher) | `cli.py`, `power_train_builder_launcher.py` | Entry points, canvas glyphs, legend, window fitting |
| [Toolbox and panel layout](powertrain_builder_toolbox) | `power_train_builder_toolbox_config_table.py` | Construction of every widget and of `BuilderState` |
| [State](powertrain_builder_state) | `power_train_builder_state.py` | `BuilderState`, constants, column layouts, data helpers |
| [Metadata](powertrain_builder_metadata) | `power_train_builder_metadata.py` | Icons, default port counts, possible positions and options |
| [Handler](powertrain_builder_handler) | `power_train_builder_handler.py` | `BuilderHandler`, callback wiring, canvas taps, deletion |
| [Canvas mode](powertrain_builder_canvas_mode) | `power_train_builder_handler_canvas_mode.py` | Startup overlay, New Design, delete mode, toolbox selection |
| [Placement](powertrain_builder_placement) | `power_train_builder_handler_placement.py` | New components, icon size, view scale |
| [Ports and edges](powertrain_builder_ports_edges) | `power_train_builder_handler_ports.py` | Hit-testing, port geometry, connections, previews |
| [Config panel](powertrain_builder_config_panel) | `power_train_builder_handler_config_panel.py` | Right-hand panel, connections table, symmetry, Apply |
| [Unknown components](powertrain_builder_unknown) | `power_train_builder_handler_unknown.py` | Free-text panel of unknown components, port types |
| [I/O](powertrain_builder_io) | `power_train_builder_handler_io.py` | Save, load, watcher path, undo / redo handlers, end session |
| [History](powertrain_builder_history) | `power_train_builder_history.py` | Undo / redo stack and snapshots |
| [YAML writer](powertrain_builder_yaml_writer) | `power_train_network_writer.py` | Canvas -> powertrain YAML |
| [YAML reader](powertrain_builder_yaml_reader) | `power_train_network_reader.py` | Powertrain YAML -> canvas, automatic layout |

---

## See also

- `fastga_he.gui.power_train_network_viewer`: read-only network plot; the builder reuses its icons, colours and string helpers.
- `fastga_he.powertrain_builder.resources.registered_components.KNOWN_COMPONENTS`: registry the toolbox, the panel and the YAML I/O are built from.