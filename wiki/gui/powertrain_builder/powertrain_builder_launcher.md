# Launcher

[<- Powertrain builder](powertrain_builder)

Modules `cli.py` and `power_train_builder_launcher.py`. They start the Bokeh server, assemble the page (toolbox, canvas, 
configuration panel, overlays) and draw every canvas glyph. All interaction logic is wired afterwards by 
[`BuilderHandler`](powertrain_builder_handler).

---

## Launching

The builder can be started in two ways. Both serve it at `http://<address>:<port>/`, open it in the default web browser, 
and block until the session ends (**End Session**, **Save & Exit**, or closing the browser tab).

#### From a terminal (CLI)

Once FAST-OAD_CS23-HE is installed, the builder is started with the `powertrain-builder` command:

```bash
powertrain-builder
powertrain-builder --port 5008 --address localhost
powertrain-builder --help
```

`powertrain-builder` is a console script declared in the `pyproject.toml` at the root of the repository:

```toml
[tool.poetry.scripts]
powertrain-builder = "fastga_he.gui.power_train_builder.cli:main"
```

Installing the package (e.g. `poetry install` or `pip install`) creates the command in the environment's scripts folder; 
it runs `main` of `cli.py`, a [Click](https://click.palletsprojects.com/) command. After adding or renaming the script in `pyproject.toml`, the package 
must be installed again for the command to exist. The same command can also be run as a module, e.g. when the 
environment's scripts folder is not on the `PATH`:

```bash
python -m fastga_he.gui.power_train_builder.cli
```

| Option | Default | Meaning |
|---|---|---|
| `--port` | `5007` | TCP port of the Bokeh server; change it if 5007 is already in use |
| `--address` | `localhost` | Address the server binds to |
| `--help` | | Show the options and their defaults, then exit |

The command only forwards its options to `PowertrainBuilderLauncher.launch` (see [`main`](#main)). To stop the server 
from the terminal, press `Ctrl+C`.

#### From Python or a notebook

```python
from fastga_he.gui.power_train_builder.power_train_builder_launcher import PowertrainBuilderLauncher

PowertrainBuilderLauncher.launch(port=5007)
```

The call blocks the Python process (or the notebook cell) until the session ends (see 
[`PowertrainBuilderLauncher.launch`](#powertrainbuilderlauncherlaunch)).

#### Remote use

The file dialogs (Load, Save, watcher CSV) open with tkinter on the machine that runs the server. The builder is 
therefore meant to be launched and used on the same computer, with the default `--address localhost`.

---

## main

```python
main(port=5007, address='localhost')
```

Command-line entry point, run by the `powertrain-builder` command. Launch the powertrain builder in the default 
web browser.

It is a [Click](https://click.palletsprojects.com/) command that forwards its two options to `PowertrainBuilderLauncher.launch`.

---

### Parameters

- **port** : *int, optional*

  TCP port of the Bokeh server, option `--port` (default: `5007`).

- **address** : *str, optional*

  Server bind address, option `--address` (default: `'localhost'`).

---

### Returns

None. The command blocks until the session ends.

---

### Examples

```bash
powertrain-builder
powertrain-builder --port 5008
python -m fastga_he.gui.power_train_builder.cli --port 5008   # same, as a module
```

---

## PowertrainBuilderLauncher.launch

```python
PowertrainBuilderLauncher.launch(port=5007, address='localhost')
```

Start the Bokeh server and open the builder in the default browser.

The call configures logging, serves the builder document at `http://<address>:<port>/` and blocks the calling thread 
until the session ends.

---

### Parameters

- **port** : *int, optional*

  TCP port of the Bokeh server (default: `5007`).

- **address** : *str, optional*

  Server bind address (default: `'localhost'`).

---

### Returns

None.

---

### See also

- [`ComponentToolboxConfigurationTableBuilder.build`](powertrain_builder_toolbox#componenttoolboxconfigurationtablebuilderbuild): builds the widgets and the `BuilderState`.
- [`BuilderHandler`](powertrain_builder_handler): wires the Python callbacks.

---

### Notes

The launch is done in four steps.

#### 1. Logging

The root logger is set to `INFO` with a timestamped format. The `bokeh` and `tornado` loggers are set to `WARNING`, so 
the console only shows the builder's own messages.

#### 2. Document factory

`make_document(doc)` builds one complete page per browser session:

1. Build the toolbox, the configuration panel and the `BuilderState`.
2. Create the **Undo** / **Redo** buttons and store them on the state, so that the handler can wire them.
3. Create the canvas figure: 800 × 950 px at first, with the ranges of the design world (`CANVAS_WORLD_X_RANGE`, 
`CANVAS_WORLD_Y_RANGE`) and no axes or grid. The title text has the background colour, so it is invisible.
4. Add the legend (`_add_canvas_legend`) and the glyphs listed below.
5. Create the `BuilderHandler` and connect the canvas `Tap` event to `on_canvas_tap`.
6. Fit the canvas to the window (`_make_canvas_fit_window`).
7. Add the roots: the main row (toolbox, canvas, panel), the startup overlay, the Undo / Redo row, the "final version?" 
overlay and the unsaved-exit overlay.

The Undo / Redo row is absolutely positioned over the top-left corner of the canvas (`left = PALETTE_WIDTH + 38 px`, 
`top = 30 px`, z-index 50), so the dialogs (z-index 100) cover it.

#### 3. Session tracking

`make_document_with_tracking` wraps the factory and registers an `on_session_destroyed` hook that stops the IO loop. 
Closing the browser tab therefore also stops the server.

#### 4. Server

A single-process `Server` serves the factory at `/`. The browser is opened 0.1 s after the IO loop starts, then 
`server.io_loop.start()` blocks.

#### Canvas glyphs

Drawn in this order, so later glyphs are on top:

| Glyph | Data source | Purpose |
|---|---|---|
| Multi-line, width 4 | `edge_source` | Committed connections |
| Dashed multi-line | `panel_preview_edge_source` | Connections picked in the panel, waiting for Apply |
| Gold scatter (`selection_ring_renderer`) | `selected_node_overlay_source` | Ring around the selected node |
| `image_url` | `placed_nodes_source` | Component icons |
| Dashed scatter | `awaiting_second_port_click_source` | Port clicked first in the two-click workflow |
| Invisible scatter (`hover_ring_renderer`) | `placed_nodes_source` | Hover ring and component tooltip |
| Scatter + hover scatter + text | `source_port_source` | Source port balls and their numbers |
| Scatter + hover scatter + text | `target_port_source` | Target port balls and their numbers |
| `LabelSet` (`node_label_set`) | `placed_nodes_source` | Component names under the icons |

Three hover tools show the component id and type, and, on the ports, the port side and number.

---

### Examples

```python
from fastga_he.gui.power_train_builder.power_train_builder_launcher import PowertrainBuilderLauncher

PowertrainBuilderLauncher.launch(port=5010)
```

---

## _add_canvas_legend

```python
_add_canvas_legend(canvas)
```

Draw the connection-type legend in the top-right corner of the canvas.

There is one row per energy type: fuel flow, mechanical power and electrical current. Each row has a small colour icon 
and a label drawn in the edge colour of that type, as in the network viewer.

---

### Parameters

- **canvas** : *bokeh.plotting.figure*

  Builder canvas to annotate.

---

### Returns

None.

---

### Notes

The legend is made of two renderers, each with its own data source and one row per entry of `_LEGEND_ENTRIES`: an 
`image_url` glyph for the icons (`icon_data_url` of `COLOR_ICON_CONFIG`) and a `LabelSet` for the coloured texts. A 
`CustomJS` callback then moves both.

Bokeh can only place images in data coordinates, but the legend must stay fixed on screen when the view is panned, 
zoomed, reset or resized. The legend is therefore laid out in screen pixels (`_LEGEND_*_PX` constants) and converted to 
data coordinates in the browser by the `_JS_PIN_LEGEND` callback. The callback runs whenever the start or end of a 
range, or the inner width or height of the canvas, changes.

| Constant | Value | Meaning |
|---|---|---|
| `_LEGEND_ICON_FROM_RIGHT_PX` | 118 | Icon centre, from the right edge of the frame |
| `_LEGEND_LABEL_FROM_RIGHT_PX` | 107 | Label left edge, from the right edge |
| `_LEGEND_TOP_PX` | 10 | Centre of the first row, from the top edge |
| `_LEGEND_ROW_STEP_PX` | 22 | Spacing between rows |
| `_LEGEND_ICON_W_PX`, `_LEGEND_ICON_H_PX` | 9, 12 | Icon size |

The initial positions assume about one data unit per pixel and are corrected as soon as the page is laid out.

---

## _make_canvas_fit_window

```python
_make_canvas_fit_window(doc, canvas, state)
```

Wire the browser-side callbacks that size the canvas to the browser window and fit its view.

---

### Parameters

- **doc** : *bokeh.document.Document*

  Bokeh document, used for its `DocumentReady` event.

- **canvas** : *bokeh.plotting.figure*

  Builder canvas.

- **state** : *BuilderState*

  Shared state, providing `placed_nodes_source`, `view_scale_input` and `fit_view_trigger`.

---

### Returns

None.

---

### See also

- [`_on_view_scale_change`](powertrain_builder_placement#_on_view_scale_change): receives the view report in Python.
- [`_request_view_fit`](powertrain_builder_io#_request_view_fit): asks the browser to refit after a load or New Design.

---

### Notes

The canvas takes the space between the two side panels, whatever the aspect ratio of the window. This is done in two 
steps, both in JavaScript.

#### 1. Canvas size (`_JS_FIT_CANVAS_SIZE`)

Runs when the document is ready and on every window resize (throttled to one animation frame). The width is the window 
width minus the canvas left position and `CONFIG_PANEL_RESERVED_WIDTH` (425 px); the height is the window height minus 
the canvas top position and an 8 px margin. Both are clamped to at least `CANVAS_MIN_WIDTH` × `CANVAS_MIN_HEIGHT` 
(480 × 420 px).

#### 2. View (`_JS_FIT_VIEW`)

Runs when the inner size of the canvas changes, and when Python flips `fit_view_trigger`.

1. Take the design world (0–800 × 0–950) and extend it to every placed node plus `CANVAS_FIT_NODE_MARGIN` 
(60 data units).
2. Use the same data-units-per-pixel `k` on both axes, so icons are not stretched: 
`k = max(width_data / width_px, height_data / height_px)`.
3. Anchor the ranges at the left and top edges; any extra room goes to the right and bottom. The ranges are also stored 
as the reset target of the **Reset** tool.
4. Write a JSON report to `view_scale_input`, for example:

   ```json
   {"k": 1.27, "area": [0, 1016.0, -66.5, 950], "area_k": 1.27}
   ```

   `k` is the scale of the current view; `area` and `area_k` are the area and scale an *empty* design would get in this 
    window, used to spread automatic layouts.
