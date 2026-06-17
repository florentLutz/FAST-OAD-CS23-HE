# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Canvas interaction handler for the powertrain builder.

:class:`PlacementHandler` wires every palette button and canvas tap event to
the appropriate action: placing a component, selecting a node for editing,
deleting a node or edge, managing port connections, and saving / loading the
canvas state.

Typical usage::

    palette_layout, table_panel, state = ComponentPaletteConfigurationTableBuilder.build()
    handler = PlacementHandler(state, canvas)
    canvas.on_event(Tap, handler.on_canvas_tap)
"""

import json
import logging
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
import bokeh.models as bkmodel
from bokeh.layouts import row
from tornado.ioloop import IOLoop

from fastga_he.gui.power_train_network_viewer import (
    ICONS_CONFIG,
    _string_cleanup,
    _url_to_base64,
    DEFAULT_COLOR,
)
from .power_train_network_writer import PowerTrainYAML
from .power_train_builder_state import (
    BuilderState,
    BUTTON_DEFAULT_COLOR_TYPE,
    BUTTON_SELECTED_COLOR_TYPE,
    NODE_RADIUS,
    PORT_RADIUS,
    _EMPTY,
)
from .power_train_builder_state import compute_ports

_LOGGER = logging.getLogger(__name__)


class PlacementHandler:
    """
    Wires palette button events to canvas placement and interaction events.

    Instantiate **after** building the palette; the constructor automatically
    wires all button ``on_click`` callbacks::

        palette_layout, table_panel, state = ComponentPaletteConfigurationTableBuilder.build()
        handler = PlacementHandler(state, main_plot)
        main_plot.on_event(Tap, handler.on_canvas_tap)
        main_plot.image_url(
            url="url", x="x", y="y", w="w", h="h",
            anchor="center", source=state.placed_nodes_source,
        )
    """

    def __init__(self, state: BuilderState, main_plot, icon_size: int = 50):
        """
        :param state: Shared :class:`BuilderState` instance.
        :param main_plot: The Bokeh ``figure`` that acts as the placement canvas.
        :param icon_size: Pixel size (width = height) used for placed icons.
        """
        self.state = state
        self.main_plot = main_plot
        self.icon_size = icon_size
        self._wire_buttons()

    # -----------------------------------------------------------------------
    # Internal wiring
    # -----------------------------------------------------------------------

    def _wire_buttons(self):
        """
        Attach ``on_click`` callbacks to every palette and action button.

        Called once from :meth:`__init__`. Connects:

        * Each component palette button → :meth:`on_palette_select`.
        * The End Session button → :meth:`_end_session`.
        * The Delete button → :meth:`_toggle_delete_mode`.
        * The New Design button → :meth:`_on_new_design`.
        * The browse_load_trigger → :meth:`_on_browse_load` (tkinter open-file dialog).
        * The browse_save_trigger → :meth:`_on_browse_save` (tkinter save-file dialogs).
        * The browse_watcher_trigger → :meth:`_on_browse_watcher` (tkinter CSV path dialog).
        """
        self.state.end_session_button.on_click(self._end_session)

        # New Design: wipe canvas and dismiss overlay immediately
        if self.state.new_design_button is not None:
            self.state.new_design_button.on_click(self._on_new_design)

        for index, button in enumerate(self.state.buttons):
            button.on_click(self._make_select_callback(index))

        if self.state.delete_button is not None:
            self.state.delete_button.on_click(self._toggle_delete_mode)

        if self.state.browse_load_trigger is not None:
            self.state.browse_load_trigger.on_change("value", self._on_browse_load)

        if self.state.browse_save_trigger is not None:
            self.state.browse_save_trigger.on_change("value", self._on_browse_save)

        if self.state.browse_watcher_trigger is not None:
            self.state.browse_watcher_trigger.on_change("value", self._on_browse_watcher)

    def _make_select_callback(self, idx: int):
        """
        Return a zero-argument closure that selects the component at *idx*.

        :param idx: Zero-based index into ``list(ICONS_CONFIG.keys())``.
        :return: Callback function for a palette button.
        """
        return lambda: self.on_palette_select(idx)

    # -----------------------------------------------------------------------
    # Startup overlay
    # -----------------------------------------------------------------------

    def _dismiss_startup_overlay(self):
        """
        Hide the startup overlay panel (and both its buttons).

        Called as soon as the user makes a choice so the canvas is unobstructed.
        """
        if self.state.startup_overlay is not None:
            self.state.startup_overlay.visible = False
        if self.state.new_design_button is not None:
            self.state.new_design_button.visible = False
        if self.state.load_design_button is not None:
            self.state.load_design_button.visible = False

    def _on_new_design(self):
        """
        Start with a blank canvas.

        Hides the startup overlay and resets every data source to empty so the
        user begins with a completely fresh powertrain design.
        """
        self._dismiss_startup_overlay()

        self.state.placed_nodes_source.data = {
            "x": [],
            "y": [],
            "url": [],
            "w": [],
            "h": [],
            "name": [],
            "node_type": [],
            "icon_type": [],
            "position": [],
            "options": [],
            "n_sources": [],
            "n_targets": [],
            "symmetry_name": [],
            "symmetry_node_index": [],
        }
        self.state.hover_source.data = dict(x=[], y=[], name=[], node_type=[])
        self.state.edge_source.data = dict(
            xs=[],
            ys=[],
            color=[],
            node_a_idx=[],
            a_label=[],
            a_kind=[],
            node_b_idx=[],
            b_label=[],
            b_kind=[],
        )
        self.state.source_port_source.data = dict(
            x=[],
            y=[],
            color=[],
            label=[],
            node_index=[],
            node_name=[],
            node_type=[],
            fill_alpha=[],
            line_alpha=[],
            connected=[],
        )
        self.state.target_port_source.data = dict(
            x=[],
            y=[],
            color=[],
            label=[],
            node_index=[],
            node_name=[],
            node_type=[],
            fill_alpha=[],
            line_alpha=[],
            connected=[],
        )
        self._clear_temp_edges()
        self.state.placed_counter.clear()
        self.state.selected_node_index = None
        self.state.selected_component = None
        self.state.delete_mode = False
        self._clear_node_table()

        if self.state.save_button is not None:
            self.state.save_button.button_type = "success"

        _LOGGER.info("New Design started – canvas cleared.")

    # -----------------------------------------------------------------------
    # Delete mode
    # -----------------------------------------------------------------------

    def _toggle_delete_mode(self):
        """
        Toggle delete mode on / off, updating button styling and status text.

        When entering delete mode any active component selection and pending
        port connection are cleared, the config panel is hidden, the Delete
        button turns red, and the status label prompts the user to click a
        node or edge to remove it.  Leaving delete mode restores the defaults.
        """
        self._cancel_pending_connection()
        self.state.delete_mode = not self.state.delete_mode
        if self.state.delete_mode:
            self.state.selected_component = None
            self.state.selected_node_index = None
            self._clear_temp_edges()
            self._clear_node_table()
            for btn in self.state.buttons:
                btn.button_type = BUTTON_DEFAULT_COLOR_TYPE
            self.state.status_div.text = (
                "<b style='color:#FF4444;font-size:14pt'>Delete mode: "
                "click an icon / a connection to remove it</b>"
            )
            self.state.delete_button.button_type = "danger"
        else:
            self.state.status_div.text = (
                "<i style='color:#aaa;font-size:14pt'>Select a component</i>"
            )
            self.state.delete_button.button_type = BUTTON_DEFAULT_COLOR_TYPE

    # -----------------------------------------------------------------------
    # Palette selection
    # -----------------------------------------------------------------------

    def on_palette_select(self, idx: int):
        """
        Select the component at position *idx* in :data:`ICONS_CONFIG`.

        A second click on the same button deselects it.  Updates button
        styling and the status label accordingly.  Can be called
        programmatically in tests without a running server.

        :param idx: Zero-based index into ``list(ICONS_CONFIG.keys())``.
        """
        component_icon_keys = list(ICONS_CONFIG.keys())
        if idx < 0 or idx >= len(component_icon_keys):
            return

        if self.state.selected_component == component_icon_keys[idx]:
            self.state.selected_component = None
            self.state.buttons[idx].button_type = BUTTON_DEFAULT_COLOR_TYPE
            self.state.status_div.text = (
                "<i style='color:#aaa;font-size:14pt'>Select a component</i>"
            )
            return

        self._cancel_pending_connection()
        self.state.selected_component = component_icon_keys[idx]

        if self.state.delete_mode:
            self.state.delete_mode = False
            if self.state.delete_button is not None:
                self.state.delete_button.button_type = BUTTON_DEFAULT_COLOR_TYPE

        for j, btn in enumerate(self.state.buttons):
            btn.button_type = BUTTON_SELECTED_COLOR_TYPE if j == idx else BUTTON_DEFAULT_COLOR_TYPE

        label = _string_cleanup(self.state.selected_component)
        self.state.status_div.text = f"<b style='color:#FFD700;font-size:14pt'>Placing: {label}</b>"

    # -----------------------------------------------------------------------
    # Save / load
    # -----------------------------------------------------------------------

    def _mark_unsaved(self):
        """
        Turn the Save button yellow to signal unsaved changes.

        Called by every method that mutates the canvas.  The button returns
        to green only after a successful save.
        """
        if self.state.save_button is not None:
            self.state.save_button.button_type = "warning"

    def _save_canvas_state(self, yaml_path: str = "", json_path: str = ""):
        """
        Serialise the current canvas to the file paths chosen by the user.

        Used only by the ``prompt()`` fallback path where the browser cannot
        write files directly and the user supplies a filesystem path manually.
        Both parameters are optional – an empty string means that file is
        skipped.

        :param yaml_path: File name (or full path) for the YAML config.
        :param json_path: File name (or full path) for the JSON backup.
        """
        nodes_data = {k: list(v) for k, v in self.state.placed_nodes_source.data.items()}
        edges_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
        source_data = {k: list(v) for k, v in self.state.source_port_source.data.items()}
        target_data = {k: list(v) for k, v in self.state.target_port_source.data.items()}

        if yaml_path:
            yaml_file = Path(yaml_path)
            yaml_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                pt_yaml = PowerTrainYAML(self.state)
                pt_yaml.set_title(yaml_file.stem)
                for node_index in range(len(nodes_data.get("name", []))):
                    pt_yaml.add_component(node_index)
                pt_yaml.add_connection()
                # Apply watcher file path when the user supplied one in the overlay.
                watcher_path = (
                    self.state.watcher_path_input.value.strip()
                    if self.state.watcher_path_input is not None
                    else ""
                )
                if watcher_path:
                    pt_yaml.set_watcher_file_path(watcher_path)
                pt_yaml.write(str(yaml_file))
                _LOGGER.info("Powertrain YAML config saved to %s", yaml_file)
            except Exception:
                _LOGGER.exception("Failed to write YAML config to %s.", yaml_file)

        if json_path:
            json_file = Path(json_path)
            json_file.parent.mkdir(parents=True, exist_ok=True)
            canvas_state = {
                "components": nodes_data,
                "connections": edges_data,
                "source_ports": source_data,
                "target_ports": target_data,
            }
            with open(json_file, "w") as f:
                json.dump(canvas_state, f, indent=2)
            _LOGGER.info("Canvas state (JSON backup) saved to %s", json_file)

        if self.state.save_button is not None:
            self.state.save_button.button_type = "success"

    def _load_canvas_state(self, json_path: str):
        """
        Restore the canvas from a JSON canvas-state backup file.

        Used only by the ``prompt()`` fallback where the user types a
        filesystem path manually.

        :param json_path: Path to the JSON file written by a previous save.
        """
        json_file = Path(json_path)
        if not json_file.exists():
            _LOGGER.error("Load failed – file not found: %s", json_file)
            return

        try:
            with open(json_file) as f:
                canvas_state = json.load(f)
        except Exception:
            _LOGGER.exception("Load failed – could not parse JSON from %s", json_file)
            return

        _LOGGER.info("Loading canvas state from file: %s", json_file)
        self._restore_canvas_from_dict(canvas_state)

    def _restore_canvas_from_dict(self, canvas_state: dict):
        """
        Shared canvas-restoration logic used by both load paths.

        Clears the current canvas entirely, then replays every node, edge, and
        port that was serialised by a previous save.

        :param canvas_state: Parsed canvas-state dict with keys ``"components"``,
            ``"connections"``, ``"source_ports"``, and ``"target_ports"``.
        """
        self.state.selected_node_index = None
        self.state.selected_component = None
        self.state.delete_mode = False
        self.state.pending_connections.clear()
        self.state.placed_counter.clear()
        self._clear_node_table()
        self._clear_temp_edges()

        nodes_data = canvas_state.get("components", {})
        edges_data = canvas_state.get("connections", {})
        source_data = canvas_state.get("source_ports", {})
        target_data = canvas_state.get("target_ports", {})

        # Ensure every expected column exists (forward-compatibility with older saves)
        _node_defaults = {
            "x": [],
            "y": [],
            "url": [],
            "w": [],
            "h": [],
            "name": [],
            "node_type": [],
            "icon_type": [],
            "position": [],
            "options": [],
            "n_sources": [],
            "n_targets": [],
            "symmetry_name": [],
            "symmetry_node_index": [],
        }
        for col, default in _node_defaults.items():
            nodes_data.setdefault(col, default)

        self.state.placed_nodes_source.data = {k: list(v) for k, v in nodes_data.items()}
        self.state.edge_source.data = {k: list(v) for k, v in edges_data.items()}
        self.state.source_port_source.data = {k: list(v) for k, v in source_data.items()}
        self.state.target_port_source.data = {k: list(v) for k, v in target_data.items()}

        self.state.hover_source.data = {
            "x": list(nodes_data.get("x", [])),
            "y": list(nodes_data.get("y", [])),
            "name": list(nodes_data.get("name", [])),
            "node_type": list(nodes_data.get("node_type", [])),
        }

        # Rebuild placed_counter so future placements continue from the right index
        for name in nodes_data.get("name", []):
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                key = parts[0]
                self.state.placed_counter[key] = max(
                    self.state.placed_counter.get(key, 0), int(parts[1])
                )

        self._rebuild_all_ports()
        self._rebuild_edges()

        if self.state.save_button is not None:
            self.state.save_button.button_type = "success"

        _LOGGER.info(
            "Canvas restored (%d nodes, %d edges).",
            len(nodes_data.get("name", [])),
            len(edges_data.get("xs", [])),
        )

    @staticmethod
    def _open_tkinter_dialog(func, **kwargs):
        """
        Run a tkinter file dialog on the calling thread and return the result.

        :param func: The ``filedialog`` function to call
            (e.g. ``filedialog.asksaveasfilename``).
        :param kwargs: Keyword arguments forwarded to *func*.
        :return: The chosen path string, or ``""`` if the user cancelled.
        """

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            result = func(parent=root, **kwargs)
        finally:
            root.destroy()
        return result or ""

    def _on_browse_load(self, attr, old, new):
        """
        Open a native OS open-file dialog (tkinter) for loading a JSON canvas
        backup, then restore the canvas from the chosen file.

        Triggered by the hidden ``browse_load_trigger`` TextInput toggling.
        """
        path = self._open_tkinter_dialog(
            filedialog.askopenfilename,
            title="Load canvas state",
            filetypes=[("JSON canvas backup", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self._dismiss_startup_overlay()
        self._load_canvas_state(json_path=path)

    def _on_browse_save(self, attr, old, new):
        """
        Open native OS save-file dialogs (tkinter) for the YAML config and JSON
        backup, then write both files.  The watcher file path is read from
        ``state.watcher_path_input`` which the user filled in the overlay.

        Triggered by the hidden ``browse_save_trigger`` TextInput toggling.
        """

        yaml_path = self._open_tkinter_dialog(
            filedialog.asksaveasfilename,
            title="Save YAML powertrain config",
            defaultextension=".yml",
            filetypes=[("YAML files", "*.yml *.yaml"), ("All files", "*.*")],
        )
        json_path = self._open_tkinter_dialog(
            filedialog.asksaveasfilename,
            title="Save JSON canvas backup",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if yaml_path or json_path:
            self._save_canvas_state(yaml_path=yaml_path, json_path=json_path)

    def _on_browse_watcher(self, attr, old, new):
        """
        Open a native OS save-file dialog (tkinter) to choose the watcher CSV
        path and write it back into ``state.watcher_path_input``.

        Triggered by the hidden ``browse_watcher_trigger`` TextInput toggling.
        """
        path = self._open_tkinter_dialog(
            filedialog.asksaveasfilename,
            title="Choose watcher CSV file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path and self.state.watcher_path_input is not None:
            self.state.watcher_path_input.value = path
            _LOGGER.info("Watcher CSV path set to: %s", path)

    # -----------------------------------------------------------------------
    # End session
    # -----------------------------------------------------------------------

    def _end_session(self):
        """
        Stop the Bokeh IO loop, terminating the server session.

        Bound to the End Session button.  After this call the browser tab will
        lose its WebSocket connection and the Python process will exit.
        """
        _LOGGER.info("Ending session and stopping server")
        IOLoop.current().stop()

    # -----------------------------------------------------------------------
    # Edge connection methods
    # -----------------------------------------------------------------------

    def _find_nearest_port(self, x: float, y: float) -> dict | None:
        """
        Return metadata for the nearest port ball within snap distance, or ``None``.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.

        :return: A dict with keys ``kind``, ``x``, ``y``, ``color``, ``label``,
            and ``node_index`` for the nearest port, or ``None``.
        """
        snap = PORT_RADIUS * 2.5
        best, best_distance = None, snap
        for kind, port_source in [
            ("source", self.state.source_port_source),
            ("target", self.state.target_port_source),
        ]:
            if port_source is None:
                continue
            data = port_source.data
            for i, (port_x, port_y) in enumerate(
                zip(list(data.get("x", [])), list(data.get("y", [])))
            ):
                distance = ((x - port_x) ** 2 + (y - port_y) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best = {
                        "kind": kind,
                        "x": port_x,
                        "y": port_y,
                        "color": list(data["color"])[i],
                        "label": list(data["label"])[i],
                        "node_index": list(data["node_index"])[i],
                    }
        return best

    def _cancel_pending_connection(self):
        """Cancel any port selection waiting for a second click."""
        self.state.pending_port = None
        if self.state.pending_port_source is not None:
            self.state.pending_port_source.data = dict(x=[], y=[], color=[])

    def _handle_port_tap(self, port: dict):
        """
        Manage the two-click port-connection workflow.

        First click stores the port as pending; second click on a compatible
        port (opposite kind, same energy colour) draws a permanent edge.
        Tapping the same port again cancels the pending selection.

        :param port: Port metadata dict with keys ``kind``, ``x``, ``y``,
            ``color``, ``label``, and ``node_index``.
        """
        self._clear_temp_edges()
        pending = self.state.pending_port
        if pending is None:
            self.state.pending_port = port
            if self.state.pending_port_source is not None:
                self.state.pending_port_source.data = dict(
                    x=[port["x"]], y=[port["y"]], color=[port["color"]]
                )
            return
        # Same port tapped again → cancel
        if (
            port["kind"] == pending["kind"]
            and port["node_index"] == pending["node_index"]
            and port["label"] == pending["label"]
        ):
            self._cancel_pending_connection()
            return
        # Connect source ↔ target of the same energy type (colour)
        if port["kind"] != pending["kind"] and port["color"] == pending["color"]:
            self._add_edge(pending, port)
            self._rebuild_all_ports()
        self._cancel_pending_connection()

    def _cursor_to_segment_distance(
        self, px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """
        Return the minimum Euclidean distance from ``(px, py)`` to the segment
        ``(x1, y1)``–``(x2, y2)``.

        Used by :meth:`_find_nearest_edge` for click-to-delete on edges.
        """
        distance_x, distance_y = x2 - x1, y2 - y1
        if distance_x == 0 and distance_y == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        projection_parameter = max(
            0.0,
            min(
                1.0,
                ((px - x1) * distance_x + (py - y1) * distance_y) / (distance_x**2 + distance_y**2),
            ),
        )
        return (
            (px - (x1 + projection_parameter * distance_x)) ** 2
            + (py - (y1 + projection_parameter * distance_y)) ** 2
        ) ** 0.5

    def _find_nearest_edge(self, x: float, y: float, snap: float = 12.0) -> int | None:
        """
        Return the index of the edge closest to ``(x, y)`` within *snap* pixels,
        or ``None`` if no edge is close enough.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.
        :param snap: Maximum allowed distance (data-units) for a hit.

        :return: Zero-based row index into ``edge_source``, or ``None``.
        """
        if self.state.edge_source is None:
            return None

        edge_data = self.state.edge_source.data
        best_index, best_distance = None, snap
        for i, (xs, ys) in enumerate(zip(edge_data.get("xs", []), edge_data.get("ys", []))):
            if len(xs) < 2:
                continue
            distance = self._cursor_to_segment_distance(x, y, xs[0], ys[0], xs[1], ys[1])
            if distance < best_distance:
                best_distance = distance
                best_index = i
        return best_index

    def _add_edge(self, port_a: dict, port_b: dict):
        """
        Append a permanent edge between two ports to ``edge_source``.

        Guards against already-occupied ports and loop connections; rejects
        the edge silently if either check fails.

        :param port_a: Source-side port metadata dict.
        :param port_b: Target-side port metadata dict.
        """
        if self.state.edge_source is None:
            return
        edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}

        for i in range(len(edge_data.get("node_a_idx", []))):
            port_a_used = (
                edge_data["node_a_idx"][i] == port_a["node_index"]
                and edge_data["a_label"][i] == port_a["label"]
                and edge_data["a_kind"][i] == port_a["kind"]
            ) or (
                edge_data["node_b_idx"][i] == port_a["node_index"]
                and edge_data["b_label"][i] == port_a["label"]
                and edge_data["b_kind"][i] == port_a["kind"]
            )
            port_b_used = (
                edge_data["node_a_idx"][i] == port_b["node_index"]
                and edge_data["a_label"][i] == port_b["label"]
                and edge_data["a_kind"][i] == port_b["kind"]
            ) or (
                edge_data["node_b_idx"][i] == port_b["node_index"]
                and edge_data["b_label"][i] == port_b["label"]
                and edge_data["b_kind"][i] == port_b["kind"]
            )
            if port_a_used or port_b_used:
                _LOGGER.info("Port already connected; edge rejected.")
                return

        loop_connection = any(
            (
                edge_data["node_a_idx"][i] == port_b["node_index"]
                and edge_data["node_b_idx"][i] == port_a["node_index"]
            )
            or (
                edge_data["node_a_idx"][i] == port_a["node_index"]
                and edge_data["node_b_idx"][i] == port_b["node_index"]
            )
            for i in range(len(edge_data.get("node_a_idx", [])))
        )
        if loop_connection:
            _LOGGER.info("Loop connection detected; edge rejected.")
            return

        edge_data["xs"].append([port_a["x"], port_b["x"]])
        edge_data["ys"].append([port_a["y"], port_b["y"]])
        edge_data["color"].append(port_a["color"])
        edge_data["node_a_idx"].append(port_a["node_index"])
        edge_data["a_label"].append(port_a["label"])
        edge_data["a_kind"].append(port_a["kind"])
        edge_data["node_b_idx"].append(port_b["node_index"])
        edge_data["b_label"].append(port_b["label"])
        edge_data["b_kind"].append(port_b["kind"])
        self.state.edge_source.data = edge_data

        if self.state.selected_node_index is not None and (
            port_a["node_index"] == self.state.selected_node_index
            or port_b["node_index"] == self.state.selected_node_index
        ):
            src_data = {k: list(v) for k, v in self.state.source_port_source.data.items()}
            tgt_data = {k: list(v) for k, v in self.state.target_port_source.data.items()}

            def _mark_connected(data, port):
                data["connected"] = [
                    "True"
                    if (
                        data["node_index"][i] == port["node_index"]
                        and data["label"][i] == port["label"]
                    )
                    else data["connected"][i]
                    for i in range(len(data["node_index"]))
                ]

            if port_a["kind"] == "source":
                _mark_connected(src_data, port_a)
                _mark_connected(tgt_data, port_b)
            else:
                _mark_connected(tgt_data, port_a)
                _mark_connected(src_data, port_b)

            self.state.source_port_source.data = src_data
            self.state.target_port_source.data = tgt_data
            self._refresh_connections_table(self.state.selected_node_index)

        self._mark_unsaved()
        _LOGGER.info(
            "Edge: %s port %s (node %d) ↔ %s port %s (node %d)",
            port_a["kind"],
            port_a["label"],
            port_a["node_index"],
            port_b["kind"],
            port_b["label"],
            port_b["node_index"],
        )

    def _add_edge_temp(self, port_a: dict, port_b: dict):
        """
        Draw a dashed preview line between two ports in ``temp_edge_source``.

        The line is **not** registered in ``edge_source``; it becomes permanent
        only when :meth:`apply_node_configurations` is called.
        """
        if self.state.temp_edge_source is None:
            return
        tdata = {k: list(v) for k, v in self.state.temp_edge_source.data.items()}
        tdata["xs"].append([port_a["x"], port_b["x"]])
        tdata["ys"].append([port_a["y"], port_b["y"]])
        tdata["color"].append(port_a["color"])
        self.state.temp_edge_source.data = tdata
        _LOGGER.info(
            "Temp edge: %s:%s → %s:%s",
            port_a["kind"],
            port_a["label"],
            port_b["kind"],
            port_b["label"],
        )

    def _clear_temp_edges(self):
        """Wipe all dashed preview edges and the pending-connection list."""
        self.state.pending_connections.clear()
        self._clear_temp_edge_visuals()

    def _clear_temp_edge_visuals(self):
        """Clear only the canvas dashed lines, leave pending_connections intact."""
        if self.state.temp_edge_source is not None:
            self.state.temp_edge_source.data = dict(xs=[], ys=[], color=[])

    def _rebuild_edges(self):
        """
        Recompute xs/ys in edge_source from stored port-identity columns.

        Called at the end of :meth:`_rebuild_all_ports` so edges track port
        positions after port-count spinner changes.  Edges whose ports no
        longer exist (e.g. port count reduced) are silently dropped.
        """
        if self.state.edge_source is None:
            return
        edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
        if not edge_data.get("node_a_idx"):
            return

        pos: dict = {}
        for kind, src in [
            ("source", self.state.source_port_source),
            ("target", self.state.target_port_source),
        ]:
            if src is None:
                continue
            d = src.data
            for i, (px, py) in enumerate(zip(list(d.get("x", [])), list(d.get("y", [])))):
                key = (kind, list(d["node_index"])[i], list(d["label"])[i])
                pos[key] = (px, py)

        new_xs, new_ys, valid = [], [], []
        for i in range(len(edge_data["node_a_idx"])):
            ka = (edge_data["a_kind"][i], edge_data["node_a_idx"][i], edge_data["a_label"][i])
            kb = (edge_data["b_kind"][i], edge_data["node_b_idx"][i], edge_data["b_label"][i])
            if ka in pos and kb in pos:
                ax, ay = pos[ka]
                bx, by = pos[kb]
                new_xs.append([ax, bx])
                new_ys.append([ay, by])
                valid.append(i)

        new_edge_data = {k: [edge_data[k][j] for j in valid] for k in edge_data}
        new_edge_data["xs"] = new_xs
        new_edge_data["ys"] = new_ys
        self.state.edge_source.data = new_edge_data

    # -----------------------------------------------------------------------
    # Port management
    # -----------------------------------------------------------------------

    def _rebuild_all_ports(self):
        """
        Recompute every port ball position from the current placed-nodes data
        and push the result into ``source_port_source`` / ``target_port_source``.

        Called whenever nodes are added, moved, deleted, or their port counts
        are changed via the spinner widgets.
        """
        if self.state.source_port_source is None or self.state.target_port_source is None:
            return

        _selected_fill_alpha, _selected_line_alpha = 0.3, 0.5

        placed_nodes_data = self.state.placed_nodes_source.data
        xs = list(placed_nodes_data.get("x", []))
        ys = list(placed_nodes_data.get("y", []))
        icon_types = list(placed_nodes_data.get("icon_type", []))
        node_types = list(placed_nodes_data.get("node_type", []))
        node_names = list(placed_nodes_data.get("name", []))
        n_sources_list = list(placed_nodes_data.get("n_sources", []))
        n_targets_list = list(placed_nodes_data.get("n_targets", []))

        source_x, source_y, source_color, source_label = [], [], [], []
        source_node_index, source_node_name, source_node_type = [], [], []
        source_fill_alpha, source_line_alpha = [], []

        target_x, target_y, target_color, target_label = [], [], [], []
        target_node_idx, target_node_name, target_node_type = [], [], []
        target_fill_alpha, target_line_alpha = [], []

        connected_ports: set = set()
        if self.state.edge_source is not None:
            edge_data = self.state.edge_source.data
            for i in range(len(edge_data.get("node_a_idx", []))):
                connected_ports.add(
                    (edge_data["a_kind"][i], edge_data["node_a_idx"][i], edge_data["a_label"][i])
                )
                connected_ports.add(
                    (edge_data["b_kind"][i], edge_data["node_b_idx"][i], edge_data["b_label"][i])
                )

        has_selected_node = self.state.selected_node_index is not None
        selected_node_index = self.state.selected_node_index

        for i, (cx, cy, icon_type, node_type, node_name) in enumerate(
            zip(xs, ys, icon_types, node_types, node_names)
        ):
            n_src = (
                int(n_sources_list[i])
                if i < len(n_sources_list)
                else self.state.default_source_count.get(node_type, 0)
            )
            n_tgt = (
                int(n_targets_list[i])
                if i < len(n_targets_list)
                else self.state.default_target_count.get(node_type, 0)
            )

            cfg = ICONS_CONFIG.get(icon_type, {})
            raw_src_color = cfg.get("source_color") or DEFAULT_COLOR
            raw_tgt_color = cfg.get("target_color") or DEFAULT_COLOR

            ports = compute_ports(cx, cy, NODE_RADIUS, PORT_RADIUS, n_src, n_tgt)
            is_selected = has_selected_node and i == selected_node_index
            fill_alpha = _selected_fill_alpha if is_selected else 0.0
            line_alpha = _selected_line_alpha if is_selected else 0.0

            for port in ports["outputs"]:
                source_x.append(port["x"])
                source_y.append(port["y"])
                source_color.append(raw_src_color)
                source_label.append(str(port["index"] + 1))
                source_node_index.append(i)
                source_node_name.append([node_name])
                source_node_type.append([node_type])
                source_fill_alpha.append(fill_alpha)
                source_line_alpha.append(line_alpha)

            for port in ports["inputs"]:
                target_x.append(port["x"])
                target_y.append(port["y"])
                target_color.append(raw_tgt_color)
                target_label.append(str(port["index"] + 1))
                target_node_idx.append(i)
                target_node_name.append([node_name])
                target_node_type.append([node_type])
                target_fill_alpha.append(fill_alpha)
                target_line_alpha.append(line_alpha)

        self.state.source_port_source.data = dict(
            x=source_x,
            y=source_y,
            color=source_color,
            label=source_label,
            node_index=source_node_index,
            node_name=source_node_name,
            node_type=source_node_type,
            fill_alpha=source_fill_alpha,
            line_alpha=source_line_alpha,
            kind=["source"] * len(source_x),
            connected=[
                "True"
                if ("source", source_node_index[i], source_label[i]) in connected_ports
                else "False"
                for i in range(len(source_x))
            ],
        )
        self.state.target_port_source.data = dict(
            x=target_x,
            y=target_y,
            color=target_color,
            label=target_label,
            node_index=target_node_idx,
            node_name=target_node_name,
            node_type=target_node_type,
            fill_alpha=target_fill_alpha,
            line_alpha=target_line_alpha,
            kind=["target"] * len(target_x),
            connected=[
                "True"
                if ("target", target_node_idx[i], target_label[i]) in connected_ports
                else "False"
                for i in range(len(target_x))
            ],
        )
        self._rebuild_edges()

        if self.state.selected_node_index is not None:
            self._refresh_connections_table(self.state.selected_node_index)

    def _best_possible_node(self, x: float, y: float):
        """
        Find the nearest placed node to canvas coordinates ``(x, y)``.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.

        :return: ``(best_idx, best_dist, current_data)`` where ``best_idx`` is the
            zero-based index of the closest node (``None`` if nothing is within snap
            distance), ``best_dist`` is the Euclidean distance, and ``current_data``
            is the raw ``placed_nodes_source.data`` dict.
        """
        current = self.state.placed_nodes_source.data
        xs = list(current.get("x", []))
        ys = list(current.get("y", []))
        if not xs:
            return None, None, current

        snap = self.icon_size
        best_idx = None
        best_dist = float("inf")
        for i, (ix, iy) in enumerate(zip(xs, ys)):
            dist = ((x - ix) ** 2 + (y - iy) ** 2) ** 0.5
            if dist < snap and dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx, best_dist, current

    # -----------------------------------------------------------------------
    # Config panel helpers
    # -----------------------------------------------------------------------

    def _populate_node_table(self, idx: int):
        """
        Fill the config panel widgets with the stored values of node *idx*.

        :param idx: Zero-based index into ``placed_nodes_source`` data arrays.
        """
        pdata = self.state.placed_nodes_source.data
        om_name = list(pdata.get("name", []))[idx]
        node_type = list(pdata.get("node_type", []))[idx]
        comp_key = list(pdata.get("icon_type", []))[idx]
        position = list(pdata.get("position", []))[idx] if pdata.get("position") else _EMPTY
        saved_options_json = list(pdata.get("options", []))[idx] if pdata.get("options") else "{}"

        self.state.options_table.visible = saved_options_json != "{}"
        self.state.name_input.value = om_name

        choices = self.state.component_type_to_icon.get(comp_key, comp_key)
        self.state.type_select.options = choices
        self.state.type_select.value = node_type if node_type in choices else choices[0]

        pos_choices = self.state.possible_position.get(node_type, [])
        self.state.position_select.options = pos_choices
        if pos_choices:
            self.state.position_select.value = (
                position if position in pos_choices else pos_choices[0]
            )
        else:
            self.state.position_select.value = ""

        saved_options: dict = {}
        try:
            saved_options = json.loads(saved_options_json) if saved_options_json else {}
        except (json.JSONDecodeError, TypeError):
            pass

        self._refresh_options_table(node_type, saved_options)

        n_src_default = self.state.default_source_count.get(node_type, 0)
        n_tgt_default = self.state.default_target_count.get(node_type, 0)
        src_editable = n_src_default == 3
        tgt_editable = n_tgt_default == 3

        current_n_src = (
            list(pdata.get("n_sources", []))[idx] if pdata.get("n_sources") else n_src_default
        )
        current_n_tgt = (
            list(pdata.get("n_targets", []))[idx] if pdata.get("n_targets") else n_tgt_default
        )

        if self.state.source_count_spinner is not None:
            self.state.source_count_spinner.visible = src_editable
            if src_editable:
                self.state.source_count_spinner.value = int(current_n_src)
        if self.state.target_count_spinner is not None:
            self.state.target_count_spinner.visible = tgt_editable
            if tgt_editable:
                self.state.target_count_spinner.value = int(current_n_tgt)
        if self.state.port_count_section is not None:
            self.state.port_count_section.visible = src_editable or tgt_editable

        if self.state.selected_node_overlay_source is not None:
            pdata = self.state.placed_nodes_source.data
            self.state.selected_node_overlay_source.data = dict(
                x=[list(pdata["x"])[idx]],
                y=[list(pdata["y"])[idx]],
            )
        self._rebuild_all_ports()
        self._refresh_connections_table(idx)
        self._refresh_symmetry_select(idx, comp_key)

    @staticmethod
    def _option_values_to_strings(value) -> str:
        """Convert an option value to its display string."""
        if value is True:
            return "True"
        if value is False:
            return "False"
        return str(value)

    @staticmethod
    def _strings_to_option_values(string: str):
        """Parse a display string back to a Python value (bool, int, float, or str)."""
        if string == "True":
            return True
        if string == "False":
            return False
        try:
            return int(string)
        except (ValueError, TypeError):
            pass
        try:
            return float(string)
        except (ValueError, TypeError):
            pass
        return string

    def _refresh_options_table(self, node_type: str, overrides: dict = None):
        """
        Rebuild the Options section of the config panel for a given node type.

        Creates one row per option (a disabled TextInput label + a Select for
        the value) and wires a shared callback so that ``options_source`` stays
        in sync with the user's current selections.

        :param node_type: Component type key used to look up ``possible_options``.
        :param overrides: Optional ``{option_name: current_value}`` mapping used
            to pre-populate widgets with previously saved values.
        """
        if overrides is None:
            overrides = {}
        options_definition = self.state.possible_options.get(node_type, {})
        option_names = list(options_definition.keys())
        opt_values = []
        new_rows = []
        value_selects = []

        for option_name, value_list in options_definition.items():
            current = (
                overrides[option_name]
                if option_name in overrides
                else (value_list[0] if value_list else _EMPTY)
            )
            current_string = self._option_values_to_strings(current)

            label_input = bkmodel.TextInput(
                value=option_name,
                width=180,
                disabled=True,
                styles={"color": "white", "font-size": "14px"},
            )
            choices = (
                [self._option_values_to_strings(choice) for choice in value_list]
                if value_list
                else [current_string]
            )
            if current_string not in choices:
                choices = [current_string] + choices

            value_select = bkmodel.Select(
                value=current_string,
                options=choices,
                width=180,
                styles={"color": "white", "font-size": "14px"},
            )
            new_rows.append(row(label_input, value_select, spacing=4))
            opt_values.append(current_string)
            value_selects.append(value_select)

        def _make_sync_callback(selects, names):
            def _on_change(attr, old, new):
                self.state.options_source.data = dict(
                    options=names,
                    value=[selected.value for selected in selects],
                )

            return _on_change

        sync_callback = _make_sync_callback(value_selects, option_names)
        for value_selected in value_selects:
            value_selected.on_change("value", sync_callback)

        if self.state.options_rows_column is not None:
            self.state.options_rows_column.children = new_rows
        self.state.options_source.data = dict(options=option_names, value=opt_values)

    def _refresh_symmetry_select(self, current_node_index: int, icon_type: str):
        """
        Populate ``symmetry_select`` with the names of all placed nodes that
        share *icon_type* with the currently selected node, excluding itself.

        :param current_node_index: Index of the node being edited.
        :param icon_type: The ``icon_type`` key of the selected node.
        """
        if self.state.symmetry_select is None:
            return

        placed_nodes_data = self.state.placed_nodes_source.data
        names = list(placed_nodes_data.get("name", []))
        icon_types = list(placed_nodes_data.get("icon_type", []))
        saved_symmetry = list(placed_nodes_data.get("symmetry_name", []))

        peers = [
            names[i]
            for i in range(len(names))
            if i != current_node_index and icon_types[i] == icon_type
        ]
        choices = [_EMPTY] + peers
        self.state.symmetry_select.options = choices

        current_symmetry_component = (
            saved_symmetry[current_node_index]
            if current_node_index < len(saved_symmetry)
            else _EMPTY
        )
        self.state.symmetry_select.value = (
            current_symmetry_component if current_symmetry_component in choices else _EMPTY
        )

    def _clear_node_table(self):
        """
        Reset all node configuration panel inputs to their empty defaults and
        hide the panel.
        """
        self.state.name_input.value = _EMPTY
        self.state.type_select.options = []
        self.state.type_select.value = _EMPTY
        self.state.position_select.options = []
        self.state.position_select.value = _EMPTY
        self.state.options_source.data = dict(options=[], value=[])
        if self.state.port_count_section is not None:
            self.state.port_count_section.visible = False
            if self.state.source_count_spinner is not None:
                self.state.source_count_spinner.visible = False
            if self.state.target_count_spinner is not None:
                self.state.target_count_spinner.visible = False
        if self.state.table_panel is not None:
            self.state.table_panel.visible = False
        if self.state.selected_node_overlay_source is not None:
            self.state.selected_node_overlay_source.data = dict(x=[], y=[])
        if self.state.connections_source is not None:
            self.state.connections_source.data = dict(my_port=[], connected_to=[], edge_idx=[])
        if self.state.connections_rows_column is not None:
            self.state.connections_rows_column.children = []
        if self.state.symmetry_select is not None:
            self.state.symmetry_select.options = [_EMPTY]
            self.state.symmetry_select.value = _EMPTY
        self._rebuild_all_ports()

    def _refresh_connections_table(self, node_idx: int):
        """
        Populate the Connections panel for node *node_idx* using dynamic rows.

        For every source port of the selected node a row is created with a
        disabled label and a Select showing compatible unconnected target ports.
        A symmetric row is created for every target port.  Selecting a new
        value immediately draws (or removes) the corresponding preview edge.

        :param node_idx: Zero-based index of the node to show connections for.
        """
        if (
            self.state.connections_rows_column is None
            or self.state.edge_source is None
            or self.state.source_port_source is None
            or self.state.target_port_source is None
        ):
            return

        source_data = {k: list(v) for k, v in self.state.source_port_source.data.items()}
        target_data = {k: list(v) for k, v in self.state.target_port_source.data.items()}
        node_data = {k: list(v) for k, v in self.state.placed_nodes_source.data.items()}
        edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}


        # Build lookup: port label → (peer_node_idx, peer_label) for every edge
        # that touches node_idx.
        connected_source: dict[str, tuple[int, str]] = {}
        connected_target: dict[str, tuple[int, str]] = {}

        for i in range(len(edge_data.get("node_a_idx", []))):
            na, nb = edge_data["node_a_idx"][i], edge_data["node_b_idx"][i]
            al, bl = edge_data["a_label"][i], edge_data["b_label"][i]
            ak, bk = edge_data["a_kind"][i], edge_data["b_kind"][i]

            if na == node_idx and ak == "source":
                connected_source[al] = (nb, bl)
            elif nb == node_idx and bk == "source":
                connected_source[bl] = (na, al)
            if na == node_idx and ak == "target":
                connected_target[al] = (nb, bl)
            elif nb == node_idx and bk == "target":
                connected_target[bl] = (na, al)

        def _peer_string(peer_node_index: int, peer_label: str, peer_kind: str) -> str:
            name = (
                node_data["name"][peer_node_index]
                if peer_node_index < len(node_data.get("name", []))
                else f"node_{peer_node_index}"
            )
            return f"{name} ({peer_kind}:{peer_label})"

        def _parse_choice(choice_string: str, candidates: list[tuple]) -> tuple | None:
            for node_index, label, kind, display_string in candidates:
                if display_string == choice_string:
                    return node_index, label, kind
            return None

        _selected_source_color = ICONS_CONFIG.get(node_data["icon_type"][node_idx], {}).get(
            "source_color", ""
        )
        free_targets: list[tuple[int, str, str, str]] = []
        for j in range(len(target_data.get("node_index", []))):
            if (
                target_data["connected"][j] == "False"
                and target_data["node_index"][j] != node_idx
                and target_data["color"][j] == _selected_source_color
            ):
                free_targets.append(
                    (
                        target_data["node_index"][j],
                        target_data["label"][j],
                        "target",
                        _peer_string(
                            target_data["node_index"][j], target_data["label"][j], "target"
                        ),
                    )
                )

        _selected_target_color = ICONS_CONFIG.get(node_data["icon_type"][node_idx], {}).get(
            "target_color", ""
        )
        free_sources: list[tuple[int, str, str, str]] = []
        for j in range(len(source_data.get("node_index", []))):
            if (
                source_data["connected"][j] == "False"
                and source_data["node_index"][j] != node_idx
                and source_data["color"][j] == _selected_target_color
            ):
                free_sources.append(
                    (
                        source_data["node_index"][j],
                        source_data["label"][j],
                        "source",
                        _peer_string(
                            source_data["node_index"][j], source_data["label"][j], "source"
                        ),
                    )
                )

        new_rows = []

        # ── Source-port rows ──────────────────────────────────────────────────
        for i in range(len(source_data.get("node_index", []))):
            if source_data["node_index"][i] != node_idx:
                continue

            src_label = source_data["label"][i]
            is_connected = source_data["connected"][i] == "True"

            label_input = bkmodel.TextInput(
                value=f"source:{src_label}",
                width=180,
                disabled=True,
                styles={"color": "white", "font-size": "14px"},
            )

            candidates = list(free_targets)
            current_str = _EMPTY
            if is_connected and src_label in connected_source:
                peer_node_i, peer_lbl = connected_source[src_label]
                current_str = _peer_string(peer_node_i, peer_lbl, "target")
                candidates = [
                    c for c in candidates if not (c[0] == peer_node_i and c[1] == peer_lbl)
                ] + [
                    (peer_node_i, peer_lbl, "target", _peer_string(peer_node_i, peer_lbl, "target"))
                ]

            choices = [_EMPTY] + [c[3] for c in candidates if c[3] != current_str]
            if current_str and current_str not in choices:
                choices = [current_str] + choices

            value_select = bkmodel.Select(
                value=current_str if current_str else _EMPTY,
                options=choices,
                width=180,
                styles={"color": "white", "font-size": "12px"},
            )

            def _make_src_callback(
                _src_label=src_label,
                _src_node_idx=node_idx,
                _src_color=source_data["color"][i],
                _src_x=source_data["x"][i],
                _src_y=source_data["y"][i],
                _candidates=candidates,
            ):
                def _on_change(attr, old, new_val):
                    self.state.pending_connections = [
                        (pa, pb)
                        for pa, pb in self.state.pending_connections
                        if not (
                            pa["kind"] == "source"
                            and pa["node_index"] == _src_node_idx
                            and pa["label"] == _src_label
                        )
                    ]
                    self._clear_temp_edge_visuals()
                    for pa, pb in list(self.state.pending_connections):
                        self._add_edge_temp(pa, pb)

                    if new_val == _EMPTY:
                        if self.state.edge_source is not None:
                            ed = {k: list(v) for k, v in self.state.edge_source.data.items()}
                            keep = [
                                i
                                for i in range(len(ed.get("node_a_idx", [])))
                                if not (
                                    (
                                        ed["node_a_idx"][i] == _src_node_idx
                                        and ed["a_label"][i] == _src_label
                                        and ed["a_kind"][i] == "source"
                                    )
                                    or (
                                        ed["node_b_idx"][i] == _src_node_idx
                                        and ed["b_label"][i] == _src_label
                                        and ed["b_kind"][i] == "source"
                                    )
                                )
                            ]
                            self.state.edge_source.data = {k: [ed[k][j] for j in keep] for k in ed}
                            self._rebuild_all_ports()
                            self._refresh_connections_table(_src_node_idx)
                        return

                    parsed = _parse_choice(new_val, _candidates)
                    if parsed is None:
                        return
                    tgt_node_i, tgt_lbl, _ = parsed

                    tgt_data = self.state.target_port_source.data
                    tgt_x, tgt_y, tgt_color = None, None, None
                    for j in range(len(tgt_data.get("node_index", []))):
                        if (
                            tgt_data["node_index"][j] == tgt_node_i
                            and tgt_data["label"][j] == tgt_lbl
                        ):
                            tgt_x = tgt_data["x"][j]
                            tgt_y = tgt_data["y"][j]
                            tgt_color = tgt_data["color"][j]
                            break
                    if tgt_x is None:
                        return

                    live_src_x, live_src_y = _src_x, _src_y
                    src_data_live = self.state.source_port_source.data
                    for j in range(len(src_data_live.get("node_index", []))):
                        if (
                            src_data_live["node_index"][j] == _src_node_idx
                            and src_data_live["label"][j] == _src_label
                        ):
                            live_src_x = src_data_live["x"][j]
                            live_src_y = src_data_live["y"][j]
                            break

                    port_a = {
                        "kind": "source",
                        "node_index": _src_node_idx,
                        "label": _src_label,
                        "x": live_src_x,
                        "y": live_src_y,
                        "color": _src_color,
                    }
                    port_b = {
                        "kind": "target",
                        "node_index": tgt_node_i,
                        "label": tgt_lbl,
                        "x": tgt_x,
                        "y": tgt_y,
                        "color": tgt_color,
                    }
                    self.state.pending_connections.append((port_a, port_b))
                    self._add_edge_temp(port_a, port_b)

                return _on_change

            value_select.on_change(
                "value",
                _make_src_callback(
                    _src_label=src_label,
                    _src_node_idx=node_idx,
                    _src_color=source_data["color"][i],
                    _src_x=source_data["x"][i],
                    _src_y=source_data["y"][i],
                    _candidates=candidates,
                ),
            )
            new_rows.append(row(label_input, value_select, spacing=4))

        # ── Target-port rows ──────────────────────────────────────────────────
        for i in range(len(target_data.get("node_index", []))):
            if target_data["node_index"][i] != node_idx:
                continue

            tgt_label = target_data["label"][i]
            is_connected = target_data["connected"][i] == "True"

            label_input = bkmodel.TextInput(
                value=f"target:{tgt_label}",
                width=180,
                disabled=True,
                styles={"color": "white", "font-size": "14px"},
            )

            candidates = list(free_sources)
            current_str = _EMPTY
            if is_connected and tgt_label in connected_target:
                peer_node_i, peer_lbl = connected_target[tgt_label]
                current_str = _peer_string(peer_node_i, peer_lbl, "source")
                candidates = [
                    c for c in candidates if not (c[0] == peer_node_i and c[1] == peer_lbl)
                ] + [
                    (peer_node_i, peer_lbl, "source", _peer_string(peer_node_i, peer_lbl, "source"))
                ]

            choices = [_EMPTY] + [c[3] for c in candidates if c[3] != current_str]
            if current_str and current_str not in choices:
                choices = [current_str] + choices

            value_select = bkmodel.Select(
                value=current_str if current_str else _EMPTY,
                options=choices,
                width=180,
                styles={"color": "white", "font-size": "12px"},
            )

            def _make_tgt_callback(
                _tgt_label=tgt_label,
                _tgt_node_idx=node_idx,
                _tgt_color=target_data["color"][i],
                _tgt_x=target_data["x"][i],
                _tgt_y=target_data["y"][i],
                _candidates=candidates,
            ):
                def _on_change(attr, old, new_val):
                    self.state.pending_connections = [
                        (pa, pb)
                        for pa, pb in self.state.pending_connections
                        if not (
                            pb["kind"] == "target"
                            and pb["node_index"] == _tgt_node_idx
                            and pb["label"] == _tgt_label
                        )
                    ]
                    self._clear_temp_edge_visuals()
                    for pa, pb in list(self.state.pending_connections):
                        self._add_edge_temp(pa, pb)

                    if new_val == _EMPTY:
                        if self.state.edge_source is not None:
                            ed = {k: list(v) for k, v in self.state.edge_source.data.items()}
                            keep = [
                                i
                                for i in range(len(ed.get("node_a_idx", [])))
                                if not (
                                    (
                                        ed["node_a_idx"][i] == _tgt_node_idx
                                        and ed["a_label"][i] == _tgt_label
                                        and ed["a_kind"][i] == "target"
                                    )
                                    or (
                                        ed["node_b_idx"][i] == _tgt_node_idx
                                        and ed["b_label"][i] == _tgt_label
                                        and ed["b_kind"][i] == "target"
                                    )
                                )
                            ]
                            self.state.edge_source.data = {k: [ed[k][j] for j in keep] for k in ed}
                            self._rebuild_all_ports()
                            self._refresh_connections_table(_tgt_node_idx)
                        return

                    parsed = _parse_choice(new_val, _candidates)
                    if parsed is None:
                        return
                    src_node_i, src_lbl, _ = parsed

                    src_data = self.state.source_port_source.data
                    src_x, src_y, src_color = None, None, None
                    for j in range(len(src_data.get("node_index", []))):
                        if (
                            src_data["node_index"][j] == src_node_i
                            and src_data["label"][j] == src_lbl
                        ):
                            src_x = src_data["x"][j]
                            src_y = src_data["y"][j]
                            src_color = src_data["color"][j]
                            break
                    if src_x is None:
                        return

                    live_tgt_x, live_tgt_y = _tgt_x, _tgt_y
                    tgt_data_live = self.state.target_port_source.data
                    for j in range(len(tgt_data_live.get("node_index", []))):
                        if (
                            tgt_data_live["node_index"][j] == _tgt_node_idx
                            and tgt_data_live["label"][j] == _tgt_label
                        ):
                            live_tgt_x = tgt_data_live["x"][j]
                            live_tgt_y = tgt_data_live["y"][j]
                            break

                    port_a = {
                        "kind": "source",
                        "node_index": src_node_i,
                        "label": src_lbl,
                        "x": src_x,
                        "y": src_y,
                        "color": src_color,
                    }
                    port_b = {
                        "kind": "target",
                        "node_index": _tgt_node_idx,
                        "label": _tgt_label,
                        "x": live_tgt_x,
                        "y": live_tgt_y,
                        "color": _tgt_color,
                    }
                    self.state.pending_connections.append((port_a, port_b))
                    self._add_edge_temp(port_a, port_b)

                return _on_change

            value_select.on_change(
                "value",
                _make_tgt_callback(
                    _tgt_label=tgt_label,
                    _tgt_node_idx=node_idx,
                    _tgt_color=target_data["color"][i],
                    _tgt_x=target_data["x"][i],
                    _tgt_y=target_data["y"][i],
                    _candidates=candidates,
                ),
            )
            new_rows.append(row(label_input, value_select, spacing=4))

        self.state.connections_rows_column.children = new_rows

        # Keep connections_source in sync
        if self.state.connections_source is not None:
            cs_my_port, cs_connected_to, cs_edge_idx = [], [], []
            for i in range(len(edge_data.get("node_a_idx", []))):
                na = edge_data["node_a_idx"][i]
                nb = edge_data["node_b_idx"][i]
                al, bl = edge_data["a_label"][i], edge_data["b_label"][i]
                ak, bk = edge_data["a_kind"][i], edge_data["b_kind"][i]

                if na == node_idx:
                    peer_name = (
                        node_data["name"][nb]
                        if nb < len(node_data.get("name", []))
                        else f"node_{nb}"
                    )
                    cs_my_port.append(f"{ak}:{al}")
                    cs_connected_to.append(f"{peer_name} ({bk}:{bl})")
                    cs_edge_idx.append(i)
                elif nb == node_idx:
                    peer_name = (
                        node_data["name"][na]
                        if na < len(node_data.get("name", []))
                        else f"node_{na}"
                    )
                    cs_my_port.append(f"{bk}:{bl}")
                    cs_connected_to.append(f"{peer_name} ({ak}:{al})")
                    cs_edge_idx.append(i)

            self.state.connections_source.data = dict(
                my_port=cs_my_port,
                connected_to=cs_connected_to,
                edge_idx=cs_edge_idx,
            )

    def _delete_selected_connection(self):
        """
        Delete the edge(s) currently selected in the Connections DataTable.

        Reads the highlighted row indices from ``connections_source.selected``,
        resolves them to actual edge indices, removes those rows from
        ``edge_source``, and refreshes the Connections panel.
        """
        if self.state.connections_source is None or self.state.edge_source is None:
            return
        selected = list(self.state.connections_source.selected.indices)
        if not selected:
            return
        edge_index_list = list(self.state.connections_source.data.get("edge_idx", []))
        to_delete = {edge_index_list[i] for i in selected if i < len(edge_index_list)}
        edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
        keep = [i for i in range(len(edge_data.get("xs", []))) if i not in to_delete]
        self.state.edge_source.data = {k: [edge_data[k][j] for j in keep] for k in edge_data}
        self._mark_unsaved()
        _LOGGER.info("Deleted connection(s) at edge indices %s", sorted(to_delete))
        if self.state.selected_node_index is not None:
            self._refresh_connections_table(self.state.selected_node_index)

    # -----------------------------------------------------------------------
    # Canvas tap handler
    # -----------------------------------------------------------------------

    def on_canvas_tap(self, event):
        """
        Handle a tap event on the main canvas.

        Behaviour depends on the current interaction mode:

        * **Delete mode** – remove the nearest placed icon or edge.
        * **Component selected** – place a new icon at the tap coordinates.
        * **Neither** – select or deselect the nearest existing node.

        :param event: Bokeh ``Tap`` event carrying ``x`` and ``y`` coordinates.
        """
        x, y = event.x, event.y

        # Port connection has highest priority in idle mode
        if not self.state.delete_mode and self.state.selected_component is None:
            nearest_port = self._find_nearest_port(x, y)
            if nearest_port is not None:
                self._handle_port_tap(nearest_port)
                return

        if self.state.delete_mode:
            best_idx, best_dist, current = self._best_possible_node(x, y)

            if best_idx is not None:
                new_data = {k: list(v) for k, v in current.items()}
                for col in new_data:
                    new_data[col].pop(best_idx)
                self.state.placed_nodes_source.data = new_data

                if self.state.hover_source is not None:
                    hdata = {k: list(v) for k, v in self.state.hover_source.data.items()}
                    for col in hdata:
                        if best_idx < len(hdata[col]):
                            hdata[col].pop(best_idx)
                    self.state.hover_source.data = hdata

                if self.state.edge_source is not None:
                    edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
                    keep = [
                        i
                        for i, (na, nb) in enumerate(
                            zip(
                                edge_data.get("node_a_idx", []),
                                edge_data.get("node_b_idx", []),
                            )
                        )
                        if na != best_idx and nb != best_idx
                    ]
                    new_edge_data = {}
                    for k, vals in edge_data.items():
                        kept = [vals[i] for i in keep]
                        if k == "node_a_idx":
                            kept = [v - 1 if v > best_idx else v for v in kept]
                        elif k == "node_b_idx":
                            kept = [v - 1 if v > best_idx else v for v in kept]
                        new_edge_data[k] = kept
                    self.state.edge_source.data = new_edge_data

                self._rebuild_all_ports()

                if self.state.selected_node_index == best_idx:
                    self.state.selected_node_index = None
                    self._clear_node_table()
                elif (
                    self.state.selected_node_index is not None
                    and self.state.selected_node_index > best_idx
                ):
                    self.state.selected_node_index -= 1

                _LOGGER.info("Deleted node at index %d", best_idx)
                self._mark_unsaved()

            else:
                edge_idx = self._find_nearest_edge(x, y)
                if edge_idx is not None and self.state.edge_source is not None:
                    edge_data = {k: list(v) for k, v in self.state.edge_source.data.items()}
                    for k in edge_data:
                        edge_data[k].pop(edge_idx)
                    self.state.edge_source.data = edge_data
                    self._mark_unsaved()
                    _LOGGER.info("Deleted edge at index %d", edge_idx)

                if self.state.selected_node_index is not None:
                    self._refresh_connections_table(self.state.selected_node_index)

            return

        if self.state.selected_component is None:
            best_idx, best_dist, current = self._best_possible_node(x, y)
            self._cancel_pending_connection()

            if best_idx is None and best_dist is None:
                return
            elif best_idx is None or self.state.selected_node_index == best_idx:
                self.state.selected_node_index = None
                self._clear_temp_edges()
                self._clear_node_table()
                return
            else:
                self._clear_temp_edges()
                self.state.selected_node_index = best_idx
                self._populate_node_table(best_idx)
                if self.state.table_panel is not None:
                    self.state.table_panel.visible = True
            return

        # Deselect any previously selected node before placing a new component
        if self.state.selected_node_index is not None:
            self.state.selected_node_index = None
            self._clear_temp_edges()
            self._clear_node_table()

        comp_key = self.state.selected_component
        count = self.state.placed_counter.get(comp_key, 0) + 1
        self.state.placed_counter[comp_key] = count
        node_name = f"{comp_key}_{count}"

        icon_path = ICONS_CONFIG[comp_key]["icon_path"]
        file_url = "file://" + str(Path(icon_path).resolve())
        b64_url = _url_to_base64(file_url)

        default_type = self.state.component_type_to_icon.get(comp_key, comp_key)[0]
        position_choices = self.state.possible_position.get(default_type, [])
        default_position = position_choices[0] if position_choices else _EMPTY

        opts_def = self.state.possible_options.get(default_type, {})
        default_opts = {
            k: (True if v_list[0] is True else (False if v_list[0] is False else v_list[0]))
            for k, v_list in opts_def.items()
            if v_list
        }
        default_opts_json = json.dumps(default_opts)

        default_n_src = self.state.default_source_count.get(default_type, 0)
        default_n_tgt = self.state.default_target_count.get(default_type, 0)

        size = self.icon_size
        current = self.state.placed_nodes_source.data
        self.state.placed_nodes_source.data = {
            "x": list(current["x"]) + [x],
            "y": list(current["y"]) + [y],
            "url": list(current["url"]) + [b64_url],
            "w": list(current["w"]) + [size],
            "h": list(current["h"]) + [size],
            "name": list(current["name"]) + [node_name],
            "icon_type": list(current.get("icon_type", [])) + [comp_key],
            "node_type": list(current.get("node_type", [])) + [default_type],
            "position": list(current.get("position", [])) + [default_position],
            "options": list(current.get("options", [])) + [default_opts_json],
            "n_sources": list(current.get("n_sources", [])) + [default_n_src],
            "n_targets": list(current.get("n_targets", [])) + [default_n_tgt],
            "symmetry_name": list(current.get("symmetry_name", [])) + [_EMPTY],
            "symmetry_node_index": list(current.get("symmetry_node_index", [])) + [-1],
        }

        if self.state.hover_source is not None:
            hdata = self.state.hover_source.data
            self.state.hover_source.data = {
                "x": list(hdata["x"]) + [x],
                "y": list(hdata["y"]) + [y],
                "name": list(hdata["name"]) + [node_name],
                "node_type": list(hdata.get("node_type", [])) + [default_type],
            }

        self._rebuild_all_ports()
        self._mark_unsaved()
        _LOGGER.info(
            "Placed %s (node_type=%s, position=%s) at (%.1f, %.1f)",
            node_name,
            default_type,
            default_position,
            x,
            y,
        )
