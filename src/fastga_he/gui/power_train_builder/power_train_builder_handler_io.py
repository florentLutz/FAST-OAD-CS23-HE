# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Save / load, end-session, and undo / redo logic.

Mixin class :class:`IOMixin` is not meant to be instantiated directly;
it is composed into :class:`PlacementHandler` via multiple inheritance.
"""

import json
import logging
from pathlib import Path
from tkinter import filedialog
import tkinter as tk
from typing import TYPE_CHECKING

from tornado.ioloop import IOLoop

from .power_train_network_writer import PowerTrainYAML
from .power_train_builder_history import make_snapshot

_LOGGER = logging.getLogger(__name__)

# For IDE type-checking only
if TYPE_CHECKING:
    from .power_train_builder_state import BuilderState


class IOMixin:
    """
    Handles serialisation (YAML + JSON), deserialisation, session termination,
    and snapshot-based undo / redo.

    Depends on ``self.state`` (a :class:`BuilderState` instance) being set by
    the concrete class before any method is called.
    """

    state: "BuilderState"

    # -----------------------------------------------------------------------
    # Save / load
    # -----------------------------------------------------------------------

    def _mark_unsaved(self):
        """
        Turn the Save button yellow to signal unsaved changes, and refresh the
        enabled / disabled appearance of the Undo and Redo buttons.

        Called by every method that mutates the canvas.  The save button
        returns to green only after a successful save.
        """
        if self.state.save_button is not None:
            self.state.save_button.button_type = "warning"
        self._refresh_undo_redo_buttons()

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
        nodes_data = {
            key: list(values) for key, values in self.state.placed_nodes_source.data.items()
        }
        edges_data = {key: list(values) for key, values in self.state.edge_source.data.items()}
        source_port_data = {
            key: list(values) for key, values in self.state.source_port_source.data.items()
        }
        target_port_data = {
            key: list(values) for key, values in self.state.target_port_source.data.items()
        }

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
                "source_ports": source_port_data,
                "target_ports": target_port_data,
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
        # Since the palette buttons are still active while loading, reset is required
        self.state.selected_node_index = None
        self.state.selected_component = None
        self.state.delete_mode = False
        self.state.pending_connections.clear()
        self.state.placed_counter.clear()
        self._clear_node_table()
        self._clear_temp_edges()

        # Read from canvas_state extract from JSON, defaulting to empty dicts if keys are missing
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

        # Fill the data sources with the loaded data
        self.state.placed_nodes_source.data = {
            key: list(value) for key, value in nodes_data.items()
        }
        self.state.edge_source.data = {key: list(value) for key, value in edges_data.items()}
        self.state.source_port_source.data = {
            key: list(value) for key, value in source_data.items()
        }
        self.state.target_port_source.data = {
            key: list(value) for key, value in target_data.items()
        }

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

        # Redraw the canvas elements based on the restored data sources
        self._rebuild_all_ports()
        self._rebuild_edges()

        if self.state.save_button is not None:
            self.state.save_button.button_type = "success"

        _LOGGER.info(
            "Canvas restored (%d nodes, %d edges).",
            len(nodes_data.get("name", [])),
            len(edges_data.get("xs", [])),
        )

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
    # Undo / redo
    # -----------------------------------------------------------------------

    def _push_undo(self):
        """
        Capture the current canvas state onto the undo stack.

        Call this **before** every action that mutates ``placed_nodes_source``,
        ``edge_source``, ``source_port_source``, or ``target_port_source`` so
        that the snapshot reflects the state the user wants to return to.
        """
        if self.state.undo_stack is None:
            return
        snapshot = make_snapshot(self.state)
        self.state.undo_stack.push(snapshot)
        self._refresh_undo_redo_buttons()

    def _restore_from_snapshot(self, snapshot: dict):
        """
        Apply a previously captured snapshot to the live canvas.

        Restores nodes, edges, ports, and the placed-counter from *snapshot*,
        then redraws all ports and edges.  The configurator panel is cleared to
        avoid stale references to nodes that may no longer exist.

        :param snapshot: Dict produced by :func:`~.power_train_builder_history.make_snapshot`.
        """
        self.state.selected_node_index = None
        self.state.selected_component = None
        self.state.delete_mode = False
        self.state.pending_connections.clear()
        self._clear_node_table()
        self._clear_temp_edges()

        nodes_data = snapshot.get("nodes", {})
        edges_data = snapshot.get("edges", {})
        source_data = snapshot.get("source_ports", {})
        target_data = snapshot.get("target_ports", {})

        self.state.placed_nodes_source.data = {
            key: list(value) for key, value in nodes_data.items()
        }
        self.state.edge_source.data = {key: list(value) for key, value in edges_data.items()}
        self.state.source_port_source.data = {
            key: list(value) for key, value in source_data.items()
        }
        self.state.target_port_source.data = {
            key: list(value) for key, value in target_data.items()
        }

        self.state.hover_source.data = {
            "x": list(nodes_data.get("x", [])),
            "y": list(nodes_data.get("y", [])),
            "name": list(nodes_data.get("name", [])),
            "node_type": list(nodes_data.get("node_type", [])),
        }

        self.state.placed_counter.clear()
        self.state.placed_counter.update(snapshot.get("placed_counter", {}))

        self._rebuild_all_ports()

        _LOGGER.info(
            "Snapshot restored (%d nodes, %d edges).",
            len(nodes_data.get("name", [])),
            len(edges_data.get("xs", [])),
        )

    def _on_undo(self):
        """
        Undo the last canvas mutation.

        Pushes the current state onto the redo stack, then pops and restores
        the most recent undo snapshot.  The save button turns yellow because
        the canvas is now different from the last saved state.
        """
        if self.state.undo_stack is None or not self.state.undo_stack.can_undo:
            return

        # Save the current state for redo before wiping it.
        current_snapshot = make_snapshot(self.state)
        self.state.undo_stack.push_redo(current_snapshot)

        snapshot = self.state.undo_stack.undo()
        if snapshot is None:
            return

        self._restore_from_snapshot(snapshot)
        self._mark_unsaved()
        _LOGGER.info(
            "Undo applied; undo depth now %d.",
            len(self.state.undo_stack),
        )

    def _on_redo(self):
        """
        Re-apply the most recently undone canvas mutation.

        Pushes the current state onto the undo stack, then pops and restores
        the top redo snapshot.  The save button turns yellow because the canvas
        is now different from the last saved state.
        """
        if self.state.undo_stack is None or not self.state.undo_stack.can_redo:
            return

        # Capture current state onto the undo stack directly (bypassing push()
        # which would wipe the redo stack).
        current_snapshot = make_snapshot(self.state)
        self.state.undo_stack._undo.append(current_snapshot)

        snapshot = self.state.undo_stack.redo()
        if snapshot is None:
            return

        self._restore_from_snapshot(snapshot)
        self._mark_unsaved()
        _LOGGER.info(
            "Redo applied; undo depth now %d.",
            len(self.state.undo_stack),
        )

    def _refresh_undo_redo_buttons(self):
        """
        Update the visual state of the Undo and Redo buttons to reflect
        whether actions are available.

        Uses ``"primary"`` when active and ``"default"`` (dimmed) when the
        corresponding stack is empty.
        """
        if self.state.undo_stack is None:
            return
        if self.state.undo_button is not None:
            self.state.undo_button.button_type = (
                "primary" if self.state.undo_stack.can_undo else "default"
            )
        if self.state.redo_button is not None:
            self.state.redo_button.button_type = (
                "primary" if self.state.undo_stack.can_redo else "default"
            )

    # -----------------------------------------------------------------------
    # End session
    # -----------------------------------------------------------------------

    def _end_session(self):
        """
        Stop the Bokeh IO loop, terminating the server session.

        Never called directly from a button ``on_click``; always invoked via
        :meth:`_on_end_session_trigger` so that the JS gate (which checks the
        save-button colour) has already run before Python acts.
        """
        _LOGGER.info("Ending session and stopping server")
        IOLoop.current().stop()

    def _on_end_session_trigger(self, attr, old, new):
        """
        Fire :meth:`_end_session` in response to the ``end_session_trigger``
        TextInput toggle.

        JS flips this trigger in exactly two situations:

        * The End Session button is clicked **and** the design is already saved
          (save button is green).
        * The **End Anyway** button in the unsaved-exit overlay is clicked.

        By routing through a trigger rather than ``on_click`` we guarantee that
        Python's stop call only runs when JS has already decided it is safe to
        do so.
        """
        self._end_session()

    def _on_end_session_save_trigger(self, attr, old, new):
        """
        Triggered by the ``end_session_save_trigger`` TextInput when the user
        clicks **Save & Exit** in the unsaved-exit overlay.

        The JS side has already opened the watcher-path overlay for the user
        to fill in; once they click **Continue to Save** the
        ``browse_save_trigger`` fires ``_on_browse_save``.  We simply set an
        internal flag here so that ``_on_browse_save`` knows it should call
        ``_end_session`` after saving.
        """
        _LOGGER.info("Save & Exit requested – will end session after save completes.")
        self._pending_exit_after_save = True

    def _on_browse_save(self, attr, old, new):
        """
        Open native OS save-file dialogs (tkinter) for the YAML config and JSON
        backup, then write both files.  The watcher file path is read from
        ``state.watcher_path_input`` which the user filled in the overlay.

        If ``_pending_exit_after_save`` is ``True`` (set by
        ``_on_end_session_save_trigger``), the session is ended automatically
        after a successful save.

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

        if getattr(self, "_pending_exit_after_save", False):
            self._pending_exit_after_save = False
            _LOGGER.info("Save & Exit: save complete – signalling browser then ending session.")
            # Flip close_window_trigger BEFORE stopping the IOLoop so Bokeh can
            # push the value change to the browser (which fires window.close() via
            # js_on_change).  _end_session() calls IOLoop.stop(), after which no
            # further document updates can be delivered to the client.
            if self.state.close_window_trigger is not None:
                self.state.close_window_trigger.value = (
                    "0" if self.state.close_window_trigger.value == "1" else "1"
                )
            # Defer the actual IOLoop stop by one tick so Bokeh has time to flush
            # the trigger update to the WebSocket before the server shuts down.
            IOLoop.current().call_later(0.3, self._end_session)
