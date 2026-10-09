# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Save / load, end-session, and undo / redo logic.

A design is saved as a single YAML file: the FAST-OAD powertrain
configuration, plus (unless the user saves it as a final version) a GUI-only
``canvas_layout`` section holding the canvas position of each component.
Loading that YAML restores the canvas (see
:class:`~.power_train_network_reader.PowerTrainYAMLReader`); without the
section, the components are arranged automatically.

Mixin class :class:`IOMixin` is not meant to be instantiated directly;
it is composed into :class:`BuilderHandler` via multiple inheritance.
"""

import html
import logging
from pathlib import Path
from tkinter import filedialog
import tkinter as tk

from tornado.ioloop import IOLoop

from .power_train_network_writer import PowerTrainYAMLWriter
from .power_train_network_reader import PowerTrainYAMLReader
from .power_train_builder_history import make_snapshot
from .power_train_builder_state import (
    SAVE_MODE_CANCEL,
    SAVE_MODE_FINAL,
    copy_column_data,
    BuilderState,
)

_YAML_SUFFIXES = (".yml", ".yaml")

_LOGGER = logging.getLogger(__name__)


class IOMixin:
    """
    Handles YAML serialisation / de-serialisation, session termination, and
    snapshot-based undo / redo.

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

        Called by every method that mutates the canvas. The save button returns
        to green only after a successful save.
        """
        self.state.save_button.button_type = "warning"
        self._refresh_undo_redo_buttons()

    def _save_canvas_state(self, yaml_path: str = "", include_canvas_layout: bool = True) -> bool:
        """
        Write the current canvas to *yaml_path*.

        The file is the FAST-OAD powertrain configuration, plus a GUI-only
        ``canvas_layout`` section when *include_canvas_layout* is ``True``, so
        it is both the analysis input and the file to re-open for later edits.
        The Save button only turns green when the write actually succeeds.

        :param yaml_path: File name (or full path) for the YAML config. An
            empty string skips the save.
        :param include_canvas_layout: ``False`` for a "final version" file
            without the ``canvas_layout`` section.
        :return: ``True`` if the file was written.
        """
        if not yaml_path:
            return False

        nodes_data = copy_column_data(self.state.placed_nodes_source.data)
        edges_data = copy_column_data(self.state.edge_source.data)
        source_port_data = copy_column_data(self.state.source_port_source.data)
        target_port_data = copy_column_data(self.state.target_port_source.data)

        yaml_file = Path(yaml_path)
        try:
            yaml_file.parent.mkdir(parents=True, exist_ok=True)
            pt_yaml = PowerTrainYAMLWriter(
                self.state,
                node_data=nodes_data,
                edge_data=edges_data,
                source_port_data=source_port_data,
                target_port_data=target_port_data,
                include_canvas_layout=include_canvas_layout,
            )
            pt_yaml.set_title(yaml_file.stem)
            for node_index in range(len(nodes_data.get("name", []))):
                pt_yaml.add_component(node_index)
            pt_yaml.add_connection()
            # Watcher CSV path from the toolbox field (or the loaded YAML), if any
            if self.state.watcher_file_path:
                pt_yaml.set_watcher_file_path(self.state.watcher_file_path)
            pt_yaml.write(str(yaml_file))
        except Exception:
            _LOGGER.exception("Failed to write YAML config to %s.", yaml_file)
            self._show_status_error(f"Could not save {yaml_file.name}; see the log for details.")
            return False

        _LOGGER.info(
            "Powertrain YAML config saved to %s (%s).",
            yaml_file,
            "with canvas layout" if include_canvas_layout else "final version, no canvas layout",
        )
        self.state.save_button.button_type = "success"
        return True

    def _load_canvas_state(self, file_path: str):
        """
        Restore the canvas from a saved design.

        Only ``.yml`` / ``.yaml`` files are accepted; they are read with
        :class:`~.power_train_network_reader.PowerTrainYAMLReader`.

        :param file_path: Path to the file chosen by the user.
        """
        input_file = Path(file_path)
        if not input_file.exists():
            _LOGGER.error("Load failed – file not found: %s", input_file)
            self._show_status_error(f"Could not load {input_file.name}: file not found.")
            return

        if input_file.suffix.lower() not in _YAML_SUFFIXES:
            _LOGGER.error("Load failed – not a YAML file: %s", input_file)
            self._show_status_error(f"Could not load {input_file.name}: not a YAML file.")
            return

        try:
            canvas_state = PowerTrainYAMLReader(self.state, self.icon_size).read(input_file)
        except Exception:
            _LOGGER.exception("Load failed – could not parse %s", input_file)
            self._show_status_error(f"Could not load {input_file.name}; see the log for details.")
            return

        _LOGGER.info("Loading canvas state from file: %s", input_file)
        self._restore_canvas_from_dict(canvas_state)

    def _show_status_error(self, message: str):
        """
        Show an error in the status line above the canvas.

        Load and save failures are otherwise only logged, so the user would see
        nothing happen. The next mode change replaces the message.

        :param message: Plain-text message (HTML special characters are escaped).
        """
        if self.state.status_div is not None:
            self.state.status_div.text = (
                f"<b style='color:#FF4444;font-size:14pt'>{html.escape(message)}</b>"
            )

    def _restore_canvas_from_dict(self, canvas_state: dict):
        """
        Replace the canvas with the design read from a YAML file.

        Clears the current canvas entirely, then replays every node and edge.
        Port balls (and edge coordinates) are always recomputed from the nodes
        by ``_rebuild_all_ports``, so they do not need to be in the file.

        :param canvas_state: Dict returned by
            :meth:`~.power_train_network_reader.PowerTrainYAMLReader.read`, with
            keys ``"components"`` (every node column), ``"connections"`` (edge
            columns) and ``"watcher_file_path"``.
        """
        # Since the toolbox buttons are still active while loading, reset is required
        self.state.selected_node_index = None
        self.state.selected_component = None
        self.state.delete_mode = False
        self.state.panel_pending_connections.clear()
        self.state.placed_counter.clear()
        self._clear_node_table()
        self._clear_temp_edges()

        # Defaulting to empty dicts if keys are missing
        nodes_data = canvas_state.get("components", {})
        edges_data = canvas_state.get("connections", {})

        # Fill the data sources with the loaded data
        self.state.placed_nodes_source.data = copy_column_data(nodes_data)
        self.state.edge_source.data = copy_column_data(edges_data)

        self.state.hover_source.data = {
            "x": list(nodes_data.get("x", [])),
            "y": list(nodes_data.get("y", [])),
            "name": list(nodes_data.get("name", [])),
            "node_type": list(nodes_data.get("node_type", [])),
        }

        # Show the YAML's watcher_file_path (if any) so a re-save keeps it
        self._set_watcher_path(str(canvas_state.get("watcher_file_path") or ""))

        # Rebuild placed_counter so future placements continue from the right index
        for node_name in nodes_data.get("name", []):
            name_parts = node_name.rsplit("_", 1)
            if len(name_parts) == 2 and name_parts[1].isdigit():
                icon_key = name_parts[0]
                self.state.placed_counter[icon_key] = max(
                    self.state.placed_counter.get(icon_key, 0), int(name_parts[1])
                )

        # Size the hover / selection rings and labels for the restored nodes
        self._apply_node_icon_size()

        # Make the browser re-fit the view so every loaded component is visible
        self._request_view_fit()

        # Redraw ports from the restored nodes; this also recomputes every
        # edge's xs / ys and drops edges pointing at non-existent ports
        self._rebuild_all_ports()

        self.state.save_button.button_type = "success"

        _LOGGER.info(
            "Canvas restored (%d nodes, %d edges).",
            len(nodes_data.get("name", [])),
            len(edges_data.get("xs", [])),
        )

    def _request_view_fit(self):
        """
        Ask the browser to re-fit the canvas view (see the launcher's
        ``_JS_FIT_VIEW``) by changing ``fit_view_trigger``. No-op without it.
        """
        if self.state.fit_view_trigger is not None:
            self.state.fit_view_trigger.value = (
                "0" if self.state.fit_view_trigger.value == "1" else "1"
            )

    def _on_browse_load(self, attr, old, new):
        """
        Open a native OS open-file dialog (tkinter) for loading a saved
        powertrain YAML, then restore the canvas from the chosen file.

        Triggered by the hidden ``browse_load_trigger`` TextInput toggling.
        """
        selected_file_path = self._open_tkinter_dialog(
            filedialog.askopenfilename,
            title="Load powertrain design",
            filetypes=[
                ("Powertrain YAML", "*.yml *.yaml"),
                ("All files", "*.*"),
            ],
        )
        if not selected_file_path:
            return
        self._dismiss_startup_overlay()
        self._load_canvas_state(selected_file_path)

    def _on_browse_watcher(self, attr, old, new):
        """
        Open a native OS save-file dialog (tkinter) to choose the watcher CSV path.

        The chosen path goes into ``state.watcher_path_input``; its change
        callback (:meth:`_on_watcher_path_change`) stores it as the
        ``watcher_file_path`` written into the YAML on the next save.

        Triggered by the hidden ``browse_watcher_trigger`` TextInput toggling.

        :param attr: Bokeh attribute name (unused, required by protocol).
        :param old: Previous trigger value (unused).
        :param new: New trigger value (unused).
        """
        watcher_csv_path = self._open_tkinter_dialog(
            filedialog.asksaveasfilename,
            title="Choose watcher CSV file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if watcher_csv_path and self.state.watcher_path_input is not None:
            self.state.watcher_path_input.value = watcher_csv_path
            _LOGGER.info("Watcher CSV path set to: %s", watcher_csv_path)

    def _on_watcher_path_change(self, attr, old, new):
        """
        Store the watcher CSV path edited in ``state.watcher_path_input``.

        The path is saved as ``watcher_file_path`` in the YAML (an empty field
        leaves it out). A real change marks the design as unsaved; the value
        set by a load or New Design matches the stored path, so it does not.

        :param attr: Bokeh attribute name (unused, required by protocol).
        :param old: Previous field value (unused).
        :param new: New field value.
        """
        watcher_csv_path = (new or "").strip()
        if watcher_csv_path == self.state.watcher_file_path:
            return
        self.state.watcher_file_path = watcher_csv_path
        self._mark_unsaved()

    def _set_watcher_path(self, watcher_csv_path: str):
        """
        Set the watcher CSV path without marking the design as unsaved.

        Used by load and New Design: the stored path is updated first, so the
        field's change callback sees no change.

        :param watcher_csv_path: Path to show and save, or an empty string.
        """
        self.state.watcher_file_path = watcher_csv_path
        if self.state.watcher_path_input is not None:
            self.state.watcher_path_input.value = watcher_csv_path

    @staticmethod
    def _open_tkinter_dialog(dialog_function, **kwargs):
        """
        Run a tkinter file dialog on the calling thread and return the result.

        :param dialog_function: The ``filedialog`` function to call
            (e.g. ``filedialog.asksaveasfilename``).
        :param kwargs: Keyword arguments forwarded to *dialog_function*.
        :return: The chosen path string, or ``""`` if the user cancelled.
        """
        tk_root = tk.Tk()
        tk_root.withdraw()
        tk_root.attributes("-topmost", True)
        try:
            result = dialog_function(parent=tk_root, **kwargs)
        finally:
            tk_root.destroy()
        return result or ""

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
        if self.state.edit_history_stack is None:
            return
        snapshot = make_snapshot(self.state)
        self.state.edit_history_stack.push_undo(snapshot)
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
        self._exit_delete_mode()
        self.state.panel_pending_connections.clear()
        self._clear_node_table()
        self._clear_temp_edges()

        nodes_data = snapshot.get("nodes", {})
        edges_data = snapshot.get("edges", {})
        source_data = snapshot.get("source_ports", {})
        target_data = snapshot.get("target_ports", {})

        self.state.placed_nodes_source.data = copy_column_data(nodes_data)
        self.state.edge_source.data = copy_column_data(edges_data)
        self.state.source_port_source.data = copy_column_data(source_data)
        self.state.target_port_source.data = copy_column_data(target_data)

        self.state.hover_source.data = {
            "x": list(nodes_data.get("x", [])),
            "y": list(nodes_data.get("y", [])),
            "name": list(nodes_data.get("name", [])),
            "node_type": list(nodes_data.get("node_type", [])),
        }

        self.state.placed_counter.clear()
        self.state.placed_counter.update(snapshot.get("placed_counter", {}))

        # The icon size is a display setting, not part of the undo history:
        # keep the slider's current size even if the snapshot predates it
        self._apply_node_icon_size()
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
        if self.state.edit_history_stack is None or not self.state.edit_history_stack.can_undo:
            return

        # Save the current state for redo before wiping it
        current_snapshot = make_snapshot(self.state)
        self.state.edit_history_stack.push_redo(current_snapshot)

        snapshot = self.state.edit_history_stack.undo()
        if snapshot is None:
            return

        self._exit_delete_mode()
        self._restore_from_snapshot(snapshot)
        self._mark_unsaved()
        _LOGGER.info(
            "Undo applied; undo depth now %d.",
            len(self.state.edit_history_stack),
        )

    def _on_redo(self):
        """
        Re-apply the most recently undone canvas mutation.

        Pushes the current state onto the undo stack, then pops and restores
        the top redo snapshot.  The save button turns yellow because the canvas
        is now different from the last saved state.
        """
        if self.state.edit_history_stack is None or not self.state.edit_history_stack.can_redo:
            return

        # Save the current state for undo, keeping the remaining redo steps
        current_snapshot = make_snapshot(self.state)
        self.state.edit_history_stack.push_undo_keep_redo(current_snapshot)

        snapshot = self.state.edit_history_stack.redo()
        if snapshot is None:
            return

        self._exit_delete_mode()
        self._restore_from_snapshot(snapshot)
        self._mark_unsaved()
        _LOGGER.info(
            "Redo applied; undo depth now %d.",
            len(self.state.edit_history_stack),
        )

    def _refresh_undo_redo_buttons(self):
        """
        Update the visual state of the Undo and Redo buttons to reflect
        whether actions are available.

        Uses ``"primary"`` when active and ``"default"`` (dimmed) when the
        corresponding stack is empty.
        """
        if self.state.edit_history_stack is None:
            return

        self.state.undo_button.button_type = (
            "primary" if self.state.edit_history_stack.can_undo else "default"
        )

        self.state.redo_button.button_type = (
            "primary" if self.state.edit_history_stack.can_redo else "default"
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

        The JS side has already opened the "final version?" overlay; once the
        user picks **Final Version** or **Edit Later**, the
        ``browse_save_trigger`` fires ``_on_browse_save``.  We simply set an
        internal flag here so that ``_on_browse_save`` knows it should call
        ``_end_session`` after saving.
        """
        _LOGGER.info("Save & Exit requested – will end session after save completes.")
        self._pending_exit_after_save = True

    def _on_browse_save(self, attr, old, new):
        """
        Open a native OS save-file dialog (tkinter) for the YAML config, then
        write it.

        Fired by the "final version?" overlay, which sets the trigger to
        ``"<mode>:<timestamp>"`` (the timestamp guarantees a value change on
        every click; the mode travels in the same change, so it cannot be read
        out of order):

        * ``"final"`` -> FAST-OAD configuration only, no ``canvas_layout``.
        * ``"editable"`` (or any other value, e.g. a legacy ``"0"`` / ``"1"``
          toggle) -> configuration + ``canvas_layout`` for later edits.
        * ``"cancel"`` -> no dialog, nothing saved; also aborts a pending
          Save & Exit so a later ordinary save does not close the session.

        Final Version and Edit Later first commit the watcher CSV field of the
        overlay, so ``state.watcher_file_path`` is up to date when the file is
        written (see :meth:`_on_watcher_path_change`).

        If ``_pending_exit_after_save`` is ``True`` (set by
        ``_on_end_session_save_trigger``), the session is ended automatically
        after a successful save. If the dialog is cancelled or the write
        fails, the session stays open so no work is lost.

        Triggered by the hidden ``browse_save_trigger`` TextInput toggling.
        """
        save_mode = str(new).split(":", 1)[0]
        if save_mode == SAVE_MODE_CANCEL:
            if getattr(self, "_pending_exit_after_save", False):
                _LOGGER.info("Save & Exit cancelled – keeping the session open.")
            self._pending_exit_after_save = False
            return

        include_canvas_layout = save_mode != SAVE_MODE_FINAL
        yaml_path = self._open_tkinter_dialog(
            filedialog.asksaveasfilename,
            title=(
                "Save powertrain design (YAML, editable)"
                if include_canvas_layout
                else "Save powertrain design (YAML, final version)"
            ),
            defaultextension=".yml",
            filetypes=[("YAML files", "*.yml *.yaml"), ("All files", "*.*")],
        )
        saved = self._save_canvas_state(
            yaml_path=yaml_path, include_canvas_layout=include_canvas_layout
        )

        if getattr(self, "_pending_exit_after_save", False):
            self._pending_exit_after_save = False
            if not saved:
                _LOGGER.info("Save & Exit: nothing was saved – keeping the session open.")
                return
            _LOGGER.info("Save & Exit: save complete – signalling browser then ending session.")
            # Flip close_window_trigger BEFORE stopping the IOLoop so Bokeh can
            # push the value change to the browser (which fires window.close() via
            # js_on_change).  _end_session() calls IOLoop.stop(), after which no
            # further document updates can be delivered to the client
            if self.state.close_window_trigger is not None:
                self.state.close_window_trigger.value = (
                    "0" if self.state.close_window_trigger.value == "1" else "1"
                )
            # Defer the actual IOLoop stop by one tick so Bokeh has time to flush
            # the trigger update to the WebSocket before the server shuts down
            IOLoop.current().call_later(0.3, self._end_session)
