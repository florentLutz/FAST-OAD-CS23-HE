# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Startup overlay, New Design, delete mode and toolbox selection.

These methods set the mode the canvas is in (startup choice, delete mode,
armed toolbox component). They prepare what the next canvas click will do, but
never place, edit or delete a component themselves, except New Design, which
clears the canvas.

Mixin class :class:`CanvasModeMixin` is not meant to be instantiated directly;
it is composed into :class:`BuilderHandler` via multiple inheritance.
"""

import logging

from fastga_he.gui.power_train_network_viewer import _string_cleanup
from .power_train_builder_state import (
    BUTTON_DEFAULT_COLOR_TYPE,
    BUTTON_SELECTED_COLOR_TYPE,
    BUILDER_ICONS_CONFIG,
    EDGE_COLUMNS,
    NODE_COLUMNS,
    STATUS_IDLE_HTML,
    empty_columns,
    BuilderState,
)

_LOGGER = logging.getLogger(__name__)


class CanvasModeMixin:
    """
    Handles the startup overlay choices, toolbox selection and delete mode.

    Placing a component is done by
    :class:`~.power_train_builder_handler_placement.PlacementMixin`.

    Depends on ``self.state`` (a :class:`BuilderState` instance) being set by
    the concrete class before any method is called.
    """

    state: "BuilderState"

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
        self.state.new_design_button.visible = False
        self.state.load_design_button.visible = False

    def _on_new_design(self):
        """
        Start a blank design. Callback of the New Design button.

        Hides the startup overlay, returns the canvas to idle, empties the
        design and the watcher path, and clears the undo / redo history. The
        Save button turns green: there is nothing to save.
        """
        self._dismiss_startup_overlay()
        self._reset_interaction_state()

        self.state.placed_nodes_source.data = empty_columns(NODE_COLUMNS)
        self.state.edge_source.data = empty_columns(EDGE_COLUMNS)
        self.state.placed_counter.clear()
        self._set_watcher_path("")
        # No nodes left: this empties the port sources
        self._rebuild_all_ports()

        # Clear edit history: a new design has no prior edits
        self.state.edit_history_stack.clear()
        self._refresh_undo_redo_buttons()

        self.state.save_button.button_type = "success"
        # Back to the default view of the empty design area
        self._request_view_fit()

        _LOGGER.info("New Design started – canvas cleared.")

    # -----------------------------------------------------------------------
    # Interaction state
    # -----------------------------------------------------------------------

    def _reset_interaction_state(self):
        """
        Return the canvas to idle: no component selected or armed for placement,
        no connection waiting for a second click or for Apply, delete mode off.

        Called when entering delete mode, and before the canvas content is
        replaced (New Design, load, undo, redo): after a replacement, a stored
        node index may point at another component.
        """
        self._cancel_pending_connection()
        # Also clears panel_pending_connections
        self._clear_temp_edges()
        self.state.selected_node_index = None
        self._clear_node_table()
        self._disarm_toolbox()
        self._exit_delete_mode()

    def _disarm_toolbox(self):
        """
        Deselect the armed toolbox component, if any.

        Resets every toolbox button and the status line.
        """
        self.state.selected_component = None
        for toolbox_button in self.state.buttons:
            toolbox_button.button_type = BUTTON_DEFAULT_COLOR_TYPE
        self.state.status_div.text = STATUS_IDLE_HTML

    # -----------------------------------------------------------------------
    # Delete mode
    # -----------------------------------------------------------------------

    def _toggle_delete_mode(self):
        """
        Toggle delete mode on / off, updating button styling and status text.

        Entering delete mode first returns the canvas to idle (see
        :meth:`_reset_interaction_state`), then the Delete button turns red and
        the status label prompts the user to click a node or edge to remove it.
        """
        if self.state.delete_mode:
            self._exit_delete_mode()
            return

        self._reset_interaction_state()
        self.state.delete_mode = True
        self.state.status_div.text = (
            "<b style='color:#FF4444;font-size:14pt'>Delete mode: "
            "click an icon / a connection to remove it</b>"
        )
        self.state.delete_button.button_type = "danger"

    def _exit_delete_mode(self):
        """
        Turn delete mode off and restore the Delete button and the status line.

        Does nothing if delete mode is already off.
        """
        if not self.state.delete_mode:
            return
        self.state.delete_mode = False
        self.state.status_div.text = STATUS_IDLE_HTML
        self.state.delete_button.button_type = BUTTON_DEFAULT_COLOR_TYPE

    # -----------------------------------------------------------------------
    # Toolbox component selection
    # -----------------------------------------------------------------------

    def on_toolbox_select(self, component_index: int):
        """
        Arm the toolbox component at *component_index* for placement, or disarm it.

        Callback of every toolbox button: a second click on the armed button
        disarms it. Arming another component cancels a pending two-click
        connection and leaves delete mode; the selected node stays selected
        until the component is placed.

        :param component_index: Zero-based index into ``list(BUILDER_ICONS_CONFIG)``;
            an index out of range is ignored.
        """
        icon_keys = list(BUILDER_ICONS_CONFIG)
        if component_index < 0 or component_index >= len(icon_keys):
            return

        # If the user clicks the same component again, deselect it
        if self.state.selected_component == icon_keys[component_index]:
            self.state.selected_component = None
            self.state.buttons[component_index].button_type = BUTTON_DEFAULT_COLOR_TYPE
            self.state.status_div.text = STATUS_IDLE_HTML
            return

        # If the user clicks a different component, cancel any pending connection and select the
        # new component
        self._cancel_pending_connection()
        self.state.selected_component = icon_keys[component_index]

        self._exit_delete_mode()

        # Find the previously selected button (if any) and reset its color to default
        previously_selected_button = next(
            (
                toolbox_button
                for toolbox_button in self.state.buttons
                if toolbox_button.button_type == BUTTON_SELECTED_COLOR_TYPE
            ),
            None,
        )
        if previously_selected_button is not None:
            previously_selected_button.button_type = BUTTON_DEFAULT_COLOR_TYPE

        # Highlight the newly selected button
        self.state.buttons[component_index].button_type = BUTTON_SELECTED_COLOR_TYPE

        label = _string_cleanup(self.state.selected_component)
        self.state.status_div.text = f"<b style='color:#FFD700;font-size:14pt'>Placing: {label}</b>"
