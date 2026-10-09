# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Startup-overlay, new-design, delete-mode, and toolbox-selection logic.

This is the "what interaction mode is the canvas in" module: every method
here sets up state for a *future* tap (which overlay button was pressed,
whether delete mode is on, which toolbox component is armed) rather than
mutating the canvas itself.

Mixin class :class:`CanvasModeMixin` is not meant to be instantiated directly;
it is composed into :class:`BuilderHandler` via multiple inheritance.
"""

import logging

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG, _string_cleanup
from .power_train_builder_state import (
    BUTTON_DEFAULT_COLOR_TYPE,
    BUTTON_SELECTED_COLOR_TYPE,
    EDGE_COLUMNS,
    HOVER_COLUMNS,
    NODE_COLUMNS,
    empty_columns,
    BuilderState,
)

_LOGGER = logging.getLogger(__name__)


class CanvasModeMixin:
    """
    Handles the startup overlay choices, toolbox button selection, and delete-mode toggling.

    Every method on this mixin arms or disarms a mode that a *subsequent*
    canvas tap will act on — it never places, deletes, or edits anything on
    the canvas itself. Actual node placement lives in
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
        Start with a blank canvas.

        Hides the startup overlay and resets every data source to empty so the
        user begins with a completely fresh powertrain design.  The undo / redo
        history is also cleared so a new session starts with a clean stack.
        """
        self._dismiss_startup_overlay()

        self.state.placed_nodes_source.data = empty_columns(NODE_COLUMNS)
        self.state.hover_source.data = empty_columns(HOVER_COLUMNS)
        self.state.edge_source.data = empty_columns(EDGE_COLUMNS)
        # The port sources are emptied by the _rebuild_all_ports call in
        # _clear_node_table below, as there are no nodes left
        self._clear_temp_edges()
        self.state.placed_counter.clear()
        self.state.selected_node_index = None
        self.state.selected_component = None
        self.state.delete_mode = False
        self._set_watcher_path("")
        self._clear_node_table()

        # Clear edit history: a new design has no prior edits
        if self.state.edit_history_stack is not None:
            self.state.edit_history_stack.clear()
        self._refresh_undo_redo_buttons()

        self.state.save_button.button_type = "success"
        # Back to the default view of the empty design area
        self._request_view_fit()

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
            for toolbox_button in self.state.buttons:
                toolbox_button.button_type = BUTTON_DEFAULT_COLOR_TYPE
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

    def _exit_delete_mode(self):
        """
        Exit delete mode unconditionally, restoring button styling and status text.

        Safe to call even when delete mode is already inactive (no-op in that case).
        Unlike :meth:`_toggle_delete_mode`, this never flips delete mode *on* and
        has no side-effects beyond resetting the relevant UI elements.
        """
        if not self.state.delete_mode:
            return
        self.state.delete_mode = False
        self.state.status_div.text = "<i style='color:#aaa;font-size:14pt'>Select a component</i>"
        self.state.delete_button.button_type = BUTTON_DEFAULT_COLOR_TYPE

    # -----------------------------------------------------------------------
    # Toolbox component selection
    # -----------------------------------------------------------------------

    def on_toolbox_select(self, component_index: int):
        """
        Select the component at position *component_index* in :data:`ICONS_CONFIG`.

        A second click on the same button deselects it. Updates button styling
        and the status label accordingly. Can be called programmatically in
        tests without a running server.

        :param component_index: Zero-based index into ``list(ICONS_CONFIG.keys())``.
        """
        icon_keys = list(ICONS_CONFIG.keys())
        if component_index < 0 or component_index >= len(icon_keys):
            return

        # If the user clicks the same component again, deselect it
        if self.state.selected_component == icon_keys[component_index]:
            self.state.selected_component = None
            self.state.buttons[component_index].button_type = BUTTON_DEFAULT_COLOR_TYPE
            self.state.status_div.text = (
                "<i style='color:#aaa;font-size:14pt'>Select a component</i>"
            )
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
