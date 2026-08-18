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
from typing import TYPE_CHECKING

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG, _string_cleanup
from .power_train_builder_state import (
    BUTTON_DEFAULT_COLOR_TYPE,
    BUTTON_SELECTED_COLOR_TYPE,
)

_LOGGER = logging.getLogger(__name__)

# For IDE type-checking only
if TYPE_CHECKING:
    from .power_train_builder_state import BuilderState


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
            starting_node_index=[],
            starting_port_label=[],
            starting_port_kind=[],
            ending_node_index=[],
            ending_port_label=[],
            ending_port_kind=[],
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

        # Clear edit history: a new design has no prior edits.
        if self.state.edit_history_stack is not None:
            self.state.edit_history_stack.clear()
        self._refresh_undo_redo_buttons()

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

        if self.state.selected_component == icon_keys[component_index]:
            self.state.selected_component = None
            self.state.buttons[component_index].button_type = BUTTON_DEFAULT_COLOR_TYPE
            self.state.status_div.text = (
                "<i style='color:#aaa;font-size:14pt'>Select a component</i>"
            )
            return

        self._cancel_pending_connection()
        self.state.selected_component = icon_keys[component_index]

        self._exit_delete_mode()

        for button_position, toolbox_button in enumerate(self.state.buttons):
            toolbox_button.button_type = (
                BUTTON_SELECTED_COLOR_TYPE
                if button_position == component_index
                else BUTTON_DEFAULT_COLOR_TYPE
            )

        label = _string_cleanup(self.state.selected_component)
        self.state.status_div.text = f"<b style='color:#FFD700;font-size:14pt'>Placing: {label}</b>"
