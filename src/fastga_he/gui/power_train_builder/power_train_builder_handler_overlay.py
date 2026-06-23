# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Startup-overlay, new-design, delete-mode, and palette-selection logic.

Mixin class :class:`OverlayMixin` is not meant to be instantiated directly;
it is composed into :class:`PlacementHandler` via multiple inheritance.
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


class OverlayMixin:
    """
    Handles the startup overlay choices, palette button selection, and delete-mode toggling.

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
        if self.state.new_design_button is not None:
            self.state.new_design_button.visible = False
        if self.state.load_design_button is not None:
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

        # Clear undo / redo history: a new design has no prior actions to undo.
        if self.state.undo_stack is not None:
            self.state.undo_stack.clear()
        self._refresh_undo_redo_buttons()

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

    def on_palette_select(self, component_index: int):
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

        if self.state.delete_mode:
            self.state.delete_mode = False
            if self.state.delete_button is not None:
                self.state.delete_button.button_type = BUTTON_DEFAULT_COLOR_TYPE

        for button_position, palette_button in enumerate(self.state.buttons):
            palette_button.button_type = (
                BUTTON_SELECTED_COLOR_TYPE
                if button_position == component_index
                else BUTTON_DEFAULT_COLOR_TYPE
            )

        label = _string_cleanup(self.state.selected_component)
        self.state.status_div.text = f"<b style='color:#FFD700;font-size:14pt'>Placing: {label}</b>"
