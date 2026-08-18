# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Node-placement logic for the powertrain builder.

This is the module that actually creates a new component on the canvas:
computing its default type / position / options / port counts, building the
icon URL, appending the new row to ``placed_nodes_source`` and
``hover_source``, and rebuilding the ports that hang off it.

Mixin class :class:`PlacementMixin` is not meant to be instantiated directly;
it is composed into :class:`BuilderHandler` via multiple inheritance.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG, _url_to_base64

from .power_train_builder_state import _EMPTY

_LOGGER = logging.getLogger(__name__)

# For IDE type-checking only
if TYPE_CHECKING:
    from .power_train_builder_state import BuilderState


class PlacementMixin:
    """
    Creates new components on the canvas.

    Depends on ``self.state`` (a :class:`BuilderState` instance) and
    ``self.icon_size`` being set by the concrete class before any method is
    called.
    """

    state: "BuilderState"
    icon_size: int

    # -----------------------------------------------------------------------
    # Node creation
    # -----------------------------------------------------------------------

    def _place_new_node(self, x: float, y: float):
        """
        Create a new component of the currently-selected palette type at ``(x, y)``.

        Deselects any previously-selected node, pushes an undo snapshot, then
        computes the new node's default component type, position, options,
        and port counts (from the shared per-type defaults on ``self.state``),
        builds its icon URL, and appends the resulting row to
        ``placed_nodes_source`` (and ``hover_source``). Finally rebuilds every
        port so the new component's port balls appear immediately.

        :param x: Canvas x coordinate (data-units) where the component is placed.
        :param y: Canvas y coordinate (data-units) where the component is placed.
        """
        # Deselect any previously selected node before placing a new component
        if self.state.selected_node_index is not None:
            self.state.selected_node_index = None
            self._clear_temp_edges()
            self._clear_node_table()

        # Capture state before placing a new component.
        self._push_undo()

        icon_key = self.state.selected_component
        placement_count = self.state.placed_counter.get(icon_key, 0) + 1
        self.state.placed_counter[icon_key] = placement_count
        node_name = f"{icon_key}_{placement_count}"

        icon_path = ICONS_CONFIG[icon_key]["icon_path"]
        file_url = "file://" + str(Path(icon_path).resolve())
        base64_url = _url_to_base64(file_url)

        default_component_type = self.state.component_type_to_icon.get(icon_key, icon_key)[0]
        position_choices = self.state.possible_position.get(default_component_type, [])
        default_position = position_choices[0] if position_choices else _EMPTY

        options_definition = self.state.possible_options.get(default_component_type, {})
        default_options = {
            option_key: (
                True
                if option_values[0] is True
                else (False if option_values[0] is False else option_values[0])
            )
            for option_key, option_values in options_definition.items()
            if option_values
        }
        default_options_json = json.dumps(default_options)

        default_source_port_count = self.state.default_source_count.get(default_component_type, 0)
        default_target_port_count = self.state.default_target_count.get(default_component_type, 0)

        icon_pixel_size = self.icon_size
        nodes_data = self.state.placed_nodes_source.data
        self.state.placed_nodes_source.data = {
            "x": list(nodes_data["x"]) + [x],
            "y": list(nodes_data["y"]) + [y],
            "url": list(nodes_data["url"]) + [base64_url],
            "w": list(nodes_data["w"]) + [icon_pixel_size],
            "h": list(nodes_data["h"]) + [icon_pixel_size],
            "name": list(nodes_data["name"]) + [node_name],
            "icon_type": list(nodes_data.get("icon_type", [])) + [icon_key],
            "node_type": list(nodes_data.get("node_type", [])) + [default_component_type],
            "position": list(nodes_data.get("position", [])) + [default_position],
            "options": list(nodes_data.get("options", [])) + [default_options_json],
            "n_sources": list(nodes_data.get("n_sources", [])) + [default_source_port_count],
            "n_targets": list(nodes_data.get("n_targets", [])) + [default_target_port_count],
            "symmetry_name": list(nodes_data.get("symmetry_name", [])) + [_EMPTY],
            "symmetry_node_index": list(nodes_data.get("symmetry_node_index", [])) + [-1],
        }

        if self.state.hover_source is not None:
            hdata = self.state.hover_source.data
            self.state.hover_source.data = {
                "x": list(hdata["x"]) + [x],
                "y": list(hdata["y"]) + [y],
                "name": list(hdata["name"]) + [node_name],
                "node_type": list(hdata.get("node_type", [])) + [default_component_type],
            }

        self._rebuild_all_ports()
        self._mark_unsaved()
        _LOGGER.info(
            "Placed %s (node_type=%s, position=%s) at (%.1f, %.1f)",
            node_name,
            default_component_type,
            default_position,
            x,
            y,
        )
