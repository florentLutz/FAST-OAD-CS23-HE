# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Canvas interaction handler for the powertrain builder.

:class:`PlacementHandler` wires every palette button and canvas tap event to
the appropriate action: placing a component on the canvas, selecting a node
for editing, deleting a node or edge, managing port connections, and saving
or loading the canvas state.

The class is assembled from four single-responsibility mixins:

* :class:`~.power_train_builder_handler_overlay.OverlayMixin` –
  startup overlay, new-design reset, delete mode, palette selection.
* :class:`~.power_train_builder_handler_io.IOMixin` –
  save / load (YAML + JSON), tkinter file dialogs, end-session triggers,
  undo / redo snapshot management.
* :class:`~.power_train_builder_handler_ports.PortEdgeMixin` –
  port hit-testing, edge add / remove / rebuild, nearest-node helpers.
* :class:`~.power_train_builder_handler_config_panel.ConfigPanelMixin` –
  node-table, options table, connections table, symmetry select.

Typical usage::

    palette_layout, table_panel, state = ComponentPaletteConfigurationTableBuilder.build()
    handler = PlacementHandler(state, canvas)
    canvas.on_event(Tap, handler.on_canvas_tap)
"""

import json
import logging
from pathlib import Path

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG, _url_to_base64

from .power_train_builder_state import BuilderState, _EMPTY
from .power_train_builder_history import UndoStack
from .power_train_builder_handler_overlay import OverlayMixin
from .power_train_builder_handler_io import IOMixin
from .power_train_builder_handler_ports import PortEdgeMixin
from .power_train_builder_handler_config_panel import ConfigPanelMixin

_LOGGER = logging.getLogger(__name__)


class PlacementHandler(OverlayMixin, IOMixin, PortEdgeMixin, ConfigPanelMixin):
    """
    Wires palette button events to canvas placement and interaction events.

    Instantiate **after** building the palette; the constructor automatically
    wires all button ``on_click`` callbacks and initialises the undo stack::

        palette_layout, table_panel, state = ComponentPaletteConfigurationTableBuilder.build()
        handler = PlacementHandler(state, main_plot)
        main_plot.on_event(Tap, handler.on_canvas_tap)
        main_plot.image_url(
            url="url", x="x", y="y", w="w", h="h",
            anchor="center", source=state.placed_nodes_source,
        )

    All public and private methods of this class operate exclusively on the
    shared :class:`BuilderState` instance passed at construction time.
    """

    def __init__(self, state: BuilderState, main_plot, icon_size: int = 50):
        """
        Initialise the handler, set up the undo stack, and wire all button callbacks.

        :param state: Shared :class:`BuilderState` instance.
        :param main_plot: The Bokeh ``figure`` that acts as the placement canvas.
        :param icon_size: Pixel size (width = height) used for placed icons.
        """
        self.state = state
        self.main_plot = main_plot
        self.icon_size = icon_size

        # Initialise the undo stack on the shared state so all mixins can
        # reach it via self.state.undo_stack.
        self.state.undo_stack = UndoStack()

        self._wire_buttons()
        # Set the initial enabled/disabled appearance of the undo/redo buttons.
        self._refresh_undo_redo_buttons()

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
        * The Undo button → :meth:`_on_undo`.
        * The Redo button → :meth:`_on_redo`.
        """
        # End Session is handled entirely via JS + end_session_trigger to avoid the
        # Python on_click firing unconditionally (before JS can gate on save state).
        if self.state.end_session_trigger is not None:
            self.state.end_session_trigger.on_change("value", self._on_end_session_trigger)

        # New Design: wipe canvas and dismiss overlay immediately
        if self.state.new_design_button is not None:
            self.state.new_design_button.on_click(self._on_new_design)

        for palette_button_index, palette_button in enumerate(self.state.buttons):
            palette_button.on_click(self._make_select_callback(palette_button_index))

        if self.state.delete_button is not None:
            self.state.delete_button.on_click(self._toggle_delete_mode)

        if self.state.browse_load_trigger is not None:
            self.state.browse_load_trigger.on_change("value", self._on_browse_load)

        if self.state.browse_save_trigger is not None:
            self.state.browse_save_trigger.on_change("value", self._on_browse_save)

        # Open a native OS file dialog to choose the watcher CSV path
        if self.state.browse_watcher_trigger is not None:
            self.state.browse_watcher_trigger.on_change("value", self._on_browse_watcher)

        if self.state.end_session_save_trigger is not None:
            self.state.end_session_save_trigger.on_change(
                "value", self._on_end_session_save_trigger
            )

        # Undo / redo buttons (created by the launcher and stored in state)
        if self.state.undo_button is not None:
            self.state.undo_button.on_click(self._on_undo)

        if self.state.redo_button is not None:
            self.state.redo_button.on_click(self._on_redo)

    def _make_select_callback(self, button_index: int):
        """
        Return a zero-argument closure that selects the component at *button_index*.

        :param button_index: Zero-based index into ``list(ICONS_CONFIG.keys())``.
        :return: Callback function suitable for wiring to a palette button.
        """
        return lambda: self.on_palette_select(button_index)

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

        An undo snapshot is pushed **before** every action that mutates the
        canvas (node placement, node deletion, edge deletion) so the user can
        reverse each step individually.

        :param event: Bokeh ``Tap`` event carrying ``x`` and ``y`` coordinates.
        """
        x, y = event.x, event.y

        # Port connection has highest priority in idle mode.
        # Undo for port connections is pushed inside _handle_port_tap, just
        # before the permanent edge is committed.
        if not self.state.delete_mode and self.state.selected_component is None:
            nearest_port = self._find_nearest_port(x, y)
            if nearest_port is not None:
                self._handle_port_tap(nearest_port)
                return

        if self.state.delete_mode:
            best_node_index, best_node_distance, nodes_data = self._best_possible_node(x, y)

            if best_node_index is not None:
                # Capture state before deleting the node.
                self._push_undo()

                new_data = {key: list(values) for key, values in nodes_data.items()}
                for column in new_data:
                    new_data[column].pop(best_node_index)
                self.state.placed_nodes_source.data = new_data

                if self.state.hover_source is not None:
                    hover_data = {
                        key: list(values) for key, values in self.state.hover_source.data.items()
                    }
                    for column in hover_data:
                        if best_node_index < len(hover_data[column]):
                            hover_data[column].pop(best_node_index)
                    self.state.hover_source.data = hover_data

                if self.state.edge_source is not None:
                    edge_data = {
                        key: list(values) for key, values in self.state.edge_source.data.items()
                    }
                    keep_indices = [
                        i
                        for i, (starting_node_index, ending_node_index) in enumerate(
                            zip(
                                edge_data.get("starting_node_index", []),
                                edge_data.get("ending_node_index", []),
                            )
                        )
                        if starting_node_index != best_node_index
                        and ending_node_index != best_node_index
                    ]
                    new_edge_data = {}
                    for column_key, column_values in edge_data.items():
                        kept_values = [column_values[i] for i in keep_indices]
                        if column_key == "starting_node_index":
                            kept_values = [v - 1 if v > best_node_index else v for v in kept_values]
                        elif column_key == "ending_node_index":
                            kept_values = [v - 1 if v > best_node_index else v for v in kept_values]
                        new_edge_data[column_key] = kept_values
                    self.state.edge_source.data = new_edge_data

                self._rebuild_all_ports()

                if self.state.selected_node_index == best_node_index:
                    self.state.selected_node_index = None
                    self._clear_node_table()
                elif (
                    self.state.selected_node_index is not None
                    and self.state.selected_node_index > best_node_index
                ):
                    self.state.selected_node_index -= 1

                _LOGGER.info("Deleted node at index %d", best_node_index)
                self._mark_unsaved()

            else:
                edge_idx = self._find_nearest_edge(x, y)
                if edge_idx is not None and self.state.edge_source is not None:
                    # Capture state before deleting the edge.
                    self._push_undo()

                    edge_data = {
                        key: list(value) for key, value in self.state.edge_source.data.items()
                    }
                    for k in edge_data:
                        edge_data[k].pop(edge_idx)
                    self.state.edge_source.data = edge_data
                    self._mark_unsaved()
                    _LOGGER.info("Deleted edge at index %d", edge_idx)

                if self.state.selected_node_index is not None:
                    self._refresh_connections_table(self.state.selected_node_index)

            return

        if self.state.selected_component is None:
            best_node_index, best_node_distance, nodes_data = self._best_possible_node(x, y)
            self._cancel_pending_connection()

            if best_node_index is None and best_node_distance is None:
                return
            elif best_node_index is None or self.state.selected_node_index == best_node_index:
                self.state.selected_node_index = None
                self._clear_temp_edges()
                self._clear_node_table()
                return
            else:
                self._clear_temp_edges()
                self.state.selected_node_index = best_node_index
                self._populate_node_table(best_node_index)
                if self.state.table_panel is not None:
                    self.state.table_panel.visible = True
            return

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
