# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Canvas interaction handler for the powertrain builder.

:class:`BuilderHandler` wires every toolbox button and canvas tap event to
the appropriate action: placing a component on the canvas, selecting a node
for editing, deleting a node or edge, managing port connections, and saving
or loading the canvas state.

The class is assembled from five single-responsibility mixins:

* :class:`~.power_train_builder_handler_canvas_mode.CanvasModeMixin` –
  startup overlay, new-design reset, delete mode, toolbox selection. These
  all arm a mode for a *future* tap rather than mutating the canvas.
* :class:`~.power_train_builder_handler_placement.PlacementMixin` –
  the actual "place a new component on the canvas" logic: computing default
  type / position / options / port counts and appending the new node.
* :class:`~.power_train_builder_handler_io.IOMixin` –
  save / load (one YAML file incl. a canvas_layout section), tkinter file
  dialogs, end-session triggers,
  undo / redo snapshot management.
* :class:`~.power_train_builder_handler_ports.PortEdgeMixin` –
  node / port / edge hit-testing, edge add / remove / rebuild.
* :class:`~.power_train_builder_handler_config_panel.ConfigPanelMixin` –
  node-table, options table, connections table, symmetry select.

Typical usage::

    toolbox_layout, config_panel_column, state = ComponentPaletteConfigurationTableBuilder.build()
    handler = BuilderHandler(state, canvas)
    canvas.on_event(Tap, handler.on_canvas_tap)
"""

import logging

from .power_train_builder_state import BuilderState, DEFAULT_NODE_ICON_SIZE, copy_column_data
from .power_train_builder_history import EditHistoryStack
from .power_train_builder_handler_canvas_mode import CanvasModeMixin
from .power_train_builder_handler_placement import PlacementMixin
from .power_train_builder_handler_io import IOMixin
from .power_train_builder_handler_ports import PortEdgeMixin
from .power_train_builder_handler_config_panel import ConfigPanelMixin

_LOGGER = logging.getLogger(__name__)


class BuilderHandler(CanvasModeMixin, PlacementMixin, IOMixin, PortEdgeMixin, ConfigPanelMixin):
    """
    Wires toolbox button events to canvas placement and interaction events.

    Instantiate **after** building the toolbox; the constructor automatically
    wires all button ``on_click`` callbacks and initialises the undo stack.

    All public and private methods of this class operate exclusively on the
    shared :class:`BuilderState` instance passed at construction time.
    """

    def __init__(self, state: BuilderState, main_plot, icon_size: int = DEFAULT_NODE_ICON_SIZE):
        """
        Initialise the handler, set up the undo stack, and wire all button callbacks.

        :param state: Shared :class:`BuilderState` instance.
        :param main_plot: The Bokeh ``figure`` that acts as the placement canvas.
        :param icon_size: Initial size (width = height, canvas data units) of
            placed icons; changed afterwards with the icon-size slider.
        """
        self.state = state
        self.main_plot = main_plot
        self.icon_size = icon_size

        # Initialise the undo stack on the shared state so all mixins can
        # reach it via self.state.edit_history_stack
        self.state.edit_history_stack = EditHistoryStack()

        self._wire_buttons_widgets()
        # Set the initial enabled/disabled appearance of the undo/redo buttons
        self._refresh_undo_redo_buttons()

    # -----------------------------------------------------------------------
    # Internal wiring
    # -----------------------------------------------------------------------

    def _wire_buttons_widgets(self):
        """
        Attach ``on_click`` callbacks of all static Bokeh buttons/ widgets in the left toolbox
        and the right component configuration table panel.

        Called once from ``__init__``. Connects:

        * Each component toolbox button -> ``on_toolbox_select``.
        * The End Session button -> ``_end_session``.
        * The Delete button -> ``_toggle_delete_mode``.
        * The New Design button -> ``_on_new_design``.
        * The browse_load_trigger -> ``_on_browse_load`` (tkinter open-file dialog).
        * The browse_save_trigger -> ``_on_browse_save`` (tkinter save-file dialog).
        * The icon-size slider -> ``_on_node_size_slider_change`` (rescales
          icons, hover / selection rings, labels and port orbits).
        * The view_scale_input -> ``_on_view_scale_change`` (window-fitted
          canvas scale, keeps rings / labels matched to the icons).
        * The Undo button -> ``_on_undo``.
        * The Redo button -> ``_on_redo``.
        * The type_select -> ``_on_type_select_change`` (refreshes
          position / options / symmetry choices as the user edits).
        * The Apply button -> ``apply_node_configurations`` (commits the
          config panel's current values to the selected node).
        * The source/target port-count spinners ->
          ``_on_port_count_spinner_change`` (live visual preview only;
          the new counts are committed on Apply, same as every other
          config-panel field).
        """
        # End Session is handled entirely via JS + end_session_trigger to avoid the
        # Python on_click firing unconditionally (before JS can gate on save state)
        self.state.end_session_trigger.on_change("value", self._on_end_session_trigger)

        # New Design: wipe canvas and dismiss overlay immediately
        self.state.new_design_button.on_click(self._on_new_design)

        for toolbox_button_index, toolbox_button in enumerate(self.state.buttons):
            toolbox_button.on_click(self._make_select_callback(toolbox_button_index))

        self.state.delete_button.on_click(self._toggle_delete_mode)

        self.state.browse_load_trigger.on_change("value", self._on_browse_load)

        self.state.browse_save_trigger.on_change("value", self._on_browse_save)

        self.state.end_session_save_trigger.on_change("value", self._on_end_session_save_trigger)

        # Icon-size slider in the toolbox header
        if self.state.node_size_slider is not None:
            self.state.node_size_slider.value = self.icon_size
            self.state.node_size_slider.on_change("value", self._on_node_size_slider_change)

        # View scale of the window-fitted canvas, reported by the browser
        if self.state.view_scale_input is not None:
            self.state.view_scale_input.on_change("value", self._on_view_scale_change)

        # Undo / redo buttons (created by the launcher and stored in state)
        self.state.undo_button.on_click(self._on_undo)

        self.state.redo_button.on_click(self._on_redo)

        # Config panel: live type-select refresh + Apply button commit
        self.state.type_select.on_change("value", self._on_type_select_change)

        self.state.apply_button.on_click(self.apply_node_configurations)

        # Port-count spinners: live visual preview only (see
        # ConfigPanelMixin._on_port_count_spinner_change) — the new counts
        # are committed to placed_nodes_source only when Apply is clicked
        self.state.source_count_spinner.on_change("value", self._on_port_count_spinner_change)
        self.state.target_count_spinner.on_change("value", self._on_port_count_spinner_change)

    def _make_select_callback(self, button_index: int):
        """
        Return a zero-argument closure that selects the component at *button_index*.

        :param button_index: Zero-based index into ``list(ICONS_CONFIG.keys())``.
        :return: Callback function suitable for wiring to a toolbox button.
        """
        return lambda: self.on_toolbox_select(button_index)

    # -----------------------------------------------------------------------
    # Canvas tap handler
    # -----------------------------------------------------------------------

    def on_canvas_tap(self, event):
        """
        Handle a tap event on the main canvas.

        Behaviour depends on the current interaction mode:

        * **Delete mode** – remove the nearest placed node or edge.
        * **Component selected** – place a new node at the tap coordinates.
        * **Neither** – select or deselect the nearest existing node.

        An undo snapshot is pushed **before** every action that mutates the
        canvas (node placement, node deletion, edge deletion) so the user can
        reverse each step individually.

        :param event: Bokeh ``Tap`` event carrying ``x`` and ``y`` coordinates.
        """
        x, y = event.x, event.y

        # Port connection has highest priority in idle mode.
        # Undo for port connections is pushed inside _handle_port_tap, just
        # before the permanent edge is committed
        if not self.state.delete_mode and self.state.selected_component is None:
            # No component is selected to place and not in delete mode: check for a port tap first
            nearest_port = self._find_nearest_port(x, y)
            if nearest_port is not None:
                # Find the port that was tapped and handle the connection
                self._handle_port_tap(nearest_port)
                return

        if self.state.delete_mode:
            # Delete mode: find if any component (node) on the canvas is tapped
            best_node_index, best_node_distance, nodes_data = self._best_possible_node(x, y)

            if best_node_index is not None:
                # Push the state into the history stack before deleting the node
                self._push_undo()

                # Remove the node from the placed_nodes_source and hover_source (if it exists)
                new_nodes_data = copy_column_data(nodes_data)
                for column_name in new_nodes_data:
                    new_nodes_data[column_name].pop(best_node_index)
                self.state.placed_nodes_source.data = new_nodes_data

                if self.state.hover_source is not None:
                    hover_data = copy_column_data(self.state.hover_source.data)
                    for column_name in hover_data:
                        if best_node_index < len(hover_data[column_name]):
                            hover_data[column_name].pop(best_node_index)
                    self.state.hover_source.data = hover_data

                # Remove any edges connected to the deleted node and adjust indices of remaining edges
                if self.state.edge_source is not None:
                    edge_data = copy_column_data(self.state.edge_source.data)
                    kept_edge_indices = [
                        edge_position
                        for edge_position, (starting_node_index, ending_node_index) in enumerate(
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
                        kept_values = [
                            column_values[edge_position] for edge_position in kept_edge_indices
                        ]
                        if column_key in ("starting_node_index", "ending_node_index"):
                            kept_values = [
                                index - 1 if index > best_node_index else index
                                for index in kept_values
                            ]

                        new_edge_data[column_key] = kept_values
                    self.state.edge_source.data = new_edge_data

                self._rebuild_all_ports()

                # If the deleted node was selected, clear the selection and the node table
                if self.state.selected_node_index == best_node_index:
                    self.state.selected_node_index = None
                    self._clear_node_table()

                # Shift the selected node index left by one if it was located after the deleted node
                # [A, B, C, D] -> delete B -> [A, C, D], so C and D shift left by one
                elif (
                    self.state.selected_node_index is not None
                    and self.state.selected_node_index > best_node_index
                ):
                    self.state.selected_node_index -= 1

                _LOGGER.info("Deleted node at index %d", best_node_index)
                self._mark_unsaved()

            else:
                edge_index = self._find_nearest_edge(x, y)
                # Check if an edge (connection) is tapped. If so, delete it
                if edge_index is not None and self.state.edge_source is not None:
                    # Capture state before deleting the edge
                    self._push_undo()

                    # Remove the edge from the edge_source data
                    edge_data = copy_column_data(self.state.edge_source.data)
                    for column_name in edge_data:
                        edge_data[column_name].pop(edge_index)
                    self.state.edge_source.data = edge_data
                    # Recompute the ports so the freed ports are no longer flagged connected
                    self._rebuild_all_ports()
                    self._mark_unsaved()
                    _LOGGER.info("Deleted edge at index %d", edge_index)

                if self.state.selected_node_index is not None:
                    self._refresh_connections_table(self.state.selected_node_index)

            return

        if self.state.selected_component is None:
            # No component is selected to place and not in delete mode: check for a node tap
            best_node_index, best_node_distance, nodes_data = self._best_possible_node(x, y)
            # Unselect port highlight and clean up the pending array
            self._cancel_pending_connection()

            if best_node_index is None and best_node_distance is None:
                return
            elif best_node_index is None or self.state.selected_node_index == best_node_index:
                # Deselect the node if the tap is not on any node or if the tap is on the already
                # selected node
                self.state.selected_node_index = None
                self._clear_temp_edges()
                self._clear_node_table()
                return
            else:
                # A node is tapped and is not the currently selected node: select it and
                # populate the config panel
                self._clear_temp_edges()
                self.state.selected_node_index = best_node_index
                self._populate_node_table(best_node_index)
                self.state.config_panel_column.visible = True
            return

        # A component is selected in the toolbox: place it at the tap coordinates
        self._place_new_node(x, y)