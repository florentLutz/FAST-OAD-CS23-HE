# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Canvas interaction handler for the powertrain builder.

:class:`BuilderHandler` wires every toolbox button and canvas tap event to
the appropriate action: placing a component on the canvas, selecting a node
for editing, deleting a node or edge, managing port connections, and saving
or loading the canvas state.

The class is assembled from six single-responsibility mixins:

* :class:`~.power_train_builder_handler_canvas_mode.CanvasModeMixin` –
  startup overlay, new-design reset, delete mode, toolbox selection. These
  all arm a mode for a *future* tap rather than mutating the canvas.
* :class:`~.power_train_builder_handler_placement.PlacementMixin` –
  the actual "place a new component on the canvas" logic: computing default
  type / position / options / port counts and appending the new node.
* :class:`~.power_train_builder_handler_io.IOMixin` –
  save / load (one YAML file incl. a canvas_layout section), tkinter file
  dialogs, end-session triggers, undo / redo.
* :class:`~.power_train_builder_handler_ports.PortEdgeMixin` –
  node / port / edge hit-testing, edge add / remove / rebuild.
* :class:`~.power_train_builder_handler_config_panel.ConfigPanelMixin` –
  node-table, options table, connections table, symmetry select.
* :class:`~.power_train_builder_handler_unknown.UnknownComponentMixin` –
  the config panel of an unknown component: free-text type / position /
  options / symmetry, "Add option" button, and port-type selects.

Typical usage::

    toolbox_layout, config_panel_column, state = ComponentToolboxConfigurationTableBuilder.build()
    handler = BuilderHandler(state, canvas)
    canvas.on_event(Tap, handler.on_canvas_tap)
"""

import functools
import logging

from .power_train_builder_state import (
    BuilderState,
    DEFAULT_NODE_ICON_SIZE,
    copy_column_data,
    drop_row,
    edge_indices_not_touching_node,
    select_edges,
    unlink_symmetry,
)
from .power_train_builder_history import EditHistoryStack
from .power_train_builder_handler_canvas_mode import CanvasModeMixin
from .power_train_builder_handler_placement import PlacementMixin
from .power_train_builder_handler_io import IOMixin
from .power_train_builder_handler_ports import PortEdgeMixin
from .power_train_builder_handler_config_panel import ConfigPanelMixin
from .power_train_builder_handler_unknown import UnknownComponentMixin

_LOGGER = logging.getLogger(__name__)


class BuilderHandler(
    CanvasModeMixin,
    PlacementMixin,
    IOMixin,
    PortEdgeMixin,
    ConfigPanelMixin,
    UnknownComponentMixin,
):
    """
    Wires every Python callback of the builder and dispatches canvas taps.

    Create it **after** the toolbox and the Undo / Redo buttons exist on the
    state. It holds no design data of its own: every mixin reads and changes
    the shared :class:`BuilderState`.
    """

    def __init__(self, state: BuilderState, main_plot, icon_size: int = DEFAULT_NODE_ICON_SIZE):
        """
        Initialise the handler, set up the undo stack, and wire all button callbacks.

        The canvas ``Tap`` event is wired by the launcher, to :meth:`on_canvas_tap`.

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
        Attach the Python callbacks of every static widget. Called once by ``__init__``.

        The End Session button has no Python callback: its JavaScript decides
        whether the session may end, then flips ``end_session_trigger``, which
        is wired here to :meth:`_on_end_session_trigger`.
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

        # Watcher CSV path: Browse dialog and direct edits of the field
        self.state.browse_watcher_trigger.on_change("value", self._on_browse_watcher)
        self.state.watcher_path_input.on_change("value", self._on_watcher_path_change)

        self.state.end_session_save_trigger.on_change("value", self._on_end_session_save_trigger)

        # Icon-size slider in the toolbox header
        self.state.node_size_slider.value = self.icon_size
        self.state.node_size_slider.on_change("value", self._on_node_size_slider_change)

        # View scale of the window-fitted canvas, reported by the browser
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

        # Unknown component: add an option row, and port types applied at once
        self.state.add_option_button.on_click(self._on_add_option)
        for side, port_type_select in (
            ("source", self.state.source_port_type_select),
            ("target", self.state.target_port_type_select),
        ):
            port_type_select.on_change(
                "value", functools.partial(self._on_port_type_change, side=side)
            )

    def _make_select_callback(self, button_index: int):
        """
        Return a zero-argument closure that selects the component at *button_index*.

        :param button_index: Zero-based index into ``list(BUILDER_ICONS_CONFIG)``.
        :return: Callback function suitable for wiring to a toolbox button.
        """
        return lambda: self.on_toolbox_select(button_index)

    # -----------------------------------------------------------------------
    # Canvas tap handler
    # -----------------------------------------------------------------------

    def on_canvas_tap(self, event):
        """
        Handle a tap on the canvas, depending on the current mode.

        * **Delete mode** – delete the node at the tap, else the edge there.
        * **Toolbox component armed** – place it at the tap.
        * **Idle** – a port within reach starts or completes a two-click
          connection (port balls overlap the icon edge, so they come first);
          otherwise select or deselect the node at the tap.

        Every branch that changes the design pushes an undo snapshot first.

        :param event: Bokeh ``Tap`` event carrying ``x`` and ``y`` coordinates.
        """
        x, y = event.x, event.y

        if self.state.delete_mode:
            self._delete_at(x, y)
        elif self.state.selected_component is not None:
            # A component is selected in the toolbox: place it at the tap coordinates
            self._place_new_node(x, y)
        else:
            # Idle mode: a port tap (connection) has priority over a node tap
            # (selection). Undo for port connections is pushed inside
            # _handle_port_tap, just before the permanent edge is committed
            nearest_port = self._find_nearest_port(x, y)
            if nearest_port is not None:
                self._handle_port_tap(nearest_port)
            else:
                self._select_at(x, y)

    # -----------------------------------------------------------------------
    # Tap actions
    # -----------------------------------------------------------------------

    def _delete_at(self, x: float, y: float):
        """
        Delete mode: remove the node at ``(x, y)``, or else the edge there.

        If a node is still selected afterwards, its Connections panel is refreshed.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.
        """
        best_node_index, _, nodes_data = self._find_nearest_node(x, y)
        if best_node_index is not None:
            self._delete_node(best_node_index, nodes_data)
            return

        edge_index = self._find_nearest_edge(x, y)
        if edge_index is not None:
            self._delete_edge(edge_index)

        if self.state.selected_node_index is not None:
            self._refresh_connections_table(self.state.selected_node_index)

    def _delete_node(self, node_index: int, nodes_data: dict):
        """
        Remove node *node_index* with its edges, as a single undoable step.

        Later nodes move up by one index, so edge endpoints, symmetry links and
        the selection are shifted to match.

        :param node_index: Zero-based index of the node to delete.
        :param nodes_data: Current ``placed_nodes_source.data``.
        """
        # Push the state into the history stack before deleting the node
        self._push_undo()

        # Remove the node from placed_nodes_source
        new_nodes_data = drop_row(nodes_data, node_index)
        _reindex_symmetry_after_deletion(new_nodes_data, node_index)
        self.state.placed_nodes_source.data = new_nodes_data

        # Remove any edges connected to the deleted node and adjust indices of remaining edges
        edge_data = copy_column_data(self.state.edge_source.data)
        new_edge_data = select_edges(
            edge_data, edge_indices_not_touching_node(edge_data, node_index)
        )
        # Nodes after the deleted one move up by one index
        for column_name in ("starting_node_index", "ending_node_index"):
            new_edge_data[column_name] = [
                index - 1 if index > node_index else index for index in new_edge_data[column_name]
            ]
        self.state.edge_source.data = new_edge_data

        self._rebuild_all_ports()

        # If the deleted node was selected, clear the selection and the node table
        if self.state.selected_node_index == node_index:
            self.state.selected_node_index = None
            self._clear_node_table()

        # Shift the selected node index left by one if it was located after the deleted node
        # [A, B, C, D] -> delete B -> [A, C, D], so C and D shift left by one
        elif (
            self.state.selected_node_index is not None
            and self.state.selected_node_index > node_index
        ):
            self.state.selected_node_index -= 1

        self._mark_unsaved()

    def _delete_edge(self, edge_index: int):
        """
        Remove edge *edge_index*, as a single undoable step.

        :param edge_index: Zero-based position of the edge in ``edge_source``.
        """
        # Capture state before deleting the edge
        self._push_undo()

        # Remove the edge from the edge_source data
        self.state.edge_source.data = drop_row(self.state.edge_source.data, edge_index)
        # Recompute the ports so the freed ports are no longer flagged connected
        self._rebuild_all_ports()
        self._mark_unsaved()

    def _select_at(self, x: float, y: float):
        """
        Idle mode: select the node at ``(x, y)``, or deselect.

        A pending two-click connection and the panel picks are cancelled first.
        Tapping empty canvas or the already selected node deselects; tapping
        another node selects it and fills the config panel.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.
        """
        best_node_index, _, _ = self._find_nearest_node(x, y)
        # Unselect port highlight and clean up the pending array
        self._cancel_pending_connection()
        self._clear_temp_edges()
        if best_node_index is None or self.state.selected_node_index == best_node_index:
            # Deselect the node if the tap is not on any node or if the tap is on the
            # already selected node
            self.state.selected_node_index = None
            self._clear_node_table()
        else:
            # A node is tapped and is not the currently selected node: select it and
            # populate the config panel
            self.state.selected_node_index = best_node_index
            self._populate_node_table(best_node_index)
            self.state.config_panel_column.visible = True


def _reindex_symmetry_after_deletion(nodes_data: dict, deleted_node_index: int) -> None:
    """
    Update the symmetry links of *nodes_data* after a node was removed.

    The peer of the deleted node loses its link, and every peer index after the
    deleted node moves up by one, like the node rows themselves.

    :param nodes_data: Node columns with the deleted row already removed; edited in place.
    :param deleted_node_index: Index the deleted node had before its removal.
    """
    symmetry_names = nodes_data.get("symmetry_name", [])
    symmetry_indices = nodes_data.get("symmetry_node_index", [])
    for node_position, peer_index in enumerate(symmetry_indices):
        if peer_index == deleted_node_index:
            unlink_symmetry(symmetry_names, symmetry_indices, [node_position])
        elif peer_index > deleted_node_index:
            symmetry_indices[node_position] = peer_index - 1
