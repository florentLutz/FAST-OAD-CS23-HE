# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Config-panel helpers: node table, options table, connections table, symmetry widget.

Mixin class :class:`ConfigPanelMixin` is not meant to be instantiated directly;
it is composed into :class:`BuilderHandler` via multiple inheritance.
"""

import functools
import json
import logging
import bokeh.models as bkmodel
from bokeh.layouts import row

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG
from .power_train_builder_state import (
    _EMPTY,
    _NO_SYMMETRY_PEER,
    copy_column_data,
    edge_indices_not_touching_node,
    edge_indices_not_touching_port,
    select_edges,
    BuilderState,
)

_LOGGER = logging.getLogger(__name__)

# Text color of the primary port label in the Connections panel
PRIMARY_PORT_LABEL_COLOR = "#ADD8E6"  # light blue

# Port label of the primary port in a power split definition: the first
# source port of splitters and gearboxes, the first target port of planetary
# gears. Component types are matched by keyword (e.g. "dc_splitter")
PRIMARY_PORT_LABEL = "1"
PRIMARY_SOURCE_PORT_TYPE_KEYWORDS = ("splitter", "gearbox")
PRIMARY_TARGET_PORT_TYPE_KEYWORDS = ("planetary_gear",)


class ConfigPanelMixin:
    """
    Handles the right-hand config panel: populating, refreshing, and clearing
    the node-name input, type/position selects, options table, connections table,
    symmetry select, and port-count spinners.

    Depends on ``self.state`` (a :class:`BuilderState` instance) being set by
    the concrete class before any method is called.
    """

    state: "BuilderState"

    # True while _populate_node_table fills the widgets, so the type_select
    # callback can tell a programmatic update from a user switching the type
    _is_populating_node_table: bool = False

    # -----------------------------------------------------------------------
    # Node table population / clearing
    # -----------------------------------------------------------------------

    def _populate_node_table(self, index: int):
        """
        Fill the config panel widgets with the stored values of node *index*.

        :param index: Zero-based index into ``placed_nodes_source`` data arrays.
        """
        placed_node_data = self.state.placed_nodes_source.data

        node_name = placed_node_data["name"][index]
        node_type = placed_node_data["node_type"][index]
        icon_key = placed_node_data["icon_type"][index]
        position = placed_node_data["position"][index]
        saved_options_json = placed_node_data["options"][index] or "{}"

        # Show the config panel and the options table if there is any options
        self.state.options_table.visible = saved_options_json != "{}"
        self.state.name_input.value = node_name

        # extract possible component type of the icon
        type_choices = self.state.icon_to_component_types.get(icon_key, icon_key)
        self.state.type_select.options = type_choices
        # Setting the value fires _on_type_select_change; flag it as programmatic so
        # selecting a node never clears its connections or symmetry
        self._is_populating_node_table = True
        try:
            self.state.type_select.value = (
                node_type if node_type in type_choices else type_choices[0]
            )
        finally:
            self._is_populating_node_table = False

        # extract possible component position in aircraft based on default/selected node type
        position_choices = self.state.possible_position.get(node_type, [])
        self.state.position_select.options = position_choices
        if position_choices:
            self.state.position_select.value = (
                position if position in position_choices else position_choices[0]
            )
        else:
            self.state.position_select.value = ""

        saved_options: dict = {}
        try:
            saved_options = json.loads(saved_options_json)
        except (json.JSONDecodeError, TypeError):
            pass

        self._refresh_options_table(node_type, saved_options)

        # Show or hide the port-count spinners based on the default counts for this node type.
        default_source_port_count = self.state.default_source_count.get(node_type, 0)
        default_target_port_count = self.state.default_target_count.get(node_type, 0)
        source_ports_are_editable = default_source_port_count == 3
        target_ports_are_editable = default_target_port_count == 3

        current_source_port_count = placed_node_data["n_sources"][index]
        current_target_port_count = placed_node_data["n_targets"][index]

        self.state.source_count_spinner.visible = source_ports_are_editable
        if source_ports_are_editable:
            self.state.source_count_spinner.value = int(current_source_port_count)

        self.state.target_count_spinner.visible = target_ports_are_editable
        if target_ports_are_editable:
            self.state.target_count_spinner.value = int(current_target_port_count)

        self.state.port_count_section.visible = (
            source_ports_are_editable or target_ports_are_editable
        )

        if self.state.selected_node_overlay_source is not None:
            self.state.selected_node_overlay_source.data = dict(
                x=[placed_node_data["x"][index]],
                y=[placed_node_data["y"][index]],
            )
        self._rebuild_all_ports()
        self._refresh_connections_table(index)
        self._refresh_symmetry_select(index, node_type)

    def _clear_node_table(self):
        """
        Reset all node configuration panel inputs to their empty defaults and
        hide the panel.
        """
        self.state.name_input.value = _EMPTY
        self.state.type_select.options = []
        self.state.type_select.value = _EMPTY
        self.state.position_select.options = []
        self.state.position_select.value = _EMPTY
        self.state.options_source.data = dict(options=[], value=[])

        self.state.port_count_section.visible = False
        self.state.source_count_spinner.visible = False
        self.state.target_count_spinner.visible = False

        self.state.config_panel_column.visible = False

        if self.state.selected_node_overlay_source is not None:
            self.state.selected_node_overlay_source.data = dict(x=[], y=[])
        self.state.connections_rows_column.children = []

        self.state.symmetry_select.options = [_EMPTY]
        self.state.symmetry_select.value = _EMPTY
        self._rebuild_all_ports()

    # -----------------------------------------------------------------------
    # Port-count spinners: live preview
    # -----------------------------------------------------------------------

    def _on_port_count_spinner_change(self, attr, old, new):
        """
        Live-preview callback wired to both port-count spinners.

        Redraws the selected node's port balls at the spinners' *current*
        values so the canvas updates as the user spins, and hides any edge
        that a shrunk count would drop — all without writing anything to
        ``placed_nodes_source`` and without permanently losing an edge from
        ``edge_source`` (see the ``override_counts`` docstring on
        :meth:`~.power_train_builder_handler_ports.PortEdgeMixin._rebuild_all_ports`
        for the snapshot/restore mechanism that makes this reversible). The
        change only becomes permanent when :meth:`apply_node_configurations`
        reads the same spinner values and commits them; simply navigating
        away (reselecting, deleting, placing a new node, or opening Delete
        mode) discards the preview for free, because every one of those
        paths already triggers a real, non-preview call to
        ``_rebuild_all_ports`` that restores the true edge list and reads
        port counts straight from the last-committed values again.

        No-op if no node is currently selected (can happen if a spinner's
        value is programmatically reset while the panel is being cleared).

        :param attr: Bokeh on_change attribute name (``"value"``); unused.
        :param old: Previous spinner value; unused.
        :param new: New spinner value; unused — both spinners are re-read
            directly since either one may be the one that changed.
        """
        node_index = self.state.selected_node_index
        if node_index is None:
            return

        placed_nodes_data = self.state.placed_nodes_source.data
        committed_n_sources = int(placed_nodes_data["n_sources"][node_index])
        committed_n_targets = int(placed_nodes_data["n_targets"][node_index])

        # Only a spinner that is currently visible represents an in-progress
        # edit; a hidden spinner's stale value must not override the
        # committed count for the other side of the node
        preview_n_sources = (
            int(self.state.source_count_spinner.value)
            if self.state.source_count_spinner.visible
            else committed_n_sources
        )
        preview_n_targets = (
            int(self.state.target_count_spinner.value)
            if self.state.target_count_spinner.visible
            else committed_n_targets
        )

        self._rebuild_all_ports(
            override_counts={node_index: (preview_n_sources, preview_n_targets)}
        )

    # -----------------------------------------------------------------------
    # Options table
    # -----------------------------------------------------------------------

    def _refresh_options_table(self, node_type: str, overrides: dict):
        """
        Rebuild the Options section of the config panel for a given node type.

        Creates one row per option (a disabled TextInput label + a Select for
        the value) and wires a shared callback so that ``options_source`` stays
        in sync with the user's current selections.

        :param node_type: Component type key used to look up ``possible_options``.
        :param overrides: Optional ``{option_name: current_value}`` mapping used
            to pre-populate widgets with previously saved values.
        """

        options_definition = self.state.possible_options.get(node_type, {})
        option_names = list(options_definition.keys())
        option_value_strings = []
        new_row_widgets = []
        value_selects = []

        for option_name, value_list in options_definition.items():
            current_values = (
                overrides[option_name]
                if option_name in overrides
                else (value_list[0] if value_list else _EMPTY)
            )
            current_value_string = _option_values_to_strings(current_values)

            label_input = bkmodel.TextInput(
                value=option_name,
                width=180,
                disabled=True,
                styles={"color": "white", "font-size": "14px"},
            )
            choices = (
                [_option_values_to_strings(choice) for choice in value_list]
                if value_list
                else [current_value_string]
            )
            if current_value_string not in choices:
                choices = [current_value_string] + choices

            value_select = bkmodel.Select(
                value=current_value_string,
                options=choices,
                width=180,
                styles={"color": "white", "font-size": "14px"},
            )
            new_row_widgets.append(row(label_input, value_select, spacing=4))
            option_value_strings.append(current_value_string)
            value_selects.append(value_select)

        self._options_value_selects = value_selects
        self._options_names = option_names

        self.state.options_rows_column.children = new_row_widgets
        # Fill the options_source
        self.state.options_source.data = dict(options=option_names, value=option_value_strings)
        # Wire the shared callback to every Select in the options table
        for value_select_widget in value_selects:
            value_select_widget.on_change("value", self._sync_options)

    def _sync_options(self, attr, old, new):
        """
        Bokeh ``on_change`` callback: keep ``options_source`` in sync with the
        options-table Select widgets.

        Wired to every Select in the options table by
        :meth:`_refresh_options_table`.  Reads the current value of each widget
        from ``self._options_value_selects`` (populated by that method) so that
        a single shared callback works regardless of which Select was changed.

        :param attr: Bokeh attribute name (unused, required by protocol).
        :param old: Previous widget value (unused).
        :param new: New widget value (unused; individual widget values are read
            directly from the stored Select references).
        """
        self.state.options_source.data = dict(
            options=self._options_names,
            value=[value_select.value for value_select in self._options_value_selects],
        )

    # -----------------------------------------------------------------------
    # Symmetry select
    # -----------------------------------------------------------------------

    def _refresh_symmetry_select(self, current_node_index: int, node_type: str):
        """
        Populate ``symmetry_select`` with the names of all placed nodes that
        share *node_type* with the currently selected node, excluding itself.

        :param current_node_index: Index of the node being edited.
        :param node_type: The ``node_type`` to filter candidates by — either
            the node's currently saved type, or the type currently selected
            in ``type_select`` if the user hasn't applied their change yet.
        """
        if self.state.symmetry_select is None:
            return

        placed_nodes_data = self.state.placed_nodes_source.data
        node_names = placed_nodes_data["name"]
        node_types = placed_nodes_data["node_type"]

        # Build a list of names of all other nodes that share the same node type
        symmetry_candidates = [
            node_names[node_position]
            for node_position in range(len(node_names))
            if node_position != current_node_index and node_types[node_position] == node_type
        ]

        choices = [_EMPTY] + symmetry_candidates
        self.state.symmetry_select.options = choices

        current_symmetry_component = placed_nodes_data["symmetry_name"][current_node_index]
        self.state.symmetry_select.value = (
            current_symmetry_component if current_symmetry_component in choices else _EMPTY
        )

    # -----------------------------------------------------------------------
    # Connections table
    # -----------------------------------------------------------------------

    def _refresh_connections_table(self, node_index: int):
        """
        Populate the Connections panel for node *node_index* using dynamic rows.

        For every source port of the selected node a row is created with a
        disabled label and a Select showing compatible unconnected target ports.
        A symmetric row is created for every target port.  Selecting a new
        value immediately draws (or removes) the corresponding preview edge.

        :param node_index: Zero-based index of the node to show connections for.
        """
        if (
            self.state.connections_rows_column is None
            or self.state.edge_source is None
            or self.state.source_port_source is None
            or self.state.target_port_source is None
        ):
            return

        source_data = copy_column_data(self.state.source_port_source.data)
        target_data = copy_column_data(self.state.target_port_source.data)
        node_data = copy_column_data(self.state.placed_nodes_source.data)
        edge_data = copy_column_data(self.state.edge_source.data)

        connected_source, connected_target = _connected_far_ports(edge_data, node_index)

        # Only free ports of the matching energy type (colour) can be offered
        icon_config = ICONS_CONFIG.get(node_data["icon_type"][node_index], {})
        free_targets = self._free_port_candidates(
            target_data, "target", node_index, icon_config.get("source_color", ""), node_data
        )
        free_sources = self._free_port_candidates(
            source_data, "source", node_index, icon_config.get("target_color", ""), node_data
        )

        # Component type of the selected node, used to highlight its primary port
        node_types = list(node_data.get("node_type", []))
        selected_node_type = node_types[node_index] if node_index < len(node_types) else _EMPTY

        # Both sides are built the same way: each port of the selected node gets a
        # label and a dropdown of the free ports on the opposite side. Sources are
        # listed first, then targets
        port_sides = (
            (source_data, "source", free_targets, connected_source),
            (target_data, "target", free_sources, connected_target),
        )
        new_row_widgets = [
            self._build_port_row(
                port_data,
                port_position,
                own_kind,
                selected_node_type,
                free_far_ports,
                connected_ports,
                node_data,
            )
            for port_data, own_kind, free_far_ports, connected_ports in port_sides
            for port_position in range(len(port_data.get("node_index", [])))
            if port_data["node_index"][port_position] == node_index
        ]

        self.state.connections_rows_column.children = new_row_widgets

    def _free_port_candidates(
        self, port_data: dict, kind: str, node_index: int, color: str, node_data: dict
    ) -> list:
        """
        Unconnected ports of other nodes that match the energy type *color*.

        :param port_data: ``source_port_source.data`` or ``target_port_source.data``.
        :param kind: ``"source"`` or ``"target"``, the side *port_data* holds.
        :param node_index: Selected node, whose own ports are excluded.
        :param color: Energy-type colour the port must have.
        :param node_data: Snapshot of ``placed_nodes_source.data``.
        :return: List of ``(node_index, label, kind, display_string)`` candidates.
        """
        return [
            (
                port_data["node_index"][port_position],
                port_data["label"][port_position],
                kind,
                self._far_string(
                    node_data,
                    port_data["node_index"][port_position],
                    port_data["label"][port_position],
                    kind,
                ),
            )
            for port_position in range(len(port_data.get("node_index", [])))
            if port_data["connected"][port_position] == "False"
            and port_data["node_index"][port_position] != node_index
            and port_data["color"][port_position] == color
        ]

    def _build_port_row(
        self,
        port_data: dict,
        port_position: int,
        own_kind: str,
        node_type: str,
        free_far_ports: list,
        connected_ports: dict,
        node_data: dict,
    ):
        """
        One Connections panel row: the port label and its connection dropdown.

        :param port_data: Data of the port source that holds the port.
        :param port_position: Position of the port in *port_data*.
        :param own_kind: ``"source"`` or ``"target"``, the side of the port.
        :param node_type: Component type of the selected node.
        :param free_far_ports: Free candidates on the opposite side
            (see :meth:`_free_port_candidates`).
        :param connected_ports: ``{own_label: (far_node_index, far_label)}`` for the
            node's connected ports on this side.
        :param node_data: Snapshot of ``placed_nodes_source.data``.
        :return: Bokeh ``row`` holding the label and the dropdown.
        """
        far_kind = "target" if own_kind == "source" else "source"
        primary_type_keywords = (
            PRIMARY_SOURCE_PORT_TYPE_KEYWORDS
            if own_kind == "source"
            else PRIMARY_TARGET_PORT_TYPE_KEYWORDS
        )
        node_index = port_data["node_index"][port_position]
        own_label = port_data["label"][port_position]
        is_connected = port_data["connected"][port_position] == "True"

        label_input = bkmodel.TextInput(
            value=f"{own_kind}:{own_label}",
            width=180,
            disabled=True,
            styles={"color": "white", "font-size": "14px"},
            stylesheets=_port_label_stylesheets(node_type, own_label, primary_type_keywords),
        )

        candidates = free_far_ports
        current_string = _EMPTY
        if is_connected and own_label in connected_ports:
            # Add the connected far port to the candidates list so that it shows up in
            # the dropdown even though it is not free. "Far" means the port on the
            # opposite side that this port of the node is (or could be) connected to
            far_node_index, far_label = connected_ports[own_label]
            current_string = self._far_string(node_data, far_node_index, far_label, far_kind)
            candidates = [
                candidate
                for candidate in candidates
                if not (candidate[0] == far_node_index and candidate[1] == far_label)
            ] + [(far_node_index, far_label, far_kind, current_string)]

        # Index 3 is the display string, used for the dropdown entries; the current
        # string is excluded here so it is not listed twice
        choices = [_EMPTY] + [
            candidate[3] for candidate in candidates if candidate[3] != current_string
        ]

        # Add the current string to the choices if it is not already present
        if current_string and current_string not in choices:
            choices = [current_string] + choices

        value_select = bkmodel.Select(
            value=current_string if current_string else _EMPTY,
            options=choices,
            width=180,
            styles={"color": "white", "font-size": "12px"},
        )

        value_select.on_change(
            "value",
            self._make_port_callback(
                _own_kind=own_kind,
                _own_label=own_label,
                _own_node_index=node_index,
                _own_color=port_data["color"][port_position],
                _own_x=port_data["x"][port_position],
                _own_y=port_data["y"][port_position],
                _candidates=candidates,
            ),
        )
        return row(label_input, value_select, spacing=4)

    @staticmethod
    def _far_string(node_data: dict, far_node_index: int, far_label: str, far_kind: str) -> str:
        """
        Format a port reference as ``"<node_name> (<kind>:<label>)"``.
        Here, "far" refers to a candidate port at the other end of the connection, which is either
        a source or a target port.

        :param node_data: Snapshot of ``placed_nodes_source.data``.
        :param far_node_index: Node index of the far-end port.
        :param far_label: Port label of the far-end port (e.g. ``"1"``).
        :param far_kind: ``"source"`` or ``"target"``.
        """
        name = (
            node_data["name"][far_node_index]
            if far_node_index < len(node_data.get("name", []))
            else f"node_{far_node_index}"
        )
        return f"{name} ({far_kind}:{far_label})"

    @staticmethod
    def _parse_choice(choice_string: str, candidates: list[tuple]) -> tuple | None:
        """
        Reverse-lookup a display string in a candidates list.

        :param choice_string: The string currently selected in the dropdown.
        :param candidates: List of ``(node_index, label, kind, display_str)`` tuples.
        :return: ``(node_index, label, kind)`` for the matching entry, or ``None``.
        """
        for node_index, label, kind, display_string in candidates:
            if display_string == choice_string:
                return node_index, label, kind
        return None

    def _make_port_callback(
        self,
        _own_kind: str,
        _own_label: str,
        _own_node_index: int,
        _own_color: str,
        _own_x: float,
        _own_y: float,
        _candidates: list,
    ):
        """
        Return a Bokeh ``on_change`` callback bound to a specific port row.

        Wraps :meth:`_on_port_select_change` via :func:`functools.partial`, pre-filling every
        port-specific argument so Bokeh receives a plain ``(attr, old, new)`` callable with no
        inner function required.

        * When ``_own_kind == "source"``:  own port = starting_port, far port = ending_port (
        target kind).
        * When ``_own_kind == "target"``:  own port = ending_port, far port = starting_port (
        source kind).

        The edge is always stored as source -> target regardless of which direction was clicked,
        so ``starting_port`` is always source-kind and ``ending_port`` is always target-kind.

        :param _own_kind: ``"source"`` or ``"target"``.
        :param _own_label: Port label of the own port (e.g. ``"1"``).
        :param _own_node_index: Node index of the own port.
        :param _own_color: Energy-type colour of the own port.
        :param _own_x: Snapshot x of the own port at callback-build time.
        :param _own_y: Snapshot y of the own port at callback-build time.
        :param _candidates: Pre-built list of ``(node_index, label, kind, display_str)`` tuples
        for the far end.
        """
        return functools.partial(
            self._on_port_select_change,
            _own_kind=_own_kind,
            _own_label=_own_label,
            _own_node_index=_own_node_index,
            _own_color=_own_color,
            _own_x=_own_x,
            _own_y=_own_y,
            _candidates=_candidates,
        )

    def _on_port_select_change(
        self,
        attr,
        old,
        new_value,
        *,
        _own_kind: str,
        _own_label: str,
        _own_node_index: int,
        _own_color: str,
        _own_x: float,
        _own_y: float,
        _candidates: list,
    ):
        """
        Handle a value change on a port-row Select widget.

        Called indirectly via :meth:`_make_port_callback` / ``functools.partial``.
        Implements the three-phase logic shared by source and target port rows:

        1. **Prune** the pending preview-edge list for the own port.
        2. **Disconnect** (if *new_value* is empty): remove the permanent edge that involves the
        own port and refresh the canvas.
        3. **Connect** (otherwise): resolve the chosen far-end port, re-fetch live coordinates
        for both ends, build source->target port dicts, and register a new preview edge.

        :param attr: Bokeh change attribute name (unused, required by protocol).
        :param old: Previous widget value (unused).
        :param new_value: Newly selected display string, or ``_EMPTY``.
        :param _own_kind: ``"source"`` or ``"target"`` — which end is "ours".
        :param _own_label: Port label of the own port.
        :param _own_node_index: Node index of the own port.
        :param _own_color: Energy-type colour of the own port.
        :param _own_x: Snapshot x coordinate captured when the row was built.
        :param _own_y: Snapshot y coordinate captured when the row was built.
        :param _candidates: Candidate ``(node_index, label, kind, display_str)`` tuples for the
        far end, built when the row was rendered.
        """
        # Drop any pending preview edge that already uses the own port. Pending
        # connections are stored as (source port, target port) pairs
        own_end = 0 if _own_kind == "source" else 1
        self.state.panel_pending_connections = [
            pending_connection
            for pending_connection in self.state.panel_pending_connections
            if not _is_same_port(
                pending_connection[own_end], _own_kind, _own_node_index, _own_label
            )
        ]

        self._clear_temp_edge_visuals()
        for starting_port, ending_port in list(self.state.panel_pending_connections):
            self._add_edge_temp(starting_port, ending_port)

        if new_value == _EMPTY:
            self._disconnect_port(_own_kind, _own_node_index, _own_label)
            return

        # Resolve the chosen far-end port from the candidates list
        parsed_choice = self._parse_choice(new_value, _candidates)
        if parsed_choice is None:
            return
        far_node_index, far_label, far_kind = parsed_choice

        # Look up the far port's live coordinates and colour
        far_port = self._live_port(far_kind, far_node_index, far_label)
        if far_port is None:
            return

        # Re-fetch own port's live coordinates (may have moved since the row was built)
        own_port = self._live_port(_own_kind, _own_node_index, _own_label) or {
            "kind": _own_kind,
            "node_index": _own_node_index,
            "label": _own_label,
            "x": _own_x,
            "y": _own_y,
        }
        own_port["color"] = _own_color

        # Build the edge always as source -> target
        starting_port, ending_port = (
            (own_port, far_port) if _own_kind == "source" else (far_port, own_port)
        )
        self.state.panel_pending_connections.append((starting_port, ending_port))
        self._add_edge_temp(starting_port, ending_port)

    def _disconnect_port(self, port_kind: str, node_index: int, port_label: str):
        """
        Remove every permanent edge attached to one port, as a single undoable step.

        :param port_kind: ``"source"`` or ``"target"``.
        :param node_index: Zero-based index of the node that owns the port.
        :param port_label: Port label (``"1"``, ``"2"``, ...).
        """
        if self.state.edge_source is None:
            return
        self._push_undo()
        edge_data = copy_column_data(self.state.edge_source.data)
        self.state.edge_source.data = select_edges(
            edge_data, edge_indices_not_touching_port(edge_data, node_index, port_label, port_kind)
        )
        self._rebuild_all_ports()
        self._mark_unsaved()

    def _live_port(self, port_kind: str, node_index: int, port_label: str) -> dict | None:
        """
        Current metadata of one port ball, read from its port data source.

        :param port_kind: ``"source"`` or ``"target"``.
        :param node_index: Zero-based index of the node that owns the port.
        :param port_label: Port label (``"1"``, ``"2"``, ...).
        :return: Dict with keys ``kind``, ``node_index``, ``label``, ``x``, ``y``
            and ``color``, or ``None`` if the port does not exist.
        """
        port_source = (
            self.state.source_port_source
            if port_kind == "source"
            else self.state.target_port_source
        )
        port_data = port_source.data
        for port_position in range(len(port_data.get("node_index", []))):
            if (
                port_data["node_index"][port_position] == node_index
                and port_data["label"][port_position] == port_label
            ):
                return {
                    "kind": port_kind,
                    "node_index": node_index,
                    "label": port_label,
                    "x": port_data["x"][port_position],
                    "y": port_data["y"][port_position],
                    "color": port_data["color"][port_position],
                }
        return None

    # -----------------------------------------------------------------------
    # Type-select live refresh
    # -----------------------------------------------------------------------

    def _on_type_select_change(self, attr, old, new):
        """
        Bokeh ``on_change`` callback wired to ``state.type_select``.

        Refreshes the choices in position select object, rebuilds the Options table
        for the newly chosen component type (pre-populated with the selected
        node's previously saved option values, if any), and refreshes the
        Symmetry select so its candidate list matches ``new`` — the type
        currently chosen in the dropdown, not the node's previously saved
        type — so the symmetry candidates update live as the user browses
        component types, before Apply is clicked.

        When the user switches the selected node to a type other than its saved
        one, every connection of the node and its symmetry link are cleared
        straight away (see :meth:`_clear_node_connections_and_symmetry`), as
        their ports and peer belong to the old type. Like a disconnect from the
        Connections panel, this is a single undoable step.

        The type, position and options themselves are not written back to
        ``placed_nodes_source`` here — that happens when the user clicks Apply
        (see :meth:`apply_node_configurations`).

        :param attr: Bokeh attribute name (unused, required by protocol).
        :param old: Previous ``type_select`` value (unused).
        :param new: Newly selected component type.
        """
        position_choices = self.state.possible_position.get(new, [])
        self.state.position_select.options = position_choices
        self.state.position_select.value = position_choices[0] if position_choices else _EMPTY

        current_selected_node_index_for_type = self.state.selected_node_index
        saved_options = {}
        if current_selected_node_index_for_type is not None:
            saved_options_json = self.state.placed_nodes_source.data["options"][
                current_selected_node_index_for_type
            ]
            try:
                saved_options = json.loads(saved_options_json or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
        self._refresh_options_table(new, saved_options)
        self.state.options_table.visible = bool(self.state.possible_options.get(new, {}))

        if current_selected_node_index_for_type is None:
            return

        # Only a user switch to another type clears the node, not the panel being
        # filled for a newly selected node or a switch back to the saved type
        saved_node_type = self.state.placed_nodes_source.data["node_type"][
            current_selected_node_index_for_type
        ]
        if not self._is_populating_node_table and new != saved_node_type:
            self._clear_node_connections_and_symmetry(current_selected_node_index_for_type)

        self._refresh_symmetry_select(current_selected_node_index_for_type, new)

    def _clear_node_connections_and_symmetry(self, node_index: int):
        """
        Remove every edge attached to node *node_index* and break its symmetry link.

        Pending preview connections of the Connections panel are discarded, and
        edges hidden by a port-count preview are restored first, so the removal
        works on the real edges. The symmetry link is cleared on both sides:
        the node itself and every node that names it as its peer. A single undo
        snapshot is pushed, and only if something is actually removed.

        :param node_index: Zero-based index of the node to clear.
        """
        # Previews belong to the old type's ports: drop them before editing
        self._clear_temp_edges()
        if self.state.edge_preview_backup is not None:
            self.state.edge_source.data = copy_column_data(self.state.edge_preview_backup)
            self.state.edge_preview_backup = None

        edge_data = (
            copy_column_data(self.state.edge_source.data)
            if self.state.edge_source is not None
            else {}
        )
        num_edges = len(edge_data.get("starting_node_index", []))
        kept_edge_indices = edge_indices_not_touching_node(edge_data, node_index)

        placed_nodes_data = copy_column_data(self.state.placed_nodes_source.data)
        node_names = list(placed_nodes_data.get("name", []))
        node_name = node_names[node_index] if node_index < len(node_names) else None
        symmetry_names, symmetry_indices = _padded_symmetry_columns(placed_nodes_data)

        # The node itself plus any node pointing back at it
        symmetry_positions_to_clear = _positions_linked_to(
            symmetry_names, symmetry_indices, {node_index}, {node_name}
        )
        node_has_link = node_index < len(symmetry_names) and (
            symmetry_names[node_index] not in (_EMPTY, None)
            or symmetry_indices[node_index] != _NO_SYMMETRY_PEER
        )
        if node_has_link and node_index not in symmetry_positions_to_clear:
            symmetry_positions_to_clear.append(node_index)

        has_edges_to_remove = len(kept_edge_indices) < num_edges
        if not has_edges_to_remove and not symmetry_positions_to_clear:
            self._refresh_connections_table(node_index)
            return

        # Capture state before clearing so a single Undo brings everything back
        self._push_undo()

        if has_edges_to_remove:
            self.state.edge_source.data = select_edges(edge_data, kept_edge_indices)

        if symmetry_positions_to_clear:
            _unlink_symmetry(symmetry_names, symmetry_indices, symmetry_positions_to_clear)
            placed_nodes_data["symmetry_name"] = symmetry_names
            placed_nodes_data["symmetry_node_index"] = symmetry_indices
            self.state.placed_nodes_source.data = placed_nodes_data

        self._rebuild_all_ports()
        self._mark_unsaved()
        _LOGGER.info(
            "Type switch on %s: removed %d connection(s) and %d symmetry link(s).",
            node_name,
            num_edges - len(kept_edge_indices),
            len(symmetry_positions_to_clear),
        )

    # -----------------------------------------------------------------------
    # Apply button
    # -----------------------------------------------------------------------

    def apply_node_configurations(self):
        """
        Commit the config panel's current widget values to the selected node.

        Wired to ``state.apply_button``. Pushes a single undo snapshot
        covering the whole operation, then writes ``name_input``,
        ``type_select``, ``position_select``, the options table, the
        port-count spinners (when visible), and the symmetry selection back
        into ``placed_nodes_source`` for the currently selected node. Also
        commits any pending port connections built up via the Connections
        panel and refreshes ports / connections afterwards.

        No-op if no node is currently selected.
        """
        current_selected_node_index = self.state.selected_node_index
        num_nodes = len(self.state.placed_nodes_source.data.get("name", []))
        if current_selected_node_index is None or current_selected_node_index >= num_nodes:
            return

        # Push undo snapshot *before* any mutation so the user can revert the
        # entire Apply operation with a single Undo click
        self._push_undo()

        placed_nodes_data = copy_column_data(self.state.placed_nodes_source.data)
        self._write_node_fields(placed_nodes_data, current_selected_node_index)
        self._write_symmetry(placed_nodes_data, current_selected_node_index)
        self.state.placed_nodes_source.data = placed_nodes_data
        self._sync_hover_source(placed_nodes_data, current_selected_node_index)

        # Commit the connections picked in the Connections panel
        for starting_port, ending_port in list(self.state.panel_pending_connections):
            self._add_edge(starting_port, ending_port)
        self._clear_temp_edges()
        self._rebuild_all_ports()

        self._mark_unsaved()

    def _write_node_fields(self, placed_nodes_data: dict, node_index: int):
        """
        Write the panel's name, type, position, options and port counts into a node row.

        :param placed_nodes_data: Copy of ``placed_nodes_source.data``, edited in place.
        :param node_index: Zero-based index of the node being applied.
        """
        new_node_type = self.state.type_select.value
        _set_cell(placed_nodes_data, "name", node_index, self.state.name_input.value)
        _set_cell(placed_nodes_data, "position", node_index, self.state.position_select.value)

        type_is_written = _set_cell(placed_nodes_data, "node_type", node_index, new_node_type)
        # Reset port counts to defaults when the type changes and spinners are hidden
        if type_is_written and not self.state.source_count_spinner.visible:
            default_source_port_count = int(self.state.default_source_count.get(new_node_type, 0))
            default_target_port_count = int(self.state.default_target_count.get(new_node_type, 0))
            _set_cell(placed_nodes_data, "n_sources", node_index, default_source_port_count)
            _set_cell(placed_nodes_data, "n_targets", node_index, default_target_port_count)

        # Place empty dict if there is no options entry for this node yet
        if "options" not in placed_nodes_data:
            placed_nodes_data["options"] = ["{}"] * len(placed_nodes_data.get("name", []))
        option_names = list(self.state.options_source.data.get("options", []))
        option_values = list(self.state.options_source.data.get("value", []))
        option_values_dict = {
            option_name: _strings_to_option_values(option_value)
            for option_name, option_value in zip(option_names, option_values, strict=True)
        }
        # Stored as a JSON string in the ColumnDataSource
        _set_cell(placed_nodes_data, "options", node_index, json.dumps(option_values_dict))

        # These two spinners show only if the node type supports multiple ports
        if self.state.source_count_spinner.visible:
            new_source_port_count = int(self.state.source_count_spinner.value)
            _set_cell(placed_nodes_data, "n_sources", node_index, new_source_port_count)
        if self.state.target_count_spinner.visible:
            new_target_port_count = int(self.state.target_count_spinner.value)
            _set_cell(placed_nodes_data, "n_targets", node_index, new_target_port_count)

    def _write_symmetry(self, placed_nodes_data: dict, node_index: int):
        """
        Link node *node_index* with the peer chosen in ``symmetry_select``.

        Both nodes point at each other afterwards, and any older link to either
        of them is removed.

        :param placed_nodes_data: Copy of ``placed_nodes_source.data`` whose name
            column already holds the applied name; edited in place.
        :param node_index: Zero-based index of the node being applied.
        """
        all_node_names = placed_nodes_data.get("name", [])
        current_node_name = all_node_names[node_index]
        new_symmetry_name = self.state.symmetry_select.value

        # Ensure symmetry columns exist and are fully sized (handles newly added nodes)
        symmetry_names, symmetry_indices = _padded_symmetry_columns(placed_nodes_data)

        # Resolve the newly chosen peer index
        new_peer_index = _NO_SYMMETRY_PEER
        if new_symmetry_name and new_symmetry_name != _EMPTY:
            new_peer_index = next(
                (
                    node_position
                    for node_position, node_name in enumerate(all_node_names)
                    if node_name == new_symmetry_name
                ),
                _NO_SYMMETRY_PEER,
            )

        # Clear stale references on both sides before writing
        #
        # Two nodes may be displaced by this Apply:
        #   (a) current node's *old* peer  — it pointed back at current_node;
        #       now current_node is leaving, so clear it.
        #   (b) new peer's *old* partner   — new_peer previously pointed at
        #       some other node D; D still points at new_peer, so clear D too.
        #
        # We handle both by wiping every node that points at the current node or
        # at the new peer, excluding the two nodes that get fresh values below.
        # Links are matched by peer index too, as the current node's name may
        # have just changed in this same Apply

        nodes_getting_fresh_write = {node_index}
        linked_node_indices = {node_index}
        linked_node_names = {current_node_name}
        if new_peer_index != _NO_SYMMETRY_PEER:
            nodes_getting_fresh_write.add(new_peer_index)
            linked_node_indices.add(new_peer_index)
            linked_node_names.add(new_symmetry_name)

        stale_positions = [
            node_position
            for node_position in _positions_linked_to(
                symmetry_names, symmetry_indices, linked_node_indices, linked_node_names
            )
            if node_position not in nodes_getting_fresh_write
        ]
        _unlink_symmetry(symmetry_names, symmetry_indices, stale_positions)

        # Write the current node's new symmetry entry
        symmetry_names[node_index] = new_symmetry_name
        symmetry_indices[node_index] = new_peer_index

        # Write the new peer's back-reference (if a peer was chosen)
        if new_peer_index != _NO_SYMMETRY_PEER:
            symmetry_names[new_peer_index] = current_node_name
            symmetry_indices[new_peer_index] = node_index

        placed_nodes_data["symmetry_name"] = symmetry_names
        placed_nodes_data["symmetry_node_index"] = symmetry_indices

    def _sync_hover_source(self, placed_nodes_data: dict, node_index: int):
        """
        Copy the applied name and type of node *node_index* into ``hover_source``.

        :param placed_nodes_data: Node columns after the Apply.
        :param node_index: Zero-based index of the node being applied.
        """
        if self.state.hover_source is None:
            return
        hover_data = copy_column_data(self.state.hover_source.data)
        for column_name in ("name", "node_type"):
            if node_index < len(placed_nodes_data.get(column_name, [])):
                _set_cell(
                    hover_data, column_name, node_index, placed_nodes_data[column_name][node_index]
                )
        self.state.hover_source.data = hover_data


def _set_cell(column_data: dict, column_name: str, row_index: int, value) -> bool:
    """
    Set one cell of column data if the column exists and the row is in range.

    :param column_data: ``ColumnDataSource.data`` copy, edited in place.
    :param column_name: Column to write.
    :param row_index: Zero-based row to write.
    :param value: New value of the cell.
    :return: ``True`` if the cell was written.
    """
    column_values = column_data.get(column_name)
    if column_values is None or row_index >= len(column_values):
        return False
    column_values[row_index] = value
    return True


def _is_same_port(port: dict, port_kind: str, node_index: int, port_label: str) -> bool:
    """
    Whether the port dict *port* is the port ``(port_kind, node_index, port_label)``.

    :param port: Port dict with keys ``kind``, ``node_index`` and ``label``.
    :param port_kind: ``"source"`` or ``"target"``.
    :param node_index: Zero-based index of the node that owns the port.
    :param port_label: Port label (``"1"``, ``"2"``, ...).
    :return: ``True`` for the same port.
    """
    return (
        port["kind"] == port_kind
        and port["node_index"] == node_index
        and port["label"] == port_label
    )


def _connected_far_ports(edge_data: dict, node_index: int) -> tuple[dict, dict]:
    """
    Far end of every connection of node *node_index*, split by the node's port side.

    An edge is stored with whichever port was clicked first as its start, so
    both ends are checked, each on its own.

    :param edge_data: ``edge_source.data`` (or a copy of it).
    :param node_index: Zero-based index of the node.
    :return: Tuple ``(connected_source, connected_target)`` of dictionaries
        mapping the node's own port label to ``(far_node_index, far_label)``.
    """
    connected_ports = {"source": {}, "target": {}}
    for edge_position in range(len(edge_data.get("starting_node_index", []))):
        starting_end = (
            edge_data["starting_node_index"][edge_position],
            edge_data["starting_port_label"][edge_position],
            edge_data["starting_port_kind"][edge_position],
        )
        ending_end = (
            edge_data["ending_node_index"][edge_position],
            edge_data["ending_port_label"][edge_position],
            edge_data["ending_port_kind"][edge_position],
        )
        for own_end, far_end in ((starting_end, ending_end), (ending_end, starting_end)):
            own_node_index, own_label, own_kind = own_end
            if own_node_index == node_index and own_kind in connected_ports:
                connected_ports[own_kind][own_label] = far_end[:2]
    return connected_ports["source"], connected_ports["target"]


def _padded_symmetry_columns(placed_nodes_data: dict) -> tuple[list, list]:
    """
    Copies of the two symmetry columns, padded to one entry per node.

    :param placed_nodes_data: ``placed_nodes_source.data`` (or a copy of it).
    :return: Tuple ``(symmetry_names, symmetry_indices)`` of new lists.
    """
    num_nodes = len(placed_nodes_data.get("name", []))
    symmetry_names = list(placed_nodes_data.get("symmetry_name", []))
    symmetry_indices = list(placed_nodes_data.get("symmetry_node_index", []))
    symmetry_names += [_EMPTY] * (num_nodes - len(symmetry_names))
    symmetry_indices += [_NO_SYMMETRY_PEER] * (num_nodes - len(symmetry_indices))
    return symmetry_names, symmetry_indices


def _positions_linked_to(
    symmetry_names: list, symmetry_indices: list, node_indices: set, node_names: set
) -> list:
    """
    Positions of the nodes whose symmetry peer is one of the given nodes.

    A link counts if it matches either the peer index or the peer name, so a
    link is still found when one of the two is out of date.

    :param symmetry_names: ``symmetry_name`` column.
    :param symmetry_indices: ``symmetry_node_index`` column.
    :param node_indices: Indices of the nodes whose incoming links are wanted.
    :param node_names: Names of the same nodes.
    :return: List of node positions, in order.
    """
    peer_names = {node_name for node_name in node_names if node_name not in (_EMPTY, None)}
    return [
        node_position
        for node_position, (peer_name, peer_index) in enumerate(
            zip(symmetry_names, symmetry_indices, strict=True)
        )
        if peer_index in node_indices or peer_name in peer_names
    ]


def _unlink_symmetry(symmetry_names: list, symmetry_indices: list, node_positions: list) -> None:
    """
    Reset the symmetry entries at *node_positions*, in place.

    :param symmetry_names: ``symmetry_name`` column, edited in place.
    :param symmetry_indices: ``symmetry_node_index`` column, edited in place.
    :param node_positions: Positions of the nodes to unlink.
    """
    for node_position in node_positions:
        symmetry_names[node_position] = _EMPTY
        symmetry_indices[node_position] = _NO_SYMMETRY_PEER


def _port_label_stylesheets(node_type: str, port_label: str, primary_type_keywords: tuple) -> list:
    """
    Stylesheets of a port label in the Connections panel.

    The primary port of a power split definition is shown in light blue, every
    other port keeps Bokeh's default text color. The color cannot go through
    ``styles``: that only styles the widget's outer element, while the text sits
    in an ``<input>`` inside the widget's shadow DOM, whose color Bokeh sets with
    its own ``.bk-input:disabled`` rule. A ``:host input:disabled`` stylesheet is
    specific enough to override it, and ``-webkit-text-fill-color`` is needed
    because Chromium and Safari paint disabled input text with it.

    :param node_type: Component type of the node that owns the port.
    :param port_label: Port label (``"1"``, ``"2"``, ...).
    :param primary_type_keywords: Component type keywords whose first port on
        this side is the primary port.
    :return: List of CSS stylesheets for the label's ``TextInput``.
    """
    is_primary_type = any(keyword in node_type for keyword in primary_type_keywords)
    if not (is_primary_type and str(port_label) == PRIMARY_PORT_LABEL):
        return []
    return [
        f":host input:disabled {{ color: {PRIMARY_PORT_LABEL_COLOR}; "
        f"-webkit-text-fill-color: {PRIMARY_PORT_LABEL_COLOR}; }}"
    ]


def _option_values_to_strings(value) -> str:
    """
    Convert an option value (bool, int, float, or str) to its display string.

    :param value: Option value to convert.
    :return: String displayed in the options table.
    """
    if value is True:
        return "True"
    if value is False:
        return "False"
    return str(value)


def _strings_to_option_values(string: str):
    """
    Parse a display string back to a typed Python value (bool, int, float, or str).

    :param string: Display string read from the options table.
    :return: The typed option value.
    """
    if string == "True":
        return True
    if string == "False":
        return False
    try:
        return int(string)
    except (ValueError, TypeError):
        pass
    try:
        return float(string)
    except (ValueError, TypeError):
        pass
    return string
