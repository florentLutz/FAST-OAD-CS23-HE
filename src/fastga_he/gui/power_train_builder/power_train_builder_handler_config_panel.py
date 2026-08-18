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
from typing import TYPE_CHECKING
import bokeh.models as bkmodel
from bokeh.layouts import row

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG
from .power_train_builder_state import _EMPTY, _NO_SYMMETRY_PEER

_LOGGER = logging.getLogger(__name__)

# For IDE type-checking only
if TYPE_CHECKING:
    from .power_train_builder_state import BuilderState


class ConfigPanelMixin:
    """
    Handles the right-hand config panel: populating, refreshing, and clearing
    the node-name input, type/position selects, options table, connections table,
    symmetry select, and port-count spinners.

    Depends on ``self.state`` (a :class:`BuilderState` instance) being set by
    the concrete class before any method is called.
    """

    state: "BuilderState"

    # -----------------------------------------------------------------------
    # Node table population / clearing
    # -----------------------------------------------------------------------

    def _populate_node_table(self, index: int):
        """
        Fill the config panel widgets with the stored values of node *index*.

        :param index: Zero-based index into ``placed_nodes_source`` data arrays.
        """
        placed_node_data = self.state.placed_nodes_source.data
        node_name = list(placed_node_data.get("name", []))[index]
        node_type = list(placed_node_data.get("node_type", []))[index]
        icon_key = list(placed_node_data.get("icon_type", []))[index]
        position = (
            list(placed_node_data.get("position", []))[index]
            if placed_node_data.get("position")
            else _EMPTY
        )
        saved_options_json = (
            list(placed_node_data.get("options", []))[index]
            if placed_node_data.get("options")
            else "{}"
        )

        self.state.options_table.visible = saved_options_json != "{}"
        self.state.name_input.value = node_name

        choices = self.state.component_type_to_icon.get(icon_key, icon_key)
        self.state.type_select.options = choices
        self.state.type_select.value = node_type if node_type in choices else choices[0]

        pos_choices = self.state.possible_position.get(node_type, [])
        self.state.position_select.options = pos_choices
        if pos_choices:
            self.state.position_select.value = (
                position if position in pos_choices else pos_choices[0]
            )
        else:
            self.state.position_select.value = ""

        saved_options: dict = {}
        try:
            saved_options = json.loads(saved_options_json) if saved_options_json else {}
        except (json.JSONDecodeError, TypeError):
            pass

        self._refresh_options_table(node_type, saved_options)

        default_source_port_count = self.state.default_source_count.get(node_type, 0)
        default_target_port_count = self.state.default_target_count.get(node_type, 0)
        source_ports_are_editable = default_source_port_count == 3
        target_ports_are_editable = default_target_port_count == 3

        current_source_port_count = (
            list(placed_node_data.get("n_sources", []))[index]
            if placed_node_data.get("n_sources")
            else default_source_port_count
        )
        current_target_port_count = (
            list(placed_node_data.get("n_targets", []))[index]
            if placed_node_data.get("n_targets")
            else default_target_port_count
        )

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
                x=[list(placed_node_data["x"])[index]],
                y=[list(placed_node_data["y"])[index]],
            )
        self._rebuild_all_ports()
        self._refresh_connections_table(index)
        self._refresh_symmetry_select(index, icon_key)

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

        self.state.table_panel.visible = False

        if self.state.selected_node_overlay_source is not None:
            self.state.selected_node_overlay_source.data = dict(x=[], y=[])
        if self.state.connections_source is not None:
            self.state.connections_source.data = dict(my_port=[], connected_to=[], edge_index=[])
        self.state.connections_rows_column.children = []

        self.state.symmetry_select.options = [_EMPTY]
        self.state.symmetry_select.value = _EMPTY
        self._rebuild_all_ports()

    # -----------------------------------------------------------------------
    # Options table
    # -----------------------------------------------------------------------

    def _refresh_options_table(self, node_type: str, overrides: dict = {}):
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
        opt_values = []
        new_rows = []
        value_selects = []

        for option_name, value_list in options_definition.items():
            current_values = (
                overrides[option_name]
                if option_name in overrides
                else (value_list[0] if value_list else _EMPTY)
            )
            current_stringing = _option_values_to_strings(current_values)

            label_input = bkmodel.TextInput(
                value=option_name,
                width=180,
                disabled=True,
                styles={"color": "white", "font-size": "14px"},
            )
            choices = (
                [_option_values_to_strings(choice) for choice in value_list]
                if value_list
                else [current_stringing]
            )
            if current_stringing not in choices:
                choices = [current_stringing] + choices

            value_select = bkmodel.Select(
                value=current_stringing,
                options=choices,
                width=180,
                styles={"color": "white", "font-size": "14px"},
            )
            new_rows.append(row(label_input, value_select, spacing=4))
            opt_values.append(current_stringing)
            value_selects.append(value_select)

        self._options_value_selects = value_selects
        self._options_names = option_names

        for value_selected in value_selects:
            value_selected.on_change("value", self._sync_options)

        self.state.options_rows_column.children = new_rows

        self.state.options_source.data = dict(options=option_names, value=opt_values)

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
            value=[s.value for s in self._options_value_selects],
        )

    # -----------------------------------------------------------------------
    # Symmetry select
    # -----------------------------------------------------------------------

    def _refresh_symmetry_select(self, current_node_index: int, icon_type: str):
        """
        Populate ``symmetry_select`` with the names of all placed nodes that
        share *icon_type* with the currently selected node, excluding itself.

        :param current_node_index: Index of the node being edited.
        :param icon_type: The ``icon_type`` key of the selected node.
        """
        if self.state.symmetry_select is None:
            return

        placed_nodes_data = self.state.placed_nodes_source.data
        names = list(placed_nodes_data.get("name", []))
        icon_types = list(placed_nodes_data.get("icon_type", []))
        saved_symmetry = list(placed_nodes_data.get("symmetry_name", []))

        symmetry_candidates = [
            names[i]
            for i in range(len(names))
            if i != current_node_index and icon_types[i] == icon_type
        ]
        choices = [_EMPTY] + symmetry_candidates
        self.state.symmetry_select.options = choices

        current_symmetry_component = (
            saved_symmetry[current_node_index]
            if current_node_index < len(saved_symmetry)
            else _EMPTY
        )
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

        source_data = {
            key: list(value) for key, value in self.state.source_port_source.data.items()
        }
        target_data = {
            key: list(value) for key, value in self.state.target_port_source.data.items()
        }
        node_data = {key: list(value) for key, value in self.state.placed_nodes_source.data.items()}
        edge_data = {key: list(value) for key, value in self.state.edge_source.data.items()}

        # Build lookup: port label → (far_node_index, far_label) for every edge
        # that touches node_index.
        connected_source: dict[str, tuple[int, str]] = {}
        connected_target: dict[str, tuple[int, str]] = {}

        for i in range(len(edge_data.get("starting_node_index", []))):
            starting_node_position = edge_data["starting_node_index"][i]
            ending_node_position = edge_data["ending_node_index"][i]
            starting_port_label_value = edge_data["starting_port_label"][i]
            ending_port_label_value = edge_data["ending_port_label"][i]
            starting_port_kind_value = edge_data["starting_port_kind"][i]
            ending_port_kind_value = edge_data["ending_port_kind"][i]

            if starting_node_position == node_index and starting_port_kind_value == "source":
                connected_source[starting_port_label_value] = (
                    ending_node_position,
                    ending_port_label_value,
                )
            elif ending_node_position == node_index and ending_port_kind_value == "source":
                connected_source[ending_port_label_value] = (
                    starting_node_position,
                    starting_port_label_value,
                )
            if starting_node_position == node_index and starting_port_kind_value == "target":
                connected_target[starting_port_label_value] = (
                    ending_node_position,
                    ending_port_label_value,
                )
            elif ending_node_position == node_index and ending_port_kind_value == "target":
                connected_target[ending_port_label_value] = (
                    starting_node_position,
                    starting_port_label_value,
                )

        selected_source_color = ICONS_CONFIG.get(node_data["icon_type"][node_index], {}).get(
            "source_color", ""
        )
        free_targets: list[tuple[int, str, str, str]] = []
        for j in range(len(target_data.get("node_index", []))):
            if (
                target_data["connected"][j] == "False"
                and target_data["node_index"][j] != node_index
                and target_data["color"][j] == selected_source_color
            ):
                free_targets.append(
                    (
                        target_data["node_index"][j],
                        target_data["label"][j],
                        "target",
                        self._far_string(
                            node_data,
                            target_data["node_index"][j],
                            target_data["label"][j],
                            "target",
                        ),
                    )
                )

        selected_target_color = ICONS_CONFIG.get(node_data["icon_type"][node_index], {}).get(
            "target_color", ""
        )
        free_sources: list[tuple[int, str, str, str]] = []
        for j in range(len(source_data.get("node_index", []))):
            if (
                source_data["connected"][j] == "False"
                and source_data["node_index"][j] != node_index
                and source_data["color"][j] == selected_target_color
            ):
                free_sources.append(
                    (
                        source_data["node_index"][j],
                        source_data["label"][j],
                        "source",
                        self._far_string(
                            node_data,
                            source_data["node_index"][j],
                            source_data["label"][j],
                            "source",
                        ),
                    )
                )

        new_rows = []

        # ── Source-port rows ──────────────────────────────────────────────────
        for i in range(len(source_data.get("node_index", []))):
            if source_data["node_index"][i] != node_index:
                continue

            source_label = source_data["label"][i]
            is_connected = source_data["connected"][i] == "True"

            label_input = bkmodel.TextInput(
                value=f"source:{source_label}",
                width=180,
                disabled=True,
                styles={"color": "white", "font-size": "14px"},
            )

            candidates = list(free_targets)
            current_string = _EMPTY
            if is_connected and source_label in connected_source:
                far_node_i, far_label = connected_source[source_label]
                current_string = self._far_string(node_data, far_node_i, far_label, "target")
                candidates = [
                    candidate
                    for candidate in candidates
                    if not (candidate[0] == far_node_i and candidate[1] == far_label)
                ] + [
                    (
                        far_node_i,
                        far_label,
                        "target",
                        self._far_string(node_data, far_node_i, far_label, "target"),
                    )
                ]

            choices = [_EMPTY] + [
                candidate[3] for candidate in candidates if candidate[3] != current_string
            ]
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
                    _own_kind="source",
                    _own_label=source_label,
                    _own_node_index=node_index,
                    _own_color=source_data["color"][i],
                    _own_x=source_data["x"][i],
                    _own_y=source_data["y"][i],
                    _candidates=candidates,
                ),
            )
            new_rows.append(row(label_input, value_select, spacing=4))

        # ── Target-port rows ──────────────────────────────────────────────────
        for i in range(len(target_data.get("node_index", []))):
            if target_data["node_index"][i] != node_index:
                continue

            target_label = target_data["label"][i]
            is_connected = target_data["connected"][i] == "True"

            label_input = bkmodel.TextInput(
                value=f"target:{target_label}",
                width=180,
                disabled=True,
                styles={"color": "white", "font-size": "14px"},
            )

            candidates = list(free_sources)
            current_string = _EMPTY
            if is_connected and target_label in connected_target:
                far_node_i, far_label = connected_target[target_label]
                current_string = self._far_string(node_data, far_node_i, far_label, "source")
                candidates = [
                    candidate
                    for candidate in candidates
                    if not (candidate[0] == far_node_i and candidate[1] == far_label)
                ] + [
                    (
                        far_node_i,
                        far_label,
                        "source",
                        self._far_string(node_data, far_node_i, far_label, "source"),
                    )
                ]

            choices = [_EMPTY] + [
                candidate[3] for candidate in candidates if candidate[3] != current_string
            ]
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
                    _own_kind="target",
                    _own_label=target_label,
                    _own_node_index=node_index,
                    _own_color=target_data["color"][i],
                    _own_x=target_data["x"][i],
                    _own_y=target_data["y"][i],
                    _candidates=candidates,
                ),
            )
            new_rows.append(row(label_input, value_select, spacing=4))

        self.state.connections_rows_column.children = new_rows

        # Keep connections_source in sync
        if self.state.connections_source is not None:
            my_port, connected_to, edge_index = [], [], []
            for i in range(len(edge_data.get("starting_node_index", []))):
                starting_node_position = edge_data["starting_node_index"][i]
                ending_node_position = edge_data["ending_node_index"][i]
                starting_port_label_value = edge_data["starting_port_label"][i]
                ending_port_label_value = edge_data["ending_port_label"][i]
                starting_port_kind_value = edge_data["starting_port_kind"][i]
                ending_port_kind_value = edge_data["ending_port_kind"][i]

                if starting_node_position == node_index:
                    far_name = (
                        node_data["name"][ending_node_position]
                        if ending_node_position < len(node_data.get("name", []))
                        else f"node_{ending_node_position}"
                    )
                    my_port.append(f"{starting_port_kind_value}:{starting_port_label_value}")
                    connected_to.append(
                        f"{far_name} ({ending_port_kind_value}:{ending_port_label_value})"
                    )
                    edge_index.append(i)
                elif ending_node_position == node_index:
                    far_name = (
                        node_data["name"][starting_node_position]
                        if starting_node_position < len(node_data.get("name", []))
                        else f"node_{starting_node_position}"
                    )
                    my_port.append(f"{ending_port_kind_value}:{ending_port_label_value}")
                    connected_to.append(
                        f"{far_name} ({starting_port_kind_value}:{starting_port_label_value})"
                    )
                    edge_index.append(i)

            self.state.connections_source.data = dict(
                my_port=my_port,
                connected_to=connected_to,
                edge_index=edge_index,
            )

    @staticmethod
    def _far_string(node_data: dict, far_node_index: int, far_label: str, far_kind: str) -> str:
        """
        Format a port reference as ``"<node_name> (<kind>:<label>)"``.

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

        Wraps :meth:`_on_port_select_change` via :func:`functools.partial`,
        pre-filling every port-specific argument so Bokeh receives a plain
        ``(attr, old, new)`` callable with no inner function required.

        * When ``_own_kind == "source"``:  own port = starting_port,
          far port = ending_port (target kind).
        * When ``_own_kind == "target"``:  own port = ending_port,
          far port = starting_port (source kind).

        The edge is always stored as source → target regardless of which
        direction was clicked, so ``starting_port`` is always source-kind
        and ``ending_port`` is always target-kind.

        :param _own_kind: ``"source"`` or ``"target"``.
        :param _own_label: Port label of the own port (e.g. ``"1"``).
        :param _own_node_index: Node index of the own port.
        :param _own_color: Energy-type colour of the own port.
        :param _own_x: Snapshot x of the own port at callback-build time.
        :param _own_y: Snapshot y of the own port at callback-build time.
        :param _candidates: Pre-built list of ``(node_index, label, kind,
            display_str)`` tuples for the far end.
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
        2. **Disconnect** (if *new_value* is empty): remove the permanent edge
           that involves the own port and refresh the canvas.
        3. **Connect** (otherwise): resolve the chosen far-end port, re-fetch
           live coordinates for both ends, build source→target port dicts, and
           register a new preview edge.

        :param attr: Bokeh change attribute name (unused, required by protocol).
        :param old: Previous widget value (unused).
        :param new_value: Newly selected display string, or ``_EMPTY``.
        :param _own_kind: ``"source"`` or ``"target"`` — which end is "ours".
        :param _own_label: Port label of the own port.
        :param _own_node_index: Node index of the own port.
        :param _own_color: Energy-type colour of the own port.
        :param _own_x: Snapshot x coordinate captured when the row was built.
        :param _own_y: Snapshot y coordinate captured when the row was built.
        :param _candidates: Candidate ``(node_index, label, kind, display_str)``
            tuples for the far end, built when the row was rendered.
        """
        _own_port_source = (
            self.state.source_port_source
            if _own_kind == "source"
            else self.state.target_port_source
        )
        _far_port_source = (
            self.state.target_port_source
            if _own_kind == "source"
            else self.state.source_port_source
        )

        # Drop any pending preview edge that has the own port at its own end
        # (starting end for source rows, ending end for target rows).
        if _own_kind == "source":
            self.state.pending_connections = [
                (sp, ep)
                for sp, ep in self.state.pending_connections
                if not (
                    sp["kind"] == "source"
                    and sp["node_index"] == _own_node_index
                    and sp["label"] == _own_label
                )
            ]
        else:
            self.state.pending_connections = [
                (sp, ep)
                for sp, ep in self.state.pending_connections
                if not (
                    ep["kind"] == "target"
                    and ep["node_index"] == _own_node_index
                    and ep["label"] == _own_label
                )
            ]

        self._clear_temp_edge_visuals()
        for sp, ep in list(self.state.pending_connections):
            self._add_edge_temp(sp, ep)

        if new_value == _EMPTY:
            # Disconnect: remove all permanent edges that touch the own port.
            if self.state.edge_source is not None:
                self._push_undo()
                edge_data = {key: list(value) for key, value in self.state.edge_source.data.items()}
                keep = [
                    i
                    for i in range(len(edge_data.get("starting_node_index", [])))
                    if not (
                        (
                            edge_data["starting_node_index"][i] == _own_node_index
                            and edge_data["starting_port_label"][i] == _own_label
                            and edge_data["starting_port_kind"][i] == _own_kind
                        )
                        or (
                            edge_data["ending_node_index"][i] == _own_node_index
                            and edge_data["ending_port_label"][i] == _own_label
                            and edge_data["ending_port_kind"][i] == _own_kind
                        )
                    )
                ]
                self.state.edge_source.data = {
                    k: [edge_data[k][j] for j in keep] for k in edge_data
                }
                self._rebuild_all_ports()
                self._refresh_connections_table(_own_node_index)
                self._mark_unsaved()
            return

        # Resolve the chosen far-end port from the candidates list.
        parsed = self._parse_choice(new_value, _candidates)
        if parsed is None:
            return
        far_node_i, far_label, _ = parsed

        # Look up the far port's live coordinates and colour.
        far_data = _far_port_source.data
        far_x, far_y, far_color = None, None, None
        for j in range(len(far_data.get("node_index", []))):
            if far_data["node_index"][j] == far_node_i and far_data["label"][j] == far_label:
                far_x = far_data["x"][j]
                far_y = far_data["y"][j]
                far_color = far_data["color"][j]
                break
        if far_x is None:
            return

        # Re-fetch own port's live coordinates (may have moved since the row was built).
        live_own_x, live_own_y = _own_x, _own_y
        own_data = _own_port_source.data
        for j in range(len(own_data.get("node_index", []))):
            if own_data["node_index"][j] == _own_node_index and own_data["label"][j] == _own_label:
                live_own_x = own_data["x"][j]
                live_own_y = own_data["y"][j]
                break

        # Build the edge always as source → target.
        if _own_kind == "source":
            starting_port = {
                "kind": "source",
                "node_index": _own_node_index,
                "label": _own_label,
                "x": live_own_x,
                "y": live_own_y,
                "color": _own_color,
            }
            ending_port = {
                "kind": "target",
                "node_index": far_node_i,
                "label": far_label,
                "x": far_x,
                "y": far_y,
                "color": far_color,
            }
        else:
            starting_port = {
                "kind": "source",
                "node_index": far_node_i,
                "label": far_label,
                "x": far_x,
                "y": far_y,
                "color": far_color,
            }
            ending_port = {
                "kind": "target",
                "node_index": _own_node_index,
                "label": _own_label,
                "x": live_own_x,
                "y": live_own_y,
                "color": _own_color,
            }

        self.state.pending_connections.append((starting_port, ending_port))
        self._add_edge_temp(starting_port, ending_port)

    # -----------------------------------------------------------------------
    # Delete selected connection
    # -----------------------------------------------------------------------

    def _delete_selected_connection(self):
        """
        Delete the edge(s) currently selected in the Connections DataTable.

        Reads the highlighted row indices from ``connections_source.selected``,
        resolves them to actual edge indices, removes those rows from
        ``edge_source``, and refreshes the Connections panel.
        """
        if self.state.connections_source is None or self.state.edge_source is None:
            return
        selected = list(self.state.connections_source.selected.indices)
        if not selected:
            return
        # Update undo stack
        self._push_undo()
        edge_index_list = list(self.state.connections_source.data.get("edge_index", []))
        to_delete = {edge_index_list[i] for i in selected if i < len(edge_index_list)}
        edge_data = {key: list(value) for key, value in self.state.edge_source.data.items()}
        keep = [i for i in range(len(edge_data.get("xs", []))) if i not in to_delete]
        self.state.edge_source.data = {k: [edge_data[k][j] for j in keep] for k in edge_data}
        self._mark_unsaved()
        _LOGGER.info("Deleted connection(s) at edge indices %s", sorted(to_delete))
        if self.state.selected_node_index is not None:
            self._refresh_connections_table(self.state.selected_node_index)

    # -----------------------------------------------------------------------
    # Type-select live refresh
    # -----------------------------------------------------------------------

    def _on_type_select_change(self, attr, old, new):
        """
        Bokeh ``on_change`` callback wired to ``state.type_select``.

        Refreshes the Position select's choices, rebuilds the Options table
        for the newly chosen component type (pre-populated with the selected
        node's previously saved option values, if any), and refreshes the
        Symmetry select so its candidate list matches the node's icon type.

        This only refreshes the config panel widgets in place; it does not
        write anything back to ``placed_nodes_source`` — that happens when
        the user clicks Apply (see :meth:`apply_node_configurations`).

        :param attr: Bokeh attribute name (unused, required by protocol).
        :param old: Previous ``type_select`` value (unused).
        :param new: Newly selected component type.
        """
        pos_choices = self.state.possible_position.get(new, [])
        self.state.position_select.options = pos_choices
        self.state.position_select.value = pos_choices[0] if pos_choices else _EMPTY

        current_selected_node_index_for_type = self.state.selected_node_index
        saved_options = {}
        if current_selected_node_index_for_type is not None:
            saved_options_json_list = list(self.state.placed_nodes_source.data.get("options", []))
            if current_selected_node_index_for_type < len(saved_options_json_list):
                try:
                    saved_options = (
                        json.loads(saved_options_json_list[current_selected_node_index_for_type])
                        or {}
                    )
                except (json.JSONDecodeError, TypeError):
                    pass
        self._refresh_options_table(new, saved_options)
        self.state.options_table.visible = bool(self.state.possible_options.get(new, {}))

        if current_selected_node_index_for_type is not None:
            icon_type_list = list(self.state.placed_nodes_source.data.get("icon_type", []))
            icon_type = (
                icon_type_list[current_selected_node_index_for_type]
                if current_selected_node_index_for_type < len(icon_type_list)
                else _EMPTY
            )
            self._refresh_symmetry_select(current_selected_node_index_for_type, icon_type)

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
        if current_selected_node_index is None:
            return

        # Push undo snapshot *before* any mutation so the user can revert the
        # entire Apply operation with a single Undo click.
        self._push_undo()

        # Get node properties from the configurator panel
        new_node_name_id = self.state.name_input.value
        new_node_type = self.state.type_select.value
        new_position = self.state.position_select.value

        option_names = list(self.state.options_source.data.get("options", []))
        option_values = list(self.state.options_source.data.get("value", []))
        option_values_dict = {
            option_name: _strings_to_option_values(option_value)
            for option_name, option_value in zip(option_names, option_values)
        }
        # convert the options dict to a JSON string for storage in the ColumnDataSource
        options_json_string = json.dumps(option_values_dict)

        placed_nodes_data = {
            key: list(values) for key, values in self.state.placed_nodes_source.data.items()
        }

        # Check first if the current_selected_node_index is within bounds for each property
        # before updating, to avoid IndexErrors.
        if current_selected_node_index < len(placed_nodes_data.get("name", [])):
            placed_nodes_data["name"][current_selected_node_index] = new_node_name_id
        if current_selected_node_index < len(placed_nodes_data.get("node_type", [])):
            placed_nodes_data["node_type"][current_selected_node_index] = new_node_type
            # Reset port counts to defaults when the type changes and spinners are hidden.
            if not self.state.source_count_spinner.visible:
                placed_nodes_data["n_sources"][current_selected_node_index] = int(
                    self.state.default_source_count.get(new_node_type, 0)
                )
                placed_nodes_data["n_targets"][current_selected_node_index] = int(
                    self.state.default_target_count.get(new_node_type, 0)
                )
        if current_selected_node_index < len(placed_nodes_data.get("position", [])):
            placed_nodes_data["position"][current_selected_node_index] = new_position
        # Place empty dict if there is no options entry for this node yet
        if "options" not in placed_nodes_data:
            placed_nodes_data["options"] = ["{}"] * len(placed_nodes_data.get("name", []))
        # Update options JSON string for this node
        if current_selected_node_index < len(placed_nodes_data["options"]):
            placed_nodes_data["options"][current_selected_node_index] = options_json_string

        # These two spinners shows only if the node type supports multiple ports
        if self.state.source_count_spinner.visible:
            new_source_port_count = int(self.state.source_count_spinner.value)
            if "n_sources" in placed_nodes_data and current_selected_node_index < len(
                placed_nodes_data["n_sources"]
            ):
                placed_nodes_data["n_sources"][current_selected_node_index] = new_source_port_count
        if self.state.target_count_spinner.visible:
            new_target_port_count = int(self.state.target_count_spinner.value)
            if "n_targets" in placed_nodes_data and current_selected_node_index < len(
                placed_nodes_data["n_targets"]
            ):
                placed_nodes_data["n_targets"][current_selected_node_index] = new_target_port_count

        # Keep the hover-tooltip source's name / node_type in sync with the edits above.
        if self.state.hover_source is not None:
            hover_data = {key: list(values) for key, values in self.state.hover_source.data.items()}
            if current_selected_node_index < len(hover_data.get("name", [])):
                hover_data["name"][current_selected_node_index] = new_node_name_id
            if current_selected_node_index < len(hover_data.get("node_type", [])):
                hover_data["node_type"][current_selected_node_index] = new_node_type
            self.state.hover_source.data = hover_data

        # ── Persist symmetry selection ────────────────────────────────

        all_node_names = placed_nodes_data.get("name", [])
        node_count = len(all_node_names)

        # Guard: current index must be valid before we touch names.
        if current_selected_node_index >= node_count:
            return

        current_node_name = all_node_names[current_selected_node_index]
        new_symmetry_name = self.state.symmetry_select.value

        # Ensure symmetry columns exist and are fully sized (handles newly added nodes).
        symmetry_names = list(placed_nodes_data.get("symmetry_name", []))
        symmetry_indices = list(placed_nodes_data.get("symmetry_node_index", []))
        if len(symmetry_names) < node_count:
            symmetry_names += [_EMPTY] * (node_count - len(symmetry_names))
        if len(symmetry_indices) < node_count:
            symmetry_indices += [_NO_SYMMETRY_PEER] * (node_count - len(symmetry_indices))

        # Resolve the newly chosen peer index.
        new_peer_index = _NO_SYMMETRY_PEER
        if new_symmetry_name and new_symmetry_name != _EMPTY:
            for node_position, node_name in enumerate(all_node_names):
                if node_name == new_symmetry_name:
                    new_peer_index = node_position
                    break

        # ── Clear stale references on both sides before writing ────────
        #
        # Two nodes may be displaced by this Apply:
        #   (a) current node's *old* peer  — it pointed back at current_node;
        #       now current_node is leaving, so clear it.
        #   (b) new peer's *old* partner   — new_peer previously pointed at
        #       some other node D; D still points at new_peer, so clear D too.
        #
        # We handle both by scanning the full table and wiping every node
        # whose symmetry_name matches current_node_name OR new_symmetry_name,
        # excluding the two nodes that will get fresh values written below.

        nodes_getting_fresh_write = {current_selected_node_index}
        if new_peer_index != _NO_SYMMETRY_PEER:
            nodes_getting_fresh_write.add(new_peer_index)

        for position in range(node_count):
            if position in nodes_getting_fresh_write:
                continue
            sym_name = symmetry_names[position]
            if sym_name == current_node_name or (
                new_symmetry_name and new_symmetry_name != _EMPTY and sym_name == new_symmetry_name
            ):
                symmetry_names[position] = _EMPTY
                symmetry_indices[position] = _NO_SYMMETRY_PEER

        # Write the current node's new symmetry entry.
        symmetry_names[current_selected_node_index] = new_symmetry_name
        symmetry_indices[current_selected_node_index] = new_peer_index

        # Write the new peer's back-reference (if a peer was chosen).
        if new_peer_index != _NO_SYMMETRY_PEER:
            symmetry_names[new_peer_index] = current_node_name
            symmetry_indices[new_peer_index] = current_selected_node_index

        placed_nodes_data["symmetry_name"] = symmetry_names
        placed_nodes_data["symmetry_node_index"] = symmetry_indices

        # Update the placed_nodes_source with the modified data
        self.state.placed_nodes_source.data = placed_nodes_data

        # ── Commit pending connections ─────────────────────────────────
        for starting_port, ending_port in list(self.state.pending_connections):
            self._add_edge(starting_port, ending_port)
        self._clear_temp_edges()
        self._rebuild_all_ports()
        self._refresh_connections_table(current_selected_node_index)

        self._mark_unsaved()


def _option_values_to_strings(value) -> str:
    """Convert an option value (bool, int, float, or str) to its display string."""
    if value is True:
        return "True"
    if value is False:
        return "False"
    return str(value)


def _strings_to_option_values(string: str):
    """Parse a display string back to a typed Python value (bool, int, float, or str)."""
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
