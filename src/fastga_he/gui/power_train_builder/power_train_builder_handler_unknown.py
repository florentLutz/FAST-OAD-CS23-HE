# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Config-panel behaviour of an unknown component.

An unknown component is one the builder has no definition for: it is placed
from the toolbox (the "unknown" icon) or loaded from a file whose component id
is not registered. Its canvas behaviour is the same as any other component;
only its config panel differs, because there is no list of possible values to
choose from:

* the component type, position and symmetry peer are free text;
* every option is a pair of free-text boxes (name, value), and the
  "Add option" button adds an empty pair;
* both port counts are editable (0 allowed); a warning is shown while both
  are 0, since such a component cannot be connected to anything;
* two selects set the energy type of its source and target ports. Unlike
  every other panel field, a change there applies at once, without Apply.
  Ports of undefined type cannot be connected until a type is chosen.

Everything else is committed by Apply, like for any component.

Mixin class :class:`UnknownComponentMixin` is not meant to be instantiated
directly; it is composed into :class:`BuilderHandler` via multiple inheritance.
"""

import ast
import json
import logging

import bokeh.models as bkmodel
from bokeh.layouts import row

from .power_train_builder_handler_config_panel import _option_values_to_strings
from .power_train_builder_state import (
    UNDEFINED_PORT_TYPE,
    UNKNOWN_COMPONENT_TYPE,
    _EMPTY,
    copy_column_data,
    edge_end_ports,
    has_no_port,
    is_unknown_node,
    select_edges,
    BuilderState,
)

_LOGGER = logging.getLogger(__name__)

_OPTION_TEXT_STYLE = {"color": "white", "font-size": "14px"}


class UnknownComponentMixin:
    """
    Fills, reads and reacts to the config-panel widgets of an unknown component.

    Depends on ``self.state`` (a :class:`BuilderState` instance) being set by
    the concrete class, and on the config-panel, port and history methods of
    the other mixins.
    """

    state: "BuilderState"

    # (name TextInput, value TextInput) of every option row of the unknown
    # component being edited, in panel order
    _unknown_option_rows: tuple = ()

    # -----------------------------------------------------------------------
    # Showing and filling the panel
    # -----------------------------------------------------------------------

    def _show_unknown_widgets(self, is_unknown: bool):
        """
        Show the free-text widgets of an unknown component, or the selects of a known one.

        :param is_unknown: ``True`` when the selected node is an unknown component.
        """
        state = self.state
        # Shown again by _refresh_unknown_port_warning once the panel is filled
        state.unknown_no_port_warning_div.visible = False
        for known_widget, unknown_widget in (
            (state.type_select, state.unknown_type_input),
            (state.position_select, state.unknown_position_input),
            (state.symmetry_select, state.unknown_symmetry_input),
        ):
            known_widget.visible = not is_unknown
            unknown_widget.visible = is_unknown
        for unknown_only_widget in (
            state.add_option_button,
            state.source_port_type_select,
            state.target_port_type_select,
        ):
            unknown_only_widget.visible = is_unknown

    def _populate_unknown_fields(self, node_index: int):
        """
        Fill the unknown-component widgets with the stored values of node *node_index*.

        :param node_index: Zero-based index of an unknown component.
        """
        nodes_data = self.state.placed_nodes_source.data
        self.state.unknown_type_input.value = nodes_data["node_type"][node_index]
        self.state.unknown_position_input.value = nodes_data["position"][node_index]
        self.state.unknown_symmetry_input.value = nodes_data["symmetry_name"][node_index]

        self._refresh_unknown_options_table(_saved_options(nodes_data, node_index))
        self.state.options_table.visible = True

        # Both sides are editable, down to no port at all (e.g. a tank or a propeller)
        for spinner, column in (
            (self.state.source_count_spinner, "n_sources"),
            (self.state.target_count_spinner, "n_targets"),
        ):
            spinner.low = 0
            spinner.visible = True
            spinner.value = int(nodes_data[column][node_index])
        self.state.port_count_section.visible = True

        # Filling the selects is not a user change: it must not apply anything
        self._is_populating_node_table = True
        try:
            for port_type_select, column in (
                (self.state.source_port_type_select, "source_port_type"),
                (self.state.target_port_type_select, "target_port_type"),
            ):
                port_type_select.value = nodes_data[column][node_index] or UNDEFINED_PORT_TYPE
        finally:
            self._is_populating_node_table = False

        # Both spinners hold the stored counts now
        self._refresh_unknown_port_warning()

    def _refresh_unknown_port_warning(self):
        """
        Show the "no port" warning while the selected unknown component has
        neither a source nor a target port, and hide it otherwise.

        The spinner values are read, not the stored counts, so the warning
        follows the spinners live, before Apply.
        """
        node_index = self.state.selected_node_index
        nodes_data = self.state.placed_nodes_source.data
        self.state.unknown_no_port_warning_div.visible = (
            node_index is not None
            and node_index < len(nodes_data.get("name", []))
            and is_unknown_node(nodes_data, node_index)
            and has_no_port(
                self.state.source_count_spinner.value, self.state.target_count_spinner.value
            )
        )

    def _warn_if_unknown_without_port(self, nodes_data: dict, node_index: int):
        """
        Log a warning if node *node_index* is an unknown component without any port.

        :param nodes_data: ``placed_nodes_source.data`` (or a copy of it).
        :param node_index: Zero-based index of the node.
        """
        if is_unknown_node(nodes_data, node_index) and has_no_port(
            nodes_data["n_sources"][node_index], nodes_data["n_targets"][node_index]
        ):
            _LOGGER.warning(
                "Unknown component %s has neither a source nor a target port; "
                "it cannot be connected to anything.",
                nodes_data["name"][node_index],
            )

    def _clear_unknown_fields(self):
        """Empty and hide the unknown-component widgets."""
        for text_input in (
            self.state.unknown_type_input,
            self.state.unknown_position_input,
            self.state.unknown_symmetry_input,
        ):
            text_input.value = _EMPTY
        self._unknown_option_rows = ()
        # Also hides the "no port" warning
        self._show_unknown_widgets(False)

    # -----------------------------------------------------------------------
    # Options
    # -----------------------------------------------------------------------

    def _refresh_unknown_options_table(self, options: dict):
        """
        Rebuild the Options rows as editable (name, value) text pairs.

        :param options: Stored options of the unknown component.
        """
        self._unknown_option_rows = ()
        self.state.options_rows_column.children = []
        for option_name, option_value in options.items():
            self._add_unknown_option_row(option_name, _option_values_to_strings(option_value))

    def _add_unknown_option_row(self, option_name: str = _EMPTY, option_value: str = _EMPTY):
        """
        Append one (name, value) option row to the Options table.

        :param option_name: Initial option name.
        :param option_value: Initial option value, as displayed.
        """
        name_input = bkmodel.TextInput(
            value=option_name, placeholder="option name", width=180, styles=_OPTION_TEXT_STYLE
        )
        value_input = bkmodel.TextInput(
            value=option_value, placeholder="value", width=180, styles=_OPTION_TEXT_STYLE
        )
        self._unknown_option_rows = (*self._unknown_option_rows, (name_input, value_input))
        # A new list, so Bokeh sends the change to the browser
        self.state.options_rows_column.children = [
            *self.state.options_rows_column.children,
            row(name_input, value_input, spacing=4),
        ]

    def _on_add_option(self):
        """
        ``on_click`` callback of the "Add option" button: add an empty option row.

        The option is written to the component on Apply; a row whose name is
        left empty is ignored, which is also how an option is removed.
        """
        node_index = self.state.selected_node_index
        nodes_data = self.state.placed_nodes_source.data
        if node_index is None or not is_unknown_node(nodes_data, node_index):
            return
        self._add_unknown_option_row()

    # -----------------------------------------------------------------------
    # Apply
    # -----------------------------------------------------------------------

    def _unknown_panel_fields(self) -> dict:
        """
        Type, position and options typed in the panel, as Apply writes them.

        An empty type falls back to ``"unknown"``. Option values are read as
        Python literals where possible (``True``, ``3``, ``2.5``, ``[1, 2]``),
        and kept as text otherwise.

        :return: ``{"node_type": str, "position": str, "options": dict}``.
        """
        return {
            "node_type": self.state.unknown_type_input.value.strip() or UNKNOWN_COMPONENT_TYPE,
            "position": self.state.unknown_position_input.value.strip(),
            "options": {
                name_input.value.strip(): _text_to_option_value(value_input.value)
                for name_input, value_input in self._unknown_option_rows
                if name_input.value.strip()
            },
        }

    # -----------------------------------------------------------------------
    # Port types (applied at once)
    # -----------------------------------------------------------------------

    def _on_port_type_change(self, attr, old, new, *, side: str):
        """
        ``on_change`` callback of the source / target port-type selects.

        Applies at once, as one undoable step: stores the new energy type of
        that side of the selected unknown component, removes every connection
        of that side (they carried the old type, so none fits the new one), and
        redraws its ports in the new colour.

        :param attr: Bokeh attribute name (unused, required by protocol).
        :param old: Previous value (unused).
        :param new: New port type, a key of ``PORT_TYPE_COLORS``.
        :param side: ``"source"`` or ``"target"``, bound when wiring the select.
        """
        node_index = self.state.selected_node_index
        if node_index is None or self._is_populating_node_table:
            return
        nodes_data = self.state.placed_nodes_source.data
        column = f"{side}_port_type"
        if not is_unknown_node(nodes_data, node_index) or nodes_data[column][node_index] == new:
            return

        self._push_undo()
        new_nodes_data = copy_column_data(nodes_data)
        new_nodes_data[column][node_index] = new
        self.state.placed_nodes_source.data = new_nodes_data
        self._drop_side_connections(node_index, side)
        self._rebuild_all_ports()
        self._mark_unsaved()
        _LOGGER.info("%s ports of node %d set to %s.", side.capitalize(), node_index, new)

    def _drop_side_connections(self, node_index: int, side: str):
        """
        Remove the connections and panel picks of one port side whose type changed.

        Each of them joined two ports of the old type, so none fits the new one.

        :param node_index: Zero-based index of the unknown component.
        :param side: ``"source"`` or ``"target"``, the side whose type changed.
        """

        def fits(end_ports) -> bool:
            return all(port[:2] != (side, node_index) for port in end_ports)

        edge_data = copy_column_data(self.state.edge_source.data)
        self.state.edge_source.data = select_edges(
            edge_data,
            [
                edge_position
                for edge_position, end_ports in enumerate(edge_end_ports(edge_data))
                if fits(end_ports)
            ],
        )
        self.state.panel_pending_connections = [
            (starting_port, ending_port)
            for starting_port, ending_port in self.state.panel_pending_connections
            if fits(tuple(_port_key(port) for port in (starting_port, ending_port)))
        ]


def _saved_options(nodes_data: dict, node_index: int) -> dict:
    """
    Stored options of node *node_index*, decoded from their JSON string.

    :param nodes_data: ``placed_nodes_source.data`` (or a copy of it).
    :param node_index: Zero-based index of the node.
    :return: ``{option_name: value}``; empty if the JSON cannot be read.
    """
    try:
        options = json.loads(nodes_data["options"][node_index] or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return options if isinstance(options, dict) else {}


def _text_to_option_value(text: str):
    """
    Option value typed in a text box, as a Python value.

    :param text: Text of the value box.
    :return: The Python literal it spells (bool, number, list, dict, quoted
        string, ...), or the stripped text itself if it is not one.
    """
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return text


def _port_key(port: dict) -> tuple:
    """``(kind, node_index, label)`` key of a port dict."""
    return port["kind"], port["node_index"], port["label"]
