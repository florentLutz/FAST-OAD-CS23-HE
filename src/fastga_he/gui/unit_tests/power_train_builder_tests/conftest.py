# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Shared fixtures for the powertrain builder tests.

The :class:`CanvasDriver` fixture drives a real :class:`BuilderHandler` through
the same entry points the GUI uses (toolbox selection, canvas taps, config
panel widgets and the Apply button), so the tests exercise the actual
callback paths rather than internal helpers.
"""

from types import SimpleNamespace

import bokeh.models as bkmodel
import bokeh.plotting as bkplot
import pytest

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG

from fastga_he.gui.power_train_builder.power_train_builder_handler import BuilderHandler
from fastga_he.gui.power_train_builder.power_train_builder_toolbox import (
    ComponentToolboxConfigurationTableBuilder,
)


class CanvasDriver:
    """
    Thin test driver around a :class:`BuilderHandler`.

    Every action goes through a public GUI entry point (tap, toolbox click,
    widget value change, Apply), so assertions reflect what a user would get.
    """

    def __init__(self, handler: BuilderHandler):
        self.handler = handler
        self.state = handler.state

    # -- Canvas actions -------------------------------------------------------

    def tap(self, x: float, y: float):
        """Tap the canvas at ``(x, y)``."""
        self.handler.on_canvas_tap(SimpleNamespace(x=x, y=y))

    def place(self, icon_key: str, x: float, y: float) -> int:
        """
        Place a component with the toolbox and return its node index.

        :param icon_key: Key of the icon in ``ICONS_CONFIG``.
        :param x: Canvas x coordinate of the new component.
        :param y: Canvas y coordinate of the new component.
        :return: Zero-based node index of the placed component.
        """
        component_index = list(ICONS_CONFIG).index(icon_key)
        self.handler.on_toolbox_select(component_index)
        self.tap(x, y)
        # Deselect the toolbox button so later taps select instead of place
        self.handler.on_toolbox_select(component_index)
        return len(self.state.placed_nodes_source.data["name"]) - 1

    def port(self, kind: str, node_index: int, label: str = "1") -> tuple[float, float]:
        """
        Canvas coordinates of a port ball.

        :param kind: ``"source"`` or ``"target"``.
        :param node_index: Node index that owns the port.
        :param label: Port label (``"1"``, ``"2"``, ...).
        :return: Tuple ``(x, y)`` of the port ball.
        """
        port_source = (
            self.state.source_port_source if kind == "source" else self.state.target_port_source
        )
        data = port_source.data
        for port_position, (owner_index, port_label) in enumerate(
            zip(data["node_index"], data["label"], strict=True)
        ):
            if owner_index == node_index and str(port_label) == str(label):
                return data["x"][port_position], data["y"][port_position]
        raise AssertionError(f"No {kind} port {label} on node {node_index}")

    def connect(self, first_port: tuple, second_port: tuple):
        """
        Connect two ports with two canvas taps.

        :param first_port: ``(kind, node_index, label)`` of the first port.
        :param second_port: ``(kind, node_index, label)`` of the second port.
        """
        self.tap(*self.port(*first_port))
        self.tap(*self.port(*second_port))

    def select(self, node_index: int):
        """Select a node by tapping its centre."""
        nodes_data = self.state.placed_nodes_source.data
        if self.state.selected_node_index == node_index:
            return
        self.tap(nodes_data["x"][node_index], nodes_data["y"][node_index])
        assert self.state.selected_node_index == node_index

    # -- Read-only views --------------------------------------------------------

    def edges(self) -> list[tuple]:
        """
        Edges as sorted ``((node, kind, label), (node, kind, label))`` pairs.

        An edge is stored with whichever port was clicked first as its start,
        so both ends are sorted to compare connections regardless of direction.
        """
        edge_data = self.state.edge_source.data
        return sorted(
            tuple(
                sorted(
                    [
                        (
                            edge_data["starting_node_index"][edge_position],
                            edge_data["starting_port_kind"][edge_position],
                            str(edge_data["starting_port_label"][edge_position]),
                        ),
                        (
                            edge_data["ending_node_index"][edge_position],
                            edge_data["ending_port_kind"][edge_position],
                            str(edge_data["ending_port_label"][edge_position]),
                        ),
                    ]
                )
            )
            for edge_position in range(len(edge_data["starting_node_index"]))
        )

    def connected_flags(self, kind: str, node_index: int) -> dict:
        """``{port_label: is_connected}`` for one side of a node."""
        port_source = (
            self.state.source_port_source if kind == "source" else self.state.target_port_source
        )
        data = port_source.data
        return {
            data["label"][port_position]: data["connected"][port_position] == "True"
            for port_position in range(len(data["label"]))
            if data["node_index"][port_position] == node_index
        }

    def connection_rows(self) -> list[tuple[str, str]]:
        """Connections panel rows as ``(label_text, current_dropdown_value)`` tuples."""
        return [
            (row_widget.children[0].value, row_widget.children[1].value)
            for row_widget in self.state.connections_rows_column.children
        ]

    def node_column(self, column_name: str) -> list:
        """One column of ``placed_nodes_source``."""
        return list(self.state.placed_nodes_source.data[column_name])


def default_component_type(state, icon_key: str) -> str:
    """Component type a newly placed *icon_key* component gets (the first choice)."""
    return state.icon_to_component_types.get(icon_key, [icon_key])[0]


def icon_with_options(state) -> str:
    """
    An icon whose default type has an option with at least two values.

    Skips the calling test when the component registry has none.
    """
    for icon_key in ICONS_CONFIG:
        option_definitions = state.possible_options.get(default_component_type(state, icon_key))
        if any(len(option_values) >= 2 for option_values in (option_definitions or {}).values()):
            return icon_key
    pytest.skip("No component type with a multi-valued option in this registry")


def icon_with_several_types(state) -> str:
    """An icon offering at least two component types; skips the test if there is none."""
    for icon_key, component_types in state.icon_to_component_types.items():
        if icon_key in ICONS_CONFIG and len(component_types) >= 2:
            return icon_key
    pytest.skip("No icon with several component types in this registry")


@pytest.fixture
def builder_state():
    """
    A fresh :class:`BuilderState` built as the launcher builds it.

    The launcher also creates the Undo / Redo buttons before the handler; the
    canvas renderers it adds are optional and left as ``None`` here.
    """
    state = ComponentToolboxConfigurationTableBuilder.build()[-1]
    state.undo_button = bkmodel.Button(label="Undo")
    state.redo_button = bkmodel.Button(label="Redo")
    return state


@pytest.fixture
def canvas(builder_state) -> CanvasDriver:
    """A :class:`CanvasDriver` around a fresh handler on a fresh state."""
    return CanvasDriver(BuilderHandler(builder_state, bkplot.figure()))
