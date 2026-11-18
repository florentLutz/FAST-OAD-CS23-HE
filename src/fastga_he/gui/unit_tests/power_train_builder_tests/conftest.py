# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Shared fixtures for the powertrain builder tests.

The :class:`CanvasDriver` fixture drives a real :class:`BuilderHandler` the way a
user does: clicking the toolbox and the action buttons, tapping the canvas, and
filling the config panel. Each click is sent as a ``ButtonClick`` event, so the
tests also check that every button is wired to the right callback.
"""

import shutil
from types import SimpleNamespace

import bokeh.models as bkmodel
import bokeh.plotting as bkplot
import pytest
from bokeh.events import ButtonClick

from fastga_he.gui.power_train_builder.power_train_builder_handler import BuilderHandler
from fastga_he.gui.power_train_builder.power_train_builder_state import BUILDER_ICONS_CONFIG
from fastga_he.gui.power_train_builder.power_train_builder_toolbox_config_table import (
    ComponentToolboxConfigurationTableBuilder,
)


class CanvasDriver:
    """
    Test driver around a :class:`BuilderHandler`: what a user can do, and what they see.

    Actions, grouped as on screen:

    * **Toolbox**: :meth:`click_toolbox`, :meth:`set_icon_size`.
    * **Canvas**: :meth:`tap`, :meth:`tap_node`, :meth:`tap_port`, and the
      shortcuts :meth:`place`, :meth:`connect` and :meth:`select`.
    * **Buttons**: :meth:`click` for any button, and :meth:`apply`,
      :meth:`undo`, :meth:`redo`, :meth:`toggle_delete_mode`, :meth:`new_design`.
    * **Config panel**: :meth:`connection_dropdown`, :meth:`add_unknown_option`.
      Other fields are set directly on their widget, e.g.
      ``canvas.state.name_input.value = "left_motor"``.
    * **Files**: :meth:`save` and :meth:`load`. They call the handler directly,
      as the real buttons go through a native file dialog.

    Read-only views: :meth:`node_column`, :meth:`edges`, :meth:`port`,
    :meth:`connected_flags`, :meth:`connection_rows`.

    ``handler`` and ``state`` stay available for the few tests that check an
    internal guard on purpose.
    """

    def __init__(self, handler: BuilderHandler):
        self.handler = handler
        self.state = handler.state

    # -- Toolbox ----------------------------------------------------------------

    def click_toolbox(self, icon_key: str):
        """
        Click the toolbox button of *icon_key*: arms it, or disarms it if it is armed.

        :param icon_key: Key of the icon in ``BUILDER_ICONS_CONFIG`` (``"unknown"``
            for the unknown component).
        """
        self.click(self.state.buttons[list(BUILDER_ICONS_CONFIG).index(icon_key)])

    def set_icon_size(self, icon_size: int):
        """Move the icon-size slider of the toolbox to *icon_size*."""
        self.state.node_size_slider.value = icon_size

    # -- Canvas -----------------------------------------------------------------

    def tap(self, x: float, y: float):
        """Tap the canvas at ``(x, y)``."""
        self.handler.on_canvas_tap(SimpleNamespace(x=x, y=y))

    def tap_node(self, node_index: int):
        """Tap the centre of node *node_index*."""
        nodes_data = self.state.placed_nodes_source.data
        self.tap(nodes_data["x"][node_index], nodes_data["y"][node_index])

    def tap_port(self, kind: str, node_index: int, label: str = "1"):
        """Tap a port ball (see :meth:`port`)."""
        self.tap(*self.port(kind, node_index, label))

    def place(self, icon_key: str, x: float, y: float) -> int:
        """
        Place a component: arm it in the toolbox, tap the canvas, disarm it.

        :param icon_key: Key of the icon in ``BUILDER_ICONS_CONFIG``.
        :param x: Canvas x coordinate of the new component.
        :param y: Canvas y coordinate of the new component.
        :return: Zero-based node index of the placed component.
        """
        self.click_toolbox(icon_key)
        self.tap(x, y)
        # Disarm, so that later taps select instead of placing
        self.click_toolbox(icon_key)
        return len(self.state.placed_nodes_source.data["name"]) - 1

    def connect(self, first_port: tuple, second_port: tuple):
        """
        Connect two ports by tapping one, then the other.

        :param first_port: ``(kind, node_index, label)`` of the port tapped first.
        :param second_port: ``(kind, node_index, label)`` of the port tapped second.
        """
        self.tap_port(*first_port)
        self.tap_port(*second_port)

    def select(self, node_index: int):
        """Select node *node_index* by tapping it, unless it is already selected."""
        if self.state.selected_node_index == node_index:
            return
        self.tap_node(node_index)
        assert self.state.selected_node_index == node_index

    # -- Buttons ----------------------------------------------------------------

    def click(self, button: bkmodel.Button):
        """Click *button*, through the callbacks wired to it, as the browser does."""
        button._trigger_event(ButtonClick(button))

    def apply(self):
        """Click Apply in the config panel."""
        self.click(self.state.apply_button)

    def undo(self):
        """Click Undo."""
        self.click(self.state.undo_button)

    def redo(self):
        """Click Redo."""
        self.click(self.state.redo_button)

    def toggle_delete_mode(self):
        """Click Delete: enter delete mode, or leave it."""
        self.click(self.state.delete_button)

    def new_design(self):
        """Click New Design."""
        self.click(self.state.new_design_button)

    # -- Config panel -------------------------------------------------------------

    def connection_dropdown(self, port_text: str) -> bkmodel.Select:
        """
        Connections panel dropdown of a port of the selected node.

        :param port_text: Label of the port's row, e.g. ``"source:1"``.
        :return: The dropdown; set its ``value`` to pick a connection.
        """
        for row_widget in self.state.connections_rows_column.children:
            if row_widget.children[0].value == port_text:
                return row_widget.children[1]
        raise AssertionError(f"No Connections row {port_text}")

    def add_unknown_option(self, option_name: str, option_value: str):
        """
        Click "Add option" for the selected unknown component, then fill the new row.

        :param option_name: Text typed in the name box.
        :param option_value: Text typed in the value box.
        """
        self.click(self.state.add_option_button)
        name_input, value_input = self.handler._unknown_option_rows[-1]
        name_input.value = option_name
        value_input.value = option_value

    # -- Files ------------------------------------------------------------------

    def save(self, yaml_path, final: bool = False) -> bool:
        """
        Save the design to *yaml_path*, skipping the "final version?" overlay and the dialog.

        :param yaml_path: Path of the YAML file to write.
        :param final: ``True`` to save a final version (no ``canvas_layout`` section).
        :return: ``True`` if the file was written.
        """
        return self.handler._save_canvas_state(str(yaml_path), include_canvas_layout=not final)

    def load(self, yaml_path):
        """Load the design in *yaml_path*, skipping the file dialog."""
        self.handler._load_canvas_state(str(yaml_path))

    # -- Read-only views ----------------------------------------------------------

    def node_column(self, column_name: str) -> list:
        """One column of ``placed_nodes_source``, e.g. ``"name"``."""
        return list(self.state.placed_nodes_source.data[column_name])

    def port(self, kind: str, node_index: int, label: str = "1") -> tuple[float, float]:
        """
        Canvas coordinates of a port ball.

        :param kind: ``"source"`` or ``"target"``.
        :param node_index: Node index that owns the port.
        :param label: Port label (``"1"``, ``"2"``, ...).
        :return: Tuple ``(x, y)`` of the port ball.
        """
        data = self._port_data(kind)
        for port_position in range(len(data["label"])):
            if data["node_index"][port_position] == node_index and str(
                data["label"][port_position]
            ) == str(label):
                return data["x"][port_position], data["y"][port_position]
        raise AssertionError(f"No {kind} port {label} on node {node_index}")

    def edges(self) -> list[tuple]:
        """
        Every edge as a ``((node, kind, label), (node, kind, label))`` pair of port ends.

        The two ends of each edge, and the edges themselves, are sorted, so
        tests can compare connections without caring which port was clicked
        first or which edge was created first.

        :return: Sorted list of edges, each a sorted pair of port ends.
        """
        edge_data = self.state.edge_source.data
        edges = []
        for edge_position in range(len(edge_data["starting_node_index"])):
            starting_end = (
                edge_data["starting_node_index"][edge_position],
                edge_data["starting_port_kind"][edge_position],
                str(edge_data["starting_port_label"][edge_position]),
            )
            ending_end = (
                edge_data["ending_node_index"][edge_position],
                edge_data["ending_port_kind"][edge_position],
                str(edge_data["ending_port_label"][edge_position]),
            )
            # Same order of the two ends, whichever port was clicked first
            first_end, second_end = sorted([starting_end, ending_end])
            edges.append((first_end, second_end))

        # Same order of the edges, whichever was created first
        return sorted(edges)

    def connected_flags(self, kind: str, node_index: int) -> dict:
        """``{port_label: is_connected}`` for one side of a node."""
        data = self._port_data(kind)
        flags = {}
        for port_position in range(len(data["label"])):
            if data["node_index"][port_position] == node_index:
                flags[data["label"][port_position]] = data["connected"][port_position] == "True"
        return flags

    def connection_rows(self) -> list[tuple[str, str]]:
        """Connections panel rows as ``(port label, current dropdown value)`` tuples."""
        rows = []
        for row_widget in self.state.connections_rows_column.children:
            port_label_input, dropdown = row_widget.children[0], row_widget.children[1]
            rows.append((port_label_input.value, dropdown.value))
        return rows

    def _port_data(self, kind: str) -> dict:
        """Data of the source or target port balls."""
        if kind == "source":
            return self.state.source_port_source.data
        return self.state.target_port_source.data


def default_component_type(state, icon_key: str) -> str:
    """Component type a newly placed *icon_key* component gets (the first choice)."""
    return state.icon_to_component_types.get(icon_key, [icon_key])[0]


def find_icon_with_options(state) -> str | None:
    """
    An icon whose default type has an option with at least two values.

    :return: The icon key, or ``None`` when the component registry has none.
    """
    for icon_key in BUILDER_ICONS_CONFIG:
        option_definitions = state.possible_options.get(default_component_type(state, icon_key))
        if any(len(option_values) >= 2 for option_values in (option_definitions or {}).values()):
            return icon_key
    return None


def icon_with_options(state) -> str:
    """
    Same as :func:`find_icon_with_options`, but skips the calling test when there is none.
    """
    icon_key = find_icon_with_options(state)
    if icon_key is None:
        pytest.skip("No component type with a multi-valued option in this registry")
    return icon_key


def icon_with_several_types(state) -> str:
    """An icon offering at least two component types; skips the test if there is none."""
    for icon_key, component_types in state.icon_to_component_types.items():
        if icon_key in BUILDER_ICONS_CONFIG and len(component_types) >= 2:
            return icon_key
    pytest.skip("No icon with several component types in this registry")


def make_builder_state():
    """
    A fresh :class:`BuilderState` built as the launcher builds it.

    The launcher also creates the Undo / Redo buttons before the handler; the
    canvas renderers it adds are optional and left as ``None`` here.
    """
    state = ComponentToolboxConfigurationTableBuilder.build()[-1]
    state.undo_button = bkmodel.Button(label="Undo")
    state.redo_button = bkmodel.Button(label="Redo")
    return state


def make_canvas(state=None) -> CanvasDriver:
    """
    A :class:`CanvasDriver` around a fresh handler.

    The ``canvas`` fixture gives one per test; call this directly when a test
    needs several fresh canvases.

    :param state: State to drive; a fresh one if ``None``.
    """
    return CanvasDriver(BuilderHandler(state or make_builder_state(), bkplot.figure()))


@pytest.fixture
def tmp_path(tmp_path):
    """
    pytest's own ``tmp_path``, deleted with everything in it as soon as the test ends.

    Overrides the built-in fixture of the same name for every test in this
    folder, so the YAML files (and the other files) a test writes do not stay
    in the system temp folder.
    """
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def builder_state():
    """A fresh :class:`BuilderState` (see :func:`make_builder_state`)."""
    return make_builder_state()


@pytest.fixture
def canvas(builder_state) -> CanvasDriver:
    """A :class:`CanvasDriver` around a fresh handler on the ``builder_state`` fixture."""
    return make_canvas(builder_state)
