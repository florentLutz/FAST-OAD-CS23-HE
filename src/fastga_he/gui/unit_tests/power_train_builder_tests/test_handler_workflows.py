# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Workflow tests for :class:`BuilderHandler`, driven through the GUI entry points.

Each test performs what a user would do (place, tap ports, pick values in the
config panel, Apply, Undo...) and checks the resulting canvas data.
"""

import json

import pytest

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG

from fastga_he.gui.power_train_builder.power_train_builder_handler_config_panel import (
    PRIMARY_SOURCE_PORT_TYPE_KEYWORDS,
)
from .conftest import default_component_type, icon_with_options, icon_with_several_types


@pytest.fixture
def chain(canvas):
    """
    Battery -> cable -> motor -> propeller, fully connected, shared between the tests below.
    """
    nodes = {
        "battery": canvas.place("battery", 100, 100),
        "cable": canvas.place("cable", 100, 300),
        "motor": canvas.place("e_motor", 100, 500),
        "propeller": canvas.place("propeller", 100, 700),
    }
    canvas.connect(("target", nodes["battery"], "1"), ("source", nodes["cable"], "1"))
    canvas.connect(("target", nodes["cable"], "1"), ("source", nodes["motor"], "1"))
    canvas.connect(("target", nodes["motor"], "1"), ("source", nodes["propeller"], "1"))
    return nodes


def _edge(first_end: tuple, second_end: tuple) -> tuple:
    return tuple(sorted([first_end, second_end]))


# ---------------------------------------------------------------------------
# Placing and connecting on the canvas
# ---------------------------------------------------------------------------


def test_placed_nodes_get_type_defaults_and_unique_names(canvas):
    # Add two motor node
    canvas.place("e_motor", 100, 100)
    canvas.place("e_motor", 300, 100)

    # Get the expected type, positions, and options for the default component type of "e_motor"
    expected_type = default_component_type(canvas.state, "e_motor")
    expected_positions = canvas.state.possible_position.get(expected_type) or [""]
    expected_options = {
        option_name: option_values[0]
        for option_name, option_values in canvas.state.possible_options.get(
            expected_type, {}
        ).items()
        if option_values
    }

    assert canvas.node_column("name") == ["e_motor_1", "e_motor_2"]
    assert canvas.node_column("node_type") == [expected_type, expected_type]
    assert canvas.node_column("position") == [expected_positions[0]] * 2
    assert json.loads(canvas.node_column("options")[0]) == expected_options
    assert canvas.state.save_button.button_type == "warning"


def test_connecting_two_compatible_ports_draws_one_edge(chain, canvas):
    assert canvas.edges() == sorted(
        [
            _edge((chain["battery"], "target", "1"), (chain["cable"], "source", "1")),
            _edge((chain["cable"], "target", "1"), (chain["motor"], "source", "1")),
            _edge((chain["motor"], "target", "1"), (chain["propeller"], "source", "1")),
        ]
    )
    assert canvas.connected_flags("target", chain["battery"]) == {"1": True}


def test_incompatible_or_reused_ports_are_rejected(chain, canvas):
    extra_cable = canvas.place("cable", 400, 300)
    num_edges_before = len(canvas.edges())

    # Battery port 1 is already used
    canvas.connect(("target", chain["battery"], "1"), ("source", extra_cable, "1"))
    # Electric source on the cable vs mechanical target on a free motor: colours differ
    free_motor = canvas.place("e_motor", 400, 500)
    canvas.connect(("source", extra_cable, "1"), ("target", free_motor, "1"))
    # Same kind on both ends
    canvas.connect(("source", extra_cable, "1"), ("source", chain["motor"], "1"))

    assert len(canvas.edges()) == num_edges_before


def test_tapping_the_same_port_twice_cancels_the_pending_connection(canvas):
    battery = canvas.place("battery", 100, 100)

    canvas.tap(*canvas.port("target", battery, "1"))
    canvas.tap(*canvas.port("target", battery, "1"))

    assert canvas.state.awaiting_second_port_click is None
    assert canvas.edges() == []


# ---------------------------------------------------------------------------
# Selecting a node and the Connections panel
# ---------------------------------------------------------------------------


def test_selecting_a_node_lists_its_ports_and_connections(chain, canvas):
    canvas.select(chain["motor"])

    assert canvas.connection_rows() == [
        ("source:1", "cable_1 (target:1)"),
        ("target:1", "propeller_1 (source:1)"),
    ]
    assert canvas.state.config_panel_column.visible


def test_tapping_the_selected_node_again_deselects_it(chain, canvas):
    canvas.select(chain["motor"])
    nodes_data = canvas.state.placed_nodes_source.data

    canvas.tap(nodes_data["x"][chain["motor"]], nodes_data["y"][chain["motor"]])

    assert canvas.state.selected_node_index is None


def test_clearing_a_dropdown_disconnects_immediately_and_is_undoable(chain, canvas):
    """
    Test that clearing a dropdown in the Connections panel disconnects the corresponding edge
    immediately, and that the action is recorded in the undo stack.
    """
    canvas.select(chain["motor"])
    num_undo_steps = len(canvas.state.edit_history_stack)

    # break the connection with selecting the empty string in the dropdown
    canvas.state.connections_rows_column.children[0].children[1].value = ""

    # Check the edge is removed
    assert _edge((chain["cable"], "target", "1"), (chain["motor"], "source", "1")) not in (
        canvas.edges()
    )
    # Check the port is now free
    assert canvas.connected_flags("source", chain["motor"]) == {"1": False}
    # Check that the action is recorded in the undo stack
    assert len(canvas.state.edit_history_stack) == num_undo_steps + 1


def test_connection_picked_in_panel_is_committed_on_apply(canvas):
    """
    This test check the pending connection will not register until the user click on Apply button.
    """
    battery = canvas.place("battery", 100, 100)
    cable = canvas.place("cable", 100, 300)
    canvas.select(cable)

    # Row 0 is the cable's source port; pick the battery's free target port
    canvas.state.connections_rows_column.children[0].children[1].value = "battery_1 (target:1)"
    assert canvas.edges() == []  # Only a preview until Apply
    canvas.handler.apply_node_configurations()

    assert canvas.edges() == [_edge((battery, "target", "1"), (cable, "source", "1"))]
    assert canvas.state.panel_pending_connections == []


def test_target_side_connection_picked_in_panel_is_committed_on_apply(canvas):
    motor = canvas.place("e_motor", 100, 500)
    propeller = canvas.place("propeller", 100, 700)
    canvas.select(motor)

    # Row 1 is the motor's target port; pick the propeller's free source port
    canvas.state.connections_rows_column.children[1].children[1].value = "propeller_1 (source:1)"
    canvas.handler.apply_node_configurations()

    assert canvas.edges() == [_edge((motor, "target", "1"), (propeller, "source", "1"))]


def test_changing_the_pick_before_apply_keeps_only_the_last_choice(canvas):
    canvas.place("battery", 100, 100)
    second_battery = canvas.place("battery", 300, 100)
    cable = canvas.place("cable", 100, 300)
    canvas.select(cable)
    source_dropdown = canvas.state.connections_rows_column.children[0].children[1]

    source_dropdown.value = "battery_1 (target:1)"
    source_dropdown.value = "battery_2 (target:1)"
    assert len(canvas.state.panel_pending_connections) == 1
    canvas.handler.apply_node_configurations()

    assert canvas.edges() == [_edge((second_battery, "target", "1"), (cable, "source", "1"))]


def test_primary_port_label_is_highlighted_in_the_panel(canvas):
    # The first component whose default type is a power-split source type
    icon_key = next(
        (
            icon_key
            for icon_key in ICONS_CONFIG
            if any(
                keyword in default_component_type(canvas.state, icon_key)
                for keyword in PRIMARY_SOURCE_PORT_TYPE_KEYWORDS
            )
        ),
        None,
    )
    if icon_key is None:
        pytest.skip("No splitter or gearbox type in this registry")
    node_index = canvas.place(icon_key, 100, 100)
    canvas.select(node_index)

    label_inputs = [
        row_widget.children[0] for row_widget in canvas.state.connections_rows_column.children
    ]
    highlighted = [label_input.value for label_input in label_inputs if label_input.stylesheets]

    assert highlighted == ["source:1"]


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def test_apply_writes_name_position_and_options(canvas):
    node_index = canvas.place(icon_with_options(canvas.state), 100, 500)
    canvas.select(node_index)
    position_choices = list(canvas.state.position_select.options)
    option_rows = canvas.state.options_rows_column.children
    option_select = next(
        option_row.children[1]
        for option_row in option_rows
        if len(option_row.children[1].options) >= 2
    )
    new_option_value = next(
        value for value in option_select.options if value != option_select.value
    )
    option_name = (
        option_rows[[option_row.children[1] for option_row in option_rows].index(option_select)]
        .children[0]
        .value
    )

    # Change the name, position, and one option value, then Apply
    canvas.state.name_input.value = "renamed_node"
    if position_choices:
        canvas.state.position_select.value = position_choices[-1]
    option_select.value = new_option_value
    canvas.handler.apply_node_configurations()

    assert canvas.node_column("name")[node_index] == "renamed_node"
    assert canvas.state.hover_source.data["name"][node_index] == "renamed_node"
    if position_choices:
        assert canvas.node_column("position")[node_index] == position_choices[-1]
    saved_options = json.loads(canvas.node_column("options")[node_index])
    assert str(saved_options[option_name]) == new_option_value


def test_symmetry_is_stored_on_both_nodes_and_moves_with_a_new_peer(canvas):
    first_motor = canvas.place("e_motor", 100, 100)
    second_motor = canvas.place("e_motor", 300, 100)
    third_motor = canvas.place("e_motor", 500, 100)

    canvas.select(first_motor)
    canvas.state.symmetry_select.value = "e_motor_2"
    canvas.handler.apply_node_configurations()
    assert canvas.node_column("symmetry_name") == ["e_motor_2", "e_motor_1", ""]
    assert canvas.node_column("symmetry_node_index") == [second_motor, first_motor, -1]

    # Re-pairing the first motor with the third one releases the second one
    canvas.state.symmetry_select.value = "e_motor_3"
    canvas.handler.apply_node_configurations()
    assert canvas.node_column("symmetry_name") == ["e_motor_3", "", "e_motor_1"]
    assert canvas.node_column("symmetry_node_index") == [third_motor, -1, first_motor]


def test_apply_is_a_single_undo_step(chain, canvas):
    canvas.select(chain["motor"])
    # Change the name and apply
    canvas.state.name_input.value = "front_motor"
    canvas.handler.apply_node_configurations()

    # Undo the change
    canvas.handler._on_undo()

    assert canvas.node_column("name")[chain["motor"]] == "e_motor_1"


def test_variable_port_count_is_committed_on_apply_and_prunes_edges(canvas):
    bus = canvas.place("bus_bar", 100, 300)
    motors = [canvas.place("e_motor", 100 + 200 * i, 500) for i in range(3)]
    for port_number, motor in enumerate(motors, start=1):
        canvas.connect(("target", bus, str(port_number)), ("source", motor, "1"))
    canvas.select(bus)

    canvas.state.target_count_spinner.value = 2
    # The bus has 3 (default) target ports
    assert len(canvas.node_column("n_targets")) and canvas.node_column("n_targets")[bus] == 3
    canvas.handler.apply_node_configurations()
    # After applying, the bus should have only 2 target ports
    assert canvas.node_column("n_targets")[bus] == 2
    assert len(canvas.edges()) == 2


# ---------------------------------------------------------------------------
# Switching the component type
# ---------------------------------------------------------------------------


def test_type_switch_clears_connections_and_symmetry_in_one_undo_step(canvas):
    icon_key = icon_with_several_types(canvas.state)
    switched_node = canvas.place(icon_key, 100, 300)
    other_node = canvas.place(icon_key, 300, 300)
    # Neighbours of every energy type, so the node can always be wired to one
    canvas.place("battery", 500, 300)
    canvas.place("propeller", 500, 500)
    canvas.place("fuel_tank", 500, 700)
    canvas.select(switched_node)
    # Fill up symmetry and connections
    canvas.state.symmetry_select.value = f"{icon_key}_2"
    for row_widget in canvas.state.connections_rows_column.children:
        free_choices = [choice for choice in row_widget.children[1].options if choice]
        if free_choices:
            row_widget.children[1].value = free_choices[0]
            break
    canvas.handler.apply_node_configurations()
    assert len(canvas.edges()) == 1
    num_undo_steps = len(canvas.state.edit_history_stack)
    saved_type = canvas.node_column("node_type")[switched_node]

    # Pick a different type for the same icon
    other_type = next(
        component_type
        for component_type in canvas.state.type_select.options
        if component_type != saved_type
    )

    canvas.state.type_select.value = other_type
    # Check that the type switch clears the connections and symmetry, but does not commit the new
    # type yet
    assert canvas.edges() == []
    assert set(canvas.node_column("symmetry_name")) == {""}
    assert canvas.state.symmetry_select.value == ""
    assert len(canvas.state.edit_history_stack) == num_undo_steps + 1
    # The type itself still waits for Apply
    assert canvas.node_column("node_type")[switched_node] == saved_type

    canvas.handler._on_undo()
    assert len(canvas.edges()) == 1
    assert canvas.node_column("symmetry_node_index")[other_node] == switched_node


def test_selecting_a_node_never_clears_its_connections(chain, canvas):
    canvas.select(chain["motor"])
    canvas.select(chain["cable"])
    canvas.select(chain["motor"])

    assert len(canvas.edges()) == 3


def test_selecting_a_node_with_an_unlisted_type_keeps_its_connections(chain, canvas):
    # A node whose saved type is not offered for its icon (e.g. a hand-edited file):
    # filling the panel falls back to the first choice, which must not count as
    # a user switching the type
    nodes_data = dict(canvas.state.placed_nodes_source.data)
    nodes_data["node_type"] = list(nodes_data["node_type"])
    nodes_data["node_type"][chain["motor"]] = "unlisted_motor"
    canvas.state.placed_nodes_source.data = nodes_data

    canvas.select(chain["motor"])

    assert len(canvas.edges()) == 3


# ---------------------------------------------------------------------------
# Delete mode
# ---------------------------------------------------------------------------


def test_deleting_a_node_removes_its_edges_and_reindexes_the_rest(chain, canvas):
    canvas.handler._toggle_delete_mode()
    nodes_data = canvas.state.placed_nodes_source.data
    canvas.tap(nodes_data["x"][chain["cable"]], nodes_data["y"][chain["cable"]])

    assert canvas.node_column("name") == ["battery_1", "e_motor_1", "propeller_1"]
    # The motor (now index 1) keeps its edge to the propeller (now index 2)
    assert canvas.edges() == [_edge((1, "target", "1"), (2, "source", "1"))]


def test_deleting_an_edge_frees_both_ports(chain, canvas):
    canvas.handler._toggle_delete_mode()
    edge_data = canvas.state.edge_source.data
    middle_x = sum(edge_data["xs"][0]) / 2
    middle_y = sum(edge_data["ys"][0]) / 2

    canvas.tap(middle_x, middle_y)

    assert len(canvas.edges()) == 2
    assert canvas.connected_flags("target", chain["battery"]) == {"1": False}
    assert canvas.connected_flags("source", chain["cable"]) == {"1": False}


# ---------------------------------------------------------------------------
# Undo / redo and new design
# ---------------------------------------------------------------------------


def test_undo_and_redo_walk_through_each_action(canvas):
    battery = canvas.place("battery", 100, 100)
    cable = canvas.place("cable", 100, 300)
    canvas.connect(("target", battery, "1"), ("source", cable, "1"))

    canvas.handler._on_undo()
    assert canvas.edges() == []
    canvas.handler._on_undo()
    assert canvas.node_column("name") == ["battery_1"]

    canvas.handler._on_redo()
    canvas.handler._on_redo()
    assert canvas.edges() == [_edge((battery, "target", "1"), (cable, "source", "1"))]
    assert not canvas.state.edit_history_stack.can_redo


def test_new_design_empties_the_canvas_and_history(chain, canvas):
    canvas.handler._on_new_design()

    assert canvas.node_column("name") == []
    assert canvas.edges() == []
    assert not canvas.state.edit_history_stack.can_undo


def test_deleting_a_node_keeps_other_symmetry_links_valid(canvas, tmp_path):
    canvas.place("battery", 100, 100)
    first_motor = canvas.place("e_motor", 300, 100)
    canvas.place("e_motor", 500, 100)
    canvas.select(first_motor)
    canvas.state.symmetry_select.value = "e_motor_2"
    canvas.handler.apply_node_configurations()

    # Deleting the battery moves both motors up by one index
    canvas.handler._toggle_delete_mode()
    canvas.tap(100, 100)

    assert canvas.node_column("name") == ["e_motor_1", "e_motor_2"]
    assert canvas.node_column("symmetry_node_index") == [1, 0]
    yaml_path = tmp_path / "design.yml"
    canvas.handler._save_canvas_state(str(yaml_path))
    assert "symmetrical: e_motor_1" in yaml_path.read_text()


def test_deleting_a_node_releases_its_symmetry_peer(canvas):
    first_motor = canvas.place("e_motor", 300, 100)
    canvas.place("e_motor", 500, 100)
    canvas.select(first_motor)
    canvas.state.symmetry_select.value = "e_motor_2"
    canvas.handler.apply_node_configurations()

    canvas.handler._toggle_delete_mode()
    canvas.tap(300, 100)

    assert canvas.node_column("symmetry_name") == [""]
    assert canvas.node_column("symmetry_node_index") == [-1]


def test_renaming_and_unpairing_in_one_apply_releases_the_old_peer(canvas):
    first_motor = canvas.place("e_motor", 300, 100)
    canvas.place("e_motor", 500, 100)
    canvas.select(first_motor)
    canvas.state.symmetry_select.value = "e_motor_2"
    canvas.handler.apply_node_configurations()

    canvas.state.name_input.value = "left_motor"
    canvas.state.symmetry_select.value = ""
    canvas.handler.apply_node_configurations()

    assert canvas.node_column("symmetry_name") == ["", ""]
    assert canvas.node_column("symmetry_node_index") == [-1, -1]


def test_renaming_a_paired_node_updates_its_peer(canvas):
    first_motor = canvas.place("e_motor", 300, 100)
    canvas.place("e_motor", 500, 100)
    canvas.select(first_motor)
    canvas.state.symmetry_select.value = "e_motor_2"
    canvas.handler.apply_node_configurations()

    canvas.state.name_input.value = "left_motor"
    canvas.handler.apply_node_configurations()

    assert canvas.node_column("symmetry_name") == ["e_motor_2", "left_motor"]


def test_two_undos_then_one_redo_keeps_the_second_redo(canvas):
    canvas.place("battery", 100, 100)
    canvas.place("cable", 100, 300)
    canvas.handler._on_undo()
    canvas.handler._on_undo()

    canvas.handler._on_redo()

    assert canvas.node_column("name") == ["battery_1"]
    assert canvas.state.edit_history_stack.can_redo
    canvas.handler._on_redo()
    assert canvas.node_column("name") == ["battery_1", "cable_1"]
