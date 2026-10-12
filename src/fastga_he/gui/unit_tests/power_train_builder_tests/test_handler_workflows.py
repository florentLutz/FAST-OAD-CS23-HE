# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Workflow tests for :class:`BuilderHandler`, driven through the GUI entry points.

Each test performs what a user would do (place, tap ports, pick values in the
config panel, Apply, Undo...) and checks the resulting canvas data.

Several tests guard state that must not outlive what it belongs to: a port
clicked once, a Connections panel pick waiting for Apply, a port-count preview,
or an armed toolbox / delete mode when the whole canvas is replaced.
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


def _row_dropdown(canvas, port_text: str):
    """Connections panel dropdown of the selected node's port *port_text* (e.g. ``"source:1"``)."""
    return next(
        row_widget.children[1]
        for row_widget in canvas.state.connections_rows_column.children
        if row_widget.children[0].value == port_text
    )


def _tap_node(canvas, node_index: int):
    nodes_data = canvas.state.placed_nodes_source.data
    canvas.tap(nodes_data["x"][node_index], nodes_data["y"][node_index])


def _replace_canvas(canvas, how: str, tmp_path):
    """Replace the whole canvas the way the user can: undo, New Design or load."""
    if how == "undo":
        canvas.handler._on_undo()
    elif how == "new_design":
        canvas.handler._on_new_design()
    else:
        yaml_path = tmp_path / "design.yml"
        assert canvas.handler._save_canvas_state(str(yaml_path))
        canvas.handler._load_canvas_state(str(yaml_path))


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


def test_invalid_port_taps_make_no_edge(chain, canvas):
    extra_cable = canvas.place("cable", 400, 300)
    free_motor = canvas.place("e_motor", 400, 500)
    num_edges_before = len(canvas.edges())

    # Battery port 1 is already used
    canvas.connect(("target", chain["battery"], "1"), ("source", extra_cable, "1"))
    # Electric source on the cable vs mechanical target on a free motor: colours differ
    canvas.connect(("source", extra_cable, "1"), ("target", free_motor, "1"))
    # Same kind on both ends
    canvas.connect(("source", extra_cable, "1"), ("source", chain["motor"], "1"))
    # The same port tapped twice cancels the pending connection
    canvas.connect(("source", extra_cable, "1"), ("source", extra_cable, "1"))

    assert len(canvas.edges()) == num_edges_before
    assert canvas.state.awaiting_second_port_click is None


def test_pending_port_click_never_wires_a_node_that_was_not_tapped(canvas):
    battery = canvas.place("battery", 100, 100)
    canvas.place("cable", 100, 300)
    canvas.place("cable", 100, 500)
    # Delete the first cable: the second one moves from index 2 to index 1
    canvas.handler._toggle_delete_mode()
    _tap_node(canvas, 1)
    canvas.handler._toggle_delete_mode()
    canvas.tap(*canvas.port("source", 1, "1"))  # first click on cable_2
    num_undo_steps_before = len(canvas.state.edit_history_stack)

    # Undo brings cable_1 back at index 1, cable_2 goes back to index 2
    canvas.handler._on_undo()
    canvas.tap(*canvas.port("target", battery, "1"))

    # The click after the undo starts a new connection; it must not connect
    # the battery to cable_1, which the user never tapped
    assert canvas.edges() == []
    assert len(canvas.state.edit_history_stack) == num_undo_steps_before - 1


# ---------------------------------------------------------------------------
# Selecting a node and the Connections panel
# ---------------------------------------------------------------------------


def test_selecting_a_node_fills_the_panel_and_tapping_it_again_deselects(chain, canvas):
    canvas.select(chain["motor"])

    assert canvas.connection_rows() == [
        ("source:1", "cable_1 (target:1)"),
        ("target:1", "propeller_1 (source:1)"),
    ]
    assert canvas.state.config_panel_column.visible

    _tap_node(canvas, chain["motor"])

    assert canvas.state.selected_node_index is None


def test_clearing_a_dropdown_disconnects_immediately_and_is_undoable(chain, canvas):
    """
    Test that clearing a dropdown in the Connections panel disconnects the corresponding edge
    immediately, and that the action is recorded in the undo stack.
    """
    canvas.select(chain["motor"])
    num_undo_steps = len(canvas.state.edit_history_stack)

    # break the connection with selecting the empty string in the dropdown
    _row_dropdown(canvas, "source:1").value = ""

    # Check the edge is removed
    assert _edge((chain["cable"], "target", "1"), (chain["motor"], "source", "1")) not in (
        canvas.edges()
    )
    # Check the port is now free
    assert canvas.connected_flags("source", chain["motor"]) == {"1": False}
    # Check that the action is recorded in the undo stack
    assert len(canvas.state.edit_history_stack) == num_undo_steps + 1


@pytest.mark.parametrize(
    "own_icon, far_icon, own_side",
    [("cable", "battery", "source"), ("e_motor", "propeller", "target")],
)
def test_panel_pick_is_committed_on_apply_and_only_the_last_pick_counts(
    canvas, own_icon, far_icon, own_side
):
    """
    A connection picked in the Connections panel is only a preview until Apply, and
    changing the pick before Apply keeps only the last choice. Checked from both sides.
    """
    far_side = "target" if own_side == "source" else "source"
    canvas.place(far_icon, 100, 100)
    second_far = canvas.place(far_icon, 300, 100)
    own = canvas.place(own_icon, 100, 300)
    canvas.select(own)
    dropdown = _row_dropdown(canvas, f"{own_side}:1")

    dropdown.value = f"{far_icon}_1 ({far_side}:1)"
    dropdown.value = f"{far_icon}_2 ({far_side}:1)"
    assert canvas.edges() == []
    assert len(canvas.state.panel_pending_connections) == 1
    canvas.handler.apply_node_configurations()

    assert canvas.edges() == [_edge((second_far, far_side, "1"), (own, own_side, "1"))]
    assert canvas.state.panel_pending_connections == []


@pytest.mark.parametrize("rebuild", ["icon_size", "port_count_preview"])
def test_panel_pick_survives_a_panel_rebuild(canvas, rebuild):
    """
    A pick waiting for Apply stays in its dropdown, with its dashed preview on the
    ports as drawn now, when the panel is rebuilt by the icon-size slider or a
    port-count preview; Apply then commits it.
    """
    battery = canvas.place("battery", 100, 100)
    bus = canvas.place("bus_bar", 300, 300)
    canvas.select(bus)
    _row_dropdown(canvas, "source:1").value = "battery_1 (target:1)"

    if rebuild == "icon_size":
        canvas.handler._on_node_size_slider_change("value", 50, 70)
    else:
        # Spin the target count only: the picked source port is untouched
        canvas.state.target_count_spinner.value = 2

    assert ("source:1", "battery_1 (target:1)") in canvas.connection_rows()
    preview_data = canvas.state.panel_preview_edge_source.data
    assert sorted(zip(preview_data["xs"][0], preview_data["ys"][0], strict=True)) == sorted(
        [canvas.port("source", bus, "1"), canvas.port("target", battery, "1")]
    )
    canvas.handler.apply_node_configurations()
    assert canvas.edges() == [_edge((battery, "target", "1"), (bus, "source", "1"))]


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
    if position_choices:
        assert canvas.node_column("position")[node_index] == position_choices[-1]
    saved_options = json.loads(canvas.node_column("options")[node_index])
    assert str(saved_options[option_name]) == new_option_value


def test_apply_is_a_single_undo_step(chain, canvas):
    canvas.select(chain["motor"])
    # Change the name and apply
    canvas.state.name_input.value = "front_motor"
    canvas.handler.apply_node_configurations()

    # Undo the change
    canvas.handler._on_undo()

    assert canvas.node_column("name")[chain["motor"]] == "e_motor_1"


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


def test_renaming_a_paired_node_updates_its_peer_and_unpairing_releases_it(canvas):
    first_motor = canvas.place("e_motor", 300, 100)
    canvas.place("e_motor", 500, 100)
    canvas.select(first_motor)
    canvas.state.symmetry_select.value = "e_motor_2"
    canvas.handler.apply_node_configurations()

    canvas.state.name_input.value = "left_motor"
    canvas.handler.apply_node_configurations()
    assert canvas.node_column("symmetry_name") == ["e_motor_2", "left_motor"]

    # Renaming and unpairing in the same Apply: the peer still names the old
    # name, so it must be found by index to be released
    canvas.state.name_input.value = "front_motor"
    canvas.state.symmetry_select.value = ""
    canvas.handler.apply_node_configurations()
    assert canvas.node_column("symmetry_name") == ["", ""]
    assert canvas.node_column("symmetry_node_index") == [-1, -1]


# ---------------------------------------------------------------------------
# Port counts
# ---------------------------------------------------------------------------


def test_port_count_reduction_prunes_edges_and_undo_brings_them_back(canvas):
    bus = canvas.place("bus_bar", 100, 300)
    motors = [canvas.place("e_motor", 100 + 200 * i, 500) for i in range(3)]
    for port_number, motor in enumerate(motors, start=1):
        canvas.connect(("target", bus, str(port_number)), ("source", motor, "1"))
    canvas.select(bus)

    canvas.state.target_count_spinner.value = 2
    # The spinner only previews: the bus keeps its 3 (default) target ports until Apply
    assert canvas.node_column("n_targets")[bus] == 3
    canvas.handler.apply_node_configurations()
    assert canvas.node_column("n_targets")[bus] == 2
    assert len(canvas.edges()) == 2

    canvas.handler._on_undo()

    assert len(canvas.edges()) == 3


def test_connection_removed_during_a_port_count_preview_stays_removed(canvas):
    battery = canvas.place("battery", 100, 100)
    bus = canvas.place("bus_bar", 300, 300)
    canvas.connect(("target", battery, "1"), ("source", bus, "1"))
    canvas.select(bus)
    canvas.state.target_count_spinner.value = 2  # preview only, not applied

    _row_dropdown(canvas, "source:1").value = ""
    # Deselecting ends the preview with a real rebuild
    canvas.tap(750, 900)

    assert canvas.state.selected_node_index is None
    assert canvas.edges() == []


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


def test_selecting_nodes_never_clears_their_connections(chain, canvas):
    canvas.select(chain["motor"])
    canvas.select(chain["cable"])
    canvas.select(chain["motor"])
    assert len(canvas.edges()) == 3

    # A node whose saved type is not offered for its icon (e.g. a hand-edited file):
    # filling the panel falls back to the first choice, which must not count as
    # a user switching the type
    canvas.select(chain["cable"])
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
    _tap_node(canvas, chain["cable"])

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


def test_deleting_nodes_keeps_symmetry_links_valid(canvas, tmp_path):
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

    # Deleting one motor of the pair releases the other
    canvas.tap(300, 100)
    assert canvas.node_column("symmetry_name") == [""]
    assert canvas.node_column("symmetry_node_index") == [-1]


# ---------------------------------------------------------------------------
# Undo / redo, New Design and replacing the canvas
# ---------------------------------------------------------------------------


def test_undo_and_redo_walk_through_each_action(canvas):
    battery = canvas.place("battery", 100, 100)
    cable = canvas.place("cable", 100, 300)
    canvas.connect(("target", battery, "1"), ("source", cable, "1"))

    canvas.handler._on_undo()
    assert canvas.edges() == []
    canvas.handler._on_undo()
    assert canvas.node_column("name") == ["battery_1"]

    # One redo leaves the next redo step available
    canvas.handler._on_redo()
    assert canvas.node_column("name") == ["battery_1", "cable_1"]
    assert canvas.state.edit_history_stack.can_redo
    canvas.handler._on_redo()
    assert canvas.edges() == [_edge((battery, "target", "1"), (cable, "source", "1"))]
    assert not canvas.state.edit_history_stack.can_redo


def test_new_design_empties_the_canvas_and_history(chain, canvas):
    canvas.handler._on_new_design()

    assert canvas.node_column("name") == []
    assert canvas.edges() == []
    assert not canvas.state.edit_history_stack.can_undo


def _click_a_port(canvas):
    canvas.tap(*canvas.port("target", 0, "1"))
    assert canvas.state.awaiting_second_port_click is not None


def _arm_a_component(canvas):
    canvas.handler.on_toolbox_select(list(ICONS_CONFIG).index("cable"))
    assert canvas.state.selected_component == "cable"


def _enter_delete_mode(canvas):
    canvas.handler._toggle_delete_mode()
    assert canvas.state.delete_button.button_type == "danger"


@pytest.mark.parametrize("how", ["undo", "new_design", "load"])
@pytest.mark.parametrize(
    "start_interaction",
    [_click_a_port, _arm_a_component, _enter_delete_mode],
    ids=["port_clicked_once", "component_armed", "delete_mode"],
)
def test_replacing_the_canvas_returns_to_idle(canvas, tmp_path, how, start_interaction):
    """
    A port clicked once, an armed toolbox component and delete mode all refer to the
    canvas being replaced, so undo, New Design and load must each end them.
    """
    canvas.place("battery", 100, 100)
    canvas.place("cable", 100, 300)
    start_interaction(canvas)

    _replace_canvas(canvas, how, tmp_path)

    assert canvas.state.awaiting_second_port_click is None
    assert canvas.state.awaiting_second_port_click_source.data["x"] == []
    assert canvas.state.selected_component is None
    assert all(button.button_type == "light" for button in canvas.state.buttons)
    assert not canvas.state.delete_mode
    assert canvas.state.delete_button.button_type != "danger"
    assert "Select a component" in canvas.state.status_div.text


# ---------------------------------------------------------------------------
# Window-fitted canvas (view-scale report of the browser, see the launcher's _JS_FIT_VIEW)
# ---------------------------------------------------------------------------

_FULL_AREA = dict(x0=0.0, x1=800.0, y0=0.0, y1=950.0)


@pytest.mark.parametrize(
    "report, view_scale, canvas_area",
    [
        pytest.param(
            {"k": 1.5, "area": [0, 1200, -100, 950], "area_k": 1.25},
            1.5,
            dict(x0=0.0, x1=1200.0, y0=-100.0, y1=950.0, per_px=1.25),
            id="full_report",
        ),
        pytest.param(
            {"k": 2.0, "area": [0, 800, 0, 950]},
            2.0,
            dict(_FULL_AREA, per_px=2.0),
            id="area_scale_falls_back_to_k",
        ),
        pytest.param("1.75", 1.75, None, id="bare_number_is_the_scale"),
        # Malformed scale: the scale is not changed
        pytest.param("not json", 1.0, None, id="not_json"),
        pytest.param("[1, 2]", 1.0, None, id="not_an_object"),
        pytest.param({"k": 0}, 1.0, None, id="zero_scale"),
        pytest.param({"k": -2}, 1.0, None, id="negative_scale"),
        pytest.param({"k": "wide"}, 1.0, None, id="text_scale"),
        pytest.param(
            {"area": [0, 800, 0, 950]}, 1.0, dict(_FULL_AREA, per_px=1.0), id="area_without_k"
        ),
        # Malformed area: the area is ignored, the scale still applies
        pytest.param({"k": 1.5, "area": [0, 800, 0]}, 1.5, None, id="area_of_three_values"),
        pytest.param({"k": 1.5, "area": [800, 0, 0, 950]}, 1.5, None, id="area_x_reversed"),
        pytest.param({"k": 1.5, "area": [0, 800, 950, 0]}, 1.5, None, id="area_y_reversed"),
        pytest.param({"k": 1.5, "area": [0, "wide", 0, 950]}, 1.5, None, id="area_not_numbers"),
        pytest.param(
            {"k": 1.5, "area": [0, 800, 0, 950], "area_k": 0}, 1.5, None, id="zero_area_scale"
        ),
    ],
)
def test_view_scale_report(canvas, report, view_scale, canvas_area):
    """The browser's report sets the view scale and canvas area; malformed parts are ignored."""
    canvas.state.view_scale_input.value = report if isinstance(report, str) else json.dumps(report)

    assert canvas.state.view_scale == view_scale
    assert canvas.state.canvas_area == canvas_area


def test_new_view_scale_resizes_the_rings_and_labels(canvas):
    canvas.place("battery", 100, 100)
    canvas.state.node_label_set = type("Labels", (), {"y_offset": None})()

    canvas.state.view_scale_input.value = json.dumps({"k": 2.0})
    offset_at_2 = canvas.state.node_label_set.y_offset
    canvas.state.view_scale_input.value = json.dumps({"k": 1.0})

    # Screen-pixel offsets shrink as each pixel covers more data units
    assert offset_at_2 is not None
    assert canvas.state.node_label_set.y_offset == pytest.approx(2 * offset_at_2)
