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
from fastga_he.gui.power_train_builder.power_train_builder_state import (
    BUILDER_ICONS_CONFIG,
    PORT_TYPE_COLORS,
    UNDEFINED_PORT_COLOR,
)
from .conftest import (
    default_component_type,
    icon_with_options,
    icon_with_several_types,
    make_canvas,
)


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


def _replace_canvas(canvas, how: str, tmp_path):
    """Replace the whole canvas the way the user can: undo, New Design or load."""
    if how == "undo":
        canvas.undo()
    elif how == "new_design":
        canvas.new_design()
    else:
        yaml_path = tmp_path / "design.yml"
        assert canvas.save(yaml_path)
        canvas.load(yaml_path)


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
    unknown = canvas.place("unknown", 600, 300)
    num_edges_before = len(canvas.edges())
    num_undo_steps_before = len(canvas.state.edit_history_stack)

    # Battery port 1 is already used
    canvas.connect(("target", chain["battery"], "1"), ("source", extra_cable, "1"))
    assert len(canvas.edges()) == num_edges_before
    # Electric source on the cable vs mechanical target on a free motor: colours differ
    canvas.connect(("source", extra_cable, "1"), ("target", free_motor, "1"))
    assert len(canvas.edges()) == num_edges_before
    # Same kind on both ends
    canvas.connect(("source", extra_cable, "1"), ("source", chain["motor"], "1"))
    assert len(canvas.edges()) == num_edges_before
    # The same port tapped twice cancels the pending connection
    canvas.connect(("source", extra_cable, "1"), ("source", extra_cable, "1"))
    assert len(canvas.edges()) == num_edges_before
    # A component is never connected to itself, even when its two sides carry the
    # same energy type (cable) or accept any (unknown component's undefined ports)
    canvas.connect(("source", extra_cable, "1"), ("target", extra_cable, "1"))
    canvas.connect(("target", unknown, "1"), ("source", unknown, "1"))
    assert len(canvas.edges()) == num_edges_before
    # not even when a connection is committed without the canvas taps (e.g. Apply)
    canvas.handler._add_edge(
        canvas.handler._live_port("source", unknown, "1"),
        canvas.handler._live_port("target", unknown, "1"),
    )

    assert len(canvas.edges()) == num_edges_before
    assert canvas.state.awaiting_second_port_click is None
    # A refused connection leaves nothing to undo
    assert len(canvas.state.edit_history_stack) == num_undo_steps_before


def test_pending_port_click_never_wires_a_node_that_was_not_tapped(canvas):
    battery = canvas.place("battery", 100, 100)
    canvas.place("cable", 100, 300)
    canvas.place("cable", 100, 500)
    # Delete the first cable: the second one moves from index 2 to index 1
    canvas.toggle_delete_mode()
    canvas.tap_node(1)
    canvas.toggle_delete_mode()
    canvas.tap_port("source", 1, "1")  # first click on cable_2
    num_undo_steps_before = len(canvas.state.edit_history_stack)

    # Undo brings cable_1 back at index 1, cable_2 goes back to index 2
    canvas.undo()
    canvas.tap_port("target", battery, "1")

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

    canvas.tap_node(chain["motor"])

    assert canvas.state.selected_node_index is None


def test_clearing_a_dropdown_disconnects_immediately_and_is_undoable(chain, canvas):
    """
    Test that clearing a dropdown in the Connections panel disconnects the corresponding edge
    immediately, and that the action is recorded in the undo stack.
    """
    canvas.select(chain["motor"])
    num_undo_steps = len(canvas.state.edit_history_stack)

    # break the connection with selecting the empty string in the dropdown
    canvas.connection_dropdown("source:1").value = ""

    # Check the edge is removed
    assert _edge((chain["cable"], "target", "1"), (chain["motor"], "source", "1")) not in (
        canvas.edges()
    )
    # Check the port is now free
    assert canvas.connected_flags("source", chain["motor"]) == {"1": False}
    # Check that the action is recorded in the undo stack
    assert len(canvas.state.edit_history_stack) == num_undo_steps + 1


def _check_panel_pick_is_committed_on_apply(canvas, own_icon: str, far_icon: str, own_side: str):
    """
    A connection picked in the Connections panel is only a preview until Apply, and
    changing the pick before Apply keeps only the last choice.

    :param own_icon: Component selected in the panel.
    :param far_icon: Component it is connected to; two of them are placed.
    :param own_side: ``"source"`` or ``"target"``, the side of the selected component.
    """
    far_side = "target" if own_side == "source" else "source"
    canvas.place(far_icon, 100, 100)
    second_far = canvas.place(far_icon, 300, 100)
    own = canvas.place(own_icon, 100, 300)
    canvas.select(own)
    dropdown = canvas.connection_dropdown(f"{own_side}:1")

    dropdown.value = f"{far_icon}_1 ({far_side}:1)"
    dropdown.value = f"{far_icon}_2 ({far_side}:1)"
    assert canvas.edges() == []
    assert len(canvas.state.panel_pending_connections) == 1
    canvas.apply()

    assert canvas.edges() == [_edge((second_far, far_side, "1"), (own, own_side, "1"))]
    assert canvas.state.panel_pending_connections == []


def test_panel_pick_from_a_source_port_is_committed_on_apply(canvas):
    _check_panel_pick_is_committed_on_apply(canvas, "cable", "battery", "source")


def test_panel_pick_from_a_target_port_is_committed_on_apply(canvas):
    _check_panel_pick_is_committed_on_apply(canvas, "e_motor", "propeller", "target")


def _pick_a_battery_for_a_bus(canvas) -> tuple[int, int]:
    """
    Pick, in the Connections panel of a bus, a battery for its source port 1 (not applied).

    :return: ``(battery, bus)`` node indices.
    """
    battery = canvas.place("battery", 100, 100)
    bus = canvas.place("bus_bar", 300, 300)
    canvas.select(bus)
    canvas.connection_dropdown("source:1").value = "battery_1 (target:1)"
    return battery, bus


def _check_pick_survived_and_is_applied(canvas, battery: int, bus: int):
    """
    The pick is still in its dropdown, its dashed preview is on the ports as drawn
    now, and Apply commits it.
    """
    assert ("source:1", "battery_1 (target:1)") in canvas.connection_rows()
    preview_data = canvas.state.panel_preview_edge_source.data
    assert sorted(zip(preview_data["xs"][0], preview_data["ys"][0], strict=True)) == sorted(
        [canvas.port("source", bus, "1"), canvas.port("target", battery, "1")]
    )
    canvas.apply()
    assert canvas.edges() == [_edge((battery, "target", "1"), (bus, "source", "1"))]


def test_panel_pick_survives_an_icon_size_change(canvas):
    battery, bus = _pick_a_battery_for_a_bus(canvas)

    canvas.set_icon_size(70)

    _check_pick_survived_and_is_applied(canvas, battery, bus)


def test_panel_pick_survives_a_port_count_preview(canvas):
    battery, bus = _pick_a_battery_for_a_bus(canvas)

    # Spin the target count only: the picked source port is untouched
    canvas.state.target_count_spinner.value = 2

    _check_pick_survived_and_is_applied(canvas, battery, bus)


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
    canvas.apply()

    assert canvas.node_column("name")[node_index] == "renamed_node"
    if position_choices:
        assert canvas.node_column("position")[node_index] == position_choices[-1]
    saved_options = json.loads(canvas.node_column("options")[node_index])
    assert str(saved_options[option_name]) == new_option_value


def test_apply_is_a_single_undo_step(chain, canvas):
    canvas.select(chain["motor"])
    # Change the name and apply
    canvas.state.name_input.value = "front_motor"
    canvas.apply()

    assert canvas.node_column("name")[chain["motor"]] == "front_motor"

    # Undo the change
    canvas.undo()

    assert canvas.node_column("name")[chain["motor"]] == "e_motor_1"


def test_symmetry_is_stored_on_both_nodes_and_moves_with_a_new_peer(canvas):
    first_motor = canvas.place("e_motor", 100, 100)
    second_motor = canvas.place("e_motor", 300, 100)
    third_motor = canvas.place("e_motor", 500, 100)

    canvas.select(first_motor)
    canvas.state.symmetry_select.value = "e_motor_2"
    canvas.apply()
    assert canvas.node_column("symmetry_name") == ["e_motor_2", "e_motor_1", ""]
    assert canvas.node_column("symmetry_node_index") == [second_motor, first_motor, -1]

    # Re-pairing the first motor with the third one releases the second one
    canvas.state.symmetry_select.value = "e_motor_3"
    canvas.apply()
    assert canvas.node_column("symmetry_name") == ["e_motor_3", "", "e_motor_1"]
    assert canvas.node_column("symmetry_node_index") == [third_motor, -1, first_motor]


def test_renaming_a_paired_node_updates_its_peer_and_unpairing_releases_it(canvas):
    first_motor = canvas.place("e_motor", 300, 100)
    canvas.place("e_motor", 500, 100)
    canvas.select(first_motor)
    canvas.state.symmetry_select.value = "e_motor_2"
    canvas.apply()

    canvas.state.name_input.value = "left_motor"
    canvas.apply()
    assert canvas.node_column("symmetry_name") == ["e_motor_2", "left_motor"]

    # Renaming and unpairing in the same Apply: the peer still names the old
    # name, so it must be found by index to be released
    canvas.state.name_input.value = "front_motor"
    canvas.state.symmetry_select.value = ""
    canvas.apply()
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
    canvas.apply()
    assert canvas.node_column("n_targets")[bus] == 2
    assert len(canvas.edges()) == 2

    canvas.undo()

    assert len(canvas.edges()) == 3


def test_connection_removed_during_a_port_count_preview_stays_removed(canvas):
    battery = canvas.place("battery", 100, 100)
    bus = canvas.place("bus_bar", 300, 300)
    canvas.connect(("target", battery, "1"), ("source", bus, "1"))
    canvas.select(bus)
    canvas.state.target_count_spinner.value = 2  # preview only, not applied

    canvas.connection_dropdown("source:1").value = ""
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
    canvas.apply()
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

    canvas.undo()
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
    canvas.toggle_delete_mode()
    canvas.tap_node(chain["cable"])

    assert canvas.node_column("name") == ["battery_1", "e_motor_1", "propeller_1"]
    # The motor (now index 1) keeps its edge to the propeller (now index 2)
    assert canvas.edges() == [_edge((1, "target", "1"), (2, "source", "1"))]


def test_deleting_an_edge_frees_both_ports(chain, canvas):
    canvas.toggle_delete_mode()
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
    canvas.apply()

    # Deleting the battery moves both motors up by one index
    canvas.toggle_delete_mode()
    canvas.tap(100, 100)
    assert canvas.node_column("name") == ["e_motor_1", "e_motor_2"]
    assert canvas.node_column("symmetry_node_index") == [1, 0]
    yaml_path = tmp_path / "design.yml"
    canvas.save(yaml_path)
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

    canvas.undo()
    assert canvas.edges() == []
    canvas.undo()
    assert canvas.node_column("name") == ["battery_1"]

    # One redo leaves the next redo step available
    canvas.redo()
    assert canvas.node_column("name") == ["battery_1", "cable_1"]
    assert canvas.state.edit_history_stack.can_redo
    canvas.redo()
    assert canvas.edges() == [_edge((battery, "target", "1"), (cable, "source", "1"))]
    assert not canvas.state.edit_history_stack.can_redo


def test_new_design_empties_the_canvas_and_history(chain, canvas):
    canvas.new_design()

    assert canvas.node_column("name") == []
    assert canvas.edges() == []
    assert not canvas.state.edit_history_stack.can_undo


def _click_a_port(canvas):
    canvas.tap_port("target", 0, "1")
    assert canvas.state.awaiting_second_port_click is not None


def _arm_a_component(canvas):
    canvas.click_toolbox("cable")
    assert canvas.state.selected_component == "cable"


def _enter_delete_mode(canvas):
    canvas.toggle_delete_mode()
    assert canvas.state.delete_button.button_type == "danger"


def _check_every_canvas_replacement_returns_to_idle(tmp_path, start_interaction):
    """
    Start *start_interaction*, replace the canvas, and check the canvas is idle again.

    Done once for each way of replacing the canvas (undo, New Design, load), each
    on a fresh canvas.
    """
    for how in ("undo", "new_design", "load"):
        canvas = make_canvas()
        canvas.place("battery", 100, 100)
        canvas.place("cable", 100, 300)
        start_interaction(canvas)

        _replace_canvas(canvas, how, tmp_path)

        state = canvas.state
        assert state.awaiting_second_port_click is None, how
        assert state.awaiting_second_port_click_source.data["x"] == [], how
        assert state.selected_component is None, how
        assert all(button.button_type == "light" for button in state.buttons), how
        assert not state.delete_mode, how
        assert state.delete_button.button_type != "danger", how
        assert "Select a component" in state.status_div.text, how


# A port clicked once, an armed toolbox component and delete mode all refer to
# the canvas being replaced, so undo, New Design and load must each end them


def test_replacing_the_canvas_cancels_a_port_clicked_once(tmp_path):
    _check_every_canvas_replacement_returns_to_idle(tmp_path, _click_a_port)


def test_replacing_the_canvas_disarms_the_toolbox(tmp_path):
    _check_every_canvas_replacement_returns_to_idle(tmp_path, _arm_a_component)


def test_replacing_the_canvas_leaves_delete_mode(tmp_path):
    _check_every_canvas_replacement_returns_to_idle(tmp_path, _enter_delete_mode)


# ---------------------------------------------------------------------------
# Window-fitted canvas (view-scale report of the browser, see the launcher's _JS_FIT_VIEW)
# ---------------------------------------------------------------------------

_FULL_AREA = dict(x0=0.0, x1=800.0, y0=0.0, y1=950.0)


def _send_view_report(canvas, report):
    """Write *report* into ``view_scale_input`` as the browser does (a dict is sent as JSON)."""
    canvas.state.view_scale_input.value = report if isinstance(report, str) else json.dumps(report)


def test_full_view_report_sets_the_scale_and_the_area(canvas):
    _send_view_report(canvas, {"k": 1.5, "area": [0, 1200, -100, 950], "area_k": 1.25})

    assert canvas.state.view_scale == 1.5
    assert canvas.state.canvas_area == dict(x0=0.0, x1=1200.0, y0=-100.0, y1=950.0, per_px=1.25)


def test_view_report_area_scale_falls_back_to_k(canvas):
    _send_view_report(canvas, {"k": 2.0, "area": [0, 800, 0, 950]})

    assert canvas.state.view_scale == 2.0
    assert canvas.state.canvas_area == dict(_FULL_AREA, per_px=2.0)


def test_view_report_as_a_bare_number_is_the_scale(canvas):
    _send_view_report(canvas, "1.75")

    assert canvas.state.view_scale == 1.75
    assert canvas.state.canvas_area is None


def test_view_report_area_without_k(canvas):
    _send_view_report(canvas, {"area": [0, 800, 0, 950]})

    assert canvas.state.view_scale == 1.0
    assert canvas.state.canvas_area == dict(_FULL_AREA, per_px=1.0)


def test_malformed_view_scale_is_ignored(canvas):
    # Each report is sent in turn; none of them changes the scale or sets an area
    for report in ("not json", "[1, 2]", {"k": 0}, {"k": -2}, {"k": "wide"}):
        _send_view_report(canvas, report)

        assert canvas.state.view_scale == 1.0, report
        assert canvas.state.canvas_area is None, report


def test_malformed_view_area_is_ignored_but_the_scale_applies(canvas):
    for area_report in (
        {"area": [0, 800, 0]},  # three values
        {"area": [800, 0, 0, 950]},  # x reversed
        {"area": [0, 800, 950, 0]},  # y reversed
        {"area": [0, "wide", 0, 950]},  # not numbers
        {"area": [0, 800, 0, 950], "area_k": 0},  # zero area scale
    ):
        _send_view_report(canvas, {"k": 1.5, **area_report})

        assert canvas.state.view_scale == 1.5, area_report
        assert canvas.state.canvas_area is None, area_report


def test_new_view_scale_resizes_the_rings_and_labels(canvas):
    canvas.place("battery", 100, 100)
    canvas.state.node_label_set = type("Labels", (), {"y_offset": None})()

    canvas.state.view_scale_input.value = json.dumps({"k": 2.0})
    offset_at_2 = canvas.state.node_label_set.y_offset
    canvas.state.view_scale_input.value = json.dumps({"k": 1.0})

    # Screen-pixel offsets shrink as each pixel covers more data units
    assert offset_at_2 is not None
    assert canvas.state.node_label_set.y_offset == pytest.approx(2 * offset_at_2)


# ---------------------------------------------------------------------------
# Unknown component
# ---------------------------------------------------------------------------


def _port_colors(canvas, kind: str, node_index: int) -> set:
    port_source = (
        canvas.state.source_port_source if kind == "source" else canvas.state.target_port_source
    )
    data = port_source.data
    return {
        color
        for owner_index, color in zip(data["node_index"], data["color"], strict=True)
        if owner_index == node_index
    }


def _edge_color(canvas, first_end: tuple, second_end: tuple) -> str:
    edge_data = canvas.state.edge_source.data
    for edge_position in range(len(edge_data["starting_node_index"])):
        ends = _edge(
            tuple(edge_data[f"starting_{key}"][edge_position] for key in _END_KEYS),
            tuple(edge_data[f"ending_{key}"][edge_position] for key in _END_KEYS),
        )
        if ends == _edge(first_end, second_end):
            return edge_data["color"][edge_position]
    raise AssertionError(f"No edge {first_end} - {second_end}")


_END_KEYS = ("node_index", "port_kind", "port_label")


def test_unknown_component_is_placed_with_one_undefined_port_per_side(canvas):
    # The unknown component is the last button of the toolbox
    assert list(BUILDER_ICONS_CONFIG)[-1] == "unknown"

    unknown = canvas.place("unknown", 300, 300)

    assert canvas.node_column("name") == ["unknown_1"]
    assert canvas.node_column("node_type") == ["unknown"]
    assert (canvas.node_column("n_sources"), canvas.node_column("n_targets")) == ([1], [1])
    assert _port_colors(canvas, "source", unknown) == {UNDEFINED_PORT_COLOR}
    assert _port_colors(canvas, "target", unknown) == {UNDEFINED_PORT_COLOR}


def test_undefined_ports_connect_only_once_typed(canvas):
    battery = canvas.place("battery", 100, 100)
    unknown = canvas.place("unknown", 300, 300)
    canvas.place("unknown", 500, 300)
    num_undo_steps = len(canvas.state.edit_history_stack)

    # Undefined ports connect to nothing: not on the canvas, not in the panel
    canvas.connect(("source", unknown, "1"), ("target", battery, "1"))
    canvas.connect(("source", unknown, "1"), ("target", 2, "1"))
    canvas.select(unknown)
    assert canvas.connection_dropdown("source:1").options == [""]
    assert canvas.edges() == []
    assert len(canvas.state.edit_history_stack) == num_undo_steps

    # Once typed, they connect to that energy type
    canvas.state.source_port_type_select.value = "electricity"
    assert "battery_1 (target:1)" in canvas.connection_dropdown("source:1").options
    canvas.connect(("source", unknown, "1"), ("target", battery, "1"))
    assert canvas.edges() == [_edge((unknown, "source", "1"), (battery, "target", "1"))]
    assert (
        _edge_color(canvas, (unknown, "source", "1"), (battery, "target", "1"))
        == PORT_TYPE_COLORS["electricity"]
    )


def test_unknown_config_panel_uses_text_boxes_committed_on_apply(canvas):
    battery = canvas.place("battery", 100, 100)
    unknown = canvas.place("unknown", 300, 300)
    state = canvas.state
    canvas.select(unknown)

    # Free-text widgets instead of the selects, and every port setting editable
    assert state.unknown_type_input.visible and not state.type_select.visible
    assert state.unknown_position_input.visible and not state.position_select.visible
    assert state.unknown_symmetry_input.visible and not state.symmetry_select.visible
    assert state.options_table.visible and state.add_option_button.visible
    assert state.source_count_spinner.visible and state.target_count_spinner.visible
    assert state.source_count_spinner.low == 0
    assert state.source_port_type_select.visible and state.target_port_type_select.visible

    state.unknown_type_input.value = "my_part"
    state.unknown_position_input.value = "in_the_nose"
    for option_name, option_value in (
        ("efficiency", "0.95"),
        ("adjust_sfc", "True"),
        ("voltages", "[400, 800]"),
        ("mode", "percent_split"),
        ("", "ignored: no name"),
    ):
        canvas.add_unknown_option(option_name, option_value)
    state.source_count_spinner.value = 2
    state.target_count_spinner.value = 0
    state.unknown_symmetry_input.value = "unknown_9"  # not on the canvas
    canvas.apply()

    assert canvas.node_column("node_type")[unknown] == "my_part"
    assert canvas.node_column("position")[unknown] == "in_the_nose"
    assert json.loads(canvas.node_column("options")[unknown]) == {
        "efficiency": 0.95,
        "adjust_sfc": True,
        "voltages": [400, 800],
        "mode": "percent_split",
    }
    assert (canvas.node_column("n_sources")[unknown], canvas.node_column("n_targets")[unknown]) == (
        2,
        0,
    )
    # A peer name that is not on the canvas is kept, without a peer index
    assert canvas.node_column("symmetry_name")[unknown] == "unknown_9"
    assert canvas.node_column("symmetry_node_index")[unknown] == -1

    # A known component gets its selects back
    canvas.select(battery)
    assert state.type_select.visible and not state.unknown_type_input.visible
    assert not state.add_option_button.visible and not state.source_port_type_select.visible


def test_unknown_id_prefix_is_typed_in_its_own_field(canvas):
    battery = canvas.place("battery", 100, 100)
    unknown = canvas.place("unknown", 300, 300)
    state = canvas.state
    prefix_input = state.unknown_id_prefix_input

    # A new unknown component gets the default prefix; a known one has none of its own
    assert canvas.node_column("id_prefix") == ["", "fastga_he.pt_component."]
    canvas.select(unknown)
    assert prefix_input.visible
    assert prefix_input.value == "fastga_he.pt_component."
    assert prefix_input.completions[0] == "fastga_he.pt_component."

    # Committed on Apply with its trailing dot, shown as stored, and suggested from then on
    prefix_input.value = " my_pkg.pt_component "
    canvas.apply()
    assert canvas.node_column("id_prefix")[unknown] == "my_pkg.pt_component."
    assert prefix_input.value == "my_pkg.pt_component."
    assert "my_pkg.pt_component." in prefix_input.completions

    # An empty prefix goes back to the default, and Apply is one undo step
    prefix_input.value = ""
    canvas.apply()
    assert canvas.node_column("id_prefix")[unknown] == "fastga_he.pt_component."
    canvas.undo()
    assert canvas.node_column("id_prefix")[unknown] == "my_pkg.pt_component."

    # A known component has no prefix field
    canvas.select(battery)
    assert not prefix_input.visible


def test_dot_in_unknown_type_shows_a_warning_but_is_kept(canvas):
    battery = canvas.place("battery", 100, 100)
    unknown = canvas.place("unknown", 300, 300)
    state = canvas.state
    warning = state.unknown_type_dot_warning_div
    canvas.select(unknown)
    assert not warning.visible

    # The warning follows each keystroke, before Apply
    state.unknown_type_input.value_input = "my_pkg.my_part"
    assert warning.visible
    state.unknown_type_input.value_input = "my_part"
    assert not warning.visible

    # Apply keeps the type as typed, without touching the prefix
    state.unknown_type_input.value = "my_pkg.my_part"
    canvas.apply()
    assert canvas.node_column("node_type")[unknown] == "my_pkg.my_part"
    assert canvas.node_column("id_prefix")[unknown] == "fastga_he.pt_component."
    assert warning.visible

    # Hidden for a known component, shown again when the unknown one is reselected
    canvas.select(battery)
    assert not warning.visible
    canvas.select(unknown)
    assert warning.visible


def test_port_type_change_applies_at_once_and_drops_unfitting_connections(canvas):
    battery = canvas.place("battery", 100, 100)
    unknown = canvas.place("unknown", 300, 300)
    propeller = canvas.place("propeller", 300, 500)
    canvas.select(unknown)
    canvas.state.source_port_type_select.value = "electricity"
    canvas.state.target_port_type_select.value = "mechanical"
    canvas.connect(("source", unknown, "1"), ("target", battery, "1"))
    canvas.connect(("source", propeller, "1"), ("target", unknown, "1"))
    assert len(canvas.edges()) == 2
    num_undo_steps = len(canvas.state.edit_history_stack)

    # Electric target ports no longer fit the propeller's mechanical source
    canvas.state.target_port_type_select.value = "electricity"
    assert canvas.edges() == [_edge((unknown, "source", "1"), (battery, "target", "1"))]
    assert _port_colors(canvas, "target", unknown) == {PORT_TYPE_COLORS["electricity"]}
    # Back to undefined: that side keeps no connection
    canvas.state.source_port_type_select.value = "undefined"
    assert canvas.edges() == []
    assert _port_colors(canvas, "source", unknown) == {UNDEFINED_PORT_COLOR}
    # Applied at once, no Apply needed: one undo step per change
    assert canvas.node_column("target_port_type")[unknown] == "electricity"
    assert len(canvas.state.edit_history_stack) == num_undo_steps + 2

    canvas.undo()
    canvas.undo()
    assert len(canvas.edges()) == 2
    assert _port_colors(canvas, "target", unknown) == {PORT_TYPE_COLORS["mechanical"]}
