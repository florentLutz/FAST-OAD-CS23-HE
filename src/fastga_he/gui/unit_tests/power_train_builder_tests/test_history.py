# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""Unit tests for the snapshot-based undo / redo stack."""

from types import SimpleNamespace

from fastga_he.gui.power_train_builder.power_train_builder_history import (
    EditHistoryStack,
    make_snapshot,
)


def test_undo_then_redo_returns_snapshots_in_order():
    """
    Test that if the history stack is stack with the right order with push_undo
    """
    history_stack = EditHistoryStack()
    history_stack.push_undo({"step": 1})
    history_stack.push_undo({"step": 2})

    assert history_stack.can_undo and not history_stack.can_redo
    assert history_stack.undo() == {"step": 2}

    history_stack.push_redo({"step": 3})
    assert history_stack.can_redo
    assert history_stack.redo() == {"step": 3}


def test_new_push_clears_redo_stack():
    history_stack = EditHistoryStack()
    # Make an edit, the undo stack store this step
    history_stack.push_undo({"step": 1})
    # Restore the previous step, the redo stack store
    history_stack.push_redo({"step": 2})
    # MAke a new edit, the redo stack should be cleared
    history_stack.push_undo({"step": 3})

    assert not history_stack.can_redo


def test_empty_stack_returns_none():
    history_stack = EditHistoryStack()

    assert history_stack.undo() is None
    assert history_stack.redo() is None
    assert len(history_stack) == 0


def test_max_depth_drops_oldest_snapshot():
    history_stack = EditHistoryStack(max_depth=3)
    for step in range(5):
        history_stack.push_undo({"step": step})

    assert len(history_stack) == 3
    remaining_steps = [history_stack.undo()["step"] for _ in range(3)]
    assert remaining_steps == [4, 3, 2]


def test_clear_empties_both_stacks():
    history_stack = EditHistoryStack()
    history_stack.push_undo({"step": 1})
    history_stack.push_redo({"step": 2})

    history_stack.clear()

    assert not history_stack.can_undo and not history_stack.can_redo


def test_snapshot_is_independent_of_live_data():
    # Set up a dummy state with some data
    nodes_data = {"name": ["battery_1"], "x": [1.0]}
    state = SimpleNamespace(
        placed_nodes_source=SimpleNamespace(data=nodes_data),
        edge_source=SimpleNamespace(data={"xs": [[0.0, 1.0]]}),
        source_port_source=SimpleNamespace(data={"label": ["1"]}),
        target_port_source=SimpleNamespace(data={"label": ["1"]}),
        placed_counter={"battery": 1},
    )

    # Create a snapshot and modify the live data
    snapshot = make_snapshot(state)
    nodes_data["name"].append("cable_1")
    state.edge_source.data["xs"][0].append(2.0)
    state.placed_counter["battery"] = 5

    assert snapshot["nodes"]["name"] == ["battery_1"]
    assert snapshot["edges"]["xs"] == [[0.0, 1.0]]
    assert snapshot["placed_counter"] == {"battery": 1}


def test_push_undo_keep_redo_leaves_redo_steps_available():
    history_stack = EditHistoryStack()
    # click undo
    history_stack.push_redo({"step": 2})
    # New edit, but keep redo steps available
    history_stack.push_undo_keep_redo({"step": 1})

    assert history_stack.can_undo and history_stack.can_redo
