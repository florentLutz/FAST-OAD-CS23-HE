# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Snapshot-based undo / redo stack for the powertrain builder.

:class:`UndoStack` is the single, self-contained history manager used by
:class:`PlacementHandler`.  It keeps a bounded list of canvas snapshots and
a separate *redo* stack that is populated whenever the user undoes an action.

A **snapshot** is a plain ``dict`` that freezes the four mutable
``ColumnDataSource`` data dicts at the moment a mutating action begins::

    {
        "nodes":        {col: [values], …},
        "edges":        {col: [values], …},
        "source_ports": {col: [values], …},
        "target_ports": {col: [values], …},
        "placed_counter": {key: int, …},
    }

All five sub-dicts are **deep-copied** from the live data, so later mutations
to the Bokeh sources do not affect the stored snapshot.

Typical usage inside a mutating handler method::

    self._push_undo()          # capture state *before* the mutation
    … perform mutation …
    self._mark_unsaved()

Then in :meth:`_on_undo`::

    snapshot = self.state.undo_stack.undo()
    if snapshot is not None:
        self._restore_from_snapshot(snapshot)
"""

import copy
import logging
from collections import deque

_LOGGER = logging.getLogger(__name__)

_DEFAULT_MAX_DEPTH = 50


class UndoStack:
    """
    Bounded snapshot stack supporting undo and redo.

    :param max_depth: Maximum number of undo steps to retain.  Older snapshots
        are silently discarded when the limit is reached.
    """

    def __init__(self, max_depth: int = _DEFAULT_MAX_DEPTH):
        self._undo: deque = deque(maxlen=max_depth)
        self._redo: deque = deque(maxlen=max_depth)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, snapshot: dict) -> None:
        """
        Push *snapshot* onto the undo stack and clear the redo stack.

        The redo stack is cleared because a new mutation invalidates any
        previously undone actions.

        :param snapshot: Canvas snapshot produced by :func:`make_snapshot`.
        """
        self._undo.append(snapshot)
        self._redo.clear()
        _LOGGER.debug("Undo push: depth=%d", len(self._undo))

    def undo(self) -> dict | None:
        """
        Pop and return the most recent snapshot, or ``None`` if empty.

        The *current* canvas state (captured by the caller just before calling
        this method) should be pushed onto the redo stack by the caller so that
        a subsequent :meth:`redo` can re-apply it.

        :return: The snapshot to restore, or ``None``.
        """
        if not self._undo:
            _LOGGER.debug("Undo requested but stack is empty.")
            return None
        snapshot = self._undo.pop()
        _LOGGER.debug("Undo pop: remaining=%d", len(self._undo))
        return snapshot

    def redo(self) -> dict | None:
        """
        Pop and return the most recently undone snapshot, or ``None``.

        :return: The snapshot to restore, or ``None``.
        """
        if not self._redo:
            _LOGGER.debug("Redo requested but stack is empty.")
            return None
        snapshot = self._redo.pop()
        _LOGGER.debug("Redo pop: remaining=%d", len(self._redo))
        return snapshot

    def push_redo(self, snapshot: dict) -> None:
        """
        Push *snapshot* onto the redo stack (called by the undo handler before
        restoring the previous state so the action can be re-applied).

        :param snapshot: Canvas snapshot of the state that is about to be
            replaced by the undo operation.
        """
        self._redo.append(snapshot)

    def clear(self) -> None:
        """Discard the entire undo and redo history (e.g. after New Design)."""
        self._undo.clear()
        self._redo.clear()
        _LOGGER.debug("Undo/redo stacks cleared.")

    @property
    def can_undo(self) -> bool:
        """``True`` when at least one undo snapshot is available."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """``True`` when at least one redo snapshot is available."""
        return bool(self._redo)

    def __len__(self) -> int:
        """Return the number of available undo steps."""
        return len(self._undo)


# ------------------------------------------------------------------
# Snapshot helper (module-level so it can be imported independently)
# ------------------------------------------------------------------


def make_snapshot(state) -> dict:
    """
    Capture a deep-copy snapshot of all mutable canvas state.

    Reads the four live ``ColumnDataSource`` data dicts and the
    ``placed_counter`` from *state* and returns an independent copy.  No
    Bokeh objects are stored — only plain Python dicts and lists — so the
    snapshot is cheap to pickle and safe to hold indefinitely.

    :param state: :class:`~.power_train_builder_state.BuilderState` instance.
    :return: Snapshot dict suitable for :meth:`UndoStack.push`.
    """
    return {
        "nodes": copy.deepcopy(
            {key: list(value) for key, value in state.placed_nodes_source.data.items()}
        ),
        "edges": copy.deepcopy({key: list(value) for key, value in state.edge_source.data.items()}),
        "source_ports": copy.deepcopy(
            {key: list(value) for key, value in state.source_port_source.data.items()}
        ),
        "target_ports": copy.deepcopy(
            {key: list(value) for key, value in state.target_port_source.data.items()}
        ),
        "placed_counter": copy.deepcopy(state.placed_counter),
    }
