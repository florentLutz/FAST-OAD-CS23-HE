# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Snapshot-based undo / redo stack for the powertrain builder.

:class:`EditHistoryStack` holds the undo snapshots of :class:`BuilderHandler`,
and a separate *redo* stack filled whenever the user undoes a change.

A **snapshot** is a plain ``dict`` that freezes the editable canvas state at
the moment a mutating action begins::

    {
        "nodes":          {col: [values], …},
        "edges":          {col: [values], …},
        "placed_counter": {key: int, …},
    }

All three entries are **deep-copied** from the live data, so later mutations
to the Bokeh sources do not affect the stored snapshot.

The port sources (``source_port_source`` / ``target_port_source``) are not
stored: they are derived from the nodes and edges, and are rebuilt by
``_rebuild_all_ports`` after every restore. Display settings (icon size, view)
are not stored either.

Typical usage inside a method that changes the design::

    self._push_undo()          # record the design *before* the change
    … change the design …
    self._mark_unsaved()

Undo and redo push the current design onto the other stack before restoring a
snapshot (see ``IOMixin._on_undo`` / ``IOMixin._on_redo``).
"""

import copy
import logging
from collections import deque

from .power_train_builder_state import copy_column_data

_LOGGER = logging.getLogger(__name__)

_DEFAULT_MAX_DEPTH = 50


class EditHistoryStack:
    """
    Undo stack and redo stack of design snapshots.

    :param max_depth: Number of snapshots kept on each stack; the oldest ones
        are dropped silently.
    """

    def __init__(self, max_depth: int = _DEFAULT_MAX_DEPTH):
        """
        Create empty undo and redo stacks.

        :param max_depth: Maximum number of snapshots kept on each stack.
        """
        self._undo: deque = deque(maxlen=max_depth)
        self._redo: deque = deque(maxlen=max_depth)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def push_undo(self, snapshot: dict) -> None:
        """
        Push *snapshot* onto the undo stack and clear the redo stack.

        After a new change, the undone steps can no longer be redone.

        :param snapshot: Canvas snapshot produced by :func:`make_snapshot`.
        """
        self._undo.append(snapshot)
        self._redo.clear()
        _LOGGER.debug("Undo push: depth=%d", len(self._undo))

    def undo(self) -> dict | None:
        """
        Pop and return the most recent snapshot, or ``None`` if empty.

        Before restoring it, the caller must push the current design onto the
        redo stack (:meth:`push_redo`), so the step can be redone.

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

    def push_undo_keep_redo(self, snapshot: dict) -> None:
        """
        Push *snapshot* onto the undo stack without clearing the redo stack.

        Used by the redo handler, which must save the current state for a later
        undo while the remaining redo steps stay available.

        :param snapshot: Snapshot dict produced by :func:`make_snapshot`.
        """
        self._undo.append(snapshot)

    def push_redo(self, snapshot: dict) -> None:
        """
        Push *snapshot* onto the redo stack.

        Used by the undo handler, before restoring the previous design.

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


# ============================================================================
# Snapshot helper (module-level so it can be imported independently)
# ============================================================================


def make_snapshot(state) -> dict:
    """
    Deep copy of the editable design: node data, edge data and placement counter.

    Port data is derived from these and is not recorded.

    :param state: :class:`~.power_train_builder_state.BuilderState` instance.
    :return: Snapshot dict suitable for :meth:`EditHistoryStack.push_undo`.
    """
    return {
        "nodes": copy.deepcopy(copy_column_data(state.placed_nodes_source.data)),
        "edges": copy.deepcopy(copy_column_data(state.edge_source.data)),
        "placed_counter": copy.deepcopy(state.placed_counter),
    }
