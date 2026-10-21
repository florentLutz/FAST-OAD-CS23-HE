# History

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_history.py`. It provides the undo / redo stack of the builder, made of snapshots of the design.

A **snapshot** freezes the editable design just before a change:

```python
{
    "nodes":          {column: [values], ...},   # placed_nodes_source.data
    "edges":          {column: [values], ...},   # edge_source.data
    "placed_counter": {icon_key: int, ...},
}
```

Everything is deep-copied, so later edits of the data sources do not affect it. The port sources are not recorded: they are derived from the nodes and edges and rebuilt after every restore. Display settings (icon size, view) are not recorded either.

Typical use in a method that changes the design:

```python
self._push_undo()      # record the design *before* the change
# ... change the design ...
self._mark_unsaved()
```

---

## EditHistoryStack

```python
EditHistoryStack(max_depth=50)
```

Bounded undo stack with a separate redo stack.

---

### Parameters

- **max_depth** : *int, optional*

  Number of snapshots kept on each stack (default: `50`). The oldest ones are dropped silently.

---

### Attributes

- **can_undo** : *bool* — at least one undo step is available.
- **can_redo** : *bool* — at least one redo step is available.
- `len(stack)` — number of undo steps.

---

### Notes

How the handler uses the two stacks:

| Action | Undo stack | Redo stack |
|---|---|---|
| Change (`_push_undo`) | push the design before the change | cleared |
| Undo | pop the snapshot to restore | push the current design |
| Redo | push the current design (`push_undo_keep_redo`) | pop the snapshot to restore |
| New Design | cleared | cleared |

---

### Examples

```python
stack = EditHistoryStack(max_depth=2)
stack.push_undo({"nodes": {}, "edges": {}, "placed_counter": {}})
stack.can_undo, stack.can_redo, len(stack)   # (True, False, 1)
```

---

## push_undo

```python
push_undo(snapshot)
```

Push a snapshot on the undo stack and clear the redo stack: a new change makes the undone steps invalid.

---

### Parameters

- **snapshot** : *dict*

  Snapshot from `make_snapshot`.

---

### Returns

None.

---

## undo

```python
undo()
```

Pop the most recent undo snapshot.

The caller must push the current design on the redo stack first (`push_redo`), so the step can be redone.

---

### Parameters

None.

---

### Returns

- **snapshot** : *dict or None*

  Snapshot to restore, or `None` if the stack is empty.

---

## redo

```python
redo()
```

Pop the most recently undone snapshot.

---

### Parameters

None.

---

### Returns

- **snapshot** : *dict or None*

  Snapshot to restore, or `None` if the stack is empty.

---

## push_undo_keep_redo

```python
push_undo_keep_redo(snapshot)
```

Push a snapshot on the undo stack **without** clearing the redo stack. Used by redo, which records the current design while the remaining redo steps stay available.

---

### Parameters

- **snapshot** : *dict*

---

### Returns

None.

---

## push_redo

```python
push_redo(snapshot)
```

Push a snapshot on the redo stack. Used by undo, before restoring the previous design.

---

### Parameters

- **snapshot** : *dict*

---

### Returns

None.

---

## clear

```python
clear()
```

Discard the whole undo and redo history.

---

### Parameters

None.

---

### Returns

None.

---

## make_snapshot

```python
make_snapshot(state)
```

Return a deep copy of the editable design: node data, edge data and placement counter.

Only plain dictionaries and lists are stored, no Bokeh objects, so a snapshot is independent of the live sources and can be kept indefinitely.

---

### Parameters

- **state** : *BuilderState*

---

### Returns

- **snapshot** : *dict*

  `{"nodes": ..., "edges": ..., "placed_counter": ...}`.