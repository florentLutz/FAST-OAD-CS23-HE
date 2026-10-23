# Unknown components

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_handler_unknown.py`, mixin `UnknownComponentMixin` of [`BuilderHandler`](powertrain_builder_handler).

An *unknown component* is one the builder has no definition for. It is either placed from the toolbox (the **Unknown** 
button, last in all tabs) or loaded from a file whose component `id` is not registered. On the canvas it behaves like 
any other component. Only its configuration panel differs, because there is no list of possible values to choose from:

| Field | Known component| Unknown component |
|---|--------|-----------|
| Type, position, symmetry peer | Selects| Free text |
| Options | Fixed names, value selects | Name and value text boxes; **Add option** adds a row |
| Port counts | Editable only for variable-port component types, minimum 1 | Both sides editable, 0 allowed |
| Port energy type | From the icon | Two selects; **applied once selected**, without clicking Apply |

*Ports of undefined type are grey and cannot be connected until a type is chosen.*

**Class attribute.** `_unknown_option_rows` holds the `(name TextInput, value TextInput)` pairs of the component being 
edited, in panel order.

---

## _show_unknown_widgets

```python
_show_unknown_widgets(is_unknown)
```

Show the free-text widgets of an unknown component and hide the selects of a known one, or the reverse.

The type, position and symmetry fields are swapped; the **Add option** button and the two port-type selects are shown 
for an unknown component only.

---

### Parameters

- **is_unknown** : *bool*

  Whether the selected component is unknown.

---

### Returns

None.

---

## _populate_unknown_fields

```python
_populate_unknown_fields(node_index)
```

Fill the panel of an unknown component with its stored values.

---

### Parameters

- **node_index** : *int*

  Index of an unknown component.

---

### Returns

None.

---

### Notes

1. Type, position and symmetry text fields.
2. Options as editable name / value rows, always shown.
3. Both port-count spinners shown, with a minimum of 0 (e.g. for a tank or a propeller-like component), and the *Ports* 
section shown.
4. Port-type selects set to the stored types (`"undefined"` if empty). This is flagged as programmatic, so 
`_on_port_type_change` ignores it.

---

## _clear_unknown_fields

```python
_clear_unknown_fields()
```

Empty the three text fields, forget the option rows and hide the unknown-component widgets.

---

### Parameters

None.

---

### Returns

None.

---

## _refresh_unknown_options_table

```python
_refresh_unknown_options_table(options)
```

Rebuild the Options rows as editable name / value text pairs.

---

### Parameters

- **options** : *dict*

  Stored options of the component.

---

### Returns

None.

---

### Notes

Every refresh creates new widgets, so text typed in the old rows but not applied is lost.

---

## _add_unknown_option_row

```python
_add_unknown_option_row(option_name='', option_value='')
```

Append one name / value row to the Options table.

---

### Parameters

- **option_name** : *str, optional*

  Initial option name (default: `''`).

- **option_value** : *str, optional*

  Initial value as displayed (default: `''`).

---

### Returns

None.

---

## _on_add_option

```python
_on_add_option()
```

Callback of the **Add option** button: add an empty row to the selected unknown component. Does nothing if no unknown 
component is selected.

The option is written on Apply. A row whose name is left empty is ignored, which is also how an option is removed.

---

### Parameters

None.

---

### Returns

None.

---

## _unknown_panel_fields

```python
_unknown_panel_fields()
```

Return the type, position and options typed in the panel, as Apply writes them.

---

### Parameters

None.

---

### Returns

- **fields** : *dict*

  `{"node_type": str, "position": str, "options": dict}`.

---

### Notes

- An empty type becomes `"unknown"`.
- Text is stripped.
- Rows with an empty name are skipped.
- Values are read as Python literals where possible (`_text_to_option_value`).

---

## _on_port_type_change

```python
_on_port_type_change(attr, old, new, *, side)
```

Callback of the source and target port-type selects: set the energy type of that side of the unknown component at once, 
without Apply, as one undoable step.

---

### Parameters

- **attr**, **old** : *str*

  Bokeh arguments (unused).

- **new** : *str*

  New port type, a key of `PORT_TYPE_COLORS` (`undefined`, `electricity`, `mechanical`, `fuel`).

- **side** : *str*

  `"source"` or `"target"`, bound when the select is wired.

---

### Returns

None.

---

### Notes

Ignored while the panel is being filled, when no unknown component is selected, or when the type does not change. 

Otherwise:

1. Push an undo snapshot.
2. Store the new type in `source_port_type` / `target_port_type`.
3. Remove every connection and pending pick of that side: they joined ports of the old type, so none fits the new one.
4. Rebuild the ports, drawn in the new colour, and mark the design unsaved.

---

## _drop_side_connections

```python
_drop_side_connections(node_index, side)
```

Remove the connections and the pending panel picks attached to one port side of a component.

---

### Parameters

- **node_index** : *int*

  Index of the unknown component.

- **side** : *str*

  `"source"` or `"target"`, the side whose type changed.

---

### Returns

None.

---

### Notes

The ports are not rebuilt here; `_on_port_type_change` does it afterwards.

---

## _saved_options

```python
_saved_options(nodes_data, node_index)
```

Return the stored options of a component, decoded from their JSON string.

---

### Parameters

- **nodes_data** : *dict*

  `placed_nodes_source.data` or a copy.

- **node_index** : *int*

  Index of the component.

---

### Returns

- **options** : *dict*

  `{option_name: value}`; empty if the JSON cannot be read or is not an object.

---

## _text_to_option_value

```python
_text_to_option_value(text)
```

Convert the text of an option value box to a Python value.

---

### Parameters

- **text** : *str*

  Text of the value box.

---

### Returns

- **value** : *object*

  The Python literal the text spells (boolean, number, list, dictionary, quoted string, ...), or the stripped text 
  itself.

---

### Examples

```python
_text_to_option_value("True")      # True
_text_to_option_value(" 2.5 ")     # 2.5
_text_to_option_value("[1, 2]")    # [1, 2]
_text_to_option_value("'abc'")     # 'abc'
_text_to_option_value("li_ion")    # 'li_ion'
```

---

## _port_key

```python
_port_key(port)
```

Return the `(kind, node_index, label)` key of a port dict.

---

### Parameters

- **port** : *dict*

  Port dict.

---

### Returns

- **key** : *tuple*
