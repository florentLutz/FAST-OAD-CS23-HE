# YAML writer

[<- Powertrain builder](powertrain_builder)

Module `power_train_network_writer.py`. It converts the canvas into the powertrain YAML file read by FAST-OAD_CS23-HE.

Unless the design is saved as a final version, the file also has a GUI-only `canvas_layout` section that stores the canvas position of each component, so the same file can be re-opened in the builder with the same arrangement. FAST-OAD_CS23-HE ignores unknown top-level keys (it only logs a warning), so this section does not affect the analysis.

---

## File format

```yaml
title: my_design

power_train_components:

  propeller_1:
    id: fastga_he.pt_component.propeller
    position: in_the_nose

  dc_bus_1:
    id: fastga_he.pt_component.dc_bus
    options:
      number_of_inputs: 1
      number_of_outputs: 2
    position: inside_the_wing

  ...

component_connections:

  - source: propeller_1
    target: motor_1

  - source: [dc_bus_1, 1]
    target: dc_splitter_1

watcher_file_path: results/watcher.csv

# Captured canvas positions used only by the builder GUI to reconstruct.
canvas_layout:
  propeller_1:
    x: 400.0
    y: 820.0
  unknown_1:
    x: 250.0
    y: 500.0
    n_sources: 1
    n_targets: 2
    source_port_type: electricity
    target_port_type: undefined
```

(Component names, positions and options above are only illustrative.)

| Key | Content |
|---|---|
| `title` | File name without extension |
| `power_train_components` | One entry per component: `id`, `options` (if any), `position`, `symmetrical` (if any), in that order |
| `component_connections` | One `source` / `target` pair per connection; `[name, N]` when the component has several ports on that side |
| `watcher_file_path` | Only written when set |
| `canvas_layout` | Only in "Edit Later" files; `x`, `y`, plus port counts and types for unknown components |

**Module constants.** `KEY_TITLE`, `KEY_PT_COMPONENTS`, `KEY_PT_CONNECTIONS`, `KEY_PT_WATCHER`, `KEY_CANVAS_LAYOUT` (the keys above); `SOURCE_SIDE_KEYWORDS` (`_inputs`, `_tanks`) and `TARGET_SIDE_KEYWORDS` (`_outputs`, `_engines`, `_power_sources`), shared with the reader; `COMPONENT_ID_PREFIX` (`fastga_he.pt_component.`); `UNKNOWN_LAYOUT_KEYS` (`n_sources`, `n_targets`, `source_port_type`, `target_port_type`); positions are rounded to one decimal.

---

## PowerTrainYAMLWriter

```python
PowerTrainYAMLWriter(state, node_data=None, edge_data=None, include_canvas_layout=True)
```

Assemble and write the powertrain YAML of a design.

The node and edge data are copied at construction, so the following calls work on a fixed snapshot of the design.

---

### Parameters

- **state** : *BuilderState*

  Builder state. Its data sources are copied when *node_data* / *edge_data* are not given; `possible_options` and `default_source_count` are always read from it.

- **node_data** : *dict, optional*

  Copy of `placed_nodes_source.data` already taken by the caller (default: `None`, copied from *state*).

- **edge_data** : *dict, optional*

  Copy of `edge_source.data` (default: `None`, copied from *state*).

- **include_canvas_layout** : *bool, optional*

  `True` to append the `canvas_layout` section (default); `False` for a final-version file.

---

### See also

- [`_save_canvas_state`](powertrain_builder_io#_save_canvas_state): the builder's use of this class.
- [`PowerTrainYAMLReader`](powertrain_builder_yaml_reader#powertrainyamlreader): the inverse operation.

---

### Examples

```python
from fastga_he.gui.power_train_builder.power_train_network_writer import PowerTrainYAMLWriter

writer = PowerTrainYAMLWriter(state, include_canvas_layout=False)
writer.set_title("my_design")
for node_index in range(len(state.placed_nodes_source.data["name"])):
    writer.add_component(node_index)
writer.add_connection()
writer.set_watcher_file_path("results/watcher.csv")
writer.write("my_design.yml")
```

---

## set_title

```python
set_title(title)
```

Set the `title` of the file.

---

### Parameters

- **title** : *str*

---

### Returns

None.

---

## add_component

```python
add_component(node_index)
```

Add one component to `power_train_components`, and its position to `canvas_layout`.

---

### Parameters

- **node_index** : *int*

  Index of the component in the node data.

---

### Returns

None.

---

### Notes

- **Known component:** `id` from `KNOWN_COMPONENTS`; options = stored options, completed with the type's defaults for options never touched in the GUI; for a variable-port component, its port counts as `number_of_*` options.
- **Unknown component:** `id` built from its type (`unknown_component_id`) and exactly the options typed for it.
- **Symmetry:** the canvas stores the link on both components, but the file has `symmetrical` on one of them only: the one whose peer has a lower index. A peer of an unknown component that is not on the canvas (index −1) is always written.
- **Options** are omitted when empty; `position` is always written.

---

## _layout_entry

```python
_layout_entry(node_index)
```

Return the `canvas_layout` entry of a component: its `x` and `y` rounded to one decimal, plus `UNKNOWN_LAYOUT_KEYS` for an unknown component, whose FAST-OAD entry cannot hold them.

---

### Parameters

- **node_index** : *int*

---

### Returns

- **entry** : *dict*

---

## _saved_options

```python
_saved_options(node_index)
```

Return the options stored on a component, decoded from their JSON string. Values keep their type (e.g. booleans stay booleans).

---

### Parameters

- **node_index** : *int*

---

### Returns

- **options** : *dict*

  Empty if the JSON cannot be read.

---

## _known_id_and_options

```python
_known_id_and_options(node_index)
```

Return the `id` and the options of a known component, as FAST-OAD reads them.

---

### Parameters

- **node_index** : *int*

---

### Returns

- **component_id** : *str*

  E.g. `"fastga_he.pt_component.battery_pack"`.

- **options** : *dict*

  Stored options, defaults of the type for the missing ones, and `number_of_*` options for a variable-port component.

---

### Notes

1. Find the `KNOWN_COMPONENTS` entry of the stored type (the first one, if several entries share the type).
2. Start from the stored options (`_saved_options`).
3. Add the default of every missing option (`default_options` of `possible_options`); stored values are never overwritten.
4. For a variable-port type (default source count `VARIABLE_PORT_DEFAULT`), add the port counts as `number_of_*` options (`_variable_port_options`). These overwrite any stored option of the same name.

---

## add_connection

```python
add_connection()
```

Add every connection of the design to `component_connections`.

An edge drawn on the canvas may start at either kind of port; each one is written source -> target.

---

### Parameters

None.

---

### Returns

None.

---

### Notes

1. `edge_end_ports` gives the two port keys of each edge.
2. The pair is swapped when needed so that the source port comes first.
3. Each end is written by `_connection_endpoint`, with `n_sources` for the source end and `n_targets` for the target end. That count decides between a plain name and `[name, N]`.

Connections are written in edge order.

---

## _connection_endpoint

```python
_connection_endpoint(port, port_count_column)
```

Return one end of a connection as FAST-OAD_CS23-HE expects it.

---

### Parameters

- **port** : *tuple*

  `(kind, node_index, label)` key of the port.

- **port_count_column** : *str*

  `"n_sources"` or `"n_targets"`, the port count of that side.

---

### Returns

- **endpoint** : *str*

  The component name, or `"[name, N]"` when it has several ports on that side.

---

## set_watcher_file_path

```python
set_watcher_file_path(watcher_path)
```

Set `watcher_file_path`. The watcher CSV shows how the specified performance parameters of the chosen powertrain components change at each time step of the flight mission. An empty string leaves the key out of the file.

---

### Parameters

- **watcher_path** : *str*

  Path of the watcher CSV file, or an empty string to leave the field out.

---

### Returns

None.

---

## _build_output

```python
_build_output()
```

Serialise the data and format the YAML text.

---

### Parameters

None.

---

### Returns

- **text** : *str*

---

### Notes

The data is dumped with `ruamel.yaml` (block style, indent 2 / 4 / 2), leaving out `canvas_layout` for a final version and `watcher_file_path` when empty. The text is then post-processed:

1. Remove the quotes around `[name, N]`.
2. Put a `watcher_file_path` value that was wrapped to the next line back on the key's line.
3. Split off the `canvas_layout` section, so the spacing rules only apply to the FAST-OAD part.
4. Add a blank line before every top-level key but the first, between connections, and between components.
5. Re-attach the `canvas_layout` section, compact, after a blank line and a comment marking it as GUI-only.

---

## write

```python
write(file_path)
```

Write the formatted YAML to a file (UTF-8).

---

### Parameters

- **file_path** : *str*

---

### Returns

None.

---

## write_to_stream

```python
write_to_stream(output_stream)
```

Write the formatted YAML to an open file-like object.

---

### Parameters

- **output_stream** : *file-like*

  E.g. `io.StringIO`.

---

### Returns

None.

---

### Examples

```python
import io

buffer = io.StringIO()
writer.write_to_stream(buffer)
print(buffer.getvalue())
```

---

## _variable_port_options

```python
_variable_port_options(component_attributes, source_port_count, target_port_count)
```

Return the `number_of_*` options of a variable-port component from its port counts. Each side is found with `port_side_of`.

---

### Parameters

- **component_attributes** : *list of str*

  `attributes` of the component in `KNOWN_COMPONENTS`.

- **source_port_count**, **target_port_count** : *int*

  Port counts on the canvas.

---

### Returns

- **options** : *dict*

  E.g. `{"number_of_inputs": 1, "number_of_outputs": 2}`.

---

## unknown_component_id

```python
unknown_component_id(component_type)
```

Return the `id` written for an unknown component.

---

### Parameters

- **component_type** : *str*

  Type typed in the panel (empty becomes `"unknown"`).

---

### Returns

- **component_id** : *str*

---

### Examples

```python
unknown_component_id("my_part")                         # 'fastga_he.pt_component.my_part'
unknown_component_id("my_package.pt_component.part")    # 'my_package.pt_component.part'
```

---

## unknown_component_type

```python
unknown_component_type(component_id)
```

Return the type shown for an unknown component read with *component_id*. Inverse of `unknown_component_id`: the FAST-OAD_CS23-HE prefix is removed, any other id is kept whole.

---

### Parameters

- **component_id** : *str or None*

  `id` in the file (may be missing).

---

### Returns

- **component_type** : *str*

  `"unknown"` if the id is missing or is the prefix alone.

---

## port_side_of

```python
port_side_of(attribute_name)
```

Return which port side a `number_of_*` attribute counts. This is the single rule shared by save and load, so both always agree.

---

### Parameters

- **attribute_name** : *str*

  An attribute of a `KNOWN_COMPONENTS` entry.

---

### Returns

- **side** : *str or None*

  - `"source"` if it contains `number_of_` and one of `_inputs`, `_tanks`;
  - `"target"` if it contains `number_of_` and one of `_outputs`, `_engines`, `_power_sources`;
  - `None` otherwise.

---

### Examples

```python
port_side_of("number_of_inputs")    # 'source'
port_side_of("number_of_engines")   # 'target'
port_side_of("cell_chemistry")      # None
```