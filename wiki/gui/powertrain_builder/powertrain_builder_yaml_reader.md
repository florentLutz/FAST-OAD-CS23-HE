# YAML reader

[<- Powertrain builder](powertrain_builder)

Module `power_train_network_reader.py`. It is the inverse of the [YAML writer](powertrain_builder_yaml_writer): it reads a powertrain YAML file 
and rebuilds the node and edge data the canvas needs.

Everything the canvas needs comes from the file:

| Section | Gives |
|---|---|
| `power_train_components` | Name, type (from `id`), icon, position, options, symmetry peer, port counts |
| `component_connections` | Connections; `[name, N]` is port `N`, a plain name is port `1` |
| `watcher_file_path` | Path of the CSV of per-time-step performance parameters of the powertrain components; kept, so that saving again preserves it |
| `canvas_layout` (optional) | Canvas `x`, `y` of each component; port counts and types of unknown components |

Components without a `canvas_layout` entry (final-version files, hand-written files, older files) are placed 
automatically with the network viewer's `HierarchicalLayout`: energy storage at the bottom, propulsors at the top. 
**Any valid powertrain YAML can therefore be opened in the builder.** If that layout cannot be computed, a grid is 
used instead.

Port and edge coordinates are not computed here; `_rebuild_all_ports` computes them after loading.

---

## PowerTrainYAMLReader

```python
PowerTrainYAMLReader(state, icon_size)
```

Rebuild canvas data from a powertrain YAML file.

---

### Parameters

- **state** : *BuilderState*

  Only its read-only lookups are used (`possible_position`, `possible_options`, `default_source_count`, 
  `default_target_count`) and its `canvas_area`, if the browser reported one.

- **icon_size** : *int*

  Icon size of the loaded components.

---

### See also

- [`_load_canvas_state`](powertrain_builder_io#_load_canvas_state): the builder's use of this class.

---

## read

```python
read(file_path)
```

Parse a powertrain YAML file and return the design it describes.

---

### Parameters

- **file_path** : *str or pathlib.Path*

  Path of a `.yml` / `.yaml` file.

---

### Returns

- **canvas_state** : *dict*

  `{"components": {column: [...]}, "connections": {column: [...]}, "watcher_file_path": str}`, ready for 
  `_restore_canvas_from_dict`.

---

### Raises

- **OSError** if the file cannot be opened.
- **ruamel.yaml.YAMLError** if the file is not valid YAML.

---

### Notes

The file is read in five steps.

1. Load the YAML (safe loader).
2. Build the nodes (`_build_nodes`).
3. Build the edges (`_build_edges`); this also gives undefined port sides of unknown components the type of what they 
are connected to.
4. Give unknown components without stored port counts enough ports for their connections (`_fit_unknown_port_counts`).
5. Place the nodes without a layout entry (`_place_nodes_without_layout`).

Problems in the file that do not prevent reading it (unknown component reference, incompatible connection, missing 
symmetry peer, ...) are logged as warnings and skipped.

---

### Examples

```python
from fastga_he.gui.power_train_builder.power_train_network_reader import PowerTrainYAMLReader

canvas_state = PowerTrainYAMLReader(state, icon_size=50).read("my_design.yml")
print(canvas_state["components"]["name"])
```

---

## _build_nodes

```python
_build_nodes(components, canvas_layout)
```

Build the node data from `power_train_components`.

---

### Parameters

- **components** : *dict*

  `power_train_components` of the file.

- **canvas_layout** : *dict*

  `canvas_layout` of the file (empty if missing).

---

### Returns

- **nodes_data** : *dict*

  Node columns (`NODE_COLUMNS`).

- **name_to_index** : *dict*

  `{component_name: node_index}`.

- **nodes_without_layout** : *list of int*

  Nodes without a valid layout entry, given a temporary position `(0, 0)`.

- **propulsor_names** : *set of str*

  Names of the components of class `propulsor`.

---

### Notes

For each component: a known component (`_drawable_known_component`) gets its row from `_known_node_row`, any other from 
`_unknown_node_row`. Its position, icon, size and name are added, and `symmetrical` entries are collected and linked at 
the end (`_link_symmetry_requests`), since the peer may come later in the file.

---

## _known_node_row

```python
_known_node_row(component, known_component)
```

Return the type, icon, position, options, port counts and port types of a known component of the file.

---

### Parameters

- **component** : *dict*

  Its entry under `power_train_components`.

- **known_component** : *dict*

  Its `KNOWN_COMPONENTS` entry.

---

### Returns

- **columns** : *dict*

  `node_type`, `icon_type`, `position`, `options` (JSON), `n_sources`, `n_targets`, and empty port types (the colours 
  come from the icon).

---

## _place_nodes_without_layout

```python
_place_nodes_without_layout(nodes_data, edges_data, nodes_without_layout, propulsor_names)
```

Position the nodes without a `canvas_layout` entry, in place.

---

### Parameters

- **nodes_data** : *dict*

  Node columns, edited in place.

- **edges_data** : *dict*

  Edge columns.

- **nodes_without_layout** : *list of int*

  Nodes to place.

- **propulsor_names** : *set of str*

  Names of the propulsors.

---

### Returns

None.

---

### Notes

1. Compute the available area: the canvas area of the user's window, inset by margins (`_layout_area`).
2. Lay out the **whole** graph with `HierarchicalLayout` (`_hierarchical_canvas_positions`), and keep the positions of 
the nodes to place only; nodes with a layout entry keep theirs.
3. If that fails, place the nodes on a grid over the same area (`_grid_positions`).

---

## _layout_position

```python
_layout_position(layout_entry)
```

Return the position stored in a `canvas_layout` entry. Static method.

---

### Parameters

- **layout_entry** : *object*

  Value stored under the component name.

---

### Returns

- **x**, **y** : *float or None*

  `(None, None)` if the entry is not a dictionary with numeric `x` and `y`.

---

## _position

```python
_position(component_type, file_position)
```

Return the position given in the file, or the type's default when it is missing.

---

### Parameters

- **component_type** : *str*

- **file_position** : *str or None*

---

### Returns

- **position** : *str*

---

## _options_json

```python
_options_json(component_type, file_options)
```

Return the options of a known component as stored by the GUI: the type's defaults overridden by the values of the file.

Only options known to the GUI (`possible_options`) are kept. In particular the `number_of_*` options are dropped: they 
are represented by the port counts and written again on the next save.

---

### Parameters

- **component_type** : *str*

- **file_options** : *dict*

  Options of the file.

---

### Returns

- **options** : *str*

  JSON string.

---

## _port_counts

```python
_port_counts(component_type, known_component, file_options)
```

Return the port counts of a known component: the type's defaults, except for a variable-port component, whose 
`number_of_*` options give its counts.

---

### Parameters

- **component_type** : *str*

- **known_component** : *dict*

  Its `KNOWN_COMPONENTS` entry.

- **file_options** : *dict*

  Options of the file.

---

### Returns

- **n_sources**, **n_targets** : *int*

---

### Notes

The side of each `number_of_*` attribute is given by [`port_side_of`](powertrain_builder_yaml_writer#port_side_of), the same rule the writer uses. Values that 
are not integers are ignored.

---

## _build_edges

```python
_build_edges(connections, nodes_data, name_to_index)
```

Build the edge data from `component_connections`.

---

### Parameters

- **connections** : *list of dict*

  `{"source": ..., "target": ...}` entries.

- **nodes_data** : *dict*

  Node columns; undefined port types of unknown components may be set here.

- **name_to_index** : *dict*

  `{component_name: node_index}`.

---

### Returns

- **edges_data** : *dict*

  Edge columns, always stored source -> target, with placeholder `xs` / `ys`.

---

### Notes

1. Read the end points (`_connection_endpoints`).
2. Type the undefined port sides of unknown components from their connections.
3. Keep only connections whose ports may be connected on the canvas: the source side has ports and the two colours are 
the same, defined energy type. Others are logged and skipped.

Connections to a port number above the component's port count are dropped later, when `_rebuild_all_ports` runs.

---

## _drawable_known_component

```python
_drawable_known_component(name, component)
```

Return the `KNOWN_COMPONENTS` entry of a component of the file, if the canvas can draw it.

---

### Parameters

- **name** : *str*

  Name of the component.

- **component** : *dict*

  Its entry under `power_train_components`.

---

### Returns

- **known_component** : *dict or None*

  `None`, and the component is shown as unknown, if its `id` is not registered or its icon is not in 
  `BUILDER_ICONS_CONFIG`.

---

## _connection_endpoints

```python
_connection_endpoints(connections, name_to_index)
```

Return the node index and port label of both ends of every valid connection of the file.

Malformed connections, connections to a component that is not in the file, and connections of a component to itself are 
logged and skipped.

---

### Parameters

- **connections** : *list of dict*

- **name_to_index** : *dict*

---

### Returns

- **endpoints** : *list of tuple*

  `(source_index, source_label, target_index, target_label)`.

---

## _type_unknown_ports_from_connections

```python
_type_unknown_ports_from_connections(nodes_data, endpoints)
```

Give every undefined port side of an unknown component the energy type it is connected to, in place.

---

### Parameters

- **nodes_data** : *dict*

  Node columns, edited in place.

- **endpoints** : *list of tuple*

  From `_connection_endpoints`.

---

### Returns

None.

---

### Notes

A file without `canvas_layout` does not store the port types of its unknown components, but its connections imply them. 
The first connection to a defined port decides. The pass is repeated until nothing changes, so a chain of unknown 
components is typed from its known end. A side connected to nothing defined stays undefined, and its connections are 
then refused.

---

## _type_undefined_side

```python
_type_undefined_side(nodes_data, node_index, side, far_color)
```

Set one undefined port side of an unknown component to the energy type of *far_color*.

---

### Parameters

- **nodes_data** : *dict*

  Node columns, edited in place.

- **node_index** : *int*

- **side** : *str*

  `"source"` or `"target"`.

- **far_color** : *str*

  Colour of the port at the other end.

---

### Returns

- **typed** : *bool*

  `True` if the side was typed; `False` if the node is known, the side is already typed, or *far_color* is not a 
  defined energy colour.

---

## _unknown_node_row

```python
_unknown_node_row(component, layout_entry)
```

Return the type, icon, position, options and port settings of an unknown component of the file.

---

### Parameters

- **component** : *dict*

  Its entry under `power_train_components`.

- **layout_entry** : *dict or None*

  Its `canvas_layout` entry.

---

### Returns

- **columns** : *dict*

---

### Notes

- Type: from the `id` (`unknown_component_type`).
- Options: kept exactly as written.
- Port counts and types: from the layout entry when the file was saved for later edits. Otherwise the types are 
undefined and the counts are `None`, to be set by `_fit_unknown_port_counts`.

---

## _layout_port_count

```python
_layout_port_count(layout_entry, key)
```

Return a port count stored in a `canvas_layout` entry, or `None` if missing, not an integer or negative.

---

### Parameters

- **layout_entry** : *dict*

- **key** : *str*

  `"n_sources"` or `"n_targets"`.

---

### Returns

- **count** : *int or None*

---

## _layout_port_type

```python
_layout_port_type(layout_entry, key)
```

Return a port type stored in a `canvas_layout` entry, or `"undefined"` if missing or invalid.

---

### Parameters

- **layout_entry** : *dict*

- **key** : *str*

  `"source_port_type"` or `"target_port_type"`.

---

### Returns

- **port_type** : *str*

---

## _fit_unknown_port_counts

```python
_fit_unknown_port_counts(nodes_data, edges_data)
```

Give every unknown component without stored port counts enough ports for its connections, in place: on each side, the 
highest port number used, and at least 1.

---

### Parameters

- **nodes_data** : *dict*

  Node columns, edited in place.

- **edges_data** : *dict*

  Edge columns.

---

### Returns

None.

---

### Notes

Only `None` counts are set: counts read from `canvas_layout` are kept even if a connection uses a higher port, and 
`_rebuild_all_ports` then drops that connection.

---

## _link_symmetry_requests

```python
_link_symmetry_requests(nodes_data, name_to_index, symmetry_requests)
```

Link every `symmetrical` pair of the file on both components, in place.

The writer puts `symmetrical` on one component of each pair only, but the canvas stores the link on both. A peer missing
from the file is logged and ignored, except for an unknown component, which keeps the name without a peer on the canvas.

---

### Parameters

- **nodes_data** : *dict*

  Node columns, edited in place.

- **name_to_index** : *dict*

- **symmetry_requests** : *list of tuple*

  `(node_index, peer_name)` of every `symmetrical` entry.

---

### Returns

None.

---

## _parse_endpoint

```python
_parse_endpoint(endpoint)
```

Convert a connection end of the file to `(component_name, port_label)`.

---

### Parameters

- **endpoint** : *object*

  Raw `source` or `target` value.

---

### Returns

- **endpoint** : *tuple or None*

  `None` if malformed. Labels are strings, like the port sources.

---

### Examples

```python
_parse_endpoint(["dc_bus_1", 2])    # ('dc_bus_1', '2')
_parse_endpoint("[dc_bus_1, 2]")    # ('dc_bus_1', '2')
_parse_endpoint("motor_1")          # ('motor_1', '1')
_parse_endpoint(None)               # None
```

---

## _matching_component_types

```python
_matching_component_types(references)
```

Return the component types whose type, or one of whose type classes, is in *references*. Same rule as 
`FASTGAHEPowerTrainConfigurator.get_component_distance`.

Used at import to build `_ENERGY_STORAGE_COMPONENT_TYPES` from `("tank", "battery_pack")`.

---

### Parameters

- **references** : *tuple of str*

  Component types or type classes.

---

### Returns

- **types** : *set of str*

---

## _hierarchy_layers

```python
_hierarchy_layers(graph, node_types, propulsor_names, from_propulsor=False)
```

Give every node a layer index, 0 being the top of the canvas. Follows the network viewer's 
`GraphBuilder._get_hierarchy_layers`, made robust to unfinished designs.

---

### Parameters

- **graph** : *networkx.DiGraph*

  Powertrain graph.

- **node_types** : *dict*

  `{component_name: component_type}`.

- **propulsor_names** : *set of str*

- **from_propulsor** : *bool, optional*

  Count the layers from the propulsors at once (default: `False`).

---

### Returns

- **layers** : *dict*

  `{component_name: int}` for every node.

---

### Notes

Distances ignore the direction of the edges.

1. **From storage:** `layer = max_distance − distance` to the nearest tank or battery, so storage is at the bottom 
(`_layers_from_energy_storage`).
2. This is kept only if every node farthest from storage is a propulsor. Otherwise, or if *from_propulsor* is set, the 
layers are the distance **from** the propulsors, which puts every propulsor in the top layer.
3. Unlike the viewer, a design may have no storage, no propulsor, or separate parts. If neither reference reaches every 
node, the one reaching more nodes is used (`_chosen_layers`) and the rest go to an extra bottom layer 
(`_with_unreached_nodes_at_the_bottom`).

---

## _layers_from_energy_storage

```python
_layers_from_energy_storage(undirected_graph, node_types, propulsor_names)
```

Return the layers counted from the energy storage.

---

### Parameters

- **undirected_graph** : *networkx.Graph*

- **node_types** : *dict*

- **propulsor_names** : *set of str*

---

### Returns

- **layers** : *dict*

  Layer of every node the storage reaches; empty without storage.

- **propulsors_not_on_top** : *bool*

  Whether a node farthest from storage is not a propulsor.

---

## _chosen_layers

```python
_chosen_layers(layers_from_energy, layers_from_propulsor, num_nodes, from_propulsor)
```

Choose the layering: storage if it reaches every node and propulsors are not required; else propulsors if they reach 
every node; else whichever reaches more nodes (storage on a tie).

---

### Parameters

- **layers_from_energy**, **layers_from_propulsor** : *dict*

- **num_nodes** : *int*

- **from_propulsor** : *bool*

---

### Returns

- **layers** : *dict*

  May miss nodes neither reference reaches.

---

## _with_unreached_nodes_at_the_bottom

```python
_with_unreached_nodes_at_the_bottom(graph, layers)
```

Complete the layers with the nodes they miss, all in one extra bottom layer.

---

### Parameters

- **graph** : *networkx.DiGraph*

- **layers** : *dict*

---

### Returns

- **layers** : *dict*

  Integer layer of every node.

---

## _layout_area

```python
_layout_area(canvas_area)
```

Return the area available to generated layouts: the window's canvas area, inset by margins.

---

### Parameters

- **canvas_area** : *dict*

  `{"x0", "x1", "y0", "y1", "per_px"}` in data units.

---

### Returns

- **area** : *tuple of float*

  `(x_min, x_max, y_min, y_max)`.

---

### Notes

The margins are 50 px left, right and bottom, and 110 px top (to clear the Undo / Redo buttons and the legend), 
converted to data units with `per_px`, and never less than `CANVAS_FIT_NODE_MARGIN` (60), so the loaded design fits the 
view without zooming out. Bounds are rounded inwards to 0.1. For a window too small, the area collapses onto its centre.

With the default area (800 × 950 at 1 unit per pixel), the result is `(60.0, 740.0, 60.0, 840.0)`.

---

## _spread

```python
_spread(values, low, high, step_max)
```

Map evenly spaced abstract coordinates onto `[low, high]`: spread as widely as the range allows, at most *step_max* 
apart per unit, and centred.

---

### Parameters

- **values** : *list of float*

- **low**, **high** : *float*

  Target range.

- **step_max** : *float*

  Largest distance between two neighbouring units.

---

### Returns

- **mapped** : *list of float*

  Same order, rounded to 0.1 and clamped to the range.

---

### Examples

```python
_spread([-1, 0, 1], 0, 100, 30)    # [20.0, 50.0, 80.0]
_spread([-1, 0, 1], 0, 100, 80)    # [0.0, 50.0, 100.0]
```

---

## _hierarchical_canvas_positions

```python
_hierarchical_canvas_positions(nodes_data, edges_data, propulsor_names, area, from_propulsor=False)
```

Run the viewer's `HierarchicalLayout` on the loaded design and spread the result over *area*.

---

### Parameters

- **nodes_data**, **edges_data** : *dict*

- **propulsor_names** : *set of str*

- **area** : *tuple of float*

  From `_layout_area`.

- **from_propulsor** : *bool, optional*

  Default: `False`.

---

### Returns

- **positions** : *dict*

  `{component_name: (x, y)}` for every node.

---

### Notes

1. Build a `networkx.DiGraph` with one node per component name and one edge per connection, source -> target.
2. Give every node a layer with `_hierarchy_layers`.
3. Run `HierarchicalLayout` (`orientation="TB"`, `sorting=True`) with these layers.
4. Map the abstract coordinates onto *area* with `_spread`, `x` and `y` separately.

The layout works in abstract units (x = slot in the layer, y = layer row, top-to-bottom orientation with crossing 
reduction). They are mapped with `_spread`, with at most 25 % of the area width between neighbours in `x` and 30 % of 
the height in `y`, so a small design stays compact and centred on a large screen.

---

## _grid_positions

```python
_grid_positions(count, area)
```

Return evenly spaced positions over an area, row by row from the top left. Fallback when the hierarchical layout fails.

---

### Parameters

- **count** : *int*

  Number of positions.

- **area** : *tuple of float*

  `(x_min, x_max, y_min, y_max)`.

---

### Returns

- **positions** : *list of tuple*

  `(x, y)` pairs.

---

### Notes

The number of columns follows the aspect ratio of the area: `round(sqrt(count × width / height))`, between 1 and 
*count*.
