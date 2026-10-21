# Metadata

[<- Powertrain builder](powertrain_builder)

Module `power_train_builder_metadata.py`. It derives, once at startup, everything the builder needs to know about each registered component type from `KNOWN_COMPONENTS` and from the components' own `constants.py` files: which icon it uses, how many ports it has, and its possible positions and options.

The `constants.py` files are read **without importing them**. Importing a component module runs FAST-OAD submodel registration, which raises *"Name … is already used"* errors when the same module is imported again.

---

## _map_possible_component_types_to_icons

```python
_map_possible_component_types_to_icons()
```

Map each icon key to the list of component types that use it.

---

### Parameters

None.

---

### Returns

- **icon_to_component_types** : *dict*

  `{icon_key: [components_type, ...]}`, in registry order. The first type of each list is the default type of a newly placed component.

---

## component_type_classes

```python
component_type_classes(known_component)
```

Return the type classes of a `KNOWN_COMPONENTS` entry, always as a list. `components_type_class` holds either one class name or a list of them.

---

### Parameters

- **known_component** : *dict*

  One entry of `KNOWN_COMPONENTS`.

---

### Returns

- **type_classes** : *list of str*

---

### Examples

```python
component_type_classes({"components_type_class": "propulsor"})          # ['propulsor']
component_type_classes({"components_type_class": ["source", "tank"]})    # ['source', 'tank']
```

---

## icon_data_url

```python
icon_data_url(icon_path)
```

Return the base64 `data:` URL of an icon file. Results are cached, so each file is read and encoded once.

It is used for every icon the builder draws: toolbox buttons, placed and loaded nodes, and the legend.

---

### Parameters

- **icon_path** : *str or pathlib.Path*

  Path of the icon file (`ICONS_CONFIG[...]["icon_path"]` or a `COLOR_ICON_CONFIG` value).

---

### Returns

- **url** : *str*

  The `data:` URL, or the `file://` URL if the file cannot be read.

---

## _build_port_count_defaults

```python
_build_port_count_defaults()
```

Compute the default number of source and target ports of every known component type.

---

### Parameters

None.

---

### Returns

- **default_source_count** : *dict*

  `{components_type: int}`.

- **default_target_count** : *dict*

  `{components_type: int}`.

---

### Notes

The first matching rule applies:

| Rule | Sources | Targets |
|---|---|---|
| Type `gearbox` | 1 | 2 |
| Type `propeller` or `aux_load` | 1 | 0 |
| Type `fuel_tank`, `gaseous_hydrogen_tank` or `battery_pack` | 0 | 1 |
| `attributes` is a list: the first attribute containing `number_of_` | `VARIABLE_PORT_DEFAULT` (3) | 3 |
| ... or the first attribute containing `_mode` | 2 | 1 |
| Anything else | 1 | 1 |

The attributes are checked in their order in the registry, and the first one containing `number_of_` or `_mode` decides.

A count of `VARIABLE_PORT_DEFAULT` marks a variable-port component: its counts are editable in the panel and saved as `number_of_*` options.

---

## _read_constant_from_ast

```python
_read_constant_from_ast(constants_file_path, variable_name)
```

Read one assignment from a `constants.py` file by parsing its syntax tree, without importing it.

---

### Parameters

- **constants_file_path** : *pathlib.Path*

  Path of the `constants.py` file.

- **variable_name** : *str*

  Name to look for, e.g. `"POSSIBLE_POSITION"`.

---

### Returns

- **value** : *object or None*

  Value of the variable, or `None`.

- **found** : *bool*

  Whether the variable is assigned in the file.

---

### Notes

The file is read as text and parsed with `ast.parse`. `ast.walk` then visits every node of the tree, and the first `Assign` whose target is a `Name` equal to *variable_name* is evaluated with `ast.literal_eval`. Since the whole tree is walked, an assignment inside a function or class body also counts.

There are three outcomes:

| Result | Meaning |
|---|---|
| `(value, True)` | Found, and its value is a literal that `ast.literal_eval` can evaluate |
| `(None, True)` | Found, but the value is not a literal (e.g. contains an enum member) |
| `(None, False)` | Not found, or the file is missing or cannot be parsed |

---

## _read_constant_via_cached_import

```python
_read_constant_via_cached_import(constants_file_path, components_root_path, variable_name)
```

Read a variable from a `constants.py` module that is already imported. Fallback for values the syntax tree cannot evaluate.

Reading a module already in `sys.modules` runs no registration code, so it has no side effect. A module that is not imported yet is not imported.

---

### Parameters

- **constants_file_path** : *pathlib.Path*

  Path of the `constants.py` file.

- **components_root_path** : *pathlib.Path*

  Root of the components tree. The dotted module name is built relative to its third parent, e.g. `fastga_he.models.propulsion.components.<...>.constants`.

- **variable_name** : *str*

  Attribute to read.

---

### Returns

- **value** : *object or None*

  The attribute, or `None` if the module is not imported or has no such attribute.

---

## _performances_component_types

```python
_performances_component_types(init_file_path, openmdao_name_to_component_type)
```

Return the component types whose `Performances*` class an `__init__.py` imports. The file is parsed, not imported.

`from .x import PerformancesBatteryPack` refers to the OpenMDAO component `BatteryPack`. Other imports, and names that are not in `KNOWN_COMPONENTS`, are skipped.

---

### Parameters

- **init_file_path** : *pathlib.Path*

  Path of the `__init__.py`.

- **openmdao_name_to_component_type** : *dict*

  `{OM_components_name: components_type}`.

---

### Returns

- **component_types** : *list of str*

  In import order.

---

## _read_possible_constant

```python
_read_possible_constant(constants_file_path, components_root_path, variable_name)
```

Read one `POSSIBLE_*` constant of a component: syntax tree first, then the already-imported module if the value is not a literal.

---

### Parameters

- **constants_file_path** : *pathlib.Path*

  Path of the component's `constants.py`.

- **components_root_path** : *pathlib.Path*

  Root of the components tree.

- **variable_name** : *str*

  `"POSSIBLE_POSITION"` or `"POSSIBLE_OPTION"`.

---

### Returns

- **value** : *object or None*

  The value, or `None` if it is missing or cannot be read.

---

## _get_performance_component_names

```python
_get_performance_component_names(components_root_path)
```

Scan the components tree and return the possible positions and options of every component type.

---

### Parameters

- **components_root_path** : *str or pathlib.Path*

  Root of the components tree.

---

### Returns

- **possible_position** : *dict*

  `{components_type: [position, ...]}`.

- **possible_options** : *dict*

  `{components_type: {option_name: [values, ...]}}`.

---

### Notes

1. Build the map `OM_components_name -> components_type` from `KNOWN_COMPONENTS`.
2. For every `__init__.py` under the root (sorted), find the component types it imports a `Performances*` class for.
3. For each of them, read `POSSIBLE_POSITION` and `POSSIBLE_OPTION` from the `constants.py` in the same folder. A missing value becomes `[]` / `{}`.
4. Give `[]` / `{}` to every known component the scan did not find.