# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Component metadata helpers for the powertrain builder.

Three groups of functions are provided:

* **Icon mapping** – :func:`_map_possible_component_types_to_icons` maps each
  icon key to the component types that use it.

* **Port-count defaults** – :func:`_build_port_count_defaults` derives the
  default source and target port counts for every known component type from
  the rules encoded in :data:`KNOWN_COMPONENTS`.

* **Position and option scanner** – :func:`_get_performance_component_names`
  walks the components tree, extracts ``POSSIBLE_POSITION`` and
  ``POSSIBLE_OPTION`` from each component's ``constants.py`` via a pure AST
  parse, and falls back to a cached-module lookup when the value is
  non-literal (e.g. contains enum references). No fresh imports are
  triggered, so FAST-OAD submodel registration side-effects are avoided.
"""

import sys
import ast
from functools import lru_cache
from pathlib import Path

from fastga_he.gui.power_train_network_viewer import _url_to_base64
from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from .power_train_builder_state import VARIABLE_PORT_DEFAULT

# ============================================================================
# Icon mapping
# ============================================================================


def _map_possible_component_types_to_icons() -> dict:
    """
    Build a dictionary mapping each icon key to the list of component types that use it.

    Iterates over :data:`KNOWN_COMPONENTS` and inverts the
    ``icon_for_network_graph`` -> ``components_type`` relationship so callers
    can look up all component types associated with a given icon.

    :return: A dictionary of the form ``{icon_key: [component_type, …]}``.
    """
    icon_key_to_component_type_map = {}
    for component in KNOWN_COMPONENTS:
        component_type = component["components_type"]
        icon_key = component["icon_for_network_graph"]
        if icon_key not in icon_key_to_component_type_map:
            icon_key_to_component_type_map[icon_key] = [component_type]
        else:
            icon_key_to_component_type_map[icon_key].append(component_type)
    return icon_key_to_component_type_map


def component_type_classes(known_component: dict) -> list:
    """
    Type classes of a ``KNOWN_COMPONENTS`` entry, always as a list.

    ``components_type_class`` holds either one class name or a list of them.

    :param known_component: One entry of :data:`KNOWN_COMPONENTS`.
    :return: List of type class names.
    """
    type_class = known_component.get("components_type_class")
    return list(type_class) if isinstance(type_class, list) else [type_class]


@lru_cache(maxsize=None)
def icon_data_url(icon_path) -> str:
    """
    Base64 data URL of an icon file, read and encoded once per path.

    :param icon_path: Path of the icon file (``ICONS_CONFIG[...]["icon_path"]``
        or a ``COLOR_ICON_CONFIG`` value).
    :return: The ``data:`` URL, or the ``file://`` URL if the file cannot be read.
    """
    return _url_to_base64("file://" + str(Path(icon_path).resolve()))


# ============================================================================
# Port-count defaults
# ============================================================================


def _build_port_count_defaults() -> tuple[dict, dict]:
    """
    Derive default source and target port counts for every known component type.

    The first matching rule applies:

    * ``"gearbox"`` -> 1 source, 2 targets.
    * ``"propeller"`` or ``"aux_load"`` -> 1 source, 0 targets.
    * ``"fuel_tank"``, ``"gaseous_hydrogen_tank"``, or ``"battery_pack"``
      -> 0 sources, 1 target.
    * ``attributes`` is a list: its first attribute containing ``"number_of_"``
      or ``"_mode"`` decides. ``"number_of_"`` -> ``VARIABLE_PORT_DEFAULT`` on
      both sides (variable-port component); ``"_mode"`` -> 2 sources, 1 target.
    * Everything else -> 1 source, 1 target.

    :return: A two-element tuple ``(default_source_count, default_target_count)``
        where both dictionaries are keyed by ``components_type`` string.
    """
    default_source_count: dict = {}
    default_target_count: dict = {}

    for component in KNOWN_COMPONENTS:
        component_type = component["components_type"]
        component_attribute = component["attributes"]

        if component_type == "gearbox":
            default_source_count[component_type] = 1
            default_target_count[component_type] = 2
        elif component_type in ("propeller", "aux_load"):
            default_source_count[component_type] = 1
            default_target_count[component_type] = 0
        elif component_type in ("fuel_tank", "gaseous_hydrogen_tank", "battery_pack"):
            default_source_count[component_type] = 0
            default_target_count[component_type] = 1
        elif isinstance(component_attribute, list):
            default_source_count[component_type] = 1
            default_target_count[component_type] = 1
            for attribute_name in component_attribute:
                if "number_of_" in attribute_name:
                    default_source_count[component_type] = VARIABLE_PORT_DEFAULT
                    default_target_count[component_type] = VARIABLE_PORT_DEFAULT
                    break
                elif "_mode" in attribute_name:
                    default_source_count[component_type] = 2
                    default_target_count[component_type] = 1
                    break
        else:
            default_source_count[component_type] = 1
            default_target_count[component_type] = 1

    return default_source_count, default_target_count


# ============================================================================
# AST-based constants reader
# ============================================================================


def _read_constant_from_ast(constants_file_path: Path, variable_name: str):
    """
    Read one assignment from a ``constants.py`` file by parsing its syntax
    tree, **without importing the module**.

    Importing would run FAST-OAD submodel registration again, which raises
    *"Name … is already used"* errors. The first assignment to *variable_name*
    found anywhere in the file (``ast.walk``) is used.

    The function distinguishes three outcomes:

    * ``(value, True)``  — variable found **and** its value is a plain Python
      literal that :func:`ast.literal_eval` can evaluate safely.
    * ``(None, True)``   — variable found but its value contains non-literal
      nodes (e.g. enum references). The caller may attempt a
      cached-module fallback.
    * ``(None, False)``  — variable not found, or the file is missing or cannot
      be parsed.

    :param constants_file_path: Path to the ``constants.py`` file to parse.
    :param variable_name: Name to look for (e.g.
        ``"POSSIBLE_POSITION"`` or ``"POSSIBLE_OPTION"``).
    :return: A two-element tuple ``(value_or_none, found_flag)``.
    """
    if not constants_file_path.exists():
        return None, False

    try:
        # Extract the source code of the constants.py file
        source_code = constants_file_path.read_text(encoding="utf-8")
        # Convert the source code into an AST (Abstract Syntax Tree)
        syntax_tree = ast.parse(source_code)
    except (OSError, SyntaxError):
        return None, False
    # Walk through each variable assignment in the AST and return the value of the possible
    # choices for the given variable name if found
    for syntax_node in ast.walk(syntax_tree):
        if isinstance(syntax_node, ast.Assign) and any(
            isinstance(target_node, ast.Name) and target_node.id == variable_name
            for target_node in syntax_node.targets
        ):
            try:
                return ast.literal_eval(syntax_node.value), True
            except (ValueError, TypeError):
                # Non-literal value (e.g. contains an enum reference)
                return None, True

    return None, False


def _read_constant_via_cached_import(
    constants_file_path: Path, components_root_path: Path, variable_name: str
):
    """
    Read *variable_name* from a ``constants.py`` that is **already cached**
    in ``sys.modules``.

    This is a last-resort fallback used only when
    :func:`_read_constant_from_ast` reports that the variable exists but
    cannot be evaluated statically. Importing a module already present in
    ``sys.modules`` is side-effect-free — no registration code runs again.

    Returns ``None`` silently if the module is not already cached.

    :param constants_file_path: Path to the ``constants.py`` file.
    :param components_root_path: Root of the components tree; used to build the
        dotted module path.
    :param variable_name: Attribute name to retrieve.
    :return: The attribute value, or ``None``.
    """
    try:
        module_path_without_suffix = constants_file_path.with_suffix("")
        relative_module_path = module_path_without_suffix.relative_to(
            components_root_path.parent.parent.parent
        )
        dotted_module_name = ".".join(relative_module_path.parts)
    except ValueError:
        return None

    cached_module = sys.modules.get(dotted_module_name)
    if cached_module is None:
        return None

    return getattr(cached_module, variable_name, None)


# ============================================================================
# Position / option scanner
# ============================================================================


def _performances_component_types(init_file_path: Path, openmdao_name_to_component_type: dict):
    """
    Component types whose ``Performances*`` class an ``__init__.py`` imports.

    ``from .x import PerformancesBatteryPack`` names the OpenMDAO component
    ``BatteryPack``; other imports, and names unknown to ``KNOWN_COMPONENTS``,
    are skipped.

    :param init_file_path: Path of the ``__init__.py`` to parse (not imported).
    :param openmdao_name_to_component_type: ``OM_components_name -> components_type``.
    :return: List of component types, in import order.
    """
    syntax_tree = ast.parse(init_file_path.read_text())
    imported_names = [
        import_alias.name
        for syntax_node in ast.walk(syntax_tree)
        if isinstance(syntax_node, ast.ImportFrom)
        for import_alias in syntax_node.names
    ]
    component_types = [
        openmdao_name_to_component_type.get(imported_name.removeprefix("Performances"))
        for imported_name in imported_names
        if imported_name.startswith("Performances")
    ]
    return [component_type for component_type in component_types if component_type is not None]


def _read_possible_constant(
    constants_file_path: Path, components_root_path: Path, variable_name: str
):
    """
    Read one ``POSSIBLE_*`` constant of a component, without importing it.

    The AST read (:func:`_read_constant_from_ast`) is tried first; a value it
    finds but cannot evaluate (e.g. enum references) is taken from the module
    if it is already imported (:func:`_read_constant_via_cached_import`).

    :param constants_file_path: Path of the component's ``constants.py``.
    :param components_root_path: Root of the components tree.
    :param variable_name: ``"POSSIBLE_POSITION"`` or ``"POSSIBLE_OPTION"``.
    :return: The value, or ``None`` if it is missing or cannot be read.
    """
    value, found = _read_constant_from_ast(constants_file_path, variable_name)
    if value is None and found:
        value = _read_constant_via_cached_import(
            constants_file_path, components_root_path, variable_name
        )
    return value


def _get_performance_component_names(components_root_path: str | Path) -> tuple[dict, dict]:
    """
    Scan the components tree and return per-component-type possible positions
    and possible options, both read from each component's ``constants.py``.

    No module is imported (see :func:`_read_possible_constant`). Known
    components the scan does not find get ``[]`` / ``{}``.

    :param components_root_path: Root path of the components tree.
    :return: A two-element tuple ``(possible_position, possible_options)`` where:

        * ``possible_position`` maps ``component_type -> list[str]``
        * ``possible_options``  maps ``component_type -> {option_name: list[values]}``
    """
    components_root_path = Path(components_root_path)
    position_results: dict = {}
    option_results: dict = {}

    # Build a reverse map: OM_components_name -> components_type
    openmdao_name_to_component_type = {
        component["OM_components_name"]: component["components_type"]
        for component in KNOWN_COMPONENTS
    }
    # Walk the components tree and parse each __init__.py to find the
    # Performances* imports, then read the constants.py in the same folder
    for init_file_path in sorted(components_root_path.rglob("__init__.py")):
        constants_file_path = init_file_path.parent / "constants.py"
        for component_type in _performances_component_types(
            init_file_path, openmdao_name_to_component_type
        ):
            position_results[component_type] = (
                _read_possible_constant(
                    constants_file_path, components_root_path, "POSSIBLE_POSITION"
                )
                or []
            )
            option_results[component_type] = (
                _read_possible_constant(
                    constants_file_path, components_root_path, "POSSIBLE_OPTION"
                )
                or {}
            )

    # Fill in any known components not found during the scan
    for component in KNOWN_COMPONENTS:
        position_results.setdefault(component["components_type"], [])
        option_results.setdefault(component["components_type"], {})

    return position_results, option_results
