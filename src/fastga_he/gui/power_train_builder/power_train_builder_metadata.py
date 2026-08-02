# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Component metadata helpers for the powertrain builder.

Three groups of functions are provided:

* **Icon mapping** – :func:`_map_possible_component_types_to_icons` builds a
  reverse lookup from ``components_type`` to its icon key.

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
from pathlib import Path

from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS


# ============================================================================
# Icon mapping
# ============================================================================


def _map_possible_component_types_to_icons() -> dict:
    """
    Build a dictionary mapping each icon key to the list of component types that use it.

    Iterates over :data:`KNOWN_COMPONENTS` and inverts the
    ``icon_for_network_graph`` → ``components_type`` relationship so callers
    can look up all component types associated with a given icon.

    :return: A dictionary of the form ``{icon_key: [component_type, …]}``.
    """
    component_type_to_icon_key_map = {}
    for component in KNOWN_COMPONENTS:
        component_type = component["components_type"]
        icon_key = component["icon_for_network_graph"]
        if icon_key not in component_type_to_icon_key_map:
            component_type_to_icon_key_map[icon_key] = [component_type]
        else:
            component_type_to_icon_key_map[icon_key].append(component_type)
    return component_type_to_icon_key_map


# ============================================================================
# Port-count defaults
# ============================================================================


def _build_port_count_defaults() -> tuple[dict, dict]:
    """
    Derive default source and target port counts for every known component type.

    Iterates over :data:`KNOWN_COMPONENTS` and applies the following rules in
    priority order:

    * Any attribute containing ``"number_of_"`` → 3 sources, 3 targets
      (variable-count component).
    * Any attribute containing ``"_mode"`` → 2 sources, 1 target.
    * ``"gearbox"`` → 1 source, 2 targets.
    * ``"propeller"`` or ``"aux_load"`` → 1 source, 0 targets.
    * ``"fuel_tank"``, ``"gaseous_hydrogen_tank"``, or ``"battery_pack"``
      → 0 sources, 1 target.
    * Everything else → 1 source, 1 target.

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
                    default_source_count[component_type] = 3
                    default_target_count[component_type] = 3
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
    Extract a single module-level assignment from a ``constants.py`` file
    using the AST, **without importing the module**.

    This avoids triggering FAST-OAD submodel registration side-effects that
    would raise *"Name … is already used"* errors when the same submodel
    string is declared across multiple components.

    The function distinguishes three outcomes:

    * ``(value, True)``  — variable found **and** its value is a plain Python
      literal that :func:`ast.literal_eval` can evaluate safely.
    * ``(None, True)``   — variable found but its value contains non-literal
      nodes (e.g. enum references). The caller may attempt a
      cached-module fallback.
    * ``(None, False)``  — variable not present in the file at all.

    :param constants_file_path: Path to the ``constants.py`` file to parse.
    :param variable_name: Module-level name to look for (e.g.
        ``"POSSIBLE_POSITION"`` or ``"POSSIBLE_OPTION"``).
    :return: A two-element tuple ``(value_or_none, found_flag)``.
    """
    if not constants_file_path.exists():
        return None, False

    try:
        source_code = constants_file_path.read_text(encoding="utf-8")
        syntax_tree = ast.parse(source_code)
    except (OSError, SyntaxError):
        return None, False

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


def _get_performance_component_names(
    components_root_path: str | Path,
    base_package: str = "fastga_he.models.propulsion.components",
) -> tuple[dict, dict]:
    """
    Scan the components tree and return per-component-type possible positions
    and possible options, both read from each component's ``constants.py``.

    Both values are extracted via a pure AST parse so that no module is
    imported and no FAST-OAD submodel registration side-effects are triggered.
    The cached-module fallback is invoked only when the variable exists in the
    file but its value contains non-literal nodes **and** the module is
    already present in ``sys.modules`` — so it is always side-effect-free.

    :param components_root_path: Root path of the components tree.
    :param base_package: Dotted package name used as the import root for the
        cached-module fallback.
    :return: A two-element tuple ``(possible_position, possible_options)`` where:

        * ``possible_position`` maps ``component_type → list[str]``
        * ``possible_options``  maps ``component_type → {option_name: list[values]}``
    """
    components_root_path = Path(components_root_path)
    position_results: dict = {}
    option_results: dict = {}

    # Build a reverse map: OM_components_name -> components_type
    open_mdao_name_to_component_type = {
        component["OM_components_name"]: component["components_type"]
        for component in KNOWN_COMPONENTS
    }
    # Walk the components tree and parse each __init__.py to find the
    # Performances* imports, then read the constants.py in the same folder.
    for init_file_path in sorted(components_root_path.rglob("__init__.py")):
        source_code = init_file_path.read_text()
        syntax_tree = ast.parse(source_code)

        for syntax_node in ast.walk(syntax_tree):
            if isinstance(syntax_node, ast.ImportFrom):
                for import_alias in syntax_node.names:
                    if not import_alias.name.startswith("Performances"):
                        continue

                    stripped_class_name = import_alias.name.removeprefix("Performances")
                    component_type = open_mdao_name_to_component_type.get(stripped_class_name)
                    if component_type is None:
                        continue

                    constants_file_path = init_file_path.parent / "constants.py"

                    # ── POSSIBLE_POSITION ──────────────────────────────────────
                    position_value, position_found = _read_constant_from_ast(
                        constants_file_path, "POSSIBLE_POSITION"
                    )
                    if position_value is None and position_found:
                        position_value = _read_constant_via_cached_import(
                            constants_file_path, components_root_path, "POSSIBLE_POSITION"
                        )
                    position_results[component_type] = position_value or []

                    # ── POSSIBLE_OPTION ────────────────────────────────────────
                    option_value, option_found = _read_constant_from_ast(
                        constants_file_path, "POSSIBLE_OPTION"
                    )
                    if option_value is None and option_found:
                        option_value = _read_constant_via_cached_import(
                            constants_file_path, components_root_path, "POSSIBLE_OPTION"
                        )
                    option_results[component_type] = option_value or {}

    # Fill in any known components not found during the scan
    for component in KNOWN_COMPONENTS:
        position_results.setdefault(component["components_type"], [])
        option_results.setdefault(component["components_type"], {})

    return position_results, option_results
