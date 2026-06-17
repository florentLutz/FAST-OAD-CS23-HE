# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Component-metadata helpers for the powertrain builder.

Three groups of functions are provided:

* **Icon mapping** – :func:`_map_possible_component_types_to_icons` builds the
  reverse lookup from ``components_type`` to its icon key.

* **Port-count defaults** – :func:`_build_port_count_defaults` derives the
  default source / target port counts for every known component type from the
  rules encoded in :data:`KNOWN_COMPONENTS`.

* **Position / option scanner** – :func:`_get_performance_component_names`
  walks the components tree, extracts ``POSSIBLE_POSITION`` and
  ``POSSIBLE_OPTION`` from each component's ``constants.py`` via a pure AST
  parse, and falls back to a cached-module lookup when the value is non-literal
  (e.g. contains enum references).  No fresh imports are triggered, so
  FAST-OAD submodel registration side-effects are avoided.
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
    Build a dict mapping each icon key to the list of component types that use it.

    :return: ``{icon_key: [component_type, …]}``
    """
    type_to_icon = {}
    for component in KNOWN_COMPONENTS:
        component_type = component["components_type"]
        icon_key = component["icon_for_network_graph"]
        if icon_key not in type_to_icon:
            type_to_icon[icon_key] = [component_type]
        else:
            type_to_icon[icon_key].append(component_type)
    return type_to_icon


# ============================================================================
# Port-count defaults
# ============================================================================


def _build_port_count_defaults() -> tuple[dict, dict]:
    """
    Derive default source/target port counts for every known component type.

    Iterates over :data:`KNOWN_COMPONENTS` and applies the following rules:

    * Any attribute containing ``"number_of_"`` → 3 sources, 3 targets
      (variable-count component).
    * Any attribute containing ``"_mode"`` → 2 sources, 1 target.
    * ``"gearbox"`` → 1 source, 2 targets.
    * ``"propeller"`` or ``"aux_load"`` → 1 source, 0 targets.
    * ``"fuel_tank"``, ``"gaseous_hydrogen_tank"``, or ``"battery_pack"``
      → 0 sources, 1 target.
    * Everything else → 1 source, 1 target.

    :return: ``(default_source_count, default_target_count)`` – both dicts
             keyed by ``components_type`` string.
    """
    default_source_count: dict = {}
    default_target_count: dict = {}

    for component in KNOWN_COMPONENTS:
        component_type = component["components_type"]
        attribute = component["attributes"]

        if component_type == "gearbox":
            default_source_count[component_type] = 1
            default_target_count[component_type] = 2
        elif component_type in ("propeller", "aux_load"):
            default_source_count[component_type] = 1
            default_target_count[component_type] = 0
        elif component_type in ("fuel_tank", "gaseous_hydrogen_tank", "battery_pack"):
            default_source_count[component_type] = 0
            default_target_count[component_type] = 1
        elif isinstance(attribute, list):
            default_source_count[component_type] = 1
            default_target_count[component_type] = 1
            for attr in attribute:
                if "number_of_" in attr:
                    default_source_count[component_type] = 3
                    default_target_count[component_type] = 3
                    break
                elif "_mode" in attr:
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


def _read_constant_from_ast(constants_path: Path, variable_name: str):
    """
    Extract a single module-level assignment from a ``constants.py`` file using
    the AST, **without importing the module**.

    This avoids triggering FAST-OAD submodel registration side-effects that
    would raise *"Name … is already used"* errors when the same submodel string
    is declared across multiple components.

    The function distinguishes three outcomes:

    * ``(value, True)``  — variable found **and** its value is a plain Python
      literal that could be evaluated safely by :func:`ast.literal_eval`.
    * ``(None, True)``   — variable found but its value contains non-literal
      nodes (e.g. enum references).  The caller may attempt a cached-module
      fallback.
    * ``(None, False)``  — variable not present in the file at all.

    :param constants_path: Path to the ``constants.py`` file to parse.
    :param variable_name: Module-level name to look for (e.g.
        ``"POSSIBLE_POSITION"`` or ``"POSSIBLE_OPTION"``).

    :return: ``(value_or_none, found_flag)``
    """
    if not constants_path.exists():
        return None, False

    try:
        source = constants_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None, False

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == variable_name for t in node.targets
        ):
            try:
                return ast.literal_eval(node.value), True
            except (ValueError, TypeError):
                # Non-literal value (e.g. contains an enum reference)
                return None, True

    return None, False


def _read_constant_via_cached_import(
    constants_path: Path, components_path: Path, variable_name: str
):
    """
    Read *variable_name* from a ``constants.py`` that is **already cached** in
    ``sys.modules``.

    This is a last-resort fallback used only when :func:`_read_constant_from_ast`
    reports that the variable exists but cannot be evaluated statically.
    Importing a module already in ``sys.modules`` is side-effect-free — no
    registration code runs again.

    Returns ``None`` silently if the module is not already cached.

    :param constants_path: Path to the ``constants.py`` file.
    :param components_path: Root of the components tree; used to build the
        dotted module path.
    :param variable_name: Attribute name to retrieve.

    :return: The attribute value, or ``None``.
    """
    try:
        relative = constants_path.with_suffix("").relative_to(components_path.parent.parent.parent)
        module_path = ".".join(relative.parts)
    except ValueError:
        return None

    module = sys.modules.get(module_path)
    if module is None:
        return None

    return getattr(module, variable_name, None)


# ============================================================================
# Position / option scanner
# ============================================================================


def _get_performance_component_names(
    components_path: str | Path,
    base_package: str = "fastga_he.models.propulsion.components",
) -> tuple[dict, dict]:
    """
    Scan the components tree and return per-component-type possible positions
    **and** possible options, both read from each component's ``constants.py``.

    Both values are extracted via a pure AST parse so that no module is
    imported and no FAST-OAD submodel registration side-effects are triggered.
    The cached-module fallback is invoked only when the variable exists in the
    file but its value contains non-literal nodes, **and** the module is
    already present in ``sys.modules`` — so it is always side-effect-free.

    :param components_path: Root path of the components tree.
    :param base_package: Dotted package name used as the import root for the
        cached-module fallback.

    :return: ``(possible_position, possible_options)`` where

        * ``possible_position`` maps ``component_type -> list[str]``
        * ``possible_options``  maps ``component_type -> {option_name: list[values]}``
    """
    components_path = Path(components_path)
    position_results = {}
    option_results = {}

    # Build a reverse map: OM_components_name -> components_type
    om_name_to_type = {
        component["OM_components_name"]: component["components_type"]
        for component in KNOWN_COMPONENTS
    }

    for init_file in sorted(components_path.rglob("__init__.py")):
        source = init_file.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if not alias.name.startswith("Performances"):
                        continue

                    stripped_name = alias.name.removeprefix("Performances")
                    component_type = om_name_to_type.get(stripped_name)
                    if component_type is None:
                        continue

                    constants_file = init_file.parent / "constants.py"

                    # ── POSSIBLE_POSITION ──────────────────────────────────────
                    pos_value, pos_found = _read_constant_from_ast(
                        constants_file, "POSSIBLE_POSITION"
                    )
                    if pos_value is None and pos_found:
                        pos_value = _read_constant_via_cached_import(
                            constants_file, components_path, "POSSIBLE_POSITION"
                        )
                    position_results[component_type] = pos_value or []

                    # ── POSSIBLE_OPTION ────────────────────────────────────────
                    opt_value, opt_found = _read_constant_from_ast(
                        constants_file, "POSSIBLE_OPTION"
                    )
                    if opt_value is None and opt_found:
                        opt_value = _read_constant_via_cached_import(
                            constants_file, components_path, "POSSIBLE_OPTION"
                        )
                    option_results[component_type] = opt_value or {}

    # Fill in any known components not found during the scan
    for component in KNOWN_COMPONENTS:
        position_results.setdefault(component["components_type"], [])
        option_results.setdefault(component["components_type"], {})

    return position_results, option_results
