# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Sunburst breakdowns of the aircraft Life Cycle Cost (LCC), mirroring the style of the LCA
sunburst in ``lcc_cost.py`` but built on the cost tree instead of a colon-encoded LCA variable
naming convention (cost variables are flat, so the hierarchy is declared explicitly below).

Two independent sunbursts are provided:

- :func:`lcc_production_cost_sun_breakdown`: splits the production cost per aircraft into a
  **recursive** cost branch (cost that recurs at every unit built: manufacturing, quality
  control, material, avionics, landing gear design, and the purchase cost of every power train
  component) and a **non-recursive** cost branch (one-time development-type cost, amortized
  over the units produced: engineering, tooling, flight test, development support,
  certification).
- :func:`lcc_operation_cost_sun_breakdown`: splits the annual operating cost per aircraft into a
  **fixed annual** cost branch (depreciation, insurance, loan, maintenance, miscellaneous, fuel,
  electricity, and any other yearly cost that doesn't scale with individual flights) and a
  **mission-based** cost branch (crew and airport costs, which scale with the number of flights
  performed).
"""

import pathlib
from typing import Callable, Dict, List, Optional, Union

import numpy as np

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import fastoad.api as oad

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS

PRODUCTION_PREFIX = "data:cost:production:"
OPERATION_PREFIX = "data:cost:operation:"


# ---------------------------------------------------------------------------------------------
# Small generic helpers, shared with the philosophy of lcc_cost.py's LCA sunburst
# ---------------------------------------------------------------------------------------------


def _round_value(value: float) -> float:
    if value == 0.0:
        return value
    else:
        # Same trick as in the LCA sunburst: a value rounded to zero simply won't display.
        return round(value, int(np.ceil(abs(np.log10(abs(value))))) + 5)


def _get_value(datafile: oad.DataFile, name: str, default: float = 0.0) -> float:
    """Returns the value of a variable in the datafile, or a default if it isn't present."""

    try:
        return float(datafile[name].value[0])
    except KeyError:
        return default


def _label(display_name: str, value: float, unit_suffix: str = "USD") -> str:
    return display_name + "<br> " + str(_round_value(value)) + " " + unit_suffix


_POWER_TRAIN_PREFIX = "data:propulsion:he_power_train:"


def _discover_power_train_variables(datafile: oad.DataFile, suffix: str) -> Dict[str, str]:
    """
    Auto-discovers every ``data:propulsion:he_power_train:<type>:<name><suffix>`` variable
    available in the datafile (this mirrors the dynamic ``cost_components_type`` /
    ``cost_components_name`` options of :class:`LCCRecursiveCost` and
    :class:`LCCSumOperationalCost`, without requiring the caller to know the architecture in
    advance). Used with ``suffix=":purchase_cost"`` for the production cost sunburst and
    ``suffix=":operational_cost"`` for the operation cost sunburst.

    :return: a dict mapping the variable name to a human-readable label built from the
    component type and name, e.g. "SM_PMSM<br>motor_1".
    """

    discovered_vars = {}
    for name in datafile.names():
        if name.startswith(_POWER_TRAIN_PREFIX) and name.endswith(suffix):
            middle = name[len(_POWER_TRAIN_PREFIX) : -len(suffix)]
            component_type, _, component_name = middle.partition(":")
            discovered_vars[name] = component_type + "<br>" + component_name

    return discovered_vars


def _get_color(category_key: str, color_dict: dict) -> str:
    """Every leaf belonging to the same top-level category shares the same color."""

    if category_key in color_dict:
        return color_dict[category_key]

    color = COLS[len(color_dict) % len(COLS)]
    color_dict[category_key] = color
    return color


# ---------------------------------------------------------------------------------------------
# Generic two-level (root -> category -> leaf) sunburst builder
# ---------------------------------------------------------------------------------------------


def _build_cost_sunburst(
    datafile: oad.DataFile,
    root_label: str,
    categories: List[Dict],
    rel: str = "absolute",
) -> go.Sunburst:
    """
    Builds a cost sunburst with one root, N categories, and their leaves.

    :param datafile: the FAST-OAD output datafile.
    :param root_label: display name of the root node (e.g. "Production cost per unit").
    :param categories: a list of dicts, each with the keys:
        - "key": short unique id for the category (used for coloring)
        - "label": display name of the category
        - "leaves": dict mapping {variable_name: display_label}
        - "divide_by" (optional): variable name whose value every leaf in this category is
          divided by (used to bring non-recursive project totals back to a per-unit basis)
    :param rel: "absolute" for raw USD values, "total" for percentage of the root total,
    "parent" for percentage of the immediate parent category.
    """

    figure_labels = []
    figure_parents = []
    figure_values = []
    figure_color = [None]
    color_dict = {}

    category_totals = {}
    category_leaf_values = {}

    # First pass: compute every leaf's (positive) value and each category's subtotal.
    for category in categories:
        divisor = 1.0
        if category.get("divide_by"):
            divisor = _get_value(datafile, category["divide_by"], default=1.0) or 1.0

        leaf_values = {}
        for var_name, disp_label in category["leaves"].items():
            value = _get_value(datafile, var_name) / divisor
            # Cost reductions (negative leaves) and zero/absent costs are not representable in
            # a "branchvalues=total" sunburst; they are simply left out of the visual breakdown.
            if value > 0.0:
                leaf_values[disp_label] = value

        category_leaf_values[category["key"]] = leaf_values
        category_totals[category["key"]] = sum(leaf_values.values())

    root_value = sum(category_totals.values())

    def _scaled(value: float, denom: float) -> float:
        if rel in ("total", "parent") and denom:
            return value / denom * 100.0
        return value

    # Root node
    figure_labels.append(root_label)
    figure_parents.append("")
    figure_values.append(100.0 if rel in ("total", "parent") else root_value)

    # Category + leaf nodes
    for category in categories:
        cat_total = category_totals[category["key"]]
        if cat_total <= 0.0:
            continue

        cat_denom = root_value if rel == "total" else root_value
        figure_labels.append(_label(category["label"], cat_total))
        figure_parents.append(root_label)
        figure_values.append(_scaled(cat_total, cat_denom))
        figure_color.append(_get_color(category["key"], color_dict))

        parent_denom = cat_total if rel == "parent" else root_value
        for disp_label, value in category_leaf_values[category["key"]].items():
            figure_labels.append(_label(disp_label, value))
            figure_parents.append(_label(category["label"], cat_total))
            figure_values.append(_scaled(value, parent_denom))
            figure_color.append(_get_color(category["key"], color_dict))

    return go.Sunburst(
        labels=figure_labels,
        parents=figure_parents,
        values=figure_values,
        branchvalues="total",
        sort=False,
        marker={"colors": figure_color},
    )


# ---------------------------------------------------------------------------------------------
# Wrappers, mirroring lca_impacts_sun_breakdown's single/multi-aircraft handling
# ---------------------------------------------------------------------------------------------


def _make_categories(
    datafile: oad.DataFile,
    spec: List[Dict],
) -> List[Dict]:
    """Resolves any dynamic leaf discovery (e.g. power train purchase costs) in the spec."""

    resolved = []
    for category in spec:
        leaves = dict(category["leaves"])
        if category.get("discover_suffix"):
            leaves.update(_discover_power_train_variables(datafile, category["discover_suffix"]))

        resolved.append(
            {
                "key": category["key"],
                "label": category["label"],
                "leaves": leaves,
                "divide_by": category.get("divide_by"),
            }
        )

    return resolved


def _sun_breakdown(
    aircraft_file_path: Union[Union[str, pathlib.Path], List[Union[str, pathlib.Path]]],
    root_label: str,
    spec: List[Dict],
    title_text: str,
    full_burst: bool = False,
    name_aircraft: Union[str, List[str]] = None,
    rel: str = "absolute",
) -> go.FigureWidget:
    if rel == "total":
        title_text += "<br>expressed as a percentage of the total cost"
    elif rel == "parent":
        title_text += "<br>expressed as a percentage of parent category"

    def _one_sunburst(path):
        datafile = oad.DataFile(path)
        categories = _make_categories(datafile, spec)
        return _build_cost_sunburst(datafile, root_label, categories, rel=rel)

    if isinstance(aircraft_file_path, (str, pathlib.Path)):
        fig = go.Figure()
        fig.add_trace(_one_sunburst(aircraft_file_path))

        if name_aircraft:
            fig.update_layout(title_text=name_aircraft + " " + title_text, title_x=0.5)
        else:
            fig.update_layout(title_text=title_text, title_x=0.5)

    elif len(aircraft_file_path) == 1:
        fig = go.Figure()
        fig.add_trace(_one_sunburst(aircraft_file_path[0]))

        if name_aircraft and name_aircraft[0]:
            fig.update_layout(title_text=name_aircraft[0] + " " + title_text, title_x=0.5)
        else:
            fig.update_layout(title_text=title_text, title_x=0.5)

    else:
        fig = make_subplots(
            1,
            cols=len(aircraft_file_path),
            specs=[[{"type": "domain"}] * len(aircraft_file_path)],
            subplot_titles=name_aircraft,
        )

        for idx, curr_aircraft_file_path in enumerate(aircraft_file_path):
            fig.add_trace(_one_sunburst(curr_aircraft_file_path), 1, idx + 1)
            if name_aircraft:
                fig.update_traces(row=1, col=idx + 1, name=name_aircraft[idx])

        fig.update_layout(title_text=title_text, title_x=0.5)

    if full_burst:
        fig.update_traces(selector=dict(type="sunburst"))
    else:
        # Root + category only, hiding individual leaf items.
        fig.update_traces(maxdepth=2, selector=dict(type="sunburst"))

    return go.FigureWidget(fig)


# ---------------------------------------------------------------------------------------------
# Production cost sunburst: Recursive vs Non-recursive
# ---------------------------------------------------------------------------------------------

_PRODUCTION_SPEC = [
    {
        "key": "recursive",
        "label": "Recursive cost",
        "leaves": {
            PRODUCTION_PREFIX + "manufacturing_cost_per_unit": "Manufacturing",
            PRODUCTION_PREFIX + "quality_control_cost_per_unit": "Quality control",
            PRODUCTION_PREFIX + "material_cost_per_unit": "Material",
            PRODUCTION_PREFIX + "avionics_cost_per_unit": "Avionics",
            # landing_gear_cost_reduction is a *reduction* (negative) and is filtered out of
            # the breakdown automatically since only positive leaves are displayed.
            PRODUCTION_PREFIX + "landing_gear_cost_reduction": "Landing gear reduction",
        },
        # Every data:propulsion:he_power_train:<type>:<name>:purchase_cost variable found in
        # the datafile is added here automatically (battery, motor, inverter, turboshaft, ...).
        "discover_suffix": ":purchase_cost",
    },
    {
        "key": "non_recursive",
        "label": "Non-recursive cost",
        "leaves": {
            PRODUCTION_PREFIX + "engineering_cost_per_unit": "Engineering",
            PRODUCTION_PREFIX + "tooling_cost_per_unit": "Tooling",
            PRODUCTION_PREFIX + "flight_test_cost_per_unit": "Flight test",
            PRODUCTION_PREFIX + "dev_support_cost_per_unit": "Development support",
            PRODUCTION_PREFIX + "certification_cost_per_unit": "Certification",
        },
        # These variables, once summed and multiplied by number_aircraft_5_years, give the
        # *total* non-recursive project cost. Dividing back by number_aircraft_5_years brings
        # them onto the same per-unit basis as the recursive cost branch.
        "divide_by": None,
    },
]


def lcc_production_cost_sun_breakdown(
    aircraft_file_path: Union[Union[str, pathlib.Path], List[Union[str, pathlib.Path]]],
    full_burst: bool = False,
    name_aircraft: Union[str, List[str]] = None,
    rel: str = "absolute",
) -> go.FigureWidget:
    """
    Give a breakdown of the aircraft production cost per unit under the form of a sunburst,
    split into a recursive cost branch (manufacturing, quality control, material, avionics,
    and the purchase cost of every power train component) and a non-recursive cost branch
    (engineering, tooling, flight test, development support, certification, amortized per
    unit over the planned production run).

    :param aircraft_file_path: path (or list of paths) to the FAST-OAD output file(s)
    containing the cost results.
    :param full_burst: if True, show the individual cost items; otherwise only show the
    recursive / non-recursive split.
    :param name_aircraft: name (or list of names) of the aircraft, used as chart title(s).
    :param rel: "absolute" for raw USD values, "total" for percentage of the production cost
    per unit, "parent" for percentage of the recursive/non-recursive subtotal.
    """

    return _sun_breakdown(
        aircraft_file_path=aircraft_file_path,
        root_label="Production cost per unit",
        spec=_PRODUCTION_SPEC,
        title_text="Production cost breakdown",
        full_burst=full_burst,
        name_aircraft=name_aircraft,
        rel=rel,
    )


# ---------------------------------------------------------------------------------------------
# Operation cost sunburst: Fixed annual vs Mission-based
# ---------------------------------------------------------------------------------------------

_OPERATION_SPEC = [
    {
        "key": "fixed",
        "label": "Fixed annual cost",
        "leaves": {
            # Note: annual_depreciation_cost is intentionally NOT included here. It is not
            # part of LCCSumOperationalCost's inputs, i.e. it does not feed into
            # data:cost:operation:annual_cost_per_unit - it is used elsewhere (e.g. cash flow /
            # NPV), so including it here would make this sunburst's root not reconcile with
            # the actual annual_cost_per_unit output.
            OPERATION_PREFIX + "annual_insurance_cost": "Insurance",
            OPERATION_PREFIX + "annual_loan_cost": "Loan",
            OPERATION_PREFIX + "additional_cost": "Additional",
            OPERATION_PREFIX + "maintenance_cost": "Maintenance",
            OPERATION_PREFIX + "miscellaneous_cost": "Miscellaneous",
        },
        # Every data:propulsion:he_power_train:<type>:<name>:operational_cost variable found in
        # the datafile is added here automatically (e.g. turboshaft or PEMFC stack upkeep).
        # These are yearly per-component costs, not tied to the number of flights, so they
        # belong in the fixed branch rather than the mission-based one.
        "discover_suffix": ":operational_cost",
    },
    {
        "key": "mission",
        "label": "Mission-based cost",
        "leaves": {
            OPERATION_PREFIX + "annual_fuel_cost": "Fuel",
            OPERATION_PREFIX + "annual_electricity_cost": "Electricity",
            OPERATION_PREFIX + "annual_crew_cost": "Crew",
            OPERATION_PREFIX + "annual_airport_cost": "Airport",
        },
    },
]


def lcc_operation_cost_sun_breakdown(
    aircraft_file_path: Union[Union[str, pathlib.Path], List[Union[str, pathlib.Path]]],
    full_burst: bool = False,
    name_aircraft: Union[str, List[str]] = None,
    rel: str = "absolute",
) -> go.FigureWidget:
    """
    Give a breakdown of the aircraft annual operating cost under the form of a sunburst, split
    into a fixed annual cost branch (depreciation, insurance, loan, maintenance, miscellaneous,
    fuel, electricity, and any other yearly cost that doesn't scale with the number of flights)
    and a mission-based cost branch (crew and airport cost, which scale with the number of
    flights performed per year).

    :param aircraft_file_path: path (or list of paths) to the FAST-OAD output file(s)
    containing the cost results.
    :param full_burst: if True, show the individual cost items; otherwise only show the
    fixed / mission-based split.
    :param name_aircraft: name (or list of names) of the aircraft, used as chart title(s).
    :param rel: "absolute" for raw USD/year values, "total" for percentage of the annual
    operating cost, "parent" for percentage of the fixed/mission-based subtotal.
    """

    return _sun_breakdown(
        aircraft_file_path=aircraft_file_path,
        root_label="Annual operating cost per unit",
        spec=_OPERATION_SPEC,
        title_text="Operating cost breakdown",
        full_burst=full_burst,
        name_aircraft=name_aircraft,
        rel=rel,
    )
