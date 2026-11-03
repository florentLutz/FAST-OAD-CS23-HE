# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Comparison bar charts of the aircraft Life Cycle Cost (LCC), mirroring the style of
``lca_impact.lca_impacts_bar_chart_simple``.

- :func:`lcc_production_cost_bar_chart_simple`: compares every item of the production cost per
  aircraft (manufacturing, quality control, material, avionics, engineering, tooling, flight
  test, development support, certification, powertrain, MSP, and the purchase cost of every
  power train component).
- :func:`lcc_operation_cost_bar_chart_simple`: compares every item of the annual operating cost
  per aircraft (insurance, loan, maintenance, fuel, electricity, crew, airport, ..., and the
  operational cost of every power train component).

Cost variables are flat, so the items shown in each chart are declared explicitly in
``_PRODUCTION_SPEC`` and ``_OPERATION_SPEC``.
"""

import pathlib
from typing import Dict, List, Tuple, Union

import plotly.graph_objects as go

import fastoad.api as oad

PRODUCTION_PREFIX = "data:cost:production:"
OPERATION_PREFIX = "data:cost:operation:"
_POWER_TRAIN_PREFIX = "data:propulsion:he_power_train:"


# ---------------------------------------------------------------------------------------------
# Cost items: specs and data extraction
# ---------------------------------------------------------------------------------------------

# Each spec describes the items plotted in one bar chart:
# - "leaves": dict mapping {variable_name: display_label}
# - "discover_suffix": every data:propulsion:he_power_train:<type>:<name><suffix> variable found
#   in the datafile is added automatically (this mirrors the dynamic ``cost_components_type`` /
#   ``cost_components_name`` options of LCCRecursiveCost and LCCSumOperationalCost, without
#   requiring the caller to know the architecture in advance).

_PRODUCTION_SPEC = {
    "leaves": {
        # Recursing costs (every unit built)
        PRODUCTION_PREFIX + "manufacturing_cost_per_unit": "Manufacturing",
        PRODUCTION_PREFIX + "quality_control_cost_per_unit": "Quality control",
        PRODUCTION_PREFIX + "material_cost_per_unit": "Material",
        PRODUCTION_PREFIX + "avionics_cost_per_unit": "Avionics",
        # A *reduction* (negative), left out automatically since only positive items are shown.
        PRODUCTION_PREFIX + "landing_gear_cost_reduction": "Landing gear reduction",
        # Adjust the variable names below if they differ from your actual output variables.
        PRODUCTION_PREFIX + "powertrain_cost_per_unit": "Powertrain",
        "data:cost:msp_per_unit": "MSP",
        # Non-recursing costs (one-time, amortized per unit over the production run)
        PRODUCTION_PREFIX + "engineering_cost_per_unit": "Engineering",
        PRODUCTION_PREFIX + "tooling_cost_per_unit": "Tooling",
        PRODUCTION_PREFIX + "flight_test_cost_per_unit": "Flight test",
        PRODUCTION_PREFIX + "dev_support_cost_per_unit": "Development support",
        PRODUCTION_PREFIX + "certification_cost_per_unit": "Certification",
    },
    "discover_suffix": ":purchase_cost",
}

_OPERATION_SPEC = {
    "leaves": {
        # Note: annual_depreciation_cost is intentionally NOT included here. It does not feed
        # into data:cost:operation:annual_cost_per_unit (it is used elsewhere, e.g. cash flow /
        # NPV), so including it would make the total not reconcile with that output.
        OPERATION_PREFIX + "annual_insurance_cost": "Insurance",
        OPERATION_PREFIX + "annual_loan_cost": "Loan",
        OPERATION_PREFIX + "additional_cost": "Additional",
        OPERATION_PREFIX + "maintenance_cost": "Maintenance",
        OPERATION_PREFIX + "miscellaneous_cost": "Miscellaneous",
        OPERATION_PREFIX + "annual_overhaul_deposit": "Overhaul Deposit",
        OPERATION_PREFIX + "annual_fuel_cost": "Fuel",
        OPERATION_PREFIX + "annual_electricity_cost": "Electricity",
        OPERATION_PREFIX + "annual_crew_cost": "Crew",
        OPERATION_PREFIX + "annual_airport_cost": "Airport",
    },
    "discover_suffix": ":operational_cost",
}


def _get_value(datafile: oad.DataFile, name: str, default: float = 0.0) -> float:
    """Returns the value of a variable in the datafile, or a default if it isn't present."""

    try:
        return float(datafile[name].value[0])
    except KeyError:
        return default


def _discover_power_train_variables(datafile: oad.DataFile, suffix: str) -> Dict[str, str]:
    """
    Auto-discovers every ``data:propulsion:he_power_train:<type>:<name><suffix>`` variable
    available in the datafile.

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


def _leaf_aliases(var_name: str, disp_label: str) -> List[str]:
    """
    Every string that should be accepted as a way to refer to a given item in
    ``item_filter_list``: its display label (e.g. "Fuel"), its full variable name, and, for the
    named (non-discovered) items, the tail of the variable name after the ``data:cost:...:``
    prefix (e.g. "annual_fuel_cost") since that's how the items are declared in
    ``_PRODUCTION_SPEC``/``_OPERATION_SPEC`` and how callers are likely to refer to them.
    """

    aliases = [disp_label, var_name]
    for prefix in (PRODUCTION_PREFIX, OPERATION_PREFIX, _POWER_TRAIN_PREFIX):
        if var_name.startswith(prefix):
            aliases.append(var_name[len(prefix) :])

    return aliases


def _get_cost_dict(
    aircraft_file_path: Union[str, pathlib.Path], spec: Dict
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Returns a flat dict mapping every cost item's display label (e.g. "Manufacturing",
    "Engineering", or an auto-discovered power train component) to its value, for a single
    aircraft. Mirrors ``lca_impact._get_impact_dict`` so :func:`_cost_bar_chart_simple` can
    follow the same logic as ``lca_impacts_bar_chart_simple``.

    Cost reductions (negative values) and zero/absent costs are left out.

    :param aircraft_file_path: path to the FAST-OAD output file containing the cost results.
    :param spec: either ``_PRODUCTION_SPEC`` or ``_OPERATION_SPEC``.

    :return: a tuple ``(cost_dict, leaf_aliases)`` where ``leaf_aliases`` maps every accepted
    alias (see :func:`_leaf_aliases`) of an item to its display label.
    """

    datafile = oad.DataFile(aircraft_file_path)

    leaves = dict(spec["leaves"])
    if spec.get("discover_suffix"):
        leaves.update(_discover_power_train_variables(datafile, spec["discover_suffix"]))

    cost_dict = {}
    leaf_aliases = {}
    for var_name, disp_label in leaves.items():
        value = _get_value(datafile, var_name)
        if value > 0.0:
            cost_dict[disp_label] = value
            for alias in _leaf_aliases(var_name, disp_label):
                leaf_aliases[alias] = disp_label

    return cost_dict, leaf_aliases


# ---------------------------------------------------------------------------------------------
# Generic comparison bar chart, mirroring lca_impact.lca_impacts_bar_chart_simple
# ---------------------------------------------------------------------------------------------


def _cost_bar_chart_simple(
    aircraft_file_paths: List[Union[str, pathlib.Path]],
    names_aircraft: List[str],
    spec: Dict,
    default_comparison_label: str,
    graph_title: str = None,
    item_filter_list: list = None,
    color_list: List[str] = None,
) -> go.FigureWidget:
    """
    Give a bar chart that compares multiple aircraft designs across all the cost items of
    ``spec``, plotted as raw USD values (all items already share the same unit, so no
    normalisation is needed). Can be used with only one design.

    :param aircraft_file_paths: paths to the output files that contain the cost results.
    :param names_aircraft: names of the aircraft.
    :param spec: either ``_PRODUCTION_SPEC`` or ``_OPERATION_SPEC``.
    :param default_comparison_label: used to build the default graph title when none is given,
    e.g. "production cost per unit".
    :param graph_title: title of the graph, if None are specified one is created based on the
    aircraft names.
    :param item_filter_list: filter to only show cost items in the list in output graph. Items
    can be given either as their display label (e.g. "Fuel") or as the raw variable name used
    in ``_PRODUCTION_SPEC``/``_OPERATION_SPEC`` (e.g. "annual_fuel_cost", with or without the
    ``data:cost:...:`` prefix). By default, everything is plotted.
    :param color_list: colors to use for each aircraft's bars (e.g. "#1f77b4", "rgb(31,119,180)",
    or a CSS color name), given in the same order as ``aircraft_file_paths``. By default,
    Plotly's automatic color cycling is used.
    """

    fig = go.Figure()

    for i, (aircraft_file_path, name_aircraft) in enumerate(
        zip(aircraft_file_paths, names_aircraft)
    ):
        cost_dict, leaf_aliases = _get_cost_dict(aircraft_file_path, spec)

        item_names = list(cost_dict.keys())
        item_values = list(cost_dict.values())

        if item_filter_list:
            filtered_item_names = []
            filtered_item_values = []
            for tl_item in item_filter_list:
                # Resolve the filter entry to a display label, whether it was already given as
                # one or provided as a raw variable name/suffix (see _leaf_aliases).
                resolved_label = leaf_aliases.get(tl_item, tl_item)
                if resolved_label in item_names:
                    filtered_item_names.append(resolved_label)
                    filtered_item_values.append(item_values[item_names.index(resolved_label)])
        else:
            filtered_item_names = item_names
            filtered_item_values = item_values

        marker_color = color_list[i] if color_list and i < len(color_list) else None

        bar_chart = go.Bar(
            name=name_aircraft,
            x=filtered_item_names,
            y=filtered_item_values,
            marker_color=marker_color,
        )
        fig.add_trace(bar_chart)

    if graph_title:
        title = graph_title
    else:
        title = "Comparison of " + default_comparison_label + " for " + ", ".join(names_aircraft)

    fig.update_layout(
        barmode="group",
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        title_x=0.5,
        title_text=title,
    )
    fig.update_xaxes(
        ticks="outside",
        title_font=dict(size=20),
        tickfont=dict(size=20),
        showline=True,
        linecolor="black",
        linewidth=3,
    )
    fig.update_yaxes(
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        linewidth=3,
        tickfont=dict(size=20),
        title="Cost [USD]",
    )
    fig.update_yaxes(
        title_font=dict(size=20),
    )

    return go.FigureWidget(fig)


# ---------------------------------------------------------------------------------------------
# Public bar charts
# ---------------------------------------------------------------------------------------------


def lcc_production_cost_bar_chart_simple(
    aircraft_file_paths: List[Union[str, pathlib.Path]],
    names_aircraft: List[str],
    graph_title: str = None,
    item_filter_list: list = None,
    color_list: List[str] = None,
) -> go.FigureWidget:
    """
    Give a bar chart that compares multiple aircraft designs across all the items of the
    production cost per unit (manufacturing, quality control, material, avionics, engineering,
    tooling, flight test, development support, certification, powertrain, MSP, and every power
    train component purchase cost), in USD.

    :param aircraft_file_paths: paths to the output files that contain the cost results.
    :param names_aircraft: names of the aircraft.
    :param graph_title: title of the graph, if None are specified one is created based on the
    aircraft names.
    :param item_filter_list: filter to only show cost items in the list in output graph. By
    default, everything is plotted.
    :param color_list: colors to use for each aircraft's bars (e.g. "#1f77b4"), given in the
    same order as ``aircraft_file_paths``. By default, Plotly's automatic color cycling is used.
    """

    return _cost_bar_chart_simple(
        aircraft_file_paths=aircraft_file_paths,
        names_aircraft=names_aircraft,
        spec=_PRODUCTION_SPEC,
        default_comparison_label="production cost per unit",
        graph_title=graph_title,
        item_filter_list=item_filter_list,
        color_list=color_list,
    )


def lcc_operation_cost_bar_chart_simple(
    aircraft_file_paths: List[Union[str, pathlib.Path]],
    names_aircraft: List[str],
    graph_title: str = None,
    item_filter_list: list = None,
    color_list: List[str] = None,
) -> go.FigureWidget:
    """
    Give a bar chart that compares multiple aircraft designs across all the items of the
    annual operating cost (insurance, loan, additional, maintenance, miscellaneous, overhaul
    deposit, fuel, electricity, crew, airport, and every power train component operational
    cost), in USD per year.

    :param aircraft_file_paths: paths to the output files that contain the cost results.
    :param names_aircraft: names of the aircraft.
    :param graph_title: title of the graph, if None are specified one is created based on the
    aircraft names.
    :param item_filter_list: filter to only show cost items in the list in output graph. By
    default, everything is plotted.
    :param color_list: colors to use for each aircraft's bars (e.g. "#1f77b4"), given in the
    same order as ``aircraft_file_paths``. By default, Plotly's automatic color cycling is used.
    """

    return _cost_bar_chart_simple(
        aircraft_file_paths=aircraft_file_paths,
        names_aircraft=names_aircraft,
        spec=_OPERATION_SPEC,
        default_comparison_label="annual operating cost",
        graph_title=graph_title,
        item_filter_list=item_filter_list,
        color_list=color_list,
    )
