# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Life Cycle Cost (LCC) bar charts for post-processing.

- ``lcc_cost_bar_chart_simple``: grouped bars, one group per cost item.
- ``lcc_cost_bar_chart_advanced``: one stacked bar per aircraft, all cost items, with the
  power train total replaced by its components.
- ``lcc_power_train_cost_breakdown``: one stacked bar per aircraft, power train only.

Their ``cost_type`` argument is ``"production"`` (cost per unit, USD) or ``"operation"``
(annual cost, USD/yr). Power train
components are found by scanning ``data:propulsion:he_power_train:<type>:<id>:purchase_cost`` /
``:operational_cost``, so any architecture works.
"""

import logging
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import plotly
import plotly.graph_objects as go

import fastoad.api as oad

PRODUCTION_PREFIX = "data:cost:production:"
OPERATION_PREFIX = "data:cost:operation:"
_POWER_TRAIN_PREFIX = "data:propulsion:he_power_train:"

_LOGGER = logging.getLogger(__name__)

PURCHASE_COST_SUFFIX = ":purchase_cost"
OPERATIONAL_COST_SUFFIX = ":operational_cost"

# Same color and hatch cycles as lca_impact.
COLS = plotly.colors.DEFAULT_PLOTLY_COLORS
HASH = ["/", "x", "-", "|", "+", ".", "", "\\"]


# ---------------------------------------------------------------------------------------------
# Cost items: specs and data extraction
# ---------------------------------------------------------------------------------------------

# Spec keys:
# - "leaves": {variable_name: display_label}.
# - "discover_suffix": power train variables ending with it are added automatically.
# - "power_train_total": leaf replaced by its components in the stacked charts.
# - "not_stacked": leaves that aren't a share of the total (e.g. MSP), kept out of stacked charts.

_PRODUCTION_SPEC = {
    "leaves": {
        # Recurring costs
        PRODUCTION_PREFIX + "manufacturing_cost_per_unit": "Manufacturing",
        PRODUCTION_PREFIX + "quality_control_cost_per_unit": "Quality control",
        PRODUCTION_PREFIX + "material_cost_per_unit": "Material",
        PRODUCTION_PREFIX + "avionics_cost_per_unit": "Avionics",
        # Negative: hidden in the simple charts, drawn below zero in the stacked ones
        PRODUCTION_PREFIX + "landing_gear_cost_reduction": "Landing gear reduction",
        PRODUCTION_PREFIX + "powertrain_cost_per_unit": "Powertrain",
        "data:cost:msp_per_unit": "MSP",
        # Non-recurring costs, amortized per unit
        PRODUCTION_PREFIX + "engineering_cost_per_unit": "Engineering",
        PRODUCTION_PREFIX + "tooling_cost_per_unit": "Tooling",
        PRODUCTION_PREFIX + "flight_test_cost_per_unit": "Flight test",
        PRODUCTION_PREFIX + "dev_support_cost_per_unit": "Development support",
        PRODUCTION_PREFIX + "certification_cost_per_unit": "Certification",
    },
    "discover_suffix": PURCHASE_COST_SUFFIX,
    "power_train_total": PRODUCTION_PREFIX + "powertrain_cost_per_unit",
    "not_stacked": ["data:cost:msp_per_unit"],
}

_OPERATION_SPEC = {
    "leaves": {
        # annual_depreciation_cost is left out: it isn't part of annual_cost_per_unit
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
    "discover_suffix": OPERATIONAL_COST_SUFFIX,
    "power_train_total": None,
    "not_stacked": [],
}

# Spec and default labels of each cost type, selected with the cost_type argument
_COST_TYPES = {
    "production": {
        "spec": _PRODUCTION_SPEC,
        "unit": "USD",
        "simple_title": "production cost per unit",
        "simple_y_axis": "Cost [USD]",
        "advanced_title": "Production cost per unit breakdown",
        "advanced_y_axis": "Production cost per unit [USD]",
        "power_train_title": "Power train purchase cost breakdown",
        "power_train_y_axis": "Purchase cost [USD]",
    },
    "operation": {
        "spec": _OPERATION_SPEC,
        "unit": "USD/yr",
        "simple_title": "annual operating cost",
        "simple_y_axis": "Cost [USD/yr]",
        "advanced_title": "Annual operating cost breakdown",
        "advanced_y_axis": "Annual operating cost [USD/yr]",
        "power_train_title": "Power train annual operational cost breakdown",
        "power_train_y_axis": "Operational cost [USD/yr]",
    },
}


def _get_cost_type(cost_type: str) -> Dict:
    """Settings of ``cost_type``, "production" or "operation"."""

    if cost_type not in _COST_TYPES:
        raise ValueError('cost_type must be "production" or "operation", got: ' + str(cost_type))
    return _COST_TYPES[cost_type]


def _get_value(datafile: oad.DataFile, name: str, default: float = 0.0) -> float:
    """Value of a variable, or ``default`` if it isn't in the file."""

    try:
        return float(datafile[name].value[0])
    # DataFile raises ValueError for a missing name
    except (KeyError, ValueError):
        return default


def _discover_power_train_variables(datafile: oad.DataFile, suffix: str) -> Dict[str, str]:
    """{variable name: "<type><br><id>"} for every power train variable ending with suffix."""

    discovered_vars = {}
    for name in datafile.names():
        split_name = _split_power_train_variable(name, suffix)
        if split_name:
            component_type, component_name = split_name
            discovered_vars[name] = component_type + "<br>" + component_name

    return discovered_vars


def _split_power_train_variable(name: str, suffix: str) -> Optional[Tuple[str, str]]:
    """Splits ``...:he_power_train:<type>:<id><suffix>`` into (type, id), None if no match."""

    if not (name.startswith(_POWER_TRAIN_PREFIX) and name.endswith(suffix)):
        return None

    middle = name[len(_POWER_TRAIN_PREFIX) : -len(suffix)]
    component_type, _, component_name = middle.partition(":")

    # Not the cost of one component
    if not component_type or not component_name or ":" in component_name:
        return None

    return component_type, component_name


def _leaf_aliases(var_name: str, disp_label: str) -> List[str]:
    """Accepted names for an item: label ("Fuel"), variable name, name without prefix."""

    aliases = [disp_label, var_name]
    for prefix in (PRODUCTION_PREFIX, OPERATION_PREFIX, _POWER_TRAIN_PREFIX):
        if var_name.startswith(prefix):
            aliases.append(var_name[len(prefix) :])

    return aliases


def _get_cost_dict(
    aircraft_file_path: Union[str, pathlib.Path], spec: Dict
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Positive cost items of one aircraft, for the simple charts.

    :return: ({label: cost}, {alias: label}).
    """

    datafile = oad.DataFile(aircraft_file_path)

    leaves = dict(spec["leaves"])
    # Add power train items discovered in the file
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
# Simple (grouped) bar charts
# ---------------------------------------------------------------------------------------------


def lcc_cost_bar_chart_simple(
    aircraft_file_paths: List[Union[str, pathlib.Path]],
    names_aircraft: List[str],
    cost_type: str = "production",
    graph_title: str = None,
    item_filter_list: list = None,
    color_list: List[str] = None,
) -> go.FigureWidget:
    """
    Cost items of several aircraft, as grouped bars.

    :param aircraft_file_paths: paths to the output files.
    :param names_aircraft: names of the aircraft.
    :param cost_type: "production" (cost per unit, USD) or "operation" (annual cost, USD/yr).
    :param graph_title: title, a default one is built if None.
    :param item_filter_list: items to show, by label ("Fuel") or variable name
    ("annual_fuel_cost"). Everything by default.
    :param color_list: one bar color per aircraft. Plotly's cycle by default.
    """

    settings = _get_cost_type(cost_type)
    spec = settings["spec"]
    default_comparison_label = settings["simple_title"]

    fig = go.Figure()

    for i, (aircraft_file_path, name_aircraft) in enumerate(
        zip(aircraft_file_paths, names_aircraft)
    ):
        cost_dict, leaf_aliases = _get_cost_dict(aircraft_file_path, spec)

        item_names = list(cost_dict.keys())
        item_values = list(cost_dict.values())

        # Set the filtered item names and values as labels and values for the bar chart
        if item_filter_list:
            filtered_item_names = []
            filtered_item_values = []
            for tl_item in item_filter_list:
                # Accept any alias of the item
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
        title=settings["simple_y_axis"],
    )
    fig.update_yaxes(
        title_font=dict(size=20),
    )

    return go.FigureWidget(fig)


# ---------------------------------------------------------------------------------------------
# Stacked-chart helpers and power train breakdown
# ---------------------------------------------------------------------------------------------


def _get_power_train_costs(datafile: oad.DataFile, suffix: str) -> List[Tuple[str, str, float]]:
    """(type, id, cost) of every power train component of one file, excluded or not."""

    costs = []
    for name in datafile.names():
        split_name = _split_power_train_variable(name, suffix)
        if split_name:
            component_type, component_name = split_name
            costs.append((component_type, component_name, _get_value(datafile, name)))

    return costs


def _group_power_train_costs(
    costs: List[Tuple[str, str, float]], group_by: str = "component", exclude_list: list = None
) -> Dict[str, float]:
    """
    Non-zero power train costs, by component id or by type, without the excluded components.

    :param costs: output of :func:`_get_power_train_costs`.
    :param group_by: "component" (one key per id) or "type" (ids of a type summed).
    :param exclude_list: types, ids or "<type>:<id>" to leave out, applied before grouping.
    """

    if group_by not in ("component", "type"):
        raise ValueError('group_by must be either "component" or "type", got: ' + str(group_by))

    exclude_set = set(exclude_list or [])

    cost_dict = {}
    for component_type, component_name, value in costs:
        if value == 0.0 or exclude_set & _power_train_names(component_type, component_name):
            continue
        key = component_name if group_by == "component" else component_type
        cost_dict[key] = cost_dict.get(key, 0.0) + value

    return cost_dict


def _get_power_train_cost_dict(
    aircraft_file_path: Union[str, pathlib.Path],
    suffix: str,
    group_by: str = "component",
    exclude_list: list = None,
) -> Dict[str, float]:
    """Non-zero power train costs of one aircraft, see :func:`_group_power_train_costs`."""

    costs = _get_power_train_costs(oad.DataFile(aircraft_file_path), suffix)
    return _group_power_train_costs(costs, group_by=group_by, exclude_list=exclude_list)


def _power_train_names(component_type: str, component_name: str) -> set:
    """
    Names that refer to a component in ``exclude_list``: its type ("PEMFC_stack"), its id
    ("pemfc_stack_1") and "<type>:<id>", each also with spaces instead of underscores.
    """

    names = {component_type, component_name, component_type + ":" + component_name}
    return names | {name.replace("_", " ") for name in names}


def _check_exclude_list(
    exclude_list: list,
    aircraft_file_paths: List[Union[str, pathlib.Path]],
    suffix: str,
    spec: Dict = None,
):
    """
    Raises a ValueError if an entry of ``exclude_list`` matches nothing in any file (most likely
    a typo). Valid entries are the power train component types and ids found in the files and,
    when ``spec`` is given, the names of its cost items and "Powertrain".
    """

    if not exclude_list:
        return

    valid_names = set()
    # Short list of the valid names, for the error message
    suggestions = set()
    if spec:
        for var_name, disp_label in spec["leaves"].items():
            if var_name == spec.get("power_train_total"):
                disp_label = _POWER_TRAIN_LABEL
            valid_names.update(_leaf_aliases(var_name, disp_label))
            suggestions.add(disp_label)
        valid_names.add(_POWER_TRAIN_LABEL)

    for aircraft_file_path in aircraft_file_paths:
        for component_type, component_name, _ in _get_power_train_costs(
            oad.DataFile(aircraft_file_path), suffix
        ):
            valid_names |= _power_train_names(component_type, component_name)
            suggestions.update((component_type, component_name))

    unknown = [entry for entry in exclude_list if entry not in valid_names]
    if unknown:
        raise ValueError(
            "exclude_list entries not found in any file: "
            + ", ".join(str(entry) for entry in unknown)
            + ". Valid entries include: "
            + ", ".join(sorted(suggestions, key=str.lower))
        )


def _aggregate_and_sort_contributors(
    contribution_on_each_aircraft: Dict[str, Dict[str, float]],
    names_aircraft: List[str],
    aggregate_and_sort_contributor: Dict[str, Union[str, List[str]]] = None,
    cutoff_criteria: float = None,
    sort_by_size: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Orders {contributor: {aircraft: cost}} bottom to top: groups from
    ``aggregate_and_sort_contributor`` first, then the rest by decreasing size (given order if
    ``sort_by_size`` is False). Positive items under ``cutoff_criteria`` % on every aircraft go
    into "Others", on top.
    """

    def _blank():
        return {aircraft: 0.0 for aircraft in names_aircraft}

    ordered = {}
    unsorted = {}

    if aggregate_and_sort_contributor:
        for new_name in aggregate_and_sort_contributor:
            ordered[new_name] = _blank()

    for contributor, contribution in contribution_on_each_aircraft.items():
        target = None
        # Aggregate contributors into a group if requested
        if aggregate_and_sort_contributor:
            for new_name, old_names in aggregate_and_sort_contributor.items():
                if isinstance(old_names, str):
                    old_names = [old_names]
                if contributor in old_names:
                    target = new_name
                    break

        if target is None:
            unsorted[contributor] = contribution
        else:
            for aircraft, cost in contribution.items():
                ordered[target][aircraft] += cost

    # Drop empty groups
    ordered = {name: c for name, c in ordered.items() if any(v != 0.0 for v in c.values())}

    totals = _blank()
    for contribution in contribution_on_each_aircraft.values():
        for aircraft, cost in contribution.items():
            totals[aircraft] += cost

    # Sorted lookup list of the contributor names (biggest first, or as given)
    others = _blank()
    if sort_by_size:
        contributors = sorted(unsorted, key=lambda c: abs(sum(unsorted[c].values())), reverse=True)
    else:
        contributors = list(unsorted)

    # Keep each contributor in that order, or move it into "Others" if under cutoff_criteria
    for contributor in contributors:
        contribution = unsorted[contributor]
        if cutoff_criteria and sum(contribution.values()) > 0.0:
            max_share = max(
                (
                    contribution.get(aircraft, 0.0) / totals[aircraft] * 100.0
                    for aircraft in names_aircraft
                    if totals[aircraft] > 0.0
                ),
                default=0.0,
            )
            if max_share < cutoff_criteria:
                for aircraft, cost in contribution.items():
                    others[aircraft] += cost
                continue

        ordered[contributor] = contribution

    if any(v != 0.0 for v in others.values()):
        ordered["Others"] = others

    return ordered


def _stacked_cost_bar_chart(
    ordered_contributors: Dict[str, Dict[str, float]],
    names_aircraft: List[str],
    title: str,
    y_axis_title: str,
    kept_names: List[str] = None,
    legend_rename: Dict[str, str] = None,
    show_total: bool = True,
    reference_value: float = None,
    reference_label: str = "Reference",
    color_dict: Dict[str, str] = None,
    value_unit: str = "USD",
    share_label: str = "of the total",
) -> go.FigureWidget:
    """
    Draws one stacked bar per aircraft from :func:`_aggregate_and_sort_contributors` output.
    Negative costs go below zero, totals are net.

    :param kept_names: names shown as is; the others get underscores replaced by spaces.
    :param share_label: end of the hover text giving each item's share.
    """

    kept_names = kept_names or []

    totals = {aircraft: 0.0 for aircraft in names_aircraft}
    positive_heights = {aircraft: 0.0 for aircraft in names_aircraft}
    negative_heights = {aircraft: 0.0 for aircraft in names_aircraft}
    for contribution in ordered_contributors.values():
        for aircraft, cost in contribution.items():
            totals[aircraft] += cost
            if cost > 0.0:
                positive_heights[aircraft] += cost
            else:
                negative_heights[aircraft] += cost

    fig = go.Figure()

    # Calculate the share and assign colors for each contributor
    for idx, (contributor, contribution) in enumerate(ordered_contributors.items()):
        if contributor in kept_names:
            display_name = contributor
        else:
            display_name = contributor.replace("_", " ")
        if legend_rename and display_name in legend_rename:
            display_name = legend_rename[display_name]

        values = [contribution.get(aircraft, 0.0) for aircraft in names_aircraft]
        shares = [
            value / totals[aircraft] * 100.0 if totals[aircraft] else 0.0
            for value, aircraft in zip(values, names_aircraft)
        ]

        if color_dict and contributor in color_dict:
            color = color_dict[contributor]
        elif color_dict and display_name in color_dict:
            color = color_dict[display_name]
        else:
            color = COLS[idx % len(COLS)]

        fig.add_trace(
            go.Bar(
                name=display_name,
                x=names_aircraft,
                y=values,
                customdata=shares,
                hovertemplate=(
                    "<b>" + display_name + "</b><br>%{x}<br>%{y:,.2f} " + value_unit + "<br>"
                    "%{customdata:.1f} % " + share_label + "<extra></extra>"
                ),
                marker=dict(
                    pattern_shape=HASH[(idx // len(COLS)) % len(HASH)],
                    color=color,
                ),
            )
        )

    if show_total:
        fig.add_trace(
            go.Scatter(
                x=names_aircraft,
                # Net total, written above the positive stack
                y=[positive_heights[aircraft] for aircraft in names_aircraft],
                text=[
                    _format_cost(totals[aircraft]) + " " + value_unit for aircraft in names_aircraft
                ],
                mode="text",
                textposition="top center",
                textfont=dict(size=18),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if reference_value is not None:
        fig.add_hline(
            y=reference_value,
            line_dash="dash",
            line_color="grey",
            annotation_text=reference_label + ": {:,.2f} ".format(reference_value) + value_unit,
            annotation_position="top left",
            annotation_font=dict(size=16),
            annotation_bgcolor="rgba(255,255,255,0.85)",
        )

    # Headroom for the totals
    max_height = max(list(positive_heights.values()) + [reference_value or 0.0, 0.0])
    min_height = min(list(negative_heights.values()) + [reference_value or 0.0, 0.0])
    y_range = [
        min_height * 1.12,
        max_height * 1.12 if max_height > 0.0 else 1.0,
    ]

    fig.update_layout(
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        title_x=0.5,
        title_text=title,
        # Like "stack", but negatives go below zero
        barmode="relative",
        legend=dict(traceorder="reversed"),
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
        title=y_axis_title,
        range=y_range,
        zeroline=True,
        zerolinecolor="black",
    )
    fig.update_yaxes(
        title_font=dict(size=20),
    )

    return go.FigureWidget(fig)


def _format_cost(value: float) -> str:
    """Thousands separators, decimals only below 1000."""

    if abs(value) >= 1000.0:
        return "{:,.0f}".format(value)
    return "{:,.2f}".format(value)


def lcc_power_train_cost_breakdown(
    aircraft_file_paths: Union[Union[str, pathlib.Path], List[Union[str, pathlib.Path]]],
    names_aircraft: Union[str, List[str]],
    cost_type: str = "production",
    group_by: str = "component",
    exclude_list: list = None,
    graph_title: str = None,
    legend_rename: Dict[str, str] = None,
    aggregate_and_sort_contributor: Dict[str, Union[str, List[str]]] = None,
    cutoff_criteria: float = None,
    show_total: bool = True,
    reference_value: float = None,
    reference_label: str = "Reference",
    color_dict: Dict[str, str] = None,
) -> go.FigureWidget:
    """
    Power train cost, one stacked bar per aircraft, one segment per component: purchase cost
    for "production", annual operational cost for "operation".

    :param aircraft_file_paths: paths to the output files.
    :param names_aircraft: names of the aircraft (x-axis labels).
    :param cost_type: "production" (cost per unit, USD) or "operation" (annual cost, USD/yr).
    :param group_by: "component" (one segment per id, e.g. "motor_1") or "type" (ids of the
    same type class summed, e.g. "PMSM").
    :param exclude_list: components to leave out, by type ("PEMFC_stack"), id ("pemfc_stack_1")
    or "<type>:<id>". Works with either ``group_by``. Raises a ValueError for an unknown entry.
    :param graph_title: title, a default one is used if None.
    :param legend_rename: {legend name: new name}.
    :param aggregate_and_sort_contributor: {new name: item or list of items} to sum. Groups are
    stacked first, from the bottom, in the given order.
    :param cutoff_criteria: items under this % of the total on every aircraft go into "Others".
    :param show_total: writes the total above each bar.
    :param reference_value: draws a dashed line at this value (e.g. a budget or the revenue).
    :param reference_label: label of that line.
    :param color_dict: {item: color} to force some colors.
    """

    settings = _get_cost_type(cost_type)
    suffix = settings["spec"]["discover_suffix"]
    default_title = settings["power_train_title"]
    y_axis_title = settings["power_train_y_axis"]
    value_unit = settings["unit"]

    if isinstance(aircraft_file_paths, (str, pathlib.Path)):
        aircraft_file_paths = [aircraft_file_paths]
    if isinstance(names_aircraft, str):
        names_aircraft = [names_aircraft]
    if len(aircraft_file_paths) != len(names_aircraft):
        raise ValueError(
            "aircraft_file_paths and names_aircraft must have the same length, got "
            + str(len(aircraft_file_paths))
            + " and "
            + str(len(names_aircraft))
        )

    _check_exclude_list(exclude_list, aircraft_file_paths, suffix)

    # {contributor: {aircraft_name: cost}}
    contribution_on_each_aircraft = {}
    for aircraft_file_path, name_aircraft in zip(aircraft_file_paths, names_aircraft):
        cost_dict = _get_power_train_cost_dict(
            aircraft_file_path,
            suffix,
            group_by=group_by,
            exclude_list=exclude_list,
        )
        for contributor, cost in cost_dict.items():
            contribution_on_each_aircraft.setdefault(contributor, {})[name_aircraft] = cost

    ordered_contributors = _aggregate_and_sort_contributors(
        contribution_on_each_aircraft,
        names_aircraft,
        aggregate_and_sort_contributor=aggregate_and_sort_contributor,
        cutoff_criteria=cutoff_criteria,
    )

    return _stacked_cost_bar_chart(
        ordered_contributors,
        names_aircraft,
        title=graph_title if graph_title else default_title,
        y_axis_title=y_axis_title,
        kept_names=list(aggregate_and_sort_contributor or {}),
        legend_rename=legend_rename,
        show_total=show_total,
        reference_value=reference_value,
        reference_label=reference_label,
        color_dict=color_dict,
        value_unit=value_unit,
        share_label="of the power train cost",
    )


# ---------------------------------------------------------------------------------------------
# Advanced (stacked) charts of all the cost items
# ---------------------------------------------------------------------------------------------

_POWER_TRAIN_LABEL = "Powertrain"
_POWER_TRAIN_DETAILS = ("component", "type", "aggregate")
# Gap between power_train_total and the sum of its components that triggers a warning
_RECONCILIATION_TOLERANCE = 0.01


def _get_cost_contributions(
    aircraft_file_path: Union[str, pathlib.Path],
    spec: Dict,
    power_train_detail: str = "component",
    exclude_list: list = None,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Non-zero cost items of one aircraft, ready to stack: ``power_train_total`` is replaced by
    its components (kept as is only if the file has none), ``not_stacked`` leaves are dropped
    and negative costs are kept.

    :param power_train_detail: "component", "type" or "aggregate" (one "Powertrain" item).
    :param exclude_list: item names or component types/ids, "Powertrain" for the whole power
    train.
    :return: ({item: cost}, {alias: item}).
    """

    if power_train_detail not in _POWER_TRAIN_DETAILS:
        raise ValueError(
            "power_train_detail must be one of "
            + ", ".join(_POWER_TRAIN_DETAILS)
            + ", got: "
            + str(power_train_detail)
        )

    datafile = oad.DataFile(aircraft_file_path)

    power_train_total_name = spec.get("power_train_total")
    not_stacked = set(spec.get("not_stacked") or [])

    contributions = {}
    aliases = {}
    exclude_set = set(exclude_list or [])

    for var_name, disp_label in spec["leaves"].items():
        # Skip the not_stacked and the total cost of the powertrain
        if var_name in not_stacked or var_name == power_train_total_name:
            continue

        # Skip the items in exclude_list, under any of their names
        if exclude_set.intersection(_leaf_aliases(var_name, disp_label)):
            continue

        value = _get_value(datafile, var_name)
        if value != 0.0:
            contributions[disp_label] = value
            for alias in _leaf_aliases(var_name, disp_label):
                aliases[alias] = disp_label

    # Every component cost of the file, then the ones kept for the chart
    all_component_costs = _get_power_train_costs(datafile, spec["discover_suffix"])
    components = _group_power_train_costs(
        all_component_costs,
        group_by="type" if power_train_detail == "type" else "component",
        exclude_list=exclude_list,
    )

    power_train_total = (
        _get_value(datafile, power_train_total_name) if power_train_total_name else 0.0
    )
    components_sum = sum(components.values())
    # Fall back on power_train_total only if the file has no components at all, otherwise it
    # would bring excluded costs back
    all_components_sum = sum(value for _, _, value in all_component_costs)
    components_found = any(value != 0.0 for _, _, value in all_component_costs)

    if components_found and power_train_total:
        gap = abs(all_components_sum - power_train_total) / abs(power_train_total)
        if gap > _RECONCILIATION_TOLERANCE:
            _LOGGER.warning(
                "In %s, the power train components sum up to %.2f but %s is %.2f (%.1f %% gap). "
                "The components are plotted, the stacked total won't match the one in the file.",
                aircraft_file_path,
                all_components_sum,
                power_train_total_name,
                power_train_total,
                gap * 100.0,
            )

    power_train_aliases = [_POWER_TRAIN_LABEL]
    if power_train_total_name:
        power_train_aliases += _leaf_aliases(power_train_total_name, _POWER_TRAIN_LABEL)

    if exclude_set.intersection(power_train_aliases):
        # The whole power train is excluded
        pass
    elif power_train_detail != "aggregate" and components:
        contributions.update(components)
        for key in components:
            aliases[key] = key
            aliases[key.replace("_", " ")] = key
    else:
        value = components_sum if components_found else power_train_total
        if value != 0.0:
            contributions[_POWER_TRAIN_LABEL] = value
            for alias in power_train_aliases:
                aliases[alias] = _POWER_TRAIN_LABEL

    return contributions, aliases


def lcc_cost_bar_chart_advanced(
    aircraft_file_paths: Union[Union[str, pathlib.Path], List[Union[str, pathlib.Path]]],
    names_aircraft: Union[str, List[str]],
    cost_type: str = "production",
    power_train_detail: str = "component",
    item_filter_list: list = None,
    exclude_list: list = None,
    normalisation_factors: List[float] = None,
    graph_title: str = None,
    y_axis_title: str = None,
    value_unit: str = None,
    legend_rename: Dict[str, str] = None,
    aggregate_and_sort_contributor: Dict[str, Union[str, List[str]]] = None,
    cutoff_criteria: float = None,
    sort_by_size: bool = True,
    show_total: bool = True,
    reference_value: float = None,
    reference_label: str = "Reference",
    color_dict: Dict[str, str] = None,
) -> go.FigureWidget:
    """
    All cost items, one stacked bar per aircraft, one segment per item. The power train total
    is replaced by its components, MSP is left out, reductions go below zero.

    Items can be named by label ("Tooling"), variable name ("tooling_cost_per_unit"), component
    id ("motor_1") or "Powertrain". For an operating cost per passenger per flight, use
    ``normalisation_factors=[flights_per_year * pax, ...]``, ``value_unit="USD/PAX/flight"`` and
    the revenue as ``reference_value``.

    :param aircraft_file_paths: paths to the output files.
    :param names_aircraft: names of the aircraft (x-axis labels).
    :param cost_type: "production" (cost per unit, USD) or "operation" (annual cost, USD/yr).
    :param power_train_detail: "component" (one segment per id), "type" (per type class) or
    "aggregate" (one "Powertrain" segment).
    :param item_filter_list: items to keep. Everything by default.
    :param exclude_list: items to leave out, or component types ("PEMFC_stack") and
    "<type>:<id>". Raises a ValueError for an unknown entry.
    :param normalisation_factors: one divisor per aircraft (e.g. for a cost per seat).
    :param y_axis_title: y-axis title, a default one is used if None.
    :param value_unit: unit of the totals, hover and reference line (USD or USD/yr by default).
    :param graph_title: title, a default one is used if None.
    :param legend_rename: {legend name: new name}.
    :param aggregate_and_sort_contributor: {new name: item or list of items} to sum. Groups are
    stacked first, from the bottom, in the given order.
    :param cutoff_criteria: items under this % of the total on every aircraft go into "Others".
    :param show_total: writes the total above each bar.
    :param reference_value: draws a dashed line at this value (e.g. a budget or the revenue).
    :param reference_label: label of that line.
    :param color_dict: {item: color} to force some colors.
    :param sort_by_size: biggest items at the bottom. If False, keeps the spec order.
    """

    settings = _get_cost_type(cost_type)
    spec = settings["spec"]
    default_title = settings["advanced_title"]
    default_y_axis_title = settings["advanced_y_axis"]
    default_value_unit = settings["unit"]

    # wrap single values in lists for uniform processing
    if isinstance(aircraft_file_paths, (str, pathlib.Path)):
        aircraft_file_paths = [aircraft_file_paths]
    if isinstance(names_aircraft, str):
        names_aircraft = [names_aircraft]

    if len(aircraft_file_paths) != len(names_aircraft):
        raise ValueError(
            "aircraft_file_paths and names_aircraft must have the same length, got "
            + str(len(aircraft_file_paths))
            + " and "
            + str(len(names_aircraft))
        )
    if normalisation_factors is not None and len(normalisation_factors) != len(names_aircraft):
        raise ValueError("normalisation_factors must give one factor per aircraft")

    _check_exclude_list(exclude_list, aircraft_file_paths, spec["discover_suffix"], spec)

    contribution_on_each_aircraft = {}
    all_aliases = {}

    for idx, (aircraft_file_path, name_aircraft) in enumerate(
        zip(aircraft_file_paths, names_aircraft)
    ):
        contributions, aliases = _get_cost_contributions(
            aircraft_file_path,
            spec,
            power_train_detail=power_train_detail,
            exclude_list=exclude_list,
        )
        all_aliases.update(aliases)

        if item_filter_list:
            wanted = {aliases.get(item, item) for item in item_filter_list}
            contributions = {k: v for k, v in contributions.items() if k in wanted}

        factor = normalisation_factors[idx] if normalisation_factors is not None else 1.0
        for contributor, cost in contributions.items():
            contribution_on_each_aircraft.setdefault(contributor, {})[name_aircraft] = cost / factor

    # Accept any alias in aggregate_and_sort_contributor
    resolved_aggregation = None
    if aggregate_and_sort_contributor:
        resolved_aggregation = {}
        for new_name, old_names in aggregate_and_sort_contributor.items():
            if isinstance(old_names, str):
                old_names = [old_names]
            resolved_aggregation[new_name] = [all_aliases.get(name, name) for name in old_names]

    ordered_contributors = _aggregate_and_sort_contributors(
        contribution_on_each_aircraft,
        names_aircraft,
        aggregate_and_sort_contributor=resolved_aggregation,
        cutoff_criteria=cutoff_criteria,
        sort_by_size=sort_by_size,
    )

    return _stacked_cost_bar_chart(
        ordered_contributors,
        names_aircraft,
        title=graph_title if graph_title else default_title,
        y_axis_title=y_axis_title if y_axis_title else default_y_axis_title,
        kept_names=list(resolved_aggregation or {}) + [_POWER_TRAIN_LABEL],
        legend_rename=legend_rename,
        show_total=show_total,
        reference_value=reference_value,
        reference_label=reference_label,
        color_dict=color_dict,
        value_unit=value_unit if value_unit else default_value_unit,
        share_label="of the total cost",
    )
