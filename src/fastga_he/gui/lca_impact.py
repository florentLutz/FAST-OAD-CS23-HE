# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import pathlib

from collections import OrderedDict

from typing import List, Union, Dict

import numpy as np

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import fastoad.api as oad

from fastga_he.exceptions import ImpactUnavailableForPlotError
from ..models.environmental_impacts.resources.constants import LCA_PREFIX

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS
HASH = ["/", "x", "-", "|", "+", ".", "", "\\"]


def lca_impacts_sun_breakdown(
    aircraft_file_path: Union[str, List[str]],
    full_burst: bool = False,
    name_aircraft: Union[str, List[str]] = None,
    rel: str = "absolute",
) -> go.FigureWidget:
    """
    Give a breakdown of the single score impact of the aircraft under the form of a sun breakdown.

    :param aircraft_file_path: path to the output file that contains the weighted and aggregated
    impacts
    :param full_burst: boolean to display all levels of impacts
    :param name_aircraft: name of the aircraft
    :param rel: string to display impacts in a relative form. By default, it is not done. Can be
    relative to the "single_score" or relative to the "parent".
    """

    title_text = "Single score breakdown"
    if rel == "single_score":
        title_text += "<br>expressed as a percentage of the total score"
    elif rel == "parent":
        title_text += "<br>expressed as a percentage of parent category"

    if isinstance(aircraft_file_path, str):
        fig = go.Figure()
        fig.add_trace(_get_impact_sunburst(aircraft_file_path, rel))

        if name_aircraft:
            fig.update_layout(title_text=name_aircraft + " " + title_text, title_x=0.5)
        else:
            fig.update_layout(title_text=title_text, title_x=0.5)

    elif len(aircraft_file_path) == 1:
        fig = go.Figure()
        fig.add_trace(_get_impact_sunburst(aircraft_file_path[0], rel))

        if name_aircraft[0]:
            fig.update_layout(title_text=name_aircraft[0] + " " + title_text, title_x=0.5)
        else:
            fig.update_layout(title_text=title_text, title_x=0.5)

    else:
        fig = make_subplots(
            1,
            cols=len(aircraft_file_path),
            specs=[[{"type": "domain"}, {"type": "domain"}]],
            subplot_titles=name_aircraft,
        )
        fig_counter = 1

        for idx, curr_aircraft_file_path in enumerate(aircraft_file_path):
            fig.add_trace(_get_impact_sunburst(curr_aircraft_file_path, rel=rel), 1, fig_counter)
            fig.update_traces(row=1, col=fig_counter, name=name_aircraft[idx])
            fig_counter += 1

        fig.update_layout(title_text=title_text, title_x=0.5)

    if full_burst:
        fig.update_traces(selector=dict(type="sunburst"))
    else:
        fig.update_traces(maxdepth=2, selector=dict(type="sunburst"))

    fig = go.FigureWidget(fig)

    return fig


def _get_impact_variable_list(aircraft_file_path: Union[str, pathlib.Path]) -> list:
    """
    Returns a list of the name of the variable associated with the weighted impacts and available
    in the output file.
    :param aircraft_file_path: path to the output file path.

    :return: a list of all weighted impact available in the output file path.
    """

    datafile = oad.DataFile(aircraft_file_path)
    names = datafile.names()
    names_variables_lca = []

    for name in names:
        if LCA_PREFIX in name:
            if "weighted" in name or "single_score" in name:
                names_variables_lca.append(name)

    return names_variables_lca


def _get_weighted_impact_dict(aircraft_file_path: Union[str, pathlib.Path]) -> dict:
    """
    Returns a dict of weighted impacts categories available in the output file and their value.
    :param aircraft_file_path: path to the output file path.

    :return: a dict of all weighted impact available in the output file path and their value.
    """

    names_variable_lca = _get_impact_variable_list(aircraft_file_path)
    names_impact_categories = {}
    datafile = oad.DataFile(aircraft_file_path)
    for name_variable_lca in names_variable_lca:
        if _depth_lca_detail(name_variable_lca) <= 2:
            impact_score = datafile[name_variable_lca].value[0]
            impact_name = name_variable_lca.replace(LCA_PREFIX, "").replace("_weighted:sum", "")
            names_impact_categories[impact_name] = impact_score

    return names_impact_categories


def _get_impact_sunburst(aircraft_file_path: str, rel: str = "absolute") -> go.Sunburst:
    names_variables_lca = _get_impact_variable_list(aircraft_file_path)
    datafile = oad.DataFile(aircraft_file_path)

    if len(names_variables_lca) == 0:
        sunburst = go.Sunburst()
        return sunburst

    max_depth = 0
    for name in names_variables_lca:
        curr_depth = _depth_lca_detail(name)
        if curr_depth > max_depth:
            max_depth = curr_depth

    # Because it's the earliest parent ;)
    label_ancestor = _get_ancestor_label(datafile)

    figure_labels = [label_ancestor]
    figure_parents = [""]
    figure_color = [None]
    color_dict = {}
    if rel == "single_score" or rel == "parent":
        figure_values = [100.0]  # In percent
    else:
        figure_values = [datafile[LCA_PREFIX + "single_score"].value[0]]

    names_variables_lca.remove(LCA_PREFIX + "single_score")

    for name in names_variables_lca:
        figure_labels.append(_name_to_label(name, datafile, rel=rel))
        figure_parents.append(_get_parent_label(name, datafile, rel=rel))
        if rel == "single_score" or rel == "parent":
            figure_values.append(
                datafile[name].value[0] / datafile[LCA_PREFIX + "single_score"].value[0] * 100.0
            )
        else:
            figure_values.append(datafile[name].value[0])
        figure_color.append(_get_color(name, color_dict))

    return go.Sunburst(
        labels=figure_labels,
        parents=figure_parents,
        values=figure_values,
        branchvalues="total",
        sort=False,
        marker={"colors": figure_color},
    )


def _depth_lca_detail(name_variable: str) -> int:
    if "single_score" in name_variable:
        return 1

    tmp_name = name_variable.replace(LCA_PREFIX, "")
    depth_lca = len(tmp_name.split(":"))
    if "sum" not in tmp_name:
        depth_lca += 1

    return depth_lca


def _name_to_label(name_variable: str, datafile: oad.DataFile, rel: str = "absolute") -> str:
    if name_variable == LCA_PREFIX + "single_score":
        return _get_ancestor_label(datafile)

    if "sum" not in name_variable:
        depth = -1
    else:
        depth = -2

    clean_name = name_variable.split(":")[depth]

    if rel == "single_score":
        value = (
            datafile[name_variable].value[0]
            / datafile[LCA_PREFIX + "single_score"].value[0]
            * 100.0
        )
        label = clean_name + "<br> " + str(_round_value(value)) + " %"
    elif rel == "parent":
        parent_value = _get_parent_score(name_variable, datafile)
        value = datafile[name_variable].value[0] / parent_value * 100.0
        label = clean_name + "<br> " + str(_round_value(value)) + " %"
    else:
        value = datafile[name_variable].value[0]
        label = clean_name + "<br> " + str(_round_value(value)) + " pt"

    return label


def _get_parent_label(name_variable: str, datafile: oad.DataFile, rel: str = "absolute") -> str:
    parent_name = _get_parent_name(name_variable)

    return _name_to_label(parent_name, datafile, rel=rel)


def _get_parent_score(name_variable: str, datafile: oad.DataFile) -> float:
    parent_name = _get_parent_name(name_variable)
    parent_score = datafile[parent_name].value[0]

    return parent_score


def _get_parent_name(name_variable: str) -> str:
    if _depth_lca_detail(name_variable) == 2:
        return LCA_PREFIX + "single_score"

    if "sum" not in name_variable:
        parent_name = ":".join(name_variable.split(":")[:-1]) + ":sum"
    else:
        parent_name = ":".join(name_variable.split(":")[:-2]) + ":sum"

    return parent_name


def _get_ancestor_label(datafile: oad.DataFile) -> str:
    return (
        "single_score <br> "
        + str(_round_value(datafile[LCA_PREFIX + "single_score"].value[0]))
        + " pt"
    )


def _round_value(value: float) -> float:
    if value == 0.0:
        return value
    else:
        return round(value, int(np.ceil(abs(np.log10(value))) + 5))


def _get_first_parent_name(name_variable: str) -> str:
    if _depth_lca_detail(name_variable) <= 2:
        return name_variable
    else:
        return _get_first_parent_name(_get_parent_name(name_variable))


def _get_color(name_variable: str, color_dict: dict) -> str:
    first_parent = _get_first_parent_name(name_variable)
    if first_parent in color_dict:
        return color_dict[first_parent]
    else:
        color = COLS[len(color_dict) % len(COLS)]
        color_dict[name_variable] = color
        return color


def lca_score_sensitivity_simple(
    results_folder_path: Union[str, pathlib.Path],
    prefix: str,
    name: str = None,
    impact_to_plot: str = "single_score",
    fig: go.Figure = None,
) -> go.Figure:
    """
    Displays the evolution of the impacts of an aircraft with respect to its lifespan. This
    method is a bit sensitive to use as it requires the results to be stored under the form of
    FAST-OAD output files, all in the same folder and all with the same prefix. It also requires
    the user to know and input said prefix. Results can be superimposed to an existing figure, but
    it is recommended to only put results computed on the same lifespan.

    :param results_folder_path: path to the folder that contains the output files that contains
    the results.
    :param prefix: prefix of the output file for the aircraft.
    :param impact_to_plot: Name of the impact to plot.
    :param name: name of the aircraft, to be displayed on the figure.
    :param fig: figure with existing results.

    :return: plotly figure with the evolution of the impact as a function of the lifespan.
    """

    aircraft_lifespan_list = []
    impact_list = []
    names_variables_lca = []

    for dirpath, _, filenames in os.walk(results_folder_path):
        for filename in filenames:
            if filename.startswith(prefix):
                if not names_variables_lca:
                    # Fetch the name of available impacts for plotting
                    names_variables_lca = list(
                        _get_weighted_impact_dict(os.path.join(dirpath, filename)).keys()
                    )

                    # Check that the impact we request exists to make it fail as soon as possible
                    # if it needs to fail
                    if impact_to_plot not in names_variables_lca:
                        raise ImpactUnavailableForPlotError(
                            "Impact "
                            + impact_to_plot
                            + " unavailable in the output file. Available impacts include: "
                            + ", ".join(names_variables_lca)
                        )

                datafile = oad.DataFile(os.path.join(dirpath, filename))
                aircraft_lifespan = datafile["data:TLAR:max_airframe_hours"].value[0]
                aircraft_lifespan_list.append(aircraft_lifespan)
                if impact_to_plot == "single_score":
                    variable_name = LCA_PREFIX + "single_score"
                else:
                    variable_name = LCA_PREFIX + impact_to_plot + "_weighted:sum"
                impact_score = datafile[variable_name].value[0]
                impact_list.append(impact_score)

    aircraft_lifespan_list, impact_list = zip(*sorted(zip(aircraft_lifespan_list, impact_list)))

    if fig is None:
        orig_fig = True
        fig = go.Figure()
    else:
        orig_fig = False

    scatter = go.Scatter(x=aircraft_lifespan_list, y=impact_list, name=name, showlegend=True)
    beautified_impact_score = impact_to_plot.replace("_", " ")

    fig.add_trace(scatter)
    if orig_fig:
        fig.update_layout(
            title_text="Evolution of the "
            + beautified_impact_score
            + " impact with life expectancy of the aircraft",
            xaxis_title="Airframe hours [h]",
            yaxis_title="Single score [-]",
        )
        _update_fig_axis(fig)

    return fig


def lca_score_sensitivity_advanced_impact_category(
    results_folder_path: Union[str, pathlib.Path],
    prefix: str,
    cutoff_criteria: float,
    name: str = None,
) -> go.Figure:
    """
    Displays the evolution of the impacts of an aircraft in terms of single score with respect to
    its lifespan by stacking the contributing impact category. This method is a bit sensitive to
    use as it requires the results to be stored under the form of FAST-OAD output files,
    all in the same folder and all with the same prefix. It also requires the user to know and
    input said prefix. In order not to overload the diagram, we'll allow the user to set a cutoff
    criteria below which not to plot the contribution of the impact. The rest will be aggregated
    into others.

    :param results_folder_path: path to the folder that contains the output files that contains
    the results.
    :param prefix: prefix of the output file for the aircraft.
    :param name: name of the aircraft, to be displayed on the figure.
    :param cutoff_criteria: cutoff criteria, in % of the single score on the last year (e.g. enter
    5 for 5% percent not 0.05)

    :return: plotly figure with the evolution of all the impact contributing ot the single score
    as a function of the lifespan.
    """

    aircraft_lifespan_list = []
    impact_variations = {}

    for dirpath, _, filenames in os.walk(results_folder_path):
        for filename in filenames:
            if filename.startswith(prefix):
                impact_score_dict = _get_weighted_impact_dict(os.path.join(dirpath, filename))
                impact_score_dict.pop("single_score")

                datafile = oad.DataFile(os.path.join(dirpath, filename))
                aircraft_lifespan = datafile["data:TLAR:max_airframe_hours"].value[0]
                aircraft_lifespan_list.append(aircraft_lifespan)

                for impact, impact_score in impact_score_dict.items():
                    _safe_add_to_dict_of_list(impact_variations, impact, impact_score)

    new_impact_variation = _sort_and_cut_off(
        impact_variations, aircraft_lifespan_list, cutoff_criteria
    )

    fig = _prep_lasagna_plot(new_impact_variation, aircraft_lifespan_list)

    fig.update_layout(
        title_text="Evolution of the contribution of each impact to the single score of the "
        + name,
        xaxis_title="Airframe hours [h]",
        yaxis_title="Single score [-]",
    )
    _update_fig_axis(fig)

    return fig


def lca_score_sensitivity_advanced_components(
    results_folder_path: Union[str, pathlib.Path],
    prefix: str,
    cutoff_criteria: float,
    name: str = None,
) -> go.Figure:
    """
    Displays the evolution of the contribution to the single score of each component of the
    aircraft as a function of the estimated lifespan of the aircraft. This method is a bit
    sensitive to use as it requires the results to be stored under the form of FAST-OAD output
    files, all in the same folder and all with the same prefix. It also requires the user to know
    and input said prefix. In order not to overload the diagram, we'll allow the user to set a
    cutoff criteria. The rest will be aggregated into others.

    :param results_folder_path: path to the folder that contains the output files that contains
    the results.
    :param prefix: prefix of the output file for the aircraft.
    :param name: name of the aircraft, to be displayed on the figure.
    :param cutoff_criteria: cutoff criteria, in % of the single score on the last year (e.g. enter
    5 for 5% percent not 0.05)

    :return: plotly figure with the evolution of all the components contributing ot the single score
    as a function of the lifespan.
    """

    aircraft_lifespan_list = []
    contributing_components_and_variables = {}
    components_contribution = {}

    for dirpath, _, filenames in os.walk(results_folder_path):
        for filename in filenames:
            if filename.startswith(prefix):
                datafile = oad.DataFile(os.path.join(dirpath, filename))
                aircraft_lifespan = datafile["data:TLAR:max_airframe_hours"].value[0]
                aircraft_lifespan_list.append(aircraft_lifespan)

                if not contributing_components_and_variables:
                    # In that context, by components, we mean all that contributes to the different
                    # impacts.
                    contributing_components_and_variables = (
                        _get_list_contributing_components_and_variables(
                            os.path.join(dirpath, filename)
                        )
                    )

                for component, variables in contributing_components_and_variables.items():
                    impact_this_component_this_year = 0.0
                    for variable in variables:
                        impact_this_component_this_year += datafile[variable].value[0]

                    _safe_add_to_dict_of_list(
                        components_contribution, component, impact_this_component_this_year
                    )

    new_component_variation = _sort_and_cut_off(
        components_contribution, aircraft_lifespan_list, cutoff_criteria
    )

    fig = _prep_lasagna_plot(new_component_variation, aircraft_lifespan_list)

    fig.update_layout(
        title_text="Evolution of the contribution of each component to the single score of the "
        + name,
        xaxis_title="Airframe hours [h]",
        yaxis_title="Single score [-]",
    )
    _update_fig_axis(fig)

    return fig


def lca_score_sensitivity_advanced_components_and_phase(
    results_folder_path: Union[str, pathlib.Path],
    prefix: str,
    cutoff_criteria: float,
    name: str = None,
) -> go.Figure:
    """
    Displays the evolution of the contribution to the single score of each component of the
    aircraft as a function of the estimated lifespan of the aircraft and separates them by phase.
    This method is a bit sensitive to use as it requires the results to be stored under the form
    of FAST-OAD output files, all in the same folder and all with the same prefix. It also
    requires the user to know and input said prefix. In order not to overload the diagram,
    we'll allow the user to set a cutoff criteria. The rest will be aggregated into others. For
    components whose total contribution is greater than the cutoff we'll highlight if their
    biggest impact comes from the production phase or the use phase

    :param results_folder_path: path to the folder that contains the output files that contains
    the results.
    :param prefix: prefix of the output file for the aircraft.
    :param name: name of the aircraft, to be displayed on the figure.
    :param cutoff_criteria: cutoff criteria, in % of the single score on the last year (e.g. enter
    5 for 5% percent not 0.05)

    :return: plotly figure with the evolution of all the components contributing ot the single score
    as a function of the lifespan.
    """

    aircraft_lifespan_list = []
    contributing_components_and_variables = {}
    components_contribution_total = {}
    components_contribution_production = {}
    components_contribution_use = {}
    components_contribution_other = {}

    for dirpath, _, filenames in os.walk(results_folder_path):
        for filename in filenames:
            if filename.startswith(prefix):
                datafile = oad.DataFile(os.path.join(dirpath, filename))
                aircraft_lifespan = datafile["data:TLAR:max_airframe_hours"].value[0]
                aircraft_lifespan_list.append(aircraft_lifespan)

                if not contributing_components_and_variables:
                    # In that context, by components, we mean all that contributes to the different
                    # impacts.
                    contributing_components_and_variables = (
                        _get_list_contributing_components_and_variables(
                            os.path.join(dirpath, filename)
                        )
                    )

                for component, variables in contributing_components_and_variables.items():
                    impact_this_component_this_year = 0.0

                    # Based on what the LCA conf file looks like at the time this was written, the
                    # only life cycle phases where we can do a breakdown of components is the
                    # production and the use phase. This is not generic
                    impact_this_component_production_this_year = 0.0
                    impact_this_component_use_this_year = 0.0
                    impact_this_component_other_this_year = 0.0

                    for variable in variables:
                        impact_this_component_this_year += datafile[variable].value[0]
                        if ":production:" in variable:
                            impact_this_component_production_this_year += datafile[variable].value[
                                0
                            ]
                        elif ":operation:" in variable:
                            impact_this_component_use_this_year += datafile[variable].value[0]
                        else:
                            impact_this_component_other_this_year += datafile[variable].value[0]

                    _safe_add_to_dict_of_list(
                        components_contribution_total, component, impact_this_component_this_year
                    )
                    if impact_this_component_other_this_year != 0.0:
                        _safe_add_to_dict_of_list(
                            components_contribution_other,
                            component,
                            impact_this_component_other_this_year,
                        )
                    if impact_this_component_production_this_year != 0.0:
                        _safe_add_to_dict_of_list(
                            components_contribution_production,
                            component,
                            impact_this_component_production_this_year,
                        )
                    if impact_this_component_use_this_year != 0.0:
                        _safe_add_to_dict_of_list(
                            components_contribution_use,
                            component,
                            impact_this_component_use_this_year,
                        )

    # To get them sorted and cut off
    new_component_variation = _sort_and_cut_off(
        components_contribution_total, aircraft_lifespan_list, cutoff_criteria
    )

    fig = go.Figure()

    cumulated_impact = np.zeros_like(aircraft_lifespan_list)

    for idx, contributor in enumerate(new_component_variation):
        component_color = COLS[idx % len(COLS)]
        beautified_impact_score = contributor.replace("_", " ")

        if contributor in components_contribution_production:
            cumulated_impact += np.array(list(components_contribution_production[contributor]))
            scatter = go.Scatter(
                x=aircraft_lifespan_list,
                y=cumulated_impact,
                name=beautified_impact_score + ": production",
                showlegend=True,
                line=dict(color="rgb(50,50,50)", width=3),
                fill="tonexty",
                fillpattern=dict(shape="/"),
                fillcolor=component_color,
                legendgroup=beautified_impact_score,
            )
            fig.add_trace(scatter)

        if contributor in components_contribution_use:
            cumulated_impact += np.array(list(components_contribution_use[contributor]))
            scatter = go.Scatter(
                x=aircraft_lifespan_list,
                y=cumulated_impact,
                name=beautified_impact_score + ": operation",
                showlegend=True,
                line=dict(color="rgb(50,50,50)", width=3),
                fill="tonexty",
                fillpattern=dict(shape="x"),
                fillcolor=component_color,
                legendgroup=beautified_impact_score,
            )
            fig.add_trace(scatter)
        if contributor in components_contribution_other:
            cumulated_impact += np.array(list(components_contribution_other[contributor]))
            scatter = go.Scatter(
                x=aircraft_lifespan_list,
                y=cumulated_impact,
                name=beautified_impact_score + ": other phases",
                line=dict(color="rgb(50,50,50)", width=3),
                showlegend=True,
                fill="tonexty",
                fillpattern=dict(shape="o"),
                fillcolor=component_color,
                legendgroup=beautified_impact_score,
            )
            fig.add_trace(scatter)

        separator_scatter = go.Scatter(
            x=aircraft_lifespan_list,
            y=cumulated_impact,
            line=dict(color="rgb(0,0,0)", width=2),
            showlegend=False,
        )
        fig.add_trace(separator_scatter)

    cumulated_impact += new_component_variation["Others"]
    others_scatter = go.Scatter(
        x=aircraft_lifespan_list,
        y=cumulated_impact,
        name="Others",
        line=dict(color="rgb(50,50,50)", width=3),
        showlegend=True,
        fill="tonexty",
        fillcolor=COLS[len(new_component_variation) % len(COLS)],
        legendgroup="others",
    )
    fig.add_trace(others_scatter)

    scatter = go.Scatter(
        x=aircraft_lifespan_list,
        y=cumulated_impact,
        name="Single score",
        line=dict(color="black", width=5),
        showlegend=True,
    )
    fig.add_trace(scatter)

    fig.update_layout(
        title_text="Evolution of the contribution of each component to the single score of the "
        + name,
        xaxis_title="Airframe hours [h]",
        yaxis_title="Single score [-]",
    )
    _update_fig_axis(fig)

    return fig


def _prep_lasagna_plot(
    treated_data_dict: Dict[str, list], aircraft_lifespan_list: list
) -> go.Figure:
    """
    Prepares the lasagna plot by stacking impacts on top of one another and filling below. Having
    contributors sorted from biggest to smallest is recommended. Thanks @ScottDelbecq for the name
    suggestions

    :param treated_data_dict: dictionary containing the data to plot
    :param aircraft_lifespan_list: list containing the lifespan at which data were computed.
    """

    fig = go.Figure()

    cumulated_impact = np.zeros_like(aircraft_lifespan_list)

    for contributor, contributor_score in treated_data_dict.items():
        cumulated_impact += np.array(list(contributor_score))
        beautified_impact_score = contributor.replace("_", " ")

        scatter = go.Scatter(
            x=aircraft_lifespan_list,
            y=cumulated_impact,
            name=beautified_impact_score,
            showlegend=True,
            fill="tonexty",
        )
        fig.add_trace(scatter)

    scatter = go.Scatter(
        x=aircraft_lifespan_list,
        y=cumulated_impact,
        name="Single score",
        line=dict(color="black", width=5),
        showlegend=True,
    )
    fig.add_trace(scatter)

    return fig


def _sort_and_cut_off(
    untreated_dict: Dict[str, list], lifespan_list: List[float], cutoff_criteria: float
) -> Dict[str, list]:
    """
    For a lot of figures, we will only plot the most significant contributors. This function sorts
    the contributor from biggest to smallest, cuts off the smallest ones based on the criteria and
    aggregates the others. It also ensures that the data are sorted starting from the one that were
    computed with the smallest lifespan.

    :param untreated_dict: dictionary containing the untreated value, stored as a dict where key
    are the contributor and the items are the evolution of the contributor.
    :param lifespan_list: list containing the untreated list containing the value of the aircraft
    lifespan at which the analysis was conducted.
    :param cutoff_criteria: value of the cutoff criteria, in percent.
    """

    for contributor_name, contributor_value in untreated_dict.items():
        aircraft_lifespan, sorted_impact = zip(*sorted(zip(lifespan_list, contributor_value)))
        untreated_dict[contributor_name] = sorted_impact

    # In order to not overload the diagram, we'll only display a limited number of impacts.
    contributor_score_last_year = []
    contributor_name = []
    aggregated_score_last_year = 0.0
    for contributor, contributor_score in untreated_dict.items():
        contributor_score_last_year.append(contributor_score[-1])
        contributor_name.append(contributor)
        aggregated_score_last_year += contributor_score[-1]

    # Ensure the biggest impacts are inserted first so that they are at the bottom of the graph
    last_output_score, last_output_name = zip(
        *sorted(zip(contributor_score_last_year, contributor_name))
    )

    # This way they should be inserted starting from the biggest down to the smallest up plus the
    # other
    biggest_to_smallest = list(reversed(list(last_output_name)))

    treated_dict = OrderedDict()
    other = np.zeros_like(lifespan_list)
    for contributor in biggest_to_smallest:
        # We only take the biggest one
        contributor_score = untreated_dict[contributor]
        if contributor_score[-1] / aggregated_score_last_year > cutoff_criteria / 100.0:
            treated_dict[contributor] = contributor_score
        else:
            other += np.array(list(contributor_score))

    treated_dict["Others"] = other

    return treated_dict


def _update_fig_axis(fig: go.Figure):
    """
    Utility function that updates the aspect of the axis so that all figures have the same aspect.

    :param fig: figure whose layout is to be updated.
    """
    fig.update_layout(
        plot_bgcolor="white",
        title_x=0.5,
        title_font=dict(size=20),
        legend_font=dict(size=20),
    )
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        title_font=dict(size=20),
        tickfont=dict(size=20),
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        range=[0, None],
        title_font=dict(size=20),
        tickfont=dict(size=20),
        side="right",
    )
    # You may wonder why I set the y-axis to the right, well that's because if it's on the left
    # changing the tick font changes the range !
    # You could try to solve that problem, but if you don't manage to update the counter below:
    # hours_wasted: 1.5


def _get_list_contributing_components_and_variables(
    datafile_path: Union[str, pathlib.Path],
) -> dict:
    """
    Gets a list of variables names that contribute to the single score as well as the component
    they are linked to.
    """

    datafile = oad.DataFile(datafile_path)

    names = datafile.names()
    contributing_components_and_variables = {}

    for name in names:
        if LCA_PREFIX in name and "_weighted:" in name:
            if _depth_lca_detail(name) >= 4:
                component_name = _get_component_from_variable_name(name)
                _safe_add_to_dict_of_list(
                    contributing_components_and_variables, component_name, name
                )
            # This isn't very generic, but I can't find another way to do it, maybe check that there
            # aren't any other subprocesses ?
            # TODO: Update if we add any life phase to the LCA analysis that aren't detailed
            elif "manufacturing:sum" in name:
                _safe_add_to_dict_of_list(
                    contributing_components_and_variables, "manufacturing", name
                )
            elif "distribution:sum" in name:
                _safe_add_to_dict_of_list(
                    contributing_components_and_variables, "distribution", name
                )

    return contributing_components_and_variables


def _safe_add_to_dict_of_list(
    dict_to_update: Dict[str, list], dict_key: str, element_to_add: Union[str, float]
):
    """
    For dictionaries where items are meant to be lists, this function checks if the key exists.
    If it does, it appends to the list, otherwise it creates the lists.

    :param dict_to_update: dictionary in which to add element.
    :param dict_key: dictionary key at which the element is meant to be added.
    :param dict_key: dictionary key at which the element is meant to be added.
    :param element_to_add: element to add to the dictionary.
    """
    if dict_to_update.get(dict_key):
        dict_to_update[dict_key].append(element_to_add)
    else:
        dict_to_update[dict_key] = [element_to_add]


def _get_component_from_variable_name(variable_name: str) -> str:
    """
    Gets the name of the component or process based on the name of the variable it is associated to.
    We will aggregate all process related to the airframe in "airframe". It is possible because
    they are included in the analysis regardless of the propulsion chain.
    """

    component = variable_name.split(":")[4 - _depth_lca_detail(variable_name) - 1]

    # These names are assured to be in the LCA conf file regardless of the propulsion chain.
    airframe_associated_components = [
        "wing",
        "fuselage",
        "horizontal_tail",
        "vertical_tail",
        "landing_gear",
        "flight_controls",
        "assembly",
    ]
    if component in airframe_associated_components:
        return "airframe"

    else:
        return component


def lca_impacts_bar_chart_simple(
    aircraft_file_paths: Union[str, List[str]],
    names_aircraft: Union[str, List[str]] = None,
) -> go.FigureWidget:
    """
    Give a bar chart that compares multiples aircraft designs across all categories. This comparison
    is done relative to the first design given in the inputs. Can be used with only one design but
    is pointless since it will compare an aircraft to itself.

    :param aircraft_file_paths: paths to the output file that contains the impacts.
    :param names_aircraft: names of the aircraft.
    """

    fig = go.Figure()

    reference_value = _get_weighted_impact_dict(aircraft_file_paths[0])
    reference_value.pop("single_score")

    for aircraft_file_path, name_aircraft in zip(aircraft_file_paths, names_aircraft):
        impact_score_dict = _get_weighted_impact_dict(aircraft_file_path)
        impact_score_dict.pop("single_score")
        impact_scores = []
        beautified_impact_names = []

        for impact_name, impact_score in impact_score_dict.items():
            beautified_impact_name = impact_name.replace("_", " ")

            impact_scores.append(impact_score / reference_value[impact_name] * 100.0)
            beautified_impact_names.append(beautified_impact_name)

        bar_chart = go.Bar(name=name_aircraft, x=beautified_impact_names, y=impact_scores)
        fig.add_trace(bar_chart)

    fig.update_layout(
        barmode="group",
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        title_x=0.5,
        title_text="Relative score of "
        + ", ".join(names_aircraft[1:])
        + " with respect to "
        + names_aircraft[0],
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
        title="Relative score [%]",
    )
    fig.update_yaxes(
        title_font=dict(size=20),
    )

    return go.FigureWidget(fig)


def lca_impacts_bar_chart_normalised_weighted(
    aircraft_file_paths: Union[str, List[str]],
    names_aircraft: Union[str, List[str]] = None,
) -> go.FigureWidget:
    """
    Give a bar chart that compares multiples aircraft designs across all categories. This comparison
    is done relative to the first design given in the inputs. Can be used with only one design but
    is pointless since it will compare an aircraft to itself.

    :param aircraft_file_paths: paths to the output file that contains the impacts.
    :param names_aircraft: names of the aircraft.
    """

    fig = go.Figure()

    for aircraft_file_path, name_aircraft in zip(aircraft_file_paths, names_aircraft):
        impact_score_dict = _get_weighted_impact_dict(aircraft_file_path)
        impact_scores = []
        beautified_impact_names = []

        for impact_name, impact_score in impact_score_dict.items():
            beautified_impact_name = impact_name.replace("_", " ")
            impact_scores.append(impact_score)
            beautified_impact_names.append(beautified_impact_name)

        bar_chart = go.Bar(name=name_aircraft, x=beautified_impact_names, y=impact_scores)
        fig.add_trace(bar_chart)

    if len(names_aircraft) == 1:
        title_text = "Normalized and weighted scores for " + names_aircraft[0]
    else:
        title_text = (
            "Normalized and weighted scores for "
            + ", ".join(names_aircraft[0:-1])
            + " and "
            + names_aircraft[-1]
        )

    fig.update_layout(
        barmode="group",
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        title_x=0.5,
        title_text=title_text,
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
        title="Points [-]",
    )
    fig.update_yaxes(
        title_font=dict(size=20),
    )

    return go.FigureWidget(fig)


def _get_component_and_contribution(aircraft_file_path: Union[str, pathlib.Path]) -> dict:
    """
    Returns a dict of the components and their impact in each category. Also return a dict with the
    total value for each impact.

    :param aircraft_file_path: path to the output file path.
    :return: a dict of the components with their contribution to each impact category.
    """

    datafile = oad.DataFile(aircraft_file_path)

    names = datafile.names()
    component_and_impacts = {}

    for name in names:
        # We can focus on the weighted value since it'll be relative anyway
        if LCA_PREFIX in name and "_weighted:" in name:
            if _depth_lca_detail(name) >= 4:
                component_name = _get_component_from_variable_name(name)
                impact_name = name.replace(LCA_PREFIX, "").split("_weighted")[0]
                contribution = datafile[name].value[0]
                if contribution != 0.0:
                    if component_name in component_and_impacts:
                        if impact_name in component_and_impacts[component_name]:
                            component_and_impacts[component_name][impact_name] += contribution
                        else:
                            component_and_impacts[component_name][impact_name] = contribution
                    else:
                        component_and_impacts[component_name] = {impact_name: contribution}

            elif "manufacturing:sum" in name:
                component_name = "manufacturing"
                impact_name = name.replace(LCA_PREFIX, "").split("_weighted")[0]
                contribution = datafile[name].value[0]
                if contribution != 0.0:
                    if component_name in component_and_impacts:
                        # Manufacturing and distribution only contribute once each time
                        component_and_impacts[component_name][impact_name] = contribution
                    else:
                        component_and_impacts[component_name] = {impact_name: contribution}
            elif "distribution:sum" in name:
                component_name = "distribution"
                impact_name = name.replace(LCA_PREFIX, "").split("_weighted")[0]
                contribution = datafile[name].value[0]
                if contribution != 0.0:
                    if component_name in component_and_impacts:
                        # Manufacturing and distribution only contribute once each time
                        component_and_impacts[component_name][impact_name] = contribution
                    else:
                        component_and_impacts[component_name] = {impact_name: contribution}

    return component_and_impacts


def lca_impacts_bar_chart_with_contributors(
    aircraft_file_path: str,
    name_aircraft: str = None,
) -> go.FigureWidget:
    """
    Give a bar chart that plot the impact of an aircraft in each category and how each component
    contributes to it in relative terms.

    :param aircraft_file_path: path to the output file that contains the results of the LCA
    :param name_aircraft: name of the aircraft.
    """

    component_and_contribution = _get_component_and_contribution(aircraft_file_path)

    fig = go.Figure()

    impact_score_dict = _get_weighted_impact_dict(aircraft_file_path)
    impact_score_dict.pop("single_score")

    component_counter = 0

    for component, impacts in component_and_contribution.items():
        impact_contributions = []
        beautified_impact_names = []

        for impact_name, contribution in impacts.items():
            beautified_impact_name = impact_name.replace("_", " ")
            beautified_impact_names.append(beautified_impact_name)

            impact_contributions.append(contribution / impact_score_dict[impact_name] * 100.0)

        bar_chart = go.Bar(
            name=component,
            x=beautified_impact_names,
            y=impact_contributions,
            marker=dict(
                pattern_shape=HASH[component_counter // len(HASH)],
                color=COLS[component_counter % len(COLS)],
            ),
        )
        fig.add_trace(bar_chart)

        component_counter += 1

    title_text = (
        "Relative contribution of each component to each impact category for " + name_aircraft
    )

    fig.update_layout(
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        title_x=0.5,
        title_text=title_text,
        barmode="stack",
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
        title="Relative contribution [%]",
    )
    fig.update_yaxes(
        title_font=dict(size=20),
    )

    return go.FigureWidget(fig)
