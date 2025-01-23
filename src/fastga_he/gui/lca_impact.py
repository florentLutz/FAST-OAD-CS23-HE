# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import os.path as pth
import pathlib

from typing import List, Union

import numpy as np

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import fastoad.api as oad

from ..models.environmental_impacts.resources.constants import LCA_PREFIX

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS


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


def _get_impact_sunburst(aircraft_file_path: str, rel: str = "absolute") -> go.Sunburst:
    datafile = oad.DataFile(aircraft_file_path)
    names = datafile.names()
    names_variables_lca = []

    for name in names:
        if LCA_PREFIX in name:
            if "weighted" in name or "single_score" in name:
                names_variables_lca.append(name)

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
    if first_parent in list(color_dict.keys()):
        return color_dict[first_parent]
    else:
        color = COLS[len(list(color_dict.keys())) % len(COLS)]
        color_dict[name_variable] = color
        return color


def lca_score_sensitivity_simple(
    results_folder_path: Union[str, pathlib.Path],
    prefix: str,
    name: str = None,
    fig: go.Figure = None,
) -> go.Figure:
    """
    Displays the evolution of the impacts of an aircraft with respect to its lifespan. This
    method is a bit sensitive to use as it requires the results to be stored under the form of
    FAST-OAD output files, all in the same folder and all with the same prefix. It also requires
    the user to know and input said prefix. Results can be superimposed to an existing figure but
    it is recommended to only put results computed on the same lifespan.

    :param results_folder_path: path to the folder that contains the output files that contains
    the results.
    :param prefix: prefix of the output file for the aircraft.
    :param name: name of the aircraft, to be displayed on the figure.
    :param fig: figure with existing results.

    :return: plotly figure with the evolution of the impact as a function of the lifespan.
    """

    aircraft_lifespan_list = []
    impact_list = []

    for dirpath, _, filenames in os.walk(results_folder_path):
        for filename in filenames:
            if filename.startswith(prefix):
                datafile = oad.DataFile(os.path.join(dirpath, filename))
                aircraft_lifespan = datafile["data:TLAR:max_airframe_hours"].value[0]
                aircraft_lifespan_list.append(aircraft_lifespan)
                impact_score = datafile["data:environmental_impact:single_score"].value[0]
                impact_list.append(impact_score)

    aircraft_lifespan_list, impact_list = zip(*sorted(zip(aircraft_lifespan_list, impact_list)))

    if fig is None:
        orig_fig = True
        fig = go.Figure()
    else:
        orig_fig = False

    scatter = go.Scatter(x=aircraft_lifespan_list, y=impact_list, name=name, showlegend=True)

    fig.add_trace(scatter)
    if orig_fig:
        fig.update_layout(
            plot_bgcolor="white",
            title_x=0.5,
            title_text="Evolution of the single score with life expectancy of the aircraft",
            title_font=dict(size=20),
            legend_font=dict(size=20),
        )
        fig.update_xaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
            gridcolor="lightgrey",
            title="Airframe hours [h]",
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
            title="Single score [-]",
            title_font=dict(size=20),
            tickfont=dict(size=20),
            side="right",
        )
        # You may wonder why I set the y-axis to the right, well that's because if it's on the left
        # changing the tick font changes the range !
        # You could try to solve that problem, but if you don't manage to update the counter below:
        # hours_wasted: 1.5

    return fig
