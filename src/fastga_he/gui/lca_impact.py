# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

import plotly
import plotly.graph_objects as go

import fastoad.api as oad

from ..models.environmental_impacts.resources.constants import LCA_PREFIX

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS


def lca_impacts_sun_breakdown(
    aircraft_file_path: str,
    full_burst: bool = False,
    name_aircraft: str = None,
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
    datafile = oad.DataFile(aircraft_file_path)
    names = datafile.names()
    names_variables_lca = []

    fig = go.Figure()

    for name in names:
        if LCA_PREFIX in name:
            if "weighted" in name or "single_score" in name:
                names_variables_lca.append(name)

    if len(names_variables_lca) == 0:
        fig = go.FigureWidget(fig)
        return fig

    max_depth = 0
    for name in names_variables_lca:
        curr_depth = depth_lca_detail(name)
        if curr_depth > max_depth:
            max_depth = curr_depth

    # Because it's the earliest parent ;)
    label_ancestor = get_ancestor_label(datafile)

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
        figure_labels.append(name_to_label(name, datafile, rel=rel))
        figure_parents.append(get_parent_label(name, datafile, rel=rel))
        if rel == "single_score" or rel == "parent":
            figure_values.append(
                datafile[name].value[0] / datafile[LCA_PREFIX + "single_score"].value[0] * 100.0
            )
        else:
            figure_values.append(datafile[name].value[0])
        figure_color.append(get_color(name, color_dict))

    fig.add_trace(
        go.Sunburst(
            labels=figure_labels,
            parents=figure_parents,
            values=figure_values,
            branchvalues="total",
            sort=False,
            marker={"colors": figure_color}
        ),
    )
    if full_burst:
        fig.update_traces(maxdepth=max_depth, selector=dict(type="sunburst"))
    else:
        fig.update_traces(maxdepth=2, selector=dict(type="sunburst"))

    title_text = "Single score breakdown"
    if rel == "single_score":
        title_text += "<br>expressed as a percentage of the total score"
    elif rel == "parent":
        title_text += "<br>expressed as a percentage of parent category"

    if name_aircraft:
        fig.update_layout(title_text=name_aircraft + " " + title_text, title_x=0.5)
    else:
        fig.update_layout(title_text=title_text, title_x=0.5)

    fig = go.FigureWidget(fig)

    return fig


def depth_lca_detail(name_variable: str) -> int:
    if "single_score" in name_variable:
        return 1

    tmp_name = name_variable.replace(LCA_PREFIX, "")
    depth_lca = len(tmp_name.split(":"))
    if "sum" not in tmp_name:
        depth_lca += 1

    return depth_lca


def name_to_label(name_variable: str, datafile: oad.DataFile, rel: str = "absolute") -> str:
    if name_variable == LCA_PREFIX + "single_score":
        return get_ancestor_label(datafile)

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
        label = clean_name + "<br> " + str(round_value(value)) + " %"
    elif rel == "parent":
        parent_value = get_parent_score(name_variable, datafile)
        value = datafile[name_variable].value[0] / parent_value * 100.0
        label = clean_name + "<br> " + str(round_value(value)) + " %"
    else:
        value = datafile[name_variable].value[0]
        label = clean_name + "<br> " + str(round_value(value)) + " pt"

    return label


def get_parent_label(name_variable: str, datafile: oad.DataFile, rel: str = "absolute") -> str:
    parent_name = get_parent_name(name_variable)

    return name_to_label(parent_name, datafile, rel=rel)


def get_parent_score(name_variable: str, datafile: oad.DataFile) -> float:
    parent_name = get_parent_name(name_variable)
    parent_score = datafile[parent_name].value[0]

    return parent_score


def get_parent_name(name_variable: str) -> str:
    if depth_lca_detail(name_variable) == 2:
        return LCA_PREFIX + "single_score"

    if "sum" not in name_variable:
        parent_name = ":".join(name_variable.split(":")[:-1]) + ":sum"
    else:
        parent_name = ":".join(name_variable.split(":")[:-2]) + ":sum"

    return parent_name


def get_ancestor_label(datafile: oad.DataFile) -> str:
    return (
        "single_score <br> "
        + str(round_value(datafile[LCA_PREFIX + "single_score"].value[0]))
        + " pt"
    )


def round_value(value: float) -> float:
    if value == 0.0:
        return value
    else:
        return round(value, int(np.ceil(abs(np.log10(value))) + 5))


def get_first_parent_name(name_variable: str) -> str:
    if depth_lca_detail(name_variable) <= 2:
        return name_variable
    else:
        return get_first_parent_name(get_parent_name(name_variable))


def get_color(name_variable: str, color_dict: dict) -> str:

    first_parent = get_first_parent_name(name_variable)
    if first_parent in list(color_dict.keys()):
        return color_dict[first_parent]
    else:
        color = COLS[len(list(color_dict.keys())) % len(COLS)]
        color_dict[name_variable] = color
        return color

