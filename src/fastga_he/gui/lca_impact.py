# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

import plotly.graph_objects as go

import fastoad.api as oad

from ..models.environmental_impacts.resources.constants import LCA_PREFIX


def lca_impacts_sun_breakdown(aircraft_file_path: str, full_burst: bool = False) -> go.FigureWidget:
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

    # Cause it's the earliest parent ;)
    label_ancestor = (
        "single_score" + "<br> " + str(datafile[LCA_PREFIX + "single_score"].value[0]) + " pt"
    )
    figure_labels = [label_ancestor]
    figure_parents = [""]
    figure_values = [datafile[LCA_PREFIX + "single_score"].value[0]]

    # Not very generic :/ Might only work for EF3.1 to check
    for name in names_variables_lca:
        if depth_lca_detail(name) == 2 and "sum" in name:
            figure_labels.append(name_to_label(name, datafile))
            figure_parents.append(label_ancestor)
            figure_values.append(datafile[name].value[0])
        elif depth_lca_detail(name) == 3 and "sum" in name:
            figure_labels.append(name_to_label(name, datafile))
            figure_parents.append(get_parent_label(name, datafile))
            figure_values.append(datafile[name].value[0])
        elif depth_lca_detail(name) == 4:
            # Sum should not be in the name here.
            figure_labels.append(name_to_label(name, datafile, -1))
            figure_parents.append(get_parent_label(name + ":sum", datafile))
            figure_values.append(datafile[name].value[0])

    fig.add_trace(
        go.Sunburst(
            labels=figure_labels,
            parents=figure_parents,
            values=figure_values,
            branchvalues="total",
        ),
    )
    if full_burst:
        fig.update_traces(maxdepth=max_depth, selector=dict(type="sunburst"))
    else:
        fig.update_traces(maxdepth=2, selector=dict(type="sunburst"))

    fig.update_layout(title_text="Single score breakdown", title_x=0.5)

    fig = go.FigureWidget(fig)

    return fig


def depth_lca_detail(name_variable: str) -> int:
    tmp_name = name_variable.replace(LCA_PREFIX, "")
    depth_lca = len(tmp_name.split(":"))
    if "sum" not in tmp_name:
        depth_lca += 1

    return depth_lca


def name_to_label(name_variable: str, datafile: oad.DataFile, depth: int = -2) -> str:
    clean_name = name_variable.split(":")[depth]
    value = datafile[name_variable].value[0]

    return clean_name + "<br> " + str(value) + " pt"


def get_parent_label(name_variable: str, datafile: oad.DataFile) -> str:
    parent_name = ":".join(name_variable.split(":")[:-2]) + ":sum"

    return name_to_label(parent_name, datafile)
