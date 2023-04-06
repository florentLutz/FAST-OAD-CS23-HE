# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import plotly.graph_objects as go
import plotly.express as px

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

COLOR_ARRAY = px.colors.qualitative.Prism


def residuals_viewer(
    recorder_data_file_path: str,
    case: str,
    power_train_file_path: str,
    what_to_plot: str = "residuals",
) -> go.Figure:
    """
    Creates a plot with all the interesting residuals as defined in the registered_components file.

    :param recorder_data_file_path: path to the sql file that contains the recorder
    :param case: string that contain the name of the case to open, will correspond to root: + the
    path to the nonlinear_solver to the DEPEquilibrium group.
    :param power_train_file_path: path to the powertrain file
    :param what_to_plot: what value to plot, can be "residuals" or "outputs"
    """
    fig = go.Figure()

    configurator = FASTGAHEPowerTrainConfigurator()
    configurator.load(power_train_file_path)
    components_name, residuals_name = configurator.get_residuals_watcher_elements_list()

    cr = om.CaseReader(recorder_data_file_path)

    solver_case = cr.get_cases(case)

    residuals_mean = {}
    residuals_min = {}
    residuals_max = {}

    for component_name, residual_name in zip(components_name, residuals_name):
        residuals_mean[component_name + "." + residual_name] = []
        residuals_min[component_name + "." + residual_name] = []
        residuals_max[component_name + "." + residual_name] = []

    for case in solver_case:

        # Because I'm 99% sure that the key cycling will always be the same in all three dict,
        # we will just do one for loop

        if what_to_plot == "residuals":
            data_what_to_plot = case.residuals
        else:
            data_what_to_plot = case.outputs

        for key in residuals_mean:
            residuals_mean[key].append(np.mean(data_what_to_plot[key]))
            residuals_min[key].append(np.min(data_what_to_plot[key]))
            residuals_max[key].append(np.max(data_what_to_plot[key]))

    color_selector = 0

    for component_name, residual_name in zip(components_name, residuals_name):

        color = COLOR_ARRAY[color_selector]
        if color_selector < len(COLOR_ARRAY) - 1:
            color_selector += 1
        else:
            color_selector = 0

        # Add graph, for avg, min and max
        data_mean = residuals_mean[component_name + "." + residual_name]
        data_min = residuals_min[component_name + "." + residual_name]
        data_max = residuals_max[component_name + "." + residual_name]
        data_x = np.arange(len(data_mean))

        scatter_mean = go.Scatter(
            x=data_x,
            y=data_mean,
            mode="lines+markers",
            name=component_name + "." + residual_name + " : Mean residuals for variable",
            legendgroup=component_name + "." + residual_name,
            legendgrouptitle_text=component_name + "." + residual_name,
            marker=dict(color=color, symbol="circle", size=10),
        )
        fig.add_trace(scatter_mean)
        scatter_min = go.Scatter(
            x=data_x,
            y=data_min,
            mode="lines+markers",
            name=component_name + "." + residual_name + " : Min residuals for variable",
            legendgroup=component_name + "." + residual_name,
            marker=dict(symbol="triangle-up", color=color, size=10),
        )
        fig.add_trace(scatter_min)
        scatter_max = go.Scatter(
            x=data_x,
            y=data_max,
            mode="lines+markers",
            name=component_name + "." + residual_name + " : Max residuals for variable",
            legendgroup=component_name + "." + residual_name,
            marker=dict(symbol="triangle-down", color=color, size=10),
        )
        fig.add_trace(scatter_max)

    return fig
