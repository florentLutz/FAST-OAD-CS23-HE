# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

from scipy.interpolate import griddata

import plotly
import plotly.graph_objects as go

from fastoad.io import VariableIO

from fastga_he.models.performances.payload_range.payload_range_inner import (
    INVALID_COMPUTATION_RESULT,
)


def payload_range_outer(
    aircraft_file_path: str, name="", fig=None, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the outer bounds of the payload range.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :return: wing plot figure.
    """

    variables = VariableIO(aircraft_file_path, file_formatter).read()

    range_array = variables["data:mission:payload_range:range"].value
    payload_array = variables["data:mission:payload_range:payload"].value
    ef_array = variables["data:mission:payload_range:emission_factor"].value

    annotation_array = []
    for ef in ef_array:
        annotation = r"Emission factor (kgCO2/kgPAX/nm): " + str(ef)
        annotation_array.append(annotation)

    if fig is None:
        fig = go.Figure()

    scatter = go.Scatter(
        x=range_array,
        y=payload_array,
        text=annotation_array,
        mode="lines+markers",
        name=name,
        showlegend=True,
    )
    fig.add_trace(scatter)

    fig.layout = go.Layout()

    fig = go.FigureWidget(fig)

    fig.update_layout(
        title_text="Aircraft Payload-Range",
        title_x=0.5,
        xaxis_title="Range [nm]",
        yaxis_title="Payload [kg]",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    fig.update_xaxes(range=[-50, None])

    return fig


def payload_range_inner(
    ref_aircraft_file_path: str,
    sec_aircraft_file_path: str = None,
    file_formatter=None,
    smooth: bool = True,
    grid_accuracy: int = 100,
    z_filter: float = None,
) -> go.FigureWidget:
    """
    Returns a figure plot of the inner values of the payload range.
    Each design can be provided a name.

    :param ref_aircraft_file_path: path of reference aircraft data file
    :param sec_aircraft_file_path: path of secondary aircraft data file which will be compared to
    the reference
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :param smooth: boolean to trigger smoothing of the heatmap
    :param grid_accuracy: an interpolation on a grid of size (grid_accuracy, grid_accuracy) is done
    on the reference ef values from the input
    :param z_filter: filter all values of emission factor above that value or below minus that value
    when comparing aircraft
    :return: wing plot figure.
    """

    ref_variables = VariableIO(ref_aircraft_file_path, file_formatter).read()

    ref_inner_range_array = ref_variables["data:mission:inner_payload_range:range"].value
    ref_inner_payload_array = ref_variables["data:mission:inner_payload_range:payload"].value
    ref_inner_ef_array = ref_variables["data:mission:inner_payload_range:emission_factor"].value

    ref_outer_range_array = ref_variables["data:mission:payload_range:range"].value
    ref_outer_payload_array = ref_variables["data:mission:payload_range:payload"].value

    if sec_aircraft_file_path:
        sec_variables = VariableIO(sec_aircraft_file_path, file_formatter).read()

        sec_inner_range_array = sec_variables["data:mission:inner_payload_range:range"].value
        sec_inner_payload_array = sec_variables["data:mission:inner_payload_range:payload"].value
        sec_inner_ef_array = sec_variables["data:mission:inner_payload_range:emission_factor"].value

        sec_outer_range_array = sec_variables["data:mission:payload_range:range"].value
        sec_outer_payload_array = sec_variables["data:mission:payload_range:payload"].value

    if sec_aircraft_file_path:
        x_for_plot = np.linspace(
            0, max(max(ref_outer_range_array), max(sec_outer_range_array)), grid_accuracy
        )
        y_for_plot = np.linspace(
            0, max(max(ref_outer_payload_array), max(sec_outer_payload_array)), grid_accuracy
        )

        x_mesh, y_mesh = np.meshgrid(x_for_plot, y_for_plot)

        ref_z = griddata(
            (ref_inner_range_array, ref_inner_payload_array),
            ref_inner_ef_array,
            (x_mesh, y_mesh),
            method="linear",
        )
        sec_z = griddata(
            (sec_inner_range_array, sec_inner_payload_array),
            sec_inner_ef_array,
            (x_mesh, y_mesh),
            method="linear",
        )

        z_for_plot = ref_z - sec_z
    else:
        x_for_plot = np.linspace(0, max(ref_outer_range_array), grid_accuracy)
        y_for_plot = np.linspace(0, max(ref_outer_payload_array), grid_accuracy)

        x_mesh, y_mesh = np.meshgrid(x_for_plot, y_for_plot)
        z_for_plot = griddata(
            (ref_inner_range_array, ref_inner_payload_array),
            ref_inner_ef_array,
            (x_mesh, y_mesh),
            method="linear",
        )

    fig = go.Figure()

    z_max = z_filter
    if sec_aircraft_file_path:
        z_min = -z_filter
    else:
        z_min = 0

    if smooth:
        scatter = go.Heatmap(
            y=y_for_plot,
            x=x_for_plot,
            z=z_for_plot,
            colorscale="Viridis",
            zsmooth="best",
            zmin=z_min,
            zmax=z_max,
        )
    else:
        scatter = go.Heatmap(
            y=y_for_plot,
            x=x_for_plot,
            z=z_for_plot,
            colorscale="Viridis",
            zmin=z_min,
            zmax=z_max,
        )

    fig.add_trace(scatter)

    ref_scatter = go.Scatter(
        x=ref_outer_range_array,
        y=ref_outer_payload_array,
        mode="lines+markers",
        name="Reference aircraft",
        showlegend=True,
    )
    fig.add_trace(ref_scatter)

    if sec_aircraft_file_path:
        sec_scatter = go.Scatter(
            x=sec_outer_range_array,
            y=sec_outer_payload_array,
            mode="lines+markers",
            name="Secondary aircraft",
            showlegend=True,
        )
        fig.add_trace(sec_scatter)

    fig = go.FigureWidget(fig)

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))

    return fig
