# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

import plotly
import plotly.graph_objects as go

from fastoad.io import VariableIO


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
