# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np

import plotly.graph_objects as go

from fastoad.io import VariableIO


def npv_curve(
    aircraft_file_path: str,
    name: str = "",
    fig=None,
    file_formatter=None,
    period_var: int = 20,
    npv_var: str = "data:cost:production:net_present_value",
    fill_color: str = "rgba(99, 110, 250, 0.3)",
    line_color: str = "rgba(99, 110, 250, 1.0)",
) -> go.FigureWidget:
    """
    Returns a figure plot of the NPV curve (cumulative Net Present Value versus period),
    with the area between the curve and zero shaded. Different designs can be superposed
    by providing an existing fig. Each design can be given a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :param period_var: variable name of the period/year array (x-axis)
    :param npv_var: variable name of the NPV array (y-axis)
    :param fill_color: RGBA color used to shade the area between the curve and zero
    :param line_color: RGBA color used for the curve line and markers
    :return: NPV curve figure.
    """

    variables = VariableIO(aircraft_file_path, file_formatter).read()

    period_array = np.linspace(0, period_var, period_var + 1)
    npv_array = variables[npv_var].value

    if fig is None:
        fig = go.Figure()

    scatter = go.Scatter(
        x=period_array,
        y=npv_array,
        mode="lines+markers",
        name=name,
        showlegend=True,
        fill="tozeroy",
        fillcolor=fill_color,
        line=dict(color=line_color),
    )
    fig.add_trace(scatter)

    # Zero reference line to make the break-even point easy to read
    fig.add_shape(
        type="line",
        x0=min(period_array),
        x1=max(period_array),
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
    )

    fig.layout = go.Layout()

    fig = go.Figure(fig)

    fig.update_layout(
        title_text="NPV Curve",
        title_x=0.5,
        xaxis_title="Period [Years]",
        yaxis_title="NPV [$]",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig
