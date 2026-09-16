# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

from typing import List, Optional, Sequence, Union

import numpy as np

import plotly.graph_objects as go
import plotly.colors as pc

from fastoad.io import VariableIO


def npv_curve(
    aircraft_file_path: Union[str, Sequence[str]],
    name: Union[str, Sequence[str]] = "",
    fig=None,
    file_formatter=None,
    period_var: int = 20,
    npv_var: str = "data:cost:production:net_present_value",
    fill_color: Optional[Union[str, Sequence[str]]] = None,
    line_color: Optional[Union[str, Sequence[str]]] = None,
) -> go.FigureWidget:
    """
    Returns a figure plot of the NPV curve (cumulative Net Present Value versus period),
    with the area between the curve and zero shaded. Multiple designs can be plotted at
    once by passing a list of file paths (and, optionally, a matching list of names/colors),
    or superposed one at a time by calling this function repeatedly with an existing fig.

    :param aircraft_file_path: path of data file, or a list/tuple of paths to plot several
    designs on the same figure
    :param name: name (or list of names, matching aircraft_file_path) to give to the trace(s)
    added to the figure
    :param fig: existing figure to which add the plot(s)
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :param period_var: variable name of the period/year array (x-axis)
    :param npv_var: variable name of the NPV array (y-axis)
    :param fill_color: RGBA color (or list of colors, matching aircraft_file_path) used to shade
    the area between each curve and zero. If not provided, colors are auto-assigned from the
    default Plotly qualitative palette.
    :param line_color: RGBA color (or list of colors, matching aircraft_file_path) used for each
    curve's line and markers. If not provided, colors are auto-assigned from the default
    Plotly qualitative palette.
    :return: NPV curve figure.
    """

    # Normalize inputs to lists so a single file behaves exactly like before, while a
    # list/tuple of files plots one trace per file on the same figure.
    if isinstance(aircraft_file_path, str):
        file_paths = [aircraft_file_path]
    else:
        file_paths = list(aircraft_file_path)

    n_curves = len(file_paths)

    if isinstance(name, str):
        # A single string is used as-is for a single file, or reused as a common prefix/label
        # (left empty by default) for each trace when multiple files are given.
        names = [name] * n_curves
    else:
        names = list(name)
        if len(names) != n_curves:
            raise ValueError("Length of 'name' must match length of 'aircraft_file_path'.")

    fill_colors = _resolve_colors(fill_color, n_curves, alpha=0.3)
    line_colors = _resolve_colors(line_color, n_curves, alpha=1.0)

    if fig is None:
        fig = go.Figure()

    period_array = np.linspace(0, period_var, period_var + 1)

    for file_path, curve_name, f_color, l_color in zip(file_paths, names, fill_colors, line_colors):
        variables = VariableIO(file_path, file_formatter).read()
        npv_array = variables[npv_var].value

        scatter = go.Scatter(
            x=period_array,
            y=npv_array,
            mode="lines+markers",
            name=curve_name,
            showlegend=True,
            fill="tozeroy",
            fillcolor=f_color,
            line=dict(color=l_color),
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


def _resolve_colors(
    color: Optional[Union[str, Sequence[str]]], n_curves: int, alpha: float
) -> List[str]:
    """
    Resolves the fill/line color argument into a list of length n_curves. If no color is
    provided, colors are auto-assigned from the default Plotly qualitative palette (converted
    to rgba with the given alpha). If a single color string is provided, it is reused for
    every curve. If a list is provided, its length must match n_curves.
    """

    if color is None:
        palette = pc.qualitative.Plotly
        colors = []
        for i in range(n_curves):
            hex_color = palette[i % len(palette)]
            r, g, b = pc.hex_to_rgb(hex_color)
            colors.append(f"rgba({r}, {g}, {b}, {alpha})")
        return colors

    if isinstance(color, str):
        return [color] * n_curves

    colors = list(color)
    if len(colors) != n_curves:
        raise ValueError("Length of color list must match length of 'aircraft_file_path'.")
    return colors
