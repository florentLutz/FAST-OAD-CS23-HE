# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

from typing import List, Optional, Sequence, Union

import numpy as np

import plotly.graph_objects as go
import plotly.colors as pc

from fastoad.io import VariableIO


def financial_performance_curve(
    aircraft_file_path: Union[str, Sequence[str]],
    name: Union[str, Sequence[str]] = "",
    plot_title: str = None,
    fig=None,
    file_formatter=None,
    period: int = 20,
    zero_reference_value: Optional[float] = 0.0,
    discipline: Union[str, Sequence[str]] = "production",
    financial_parameter: Optional[Union[str, Sequence[str]]] = "net_present_value",
    financial_parameter_axis_label: str = "NPV [$]",
    y_variable: Optional[Union[str, Sequence[str]]] = None,
    y2_axis_label: str = "",
    fill_color: Optional[Union[str, Sequence[str]]] = None,
    line_color: Optional[Union[str, Sequence[str]]] = None,
    fill: bool = True,
) -> go.Figure:
    """
    Returns a figure plot of a financial performance curve against time periods (years), with
    an optional second y-axis for any other quantity (financial or not, e.g. number of aircraft
    delivered). The area between the main curve and zero is optionally shaded. Multiple designs
    can be plotted at once by passing a list of file paths (and, optionally, a matching list of
    names/colors), or superposed one at a time by calling this function repeatedly with an
    existing fig.

    The x-axis is always the time period array (0, 1, ..., period).

    Y variables are collected in this order and at most two are allowed:
    1. the variable(s) defined by ``discipline`` and ``financial_parameter``, i.e.
       ``data:cost:{discipline}:{financial_parameter}``,
    2. the variable(s) given by full name in ``y_variable``.
    The first one is plotted on the main (left) y-axis, the second one on the secondary (right)
    y-axis. So a financial parameter always takes the main axis over a ``y_variable``, and when
    both are defined the same way (two financial parameters, or two ``y_variable`` names with
    ``financial_parameter=None``), the order given by the user decides.

    :param aircraft_file_path: path of data file, or a list/tuple of paths to plot several
    designs on the same figure
    :param name: name (or list of names, matching aircraft_file_path) to give to the trace(s)
    added to the figure
    :param plot_title: title of the figure
    :param fig: existing figure to which add the plot(s)
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :param period: number of periods (years) of the x-axis (0, 1, ..., period)
    :param zero_reference_value: value of the horizontal reference line (e.g. break-even) on the
    main y-axis. Set to None to draw no reference line.
    :param discipline: discipline of the life cycle cost (production or operation). A single
    value is reused for every financial parameter, or give one per financial parameter.
    :param financial_parameter: name (or list of names) of the financial parameter(s). Set to
    None to define the y variables only with y_variable.
    :param financial_parameter_axis_label: label of the main y-axis
    :param y_variable: full variable name (or list of names) of extra quantity(ies) to plot, e.g.
    "data:cost:production:aircraft_delivered". Usually plotted on the secondary y-axis.
    :param y2_axis_label: label of the secondary y-axis
    :param fill_color: RGBA color (or list of colors, matching aircraft_file_path) used to shade
    the area between each main curve and zero. If not provided, colors are auto-assigned from
    the default Plotly qualitative palette.
    :param line_color: RGBA color (or list of colors, matching aircraft_file_path) used for each
    design's lines and markers (the secondary curve is dashed with the same color). If not
    provided, colors are auto-assigned from the default Plotly qualitative palette.
    :param fill: whether to shade the area between each main curve and zero
    :return: financial performance curve figure.
    """

    y_variable_names = _resolve_y_variables(discipline, financial_parameter, y_variable)
    main_variable = y_variable_names[0]
    secondary_variable = y_variable_names[1] if len(y_variable_names) > 1 else None

    # Normalize inputs to lists so a single file behaves like a list of one file.
    if isinstance(aircraft_file_path, str):
        file_paths = [aircraft_file_path]
    else:
        file_paths = list(aircraft_file_path)

    n_curves = len(file_paths)

    if isinstance(name, str):
        names = [name] * n_curves
    else:
        names = list(name)
        if len(names) != n_curves:
            raise ValueError("Length of 'name' must match length of 'aircraft_file_path'.")

    fill_colors = _resolve_colors(fill_color, n_curves, alpha=0.3)
    line_colors = _resolve_colors(line_color, n_curves, alpha=1.0)

    if fig is None:
        fig = go.Figure()

    period_array = np.linspace(0, period, period + 1)

    for file_path, curve_name, f_color, l_color in zip(file_paths, names, fill_colors, line_colors):
        variables = VariableIO(file_path, file_formatter).read()

        main_array = _read_array(variables, main_variable, period_array, file_path)
        fig.add_trace(
            go.Scatter(
                x=period_array,
                y=main_array,
                mode="lines+markers",
                name=curve_name,
                showlegend=True,
                fill="tozeroy" if fill else None,
                fillcolor=f_color if fill else None,
                line=dict(color=l_color),
            )
        )

        if secondary_variable is not None:
            secondary_array = _read_array(variables, secondary_variable, period_array, file_path)
            secondary_name = (
                " - ".join(part for part in (curve_name, y2_axis_label) if part)
                or secondary_variable
            )
            fig.add_trace(
                go.Scatter(
                    x=period_array,
                    y=secondary_array,
                    mode="lines+markers",
                    name=secondary_name,
                    showlegend=True,
                    yaxis="y2",
                    line=dict(color=l_color, dash="dash"),
                    marker=dict(symbol="diamond"),
                )
            )

    # Horizontal reference line (e.g. break-even) on the main y-axis
    if zero_reference_value is not None:
        fig.add_hline(
            y=zero_reference_value,
            line=dict(color="black", width=1, dash="dash"),
        )

    fig.update_layout(
        title_text=plot_title,
        title_x=0.5,
        xaxis_title="Period [Years]",
        yaxis_title=financial_parameter_axis_label,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    if secondary_variable is not None:
        fig.update_layout(
            yaxis2=dict(
                title=y2_axis_label,
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )

    return fig


def _resolve_y_variables(
    discipline: Union[str, Sequence[str]],
    financial_parameter: Optional[Union[str, Sequence[str]]],
    y_variable: Optional[Union[str, Sequence[str]]],
) -> List[str]:
    """
    Builds the ordered list of y variable names: financial parameters first (in the order
    given), then full-name y variables (in the order given). The first is the main-axis variable,
    the second the secondary-axis variable.
    """

    names = []

    if financial_parameter is not None:
        parameters = (
            [financial_parameter]
            if isinstance(financial_parameter, str)
            else list(financial_parameter)
        )
        if isinstance(discipline, str):
            disciplines = [discipline] * len(parameters)
        else:
            disciplines = list(discipline)
            if len(disciplines) != len(parameters):
                raise ValueError(
                    "Length of 'discipline' must match length of 'financial_parameter'."
                )
        names += [f"data:cost:{d}:{p}" for d, p in zip(disciplines, parameters)]

    if y_variable is not None:
        names += [y_variable] if isinstance(y_variable, str) else list(y_variable)

    if not names:
        raise ValueError("At least one y variable must be defined.")
    if len(names) > 2:
        raise ValueError(
            f"At most two y variables can be plotted (main and secondary axis), got {len(names)}."
        )

    return names


def _read_array(variables, variable_name: str, period_array: np.ndarray, file_path: str):
    """
    Reads a variable as a flat array and checks that it matches the time period array.
    """

    array = np.asarray(variables[variable_name].value, dtype=float).flatten()

    # Fill with zeros if the variable is delivery_count related, since there is no delivery
    # during development phase
    if "delivery_count" in variable_name:
        zero_deliveries = np.zeros(period_array.shape[0] - array.shape[0])
        array = np.append(zero_deliveries, array)

    if len(array) != len(period_array):
        raise ValueError(
            f"Variable '{variable_name}' in '{file_path}' has {len(array)} values but the period "
            f"array has {len(period_array)}. Check period."
        )
    return array


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
