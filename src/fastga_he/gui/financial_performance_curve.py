# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

from typing import List, Optional, Sequence

import pathlib
import numpy as np
import plotly
import plotly.graph_objects as go

from fastoad.io import VariableIO

from .lca_impact import _update_fig_axis


def financial_performance_curve(
    aircraft_file_path: str | pathlib.Path | list[str | pathlib.Path],
    name: str | list[str] = "",
    fig=None,
    file_formatter=None,
    period: int = 20,
    zero_reference_value: Optional[float] = 0.0,
    discipline: str | Sequence[str] = "production",
    y_variable: str | Sequence[str] = "net_present_value",
    y_axis_label: str | Sequence[str] = "NPV [$]",
    line_color: Optional[str | list[str]] = None,
    fill: bool = True,
) -> go.FigureWidget:
    """
    Plots a financial performance curve against time (years), with an optional secondary
    y-axis for another quantity (e.g. aircraft delivered). The area between the main curve and
    zero can be shaded. Several designs can be plotted at once by passing a list of file paths
    (with matching names/colors), or added one by one by passing the returned ``fig`` back in.

    The x-axis is the period array (0, 1, ..., period). At most two y variables are plotted, in
    the order given in ``y_variable``: the first goes on the main (left) axis, the second on the
    secondary (right) axis.

    :param aircraft_file_path: path of the data file, or list of paths for several designs
    :param name: trace name, as one for all designs or one per design; a shorter list is padded
    with empty names
    :param fig: existing figure to add the plot(s) to
    :param file_formatter: formatter of the data file; default format if not provided
    :param period: number of periods (years) on the x-axis
    :param zero_reference_value: y value of the horizontal reference line (e.g. break-even) on
    the main axis; None for no line
    :param discipline: life cycle cost discipline (production or operation), as one value for
    all y variables or one per y variable; only used for short names in y_variable
    :param y_variable: one or two variables to plot. A short name (no ":"), e.g.
    "net_present_value", is a financial parameter read as
    ``data:cost:{discipline}:{name}``; a full name, e.g.
    "data:cost:production:aircraft_delivered", is read as is
    :param y_axis_label: y-axis label(s), in the same order as the plotted variables (main
    axis, then secondary axis). A single string labels the main axis only; a list must have one
    label per plotted variable
    :param line_color: RGBA line and marker color, as one for all designs or one per design (the
    secondary curve is dashed in the same color); designs without a color (None, or past the end
    of a shorter list) get the next color of the Plotly palette, based on the number of designs
    already in ``fig``
    :param fill: whether to shade the area between each main curve and zero, in the Plotly
    palette at 0.3 opacity
    :return: financial performance curve figure
    """

    y_variable_names = _resolve_y_variables(discipline, y_variable)
    main_variable = y_variable_names[0]
    secondary_variable = y_variable_names[1] if len(y_variable_names) > 1 else None

    secondary_axis_label = ""
    if isinstance(y_axis_label, str):
        main_axis_label = y_axis_label
    elif secondary_variable is None:
        (main_axis_label,) = y_axis_label
    else:
        main_axis_label, secondary_axis_label = y_axis_label

    if fig is None:
        fig = go.Figure()

    if isinstance(aircraft_file_path, (str, pathlib.Path)):
        aircraft_file_path = [aircraft_file_path]
    n_designs = len(aircraft_file_path)

    # Lists shorter than the file list are padded: empty names, default palette colors (None)
    names = _per_design(name, n_designs, pad_value="")
    line_colors = _per_design(line_color, n_designs, pad_value=None)

    period_array = np.linspace(0, period, period + 1)

    for file_path, design_name, design_line_color in zip(aircraft_file_path, names, line_colors):
        # Pick the next palette color, so that each design added to the same fig gets its own
        # color. Designs are counted by their main-axis traces.
        design_index = sum(1 for trace in fig.data if trace.yaxis != "y2")
        fill_color = _palette_color(design_index, alpha=0.3)
        if design_line_color is None:
            design_line_color = _palette_color(design_index, alpha=1.0)

        variables = VariableIO(file_path, file_formatter).read()

        main_array = _read_array(variables, main_variable, period_array, file_path)
        fig.add_trace(
            go.Scatter(
                x=period_array,
                y=main_array,
                mode="lines+markers",
                name=design_name,
                showlegend=True,
                fill="tozeroy" if fill else None,
                fillcolor=fill_color if fill else None,
                line=dict(color=design_line_color),
            )
        )

        if secondary_variable is not None:
            secondary_array = _read_array(variables, secondary_variable, period_array, file_path)
            secondary_name = (
                " - ".join(part for part in (design_name, secondary_axis_label) if part)
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
                    line=dict(color=design_line_color, dash="dash"),
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
        xaxis_title="Period [Years]",
        yaxis_title=main_axis_label,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    if secondary_variable is not None:
        fig.update_layout(
            yaxis2=dict(
                title=secondary_axis_label,
                overlaying="y",
                showgrid=False,
            )
        )

    _update_fig_axis(fig)

    # _update_fig_axis starts every y-axis at 0 and puts it on the right. The main axis must go
    # below 0 (NPV is negative before break-even) and stay on the left of the secondary axis.
    fig.update_layout(yaxis=dict(range=None, side="left"))

    return fig


def _resolve_y_variables(
    discipline: str | Sequence[str],
    y_variable: str | Sequence[str],
) -> List[str]:
    """
    Builds the ordered list of full y variable names. Short names (no ":") are financial
    parameters, expanded to ``data:cost:{discipline}:{name}``; full names are kept as is. The
    first is the main-axis variable, the second the secondary-axis variable.
    """

    variables = [y_variable] if isinstance(y_variable, str) else y_variable

    if isinstance(discipline, str):
        disciplines = [discipline] * len(variables)
    else:
        disciplines = discipline
        if len(disciplines) != len(variables):
            raise ValueError("Length of 'discipline' must match length of 'y_variable'.")

    names = [
        variable if ":" in variable else f"data:cost:{d}:{variable}"
        for d, variable in zip(disciplines, variables)
    ]

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
        zero_deliveries = np.zeros(len(period_array) - len(array))
        array = np.append(zero_deliveries, array)

    if len(array) != len(period_array):
        raise ValueError(
            f"Variable '{variable_name}' in '{file_path}' has {len(array)} values but the period "
            f"array has {len(period_array)}. Please double check the period argument matches the "
            f"length of '{variable_name}'."
        )
    return array


def _per_design(value, n_designs: int, pad_value) -> list:
    """
    Turns a per-design argument into a list of length n_designs: a single value (or None) is
    reused for every design, a shorter list is padded with pad_value.
    """

    if value is None or isinstance(value, str):
        return [value] * n_designs
    return list(value) + [pad_value] * (n_designs - len(value))


def _palette_color(index: int, alpha: float) -> str:
    """
    Returns the index-th color of the default Plotly qualitative palette (cycling if needed),
    as an rgba string with the given alpha.
    """

    palette = plotly.colors.qualitative.Plotly
    r, g, b = plotly.colors.hex_to_rgb(palette[index % len(palette)])
    return f"rgba({r}, {g}, {b}, {alpha})"
