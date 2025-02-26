# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
"""
This regression model is construct based on the data provided by :cite:`hoogendoorn:2018`.
The power density of a pure PEMFC stack is utilized in dimension calculation.
"""

import numpy as np
import plotly.graph_objects as go
from scipy import optimize


# Linear fit function
def linear_func(x, m, b):
    return m * x + b


if __name__ == "__main__":
    # Data points from the graph
    x = np.array([0.7, 2.7, 10.5])
    y = np.array([486, 655, 1323])

    # Fit linear regression
    linear_params, _ = optimize.curve_fit(linear_func, x, y)

    # Calculate fitted y values and R-squared
    y_linear = linear_func(x, *linear_params)
    r2_linear = 1 - np.sum((y - y_linear) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Generate points for smooth curve
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = linear_func(x_smooth, *linear_params)

    # Create interactive plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", name="Data points", marker=dict(color="blue"))
    )
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name=f"Linear (R² = {r2_linear:.3f})",
            line=dict(color="red"),
        )
    )

    # Add function text to the plot
    function_text = f"y = {linear_params[0]:.4f}x + {linear_params[1]:.4f}"
    fig.add_annotation(
        x=max(x),
        y=min(y),
        text=function_text,
        showarrow=False,
        font=dict(size=12, color="blue"),
        xanchor="right",
        yanchor="bottom",
    )

    fig.update_layout(title="Linear Regression", xaxis_title="X", yaxis_title="Y")
    fig.show()

    print(f"Linear: y = {linear_params[0]:.4f}x + {linear_params[1]:.4f}")
