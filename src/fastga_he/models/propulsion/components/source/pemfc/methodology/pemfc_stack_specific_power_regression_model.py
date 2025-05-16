# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
"""
This regression model is construct based on the data provided by :cite:`hoogendoorn:2018`.
The specific power of a pure PEMFC stack is utilized in weight calculation.
"""

import numpy as np
import plotly.graph_objects as go

from scipy import optimize


# Logarithmic fit
def log_func(x, a, b):
    return a * np.log(x) + b


if __name__ == "__main__":
    # Data points from the graph
    x = np.array([0.012, 0.32, 0.7, 2.4, 2.7])
    y = np.array([0.307692308, 1.684210526, 1.129032258, 1.684, 2.061068702])

    log_params, _ = optimize.curve_fit(log_func, x, y)

    y_log = log_func(x, *log_params)
    r2_log = 1 - np.sum((y - y_log) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Generate points for smooth curves
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = log_func(x_smooth, *log_params)

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
            name=f"Logarithmic (R² = {r2_log:.3f})",
            line=dict(color="red"),
        )
    )

    # Add function text to the plot
    function_text = f"y = {log_params[0]:.4f}ln(x) + {log_params[1]:.4f}"
    fig.add_annotation(
        x=max(x),
        y=min(y),
        text=function_text,
        showarrow=False,
        font=dict(size=12, color="blue"),
        xanchor="right",
        yanchor="bottom",
    )

    fig.update_layout(
        title="Logarithmic Regression",
        xaxis_title="Power max (kW)",
        yaxis_title="Specific power (kW/kg)",
    )
    fig.show()

    print(f"Logarithmic: y = {log_params[0]:.4f}ln(x) + {log_params[1]:.4f}")
