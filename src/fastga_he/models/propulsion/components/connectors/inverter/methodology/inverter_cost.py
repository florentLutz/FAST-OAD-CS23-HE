# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import plotly.graph_objects as go
from scipy import optimize

# Data points
x = np.array([150, 140, 220, 150, 250, 340, 0.25, 0.5])
y = np.array([9728.25, 5100, 5176.41, 7976, 14663, 6386.31, 3545, 5165])


# Power-law function: y = a * x^b
def power_func(x, a, b):
    return a * np.power(x, b)


if __name__ == "__main__":
    # Fit the power-law model
    power_params, power_covariance = optimize.curve_fit(power_func, x, y, maxfev=100)

    y_power = power_func(x, *power_params)
    r2_power = 1 - np.sum((y - y_power) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Smooth curve for plotting
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = power_func(x_smooth, *power_params)

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
            name=f"Power-law (R² = {r2_power:.3f})",
            line=dict(color="green"),
        )
    )

    # Add function text to the plot
    function_text = f"y = {power_params[0]:.4f} * x^{power_params[1]:.4f}"
    fig.add_annotation(
        x=max(x),
        y=min(y),
        text=function_text,
        showarrow=False,
        font=dict(size=12, color="green"),
        xanchor="right",
        yanchor="bottom",
    )

    fig.update_layout(title="Power-Law Regression", xaxis_title="Power (kW)", yaxis_title="USD")
    fig.show()

    print(f"Power-law: y = {power_params[0]:.4f} * x^{power_params[1]:.4f}")
