# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import plotly.graph_objects as go
from scipy import optimize


# Logarithmic fit function
def logarithmic_func(x, a, b):
    return a + b * np.log(x + 1)  # Added +1 to handle x=0


if __name__ == "__main__":
    # Data points from the provided table
    x = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
        ]
    )
    y = np.array(
        [
            1,
            0.96,
            0.92,
            0.91,
            0.78,
            0.75,
            0.73,
            0.7,
            0.68,
            0.67,
            0.66,
            0.65,
            0.63,
            0.62,
            0.61,
            0.6,
            0.59,
            0.58,
            0.57,
            0.56,
            0.55,
            0.54,
            0.53,
            0.52,
            0.51,
            0.5,
            0.49,
            0.48,
            0.47,
        ]
    )

    # Fit logarithmic regression
    # Modified function to handle x=0 by adding 1 to all x values before taking log
    logarithmic_params, pcov = optimize.curve_fit(logarithmic_func, x, y)

    # Calculate fitted y values and R-squared
    y_logarithmic = logarithmic_func(x, *logarithmic_params)
    r2_logarithmic = 1 - np.sum((y - y_logarithmic) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Generate points for smooth curve
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = logarithmic_func(x_smooth, *logarithmic_params)

    # Create interactive plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", name="Data points", marker=dict(color="blue", size=8))
    )
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name=f"Logarithmic (R² = {r2_logarithmic:.3f})",
            line=dict(color="red", width=2),
        )
    )

    # Add function text to the plot
    function_text = f"y = {logarithmic_params[0]:.4f} + {logarithmic_params[1]:.4f} * ln(x+1)"
    fig.add_annotation(
        x=max(x) * 0.7,
        y=max(y) * 0.7,
        text=function_text,
        showarrow=False,
        font=dict(size=12, color="blue"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
    )

    fig.update_layout(
        title="Logarithmic Regression",
        xaxis_title="Year from 2022",
        yaxis_title="USD",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )

    fig.show()

    print(f"Logarithmic fit: {function_text}")
    print(f"R² value: {r2_logarithmic:.4f}")

    # Print residuals for model evaluation
    print("\nResiduals (first 5 and last 5 points):")
    for i in range(5):
        print(
            f"Point {i} (x={x[i]}): Actual={y[i]}, Fitted={y_logarithmic[i]:.2f}, Residual={y[i] - y_logarithmic[i]:.2f}"
        )
    print("...")
    for i in range(len(x) - 5, len(x)):
        print(
            f"Point {i} (x={x[i]}): Actual={y[i]}, Fitted={y_logarithmic[i]:.2f}, Residual={y[i] - y_logarithmic[i]:.2f}"
        )
