# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import plotly.graph_objects as go
from scipy import optimize


# Inverse power function y = a * x^(-b)
def power_func(x, a, b):
    # Handle x=0 by adding a small epsilon to avoid division by zero
    x_safe = x + 1e-10
    return a * x_safe ** (-b)


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

    # For the power function, we need to exclude x=0 for curve fitting
    # Let's create a version of the data without the x=0 point
    mask = x > 0
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Initial parameter guesses
    initial_params = [1.0, 0.1]  # Initial guesses for a and b

    # Fit inverse power regression to the filtered data
    power_params, pcov = optimize.curve_fit(power_func, x_filtered, y_filtered, p0=initial_params)

    # Extract parameters
    a, b = power_params

    # Calculate fitted y values for all points (including x=0)
    y_power = power_func(x, a, b)

    # Calculate R-squared for the power function
    r2_power = 1 - np.sum((y - y_power) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Generate points for smooth curve
    x_smooth = np.linspace(0.01, max(x), 100)  # Start slightly above 0 to avoid division by zero
    y_smooth = power_func(x_smooth, a, b)

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
            name=f"Power Law (R² = {r2_power:.3f})",
            line=dict(color="red", width=2),
        )
    )

    # Add function text to the plot
    function_text = f"y = {a:.4f} * x^(-{b:.4f})"
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
        title="Inverse Power Law Regression",
        xaxis_title="Year from 2022",
        yaxis_title="USD",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )

    fig.show()

    print(f"Inverse Power Law fit: {function_text}")
    print(f"R² value: {r2_power:.4f}")

    # Print residuals for model evaluation
    print("\nResiduals (first 5 and last 5 points):")
    for i in range(5):
        print(
            f"Point {i} (x={x[i]}): Actual={y[i]}, Fitted={y_power[i]:.2f}, Residual={y[i] - y_power[i]:.2f}"
        )
    print("...")
    for i in range(len(x) - 5, len(x)):
        print(
            f"Point {i} (x={x[i]}): Actual={y[i]}, Fitted={y_power[i]:.2f}, Residual={y[i] - y_power[i]:.2f}"
        )
