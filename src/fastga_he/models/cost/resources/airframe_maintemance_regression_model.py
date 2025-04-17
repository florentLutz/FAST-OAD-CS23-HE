# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
"""
This regression model is construct based on the data provided by
https://www.guardianjet.com/jet-aircraft-online-tools.
"""

import numpy as np
import plotly.graph_objects as go
from scipy import optimize


# Exponential fit function
def exponential_func(x, a, b):
    return a * np.exp(b * x)


# Power fit function (alternative model)
def power_func(x, a, b):
    return a * (x**b)


if __name__ == "__main__":
    # Data points from the graph
    x = np.array([1550.0, 2048.0, 2089.0, 3087.0, 4535.0])  # Sorted for better visualization
    y = np.array([282.0, 270.0, 374.0, 349.0, 575.0])  # Corresponding y values

    try:
        # Try exponential fit with initial parameter guesses
        # Using very small initial value for b to prevent overflow
        initial_guess = [y[0], 0.0001]
        exponential_params, pcov = optimize.curve_fit(
            exponential_func, x, y, p0=initial_guess, bounds=([0, -0.001], [1000, 0.001])
        )


        # Calculate fitted y values and R-squared
        y_exponential = exponential_func(x, *exponential_params)
        r2_exponential = 1 - np.sum((y - y_exponential) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Generate points for smooth curve
        x_smooth = np.linspace(min(x), max(x), 100)
        y_smooth = exponential_func(x_smooth, *exponential_params)

        fit_type = "Exponential"
        fit_params = exponential_params
        r2_value = r2_exponential
        y_fit = y_exponential
        function_text = f"y = {fit_params[0]:.4f} * e^({fit_params[1]:.6f}x)"

    except RuntimeError:
        print("Exponential fit failed, trying power function instead")
        # If exponential fails, try power function fit
        power_params, pcov = optimize.curve_fit(power_func, x, y)

        # Calculate fitted y values and R-squared for power function
        y_power = power_func(x, *power_params)
        r2_power = 1 - np.sum((y - y_power) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Generate points for smooth curve
        x_smooth = np.linspace(min(x), max(x), 100)
        y_smooth = power_func(x_smooth, *power_params)

        fit_type = "Power"
        fit_params = power_params
        r2_value = r2_power
        y_fit = y_power
        function_text = f"y = {fit_params[0]:.4f} * x^({fit_params[1]:.4f})"

    # Create interactive plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", name="Data points", marker=dict(color="blue", size=10))
    )
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name=f"{fit_type} (R² = {r2_value:.3f})",
            line=dict(color="red", width=2),
        )
    )

    # Add function text to the plot
    fig.add_annotation(
        x=max(x) * 0.8,
        y=min(y) * 1.2,
        text=function_text,
        showarrow=False,
        font=dict(size=12, color="blue"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
    )

    fig.update_layout(
        title="Airframe maintenance cost per flight hour",
        xaxis_title="OWE (kg)",
        yaxis_title="cost_per_hour (USD/h)",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )

    fig.show()

    print(f"{fit_type} fit: {function_text}")
    print(f"R² value: {r2_value:.4f}")

    # Print residuals for model evaluation
    print("\nResiduals:")
    for i in range(len(x)):
        print(f"Point {i + 1}: {y[i] - y_fit[i]:.2f}")
