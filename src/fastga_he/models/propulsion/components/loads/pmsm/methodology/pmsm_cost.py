# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import plotly.graph_objects as go
from scipy import optimize


# Exponential fit function - simplified two-parameter version
def exponential_func(x, a, b):
    return a * np.exp(b * x)  # a*e^(b*x) form


if __name__ == "__main__":
    # Updated data points
    x = np.array(
        [
            27,
            33,
            55,
            80,
            84,
        ]
    )
    y = np.array(
        [
            2651,
            3024.5,
            3425,
            6187,
            11484,
        ]
    )

    # Initial parameter guess for exponential fit
    # For growing data like this, typical initial guesses are:
    # a: smaller than the first y value
    # b: positive value for growth
    initial_guess = [1000, 0.03]

    # Fit exponential regression
    exponential_params, pcov = optimize.curve_fit(
        exponential_func, x, y, p0=initial_guess, maxfev=10000
    )

    # Calculate fitted y values and R-squared
    y_exponential = exponential_func(x, *exponential_params)
    r2_exponential = 1 - np.sum((y - y_exponential) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Generate points for smooth curve
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = exponential_func(x_smooth, *exponential_params)

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
            name=f"Exponential (R² = {r2_exponential:.3f})",
            line=dict(color="red", width=2),
        )
    )

    # Add function text to the plot
    function_text = f"y = {exponential_params[0]:.4f} * e^({exponential_params[1]:.4f}x)"
    fig.add_annotation(
        x=(max(x) + min(x)) / 2,
        y=max(y) * 0.7,
        text=function_text,
        showarrow=False,
        font=dict(size=12, color="blue"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
    )

    fig.update_layout(
        title="Exponential Regression",
        xaxis_title="max continuous power (kW)",
        yaxis_title="USD",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )

    fig.show()

    print(f"Exponential fit: {function_text}")
    print(f"R² value: {r2_exponential:.4f}")

    # Print residuals for model evaluation
    print("\nResiduals:")
    for i in range(len(x)):
        print(
            f"Point {i} (x={x[i]}): Actual={y[i]}, Fitted={y_exponential[i]:.2f}, Residual={y[i] - y_exponential[i]:.2f}"
        )

    # For prediction purposes, demonstrate extrapolation
    test_points = [25, 50, 85, 100]
    print("\nPredictions at specific points:")
    for point in test_points:
        prediction = exponential_func(point, *exponential_params)
        print(f"At x={point}: Predicted y={prediction:.2f}")
