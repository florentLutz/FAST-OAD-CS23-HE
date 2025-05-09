# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
from scipy import optimize
import plotly.graph_objects as go


# Polynomial function (2nd order)
def polynomial_func(x, a, b, c):
    return a * x**2 + b * x + c  # ax^2 + bx + c form


if __name__ == "__main__":
    # Updated data points
    x = np.array(
        [
            2048,
            1550,
            3087,
            4535,
            2089,
            1727,
        ]
    )
    y = np.array(
        [
            270,
            282,
            349,
            575,
            374,
            266,
        ]
    )

    # Initial parameter guess for polynomial fit
    # Starting with small values for all parameters
    initial_guess = [0.0001, 0.01, 200]

    # Fit polynomial regression
    polynomial_params, pcov = optimize.curve_fit(
        polynomial_func, x, y, p0=initial_guess, maxfev=10000
    )

    # Calculate fitted y values and R-squared
    y_polynomial = polynomial_func(x, *polynomial_params)
    r2_polynomial = 1 - np.sum((y - y_polynomial) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Generate points for smooth curve
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = polynomial_func(x_smooth, *polynomial_params)

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
            name=f"Polynomial (R² = {r2_polynomial:.3f})",
            line=dict(color="red", width=2),
        )
    )

    # Add function text to the plot
    a, b, c = polynomial_params
    function_text = f"y = {a:.6f}x² + {b:.4f}x + {c:.2f}"
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
        title="Polynomial Regression (2nd Order)",
        xaxis_title="OWE (kg)",
        yaxis_title="USD",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )

    fig.show()

    print(f"Polynomial fit: {function_text}")
    print(f"R² value: {r2_polynomial:.4f}")

    # Print residuals for model evaluation
    print("\nResiduals:")
    for i in range(len(x)):
        print(
            f"Point {i+1} (x={x[i]}): Actual={y[i]}, Fitted={y_polynomial[i]:.2f}, Residual={y[i] - y_polynomial[i]:.2f}"
        )

    # For prediction purposes, demonstrate extrapolation
    test_points = [1500, 2000, 3000, 4500]
    print("\nPredictions at specific points:")
    for point in test_points:
        prediction = polynomial_func(point, *polynomial_params)
        print(f"At x={point}: Predicted y={prediction:.2f}")
