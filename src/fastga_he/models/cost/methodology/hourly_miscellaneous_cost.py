# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import plotly.graph_objects as go

if __name__ == "__main__":
    # Data points
    x = np.array([2048, 1550, 3087, 4535, 2089, 734.83, 1115.85, 1020.6, 535.24])
    y = np.array([80, 75, 82, 200, 78, 8.8, 19.33, 13.63, 6.12])

    # Linear regression using numpy.polyfit (1st order)
    coeffs = np.polyfit(x, y, 1)  # Returns [m, b] for y = mx + b
    m, b = coeffs

    # Fitted values and R²
    y_linear = np.polyval(coeffs, x)
    r2_linear = 1 - np.sum((y - y_linear) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Smooth curve
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = np.polyval(coeffs, x_smooth)

    # Plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", name="Data points", marker=dict(color="blue", size=8))
    )
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name=f"Linear (R² = {r2_linear:.3f})",
            line=dict(color="green", width=2),
        )
    )

    # Function text
    function_text = f"y = {m:.6f}x + {b:.2f}"
    fig.add_annotation(
        x=(max(x) + min(x)) / 2,
        y=max(y) * 0.7,
        text=function_text,
        showarrow=False,
        font=dict(size=12, color="green"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
    )

    fig.update_layout(
        title="Linear Regression",
        xaxis_title="OWE (kg)",
        yaxis_title="USD",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )

    fig.show()

    print(f"Linear fit: {function_text}")
    print(f"R² value: {r2_linear:.4f}")

    # Residuals
    print("\nResiduals:")
    for i in range(len(x)):
        fitted = y_linear[i]
        print(
            f"Point {i+1} (x={x[i]}): Actual={y[i]}, Fitted={fitted:.2f}, Residual={y[i] - fitted:.2f}"
        )

    # Predictions
    test_points = [1500, 2000, 3000, 4500]
    print("\nPredictions at specific points:")
    for point in test_points:
        prediction = np.polyval(coeffs, point)
        print(f"At x={point}: Predicted y={prediction:.2f}")
