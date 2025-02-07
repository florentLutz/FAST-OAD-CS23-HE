# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
"""
This regression model is  construct based on the data provided by, Source: H3D_H2_UnmannedAviation_Brochure 2024.pptx.
The specific power is utilized in weight calculation for PEMFC.
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Data points from the graph
x = np.array([0.3, 0.8, 1.2, 2.0])
y = np.array([0.486, 0.645, 0.57, 0.667])


# Logarithmic fit
def log_func(x, a, b):
    return a * np.log(x) + b


log_params, _ = optimize.curve_fit(log_func, x, y)

y_log = log_func(x, *log_params)
r2_log = 1 - np.sum((y - y_log) ** 2) / np.sum((y - np.mean(y)) ** 2)

# Generate points for smooth curves
x_smooth = np.linspace(min(x), max(x), 100)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x_smooth, log_func(x_smooth, *log_params), "r-", label=f"Logarithmic (R² = {r2_log:.3f})")
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Logarithmic Regression")
plt.legend()

# Add function text to the plot
function_text = f"y = {log_params[0]:.4f}ln(x) + {log_params[1]:.4f}"
plt.text(0.7, 0.5, function_text, fontsize=12, color="blue", ha="left", va="bottom")

plt.show()

# Print equation
print(f"Logarithmic: y = {log_params[0]:.4f}ln(x) + {log_params[1]:.4f}")
