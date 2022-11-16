# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pandas as pd

import scipy.optimize as opt
import plotly.graph_objects as go


def open_circuit_function(parameter, p_1, alpha_1, p_2, alpha_2, p_3):

    return (
        p_1 * np.exp(alpha_1 * parameter)
        + p_2 * np.exp(alpha_2 * parameter)
        + p_3 * parameter ** 2.0
    )


if __name__ == "__main__":

    data_file = pth.join(pth.dirname(__file__), "data/OCV.csv")

    ocv_data = pd.read_csv(data_file)

    soc = ocv_data["SOC"]
    voltage = ocv_data["VOLTAGE"]

    fig = go.Figure()
    scatter = go.Scatter(x=soc, y=voltage, mode="lines+markers", name="Data")
    fig.add_trace(scatter)

    x0 = (3.637, -0.0005747, -0.3091, -0.1366, 7.033e-5)
    bnds = ((0.0, -1.0, -100.0, -1.0, -1.0), (100.0, 1.0, 0.0, 1.0, 1.0))

    p_opt, p_cov, _, _, _ = opt.curve_fit(
        open_circuit_function, soc, voltage, x0, bounds=bnds, maxfev=2000, full_output=True
    )

    np.set_printoptions(suppress=True)

    print("Optimal parameter", p_opt)

    computed_voltage = open_circuit_function(soc, p_opt[0], p_opt[1], p_opt[2], p_opt[3], p_opt[4])

    scatter_interpolated = go.Scatter(
        x=soc,
        y=computed_voltage,
        mode="lines+markers",
        name="Interpolated",
    )
    fig.add_trace(scatter_interpolated)
    fig.update_layout(
        title_text="Empirical Circuit Model for the Open Circuit Voltage",
        title_x=0.5,
        xaxis_title="State-of-Charge [%]",
        yaxis_title="Open Circuit Voltage [V]",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.show()
    mean_error = np.mean(np.abs(computed_voltage - voltage) / voltage)
    print("Mean error:", mean_error)
