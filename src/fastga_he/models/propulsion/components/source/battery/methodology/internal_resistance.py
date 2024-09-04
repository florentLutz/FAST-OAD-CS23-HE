# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pandas as pd

import scipy.optimize as opt
import plotly.graph_objects as go


def internal_resistance_result(dod):
    return (
        7.94693564e-05 * dod
        - 1.18383130e-06 * dod**2.0
        + 5.75440812e-09 * dod**3.0
        + 2.96477143e-03
    )


def voltage_curve(dod, c_rate):
    current = 20 * c_rate

    return open_circuit_voltage(100.0 - dod) - internal_resistance_result(dod) * current


def internal_resistance_function(parameter, p_1, p_2, p_3, constant):
    return p_1 * parameter + p_2 * parameter**2.0 + p_3 * parameter**3.0 + constant


def open_circuit_voltage(soc):
    return (
        94.50 * np.exp(-0.01292712 * soc) - 91.349 * np.exp(-0.01362893 * soc) + 1.472e-4 * soc**2.0
    )


if __name__ == "__main__":
    data_file = pth.join(pth.dirname(__file__), "data/discharge_curve.csv")

    discharge_curve = pd.read_csv(data_file)

    dod_1c = discharge_curve["1C_X"].to_numpy()
    voltage_1c = discharge_curve["1C_Y"].to_numpy()

    open_circuit_1c = open_circuit_voltage(100.0 - dod_1c)

    dod_2c = discharge_curve["2C_X"].to_numpy()
    voltage_2c = discharge_curve["2C_Y"].to_numpy()

    voltage_2c_interp = np.interp(dod_1c, dod_2c, voltage_2c)
    open_circuit_2c = open_circuit_voltage(100.0 - dod_2c)

    dod_3c = discharge_curve["3C_X"].to_numpy()
    voltage_3c = discharge_curve["3C_Y"].to_numpy()

    voltage_3c_interp = np.interp(dod_1c, dod_3c, voltage_3c)
    open_circuit_3c = open_circuit_voltage(100.0 - dod_3c)

    dod_5c = discharge_curve["5C_X"].to_numpy()
    voltage_5c = discharge_curve["5C_Y"].to_numpy()

    voltage_5c_interp = np.interp(dod_1c, dod_5c, voltage_5c)
    open_circuit_5c = open_circuit_voltage(100.0 - dod_5c)

    fig = go.Figure()
    scatter = go.Scatter(
        x=dod_1c,
        y=voltage_1c - voltage_2c_interp,
        mode="lines+markers",
        name="Difference between 1 and 2",
    )
    fig.add_trace(scatter)
    scatter_2 = go.Scatter(
        x=dod_1c,
        y=voltage_2c_interp - voltage_3c_interp,
        mode="lines+markers",
        name="Difference between 2 and 3",
    )
    fig.add_trace(scatter_2)
    scatter_3 = go.Scatter(
        x=dod_1c,
        y=(voltage_3c_interp - voltage_5c_interp) / 2.0,
        mode="lines+markers",
        name="Difference between 3 and 5 divided by 2",
    )
    fig.add_trace(scatter_3)
    # fig.show()

    i = 20.0  # C_rate of 1 with a battery capacity of 20 Ah

    internal_resistance_1c = -(voltage_1c - open_circuit_1c) / i
    internal_resistance_2c = -(voltage_2c - open_circuit_2c) / (2.0 * i)
    internal_resistance_3c = -(voltage_3c - open_circuit_3c) / (3.0 * i)
    internal_resistance_5c = -(voltage_5c - open_circuit_5c) / (5.0 * i)

    dod_for_fit = np.concatenate((dod_1c, dod_2c, dod_3c, dod_5c))
    ri_for_fit = np.concatenate(
        (
            internal_resistance_1c,
            internal_resistance_2c,
            internal_resistance_3c,
            internal_resistance_5c,
        )
    )

    where_valid = np.where(dod_for_fit <= 80.0)
    fig2 = go.Figure()
    scatter = go.Scatter(
        x=dod_for_fit[where_valid],
        y=ri_for_fit[where_valid],
        mode="markers",
        name="Internal resistance",
    )
    fig2.add_trace(scatter)
    # fig2.show()

    x0 = (0.5, 0.5, 0.5, 0.5)
    bnds = (
        (-1.0, -1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0, 1.0),
    )

    p_opt, p_cov, _, _, _ = opt.curve_fit(
        internal_resistance_function,
        dod_for_fit[where_valid],
        ri_for_fit[where_valid],
        x0,
        bounds=bnds,
        maxfev=2000,
        full_output=True,
    )

    print("Optimal parameter", p_opt)

    fig3 = go.Figure()

    where_valid_1c = np.where(dod_1c <= 80.0)
    scatter_1c = go.Scatter(
        x=dod_1c[where_valid_1c],
        y=voltage_1c[where_valid_1c],
        mode="lines+markers",
        name="1C - Data",
        legendgroup="1C",
        legendgrouptitle_text="1C",
    )
    fig3.add_trace(scatter_1c)
    interpolated_scatter_1c = go.Scatter(
        x=dod_1c[where_valid_1c],
        y=voltage_curve(dod_1c[where_valid_1c], 1.0),
        mode="lines+markers",
        name="1C - Interpolated",
        legendgroup="1C",
    )
    fig3.add_trace(interpolated_scatter_1c)

    where_valid_2c = np.where(dod_2c <= 80.0)
    scatter_2c = go.Scatter(
        x=dod_2c[where_valid_2c],
        y=voltage_2c[where_valid_2c],
        mode="lines+markers",
        name="2C - Data",
        legendgroup="2C",
        legendgrouptitle_text="2C",
    )
    fig3.add_trace(scatter_2c)
    interpolated_scatter_2c = go.Scatter(
        x=dod_2c[where_valid_2c],
        y=voltage_curve(dod_2c[where_valid_2c], 2.0),
        mode="lines+markers",
        name="2C - Interpolated",
        legendgroup="2C",
    )
    fig3.add_trace(interpolated_scatter_2c)

    where_valid_3c = np.where(dod_3c <= 80.0)
    scatter_3c = go.Scatter(
        x=dod_3c[where_valid_3c],
        y=voltage_3c[where_valid_3c],
        mode="lines+markers",
        name="3C - Data",
        legendgroup="3C",
        legendgrouptitle_text="3C",
    )
    fig3.add_trace(scatter_3c)
    interpolated_scatter_3c = go.Scatter(
        x=dod_3c[where_valid_3c],
        y=voltage_curve(dod_3c[where_valid_3c], 3.0),
        mode="lines+markers",
        name="3C - Interpolated",
        legendgroup="3C",
    )
    fig3.add_trace(interpolated_scatter_3c)

    where_valid_5c = np.where(dod_5c <= 80.0)
    scatter_5c = go.Scatter(
        x=dod_5c[where_valid_5c],
        y=voltage_5c[where_valid_5c],
        mode="lines+markers",
        name="5C - Data",
        legendgroup="5C",
        legendgrouptitle_text="5C",
    )
    fig3.add_trace(scatter_5c)
    interpolated_scatter_5c = go.Scatter(
        x=dod_5c[where_valid_5c],
        y=voltage_curve(dod_5c[where_valid_5c], 5.0),
        mode="lines+markers",
        name="5C - Interpolated",
        legendgroup="5C",
    )
    fig3.add_trace(interpolated_scatter_5c)

    fig3.show()
