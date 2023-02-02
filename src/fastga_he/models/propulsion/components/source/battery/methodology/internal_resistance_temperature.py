# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO
"""This module is to study the effect of temperature on battery internal resistance. One of the
reference in :cite:`chen:2021` suggests that internal resistance is a product of a function of
SOC and a function of temperature under the form of an exponential (R_int(SOC, T) = f(SOC) * g(
T)). :cite:`lai:2019` gives the evolution of internal resistance for the temperature for a 18650
cell (NCR 18650-type cylindrical Li-ion). We will make the assumption that the coefficient in the
exponential do not vary too much from one Li-ion cell to another (Linked to chemistry, catalytic
speed ?). """

import os.path as pth

import numpy as np
import pandas as pd

import scipy.optimize as opt
import plotly.graph_objects as go


def temperature_function(temperature_value, scaling_factor, ref_temperature):

    return np.exp(scaling_factor / (temperature_value + ref_temperature))


def ratio(temperature_value, scaling_factor, ref_temperature):

    return temperature_function(
        temperature_value, scaling_factor, ref_temperature
    ) / temperature_function(
        293.0,
        scaling_factor,
        ref_temperature,
    )


def ratio_expanded(temperature_value):

    return np.exp(
        (46.386005 * (293.0 - temperature_value))
        / ((temperature_value - 254.33423266) * (293.0 - 254.33423266))
    )


if __name__ == "__main__":

    data_file = pth.join(pth.dirname(__file__), "data/internal_resistance_temperature.csv")
    temp_dependency = pd.read_csv(data_file)

    x_293 = temp_dependency["293K_X"].to_numpy()
    y_293 = temp_dependency["293K_Y"].to_numpy()

    x_303 = temp_dependency["303K_X"].to_numpy()
    y_303 = temp_dependency["303K_Y"].to_numpy()

    x_313 = temp_dependency["313K_X"].to_numpy()
    y_313 = temp_dependency["313K_Y"].to_numpy()

    x_323 = temp_dependency["323K_X"].to_numpy()
    y_323 = temp_dependency["323K_Y"].to_numpy()

    x_333 = temp_dependency["333K_X"].to_numpy()
    y_333 = temp_dependency["333K_Y"].to_numpy()

    x_list = [x_293, x_303, x_313, x_323, x_333]
    y_list = [y_293, y_303, y_313, y_323, y_333]
    temperatures = [293, 303, 313, 323, 333]

    fig = go.Figure()

    for x, y, temperature in zip(x_list, y_list, temperatures):

        scatter = go.Scatter(x=x, y=y, name=str(temperature), mode="lines+markers")
        fig.add_trace(scatter)

    # fig.show()

    # First we ensure that we work at the same SOC for all temperatures, we'll do that by
    # interpolating  on the 293 graph which will serve as a default value

    y_303 = np.interp(x_293, x_303, y_303)
    y_313 = np.interp(x_293, x_313, y_313)
    y_323 = np.interp(x_293, x_323, y_323)
    y_333 = np.interp(x_293, x_333, y_333)

    # Now we can get the value of the ratio (R_int(SOC_i, T_j) / R_int(SOC_i, T293)) which leave
    # us only with the ratios of the g functions. Additionally we will drop the point of the
    # array where SOC < 0.2, too unreliable.

    proper_index = np.where(x_293 > 0.2)

    ratio_303_293 = y_303 / y_293
    ratio_303_293 = ratio_303_293[proper_index]

    ratio_313_293 = y_313 / y_293
    ratio_313_293 = ratio_313_293[proper_index]

    ratio_323_293 = y_323 / y_293
    ratio_323_293 = ratio_323_293[proper_index]

    ratio_333_293 = y_333 / y_293
    ratio_333_293 = ratio_333_293[proper_index]

    # These arrays should be more or less constant so we check that
    x_list = [x_293[proper_index], x_293[proper_index], x_293[proper_index], x_293[proper_index]]
    y_list = [ratio_303_293, ratio_313_293, ratio_323_293, ratio_333_293]
    temperatures = [303, 313, 323, 333]

    fig2 = go.Figure()

    for x, y, temperature in zip(x_list, y_list, temperatures):
        scatter = go.Scatter(
            x=x, y=y, name="ratio of " + str(temperature) + "K to 293K", mode="lines+markers"
        )
        fig2.add_trace(scatter)

    # fig2.show()
    mean_303_293 = np.mean(ratio_303_293)
    print(
        "Mean for 303:",
        mean_303_293,
        "--- Max relative Deviation:",
        max(abs(el - mean_303_293) for el in ratio_303_293) / mean_303_293,
    )
    mean_313_293 = np.mean(ratio_313_293)
    print(
        "Mean for 313:",
        mean_313_293,
        "--- Max relative Deviation:",
        max(abs(el - mean_313_293) for el in ratio_313_293) / mean_313_293,
    )
    mean_323_293 = np.mean(ratio_323_293)
    print(
        "Mean for 323:",
        mean_323_293,
        "--- Max relative Deviation:",
        max(abs(el - mean_323_293) for el in ratio_323_293) / mean_323_293,
    )
    mean_333_293 = np.mean(ratio_333_293)
    print(
        "Mean for 333:",
        mean_333_293,
        "--- Max relative Deviation:",
        max(abs(el - mean_333_293) for el in ratio_333_293) / mean_333_293,
    )

    ratio_value = [mean_303_293, mean_313_293, mean_323_293, mean_333_293]

    x0 = (-100.0, 150.0)
    bnds = (
        (-300, -300),
        (300, 300),
    )

    p_opt, p_cov, _, _, _ = opt.curve_fit(
        ratio,
        np.array(temperatures),
        np.array(ratio_value),
        x0,
        bounds=bnds,
        maxfev=2000,
        full_output=True,
    )

    print("\nOptimal parameter", p_opt)
    print("\n")
    print(ratio(np.array(temperatures), 46.386005, -254.33423266))
    print(np.array(ratio_value))

    fig3 = go.Figure()

    y_list = [y_303, y_313, y_323, y_333]

    for temperature, y_to_plot in zip(temperatures, y_list):
        x = x_293
        y_interpolated = y_293 * ratio(temperature, 46.386005, -254.33423266)
        y_interpolated_v_2 = y_293 * ratio_expanded(temperature)
        y_real = y_to_plot

        scatter = go.Scatter(
            x=x[proper_index],
            y=y_interpolated[proper_index],
            name="Interpolated internal resistance " + str(temperature),
            mode="lines+markers",
            legendgroup=str(temperature),
            legendgrouptitle_text="data for" + str(temperature) + " K",
        )
        fig3.add_trace(scatter)
        scatter_real = go.Scatter(
            x=x[proper_index],
            y=y_real[proper_index],
            name="Real internal resistance " + str(temperature),
            mode="lines+markers",
            legendgroup=str(temperature),
        )
        fig3.add_trace(scatter_real)
        scatter = go.Scatter(
            x=x[proper_index],
            y=y_interpolated_v_2[proper_index],
            name="Interpolated v2 internal resistance " + str(temperature),
            mode="lines+markers",
        )
        fig3.add_trace(scatter)

    fig3.show()

    # Just to confirm we will study the tendency of the function (should have an asymptotic curve
    # when reaching very high temperatures and should be decreasing)
    fig4 = go.Figure()
    x = np.linspace(273.15, 473.15, 300)
    scatter = go.Scatter(
        x=x,
        y=ratio(x, 46.386005, -254.33423266),
        name="Temperature contribution",
        mode="lines+markers",
    )
    fig4.add_trace(scatter)
    # fig4.show()
