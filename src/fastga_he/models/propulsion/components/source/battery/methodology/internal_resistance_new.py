# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pandas as pd

import scipy.optimize as opt
import plotly.graph_objects as go


def discharge_curve_func(dod, current):

    open_circuit_voltage = (
        -9.65121262e-10 * dod ** 5.0
        + 1.81419058e-07 * dod ** 4.0
        - 1.11814100e-05 * dod ** 3.0
        + 2.26114438e-04 * dod ** 2.0
        - 8.54619953e-03 * dod
        + 4.12
    )
    # We'll correct by putting a 4.2 since its the value in datasheet
    internal_resistance = (
        2.62771800e-11 * dod ** 5.0
        - 1.48987233e-08 * dod ** 4.0
        + 2.03615618e-06 * dod ** 3.0
        - 1.06451730e-04 * dod ** 2.0
        + 2.13818712e-03 * dod
        + 3.90444549e-02
    )
    return open_circuit_voltage - internal_resistance * current


def ocv_function_curve_fit(dod, p_1, alpha_1, p_2, alpha_2, p_3):

    return p_1 * np.exp(alpha_1 * dod) + p_2 * np.exp(alpha_2 * dod) + p_3 * dod ** 2.0


def ocv_function(dod):

    return (
        94.94621784 * np.exp(-0.00177769 * dod)
        - 90.81194028 * np.exp(-0.00177768 * dod)
        - 0.00002179 * dod ** 2.0
    )


def internal_resistance_function(parameter, p_1, p_2, p_3, constant):

    return constant + p_1 * parameter + p_2 * parameter ** 2.0 + p_3 * parameter ** 3.0


if __name__ == "__main__":

    # source = "data/discharge_curve_new.csv"
    source = "data/discharge_curve_new_new.csv"

    data_file = pth.join(pth.dirname(__file__), source)
    discharge_curve = pd.read_csv(data_file)

    # First step is to get the open circuit voltage; Since we assume that U = OCV - R * I for
    # now, we can use the difference between each curve to go back to the OCV. We'll do that
    # multiple time to round out the error. But first, we need to make sure that all curves have
    # the same x's

    depth_of_discharge_ref = np.linspace(0.0, 1.0)
    if source == "data/discharge_curve_new.csv":
        voltage_0_2_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["0_2_A_X"].to_numpy(),
            discharge_curve["0_2_A_Y"].to_numpy(),
        )
        voltage_0_5_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["0_5_A_X"].to_numpy(),
            discharge_curve["0_5_A_Y"].to_numpy(),
        )
        voltage_1_0_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["1_0_A_X"].to_numpy(),
            discharge_curve["1_0_A_Y"].to_numpy(),
        )
        voltage_2_0_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["2_0_A_X"].to_numpy(),
            discharge_curve["2_0_A_Y"].to_numpy(),
        )
        voltage_3_0_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["3_0_A_X"].to_numpy(),
            discharge_curve["3_0_A_Y"].to_numpy(),
        )
        voltage_5_0_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["5_0_A_X"].to_numpy(),
            discharge_curve["5_0_A_Y"].to_numpy(),
        )
    else:
        voltage_0_2_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["0_2_A_X"].to_numpy()
            / np.nanmax(discharge_curve["0_2_A_X"].to_numpy()),
            discharge_curve["0_2_A_Y"].to_numpy(),
        )
        voltage_0_5_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["0_5_A_X"].to_numpy()
            / np.nanmax(discharge_curve["0_5_A_X"].to_numpy()),
            discharge_curve["0_5_A_Y"].to_numpy(),
        )
        voltage_1_0_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["1_0_A_X"].to_numpy()
            / np.nanmax(discharge_curve["1_0_A_X"].to_numpy()),
            discharge_curve["1_0_A_Y"].to_numpy(),
        )
        voltage_2_0_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["2_0_A_X"].to_numpy()
            / np.nanmax(discharge_curve["2_0_A_X"].to_numpy()),
            discharge_curve["2_0_A_Y"].to_numpy(),
        )
        voltage_3_0_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["3_0_A_X"].to_numpy()
            / np.nanmax(discharge_curve["3_0_A_X"].to_numpy()),
            discharge_curve["3_0_A_Y"].to_numpy(),
        )
        voltage_5_0_A = np.interp(
            depth_of_discharge_ref,
            discharge_curve["5_0_A_X"].to_numpy()
            / np.nanmax(discharge_curve["5_0_A_X"].to_numpy()),
            discharge_curve["5_0_A_Y"].to_numpy(),
        )
    depth_of_discharge_ref *= 100.0

    # Let's start with a plot to ensure data were properly sampled
    fig = go.Figure()

    x_list = [
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
    ]
    y_list = [
        voltage_0_2_A,
        voltage_0_5_A,
        voltage_1_0_A,
        voltage_2_0_A,
        voltage_3_0_A,
        voltage_5_0_A,
    ]
    names = ["0.2A", "0.5A", "1.0A", "2.0A", "3.0A", "5.0A"]

    for x, y, name in zip(x_list, y_list, names):
        scatter = go.Scatter(x=x, y=y, name=name, mode="lines+markers")
        fig.add_trace(scatter)

    # fig.show()

    # Looks OK,
    # Let's now try to infer the OCV by doing a mean of different curve differences. Knowing
    # r_int_i_* should be negative it is actually -r_int_i_* we will clip positive values
    r_int_i_0_2_0_5 = (voltage_0_5_A - voltage_0_2_A) / 0.3
    ocv_0_2_0_5 = voltage_0_2_A - 0.2 * r_int_i_0_2_0_5

    r_int_i_1_0_0_5 = (voltage_1_0_A - voltage_0_5_A) / 0.5
    ocv_1_0_0_5 = voltage_0_5_A - 0.5 * r_int_i_1_0_0_5

    r_int_i_2_0_1_0 = voltage_2_0_A - voltage_1_0_A
    ocv_2_0_1_0 = voltage_1_0_A - r_int_i_2_0_1_0

    r_int_i_3_0_2_0 = voltage_3_0_A - voltage_2_0_A
    ocv_3_0_2_0 = voltage_2_0_A - 2.0 * r_int_i_3_0_2_0

    r_int_i_5_0_3_0 = (voltage_5_0_A - voltage_3_0_A) / 2.0
    ocv_5_0_3_0 = voltage_3_0_A - 3.0 * r_int_i_5_0_3_0

    x_list = [
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
    ]
    y_list = [
        ocv_0_2_0_5,
        ocv_1_0_0_5,
        ocv_2_0_1_0,
        ocv_3_0_2_0,
        ocv_5_0_3_0,
    ]
    names = ["0.2A to 0.5A", "0.5A to 1.0A", "1.0A to 2.0A", "2.0A to 3.0A", "3.0A to 5.0A"]

    fig2 = go.Figure()

    for x, y, name in zip(x_list, y_list, names):
        scatter = go.Scatter(x=x, y=y, name=name, mode="lines+markers")
        fig2.add_trace(scatter)

    # fig2.show()

    # Does not look too bad either ! Let's mean those
    ocv = np.mean(
        np.array(
            [
                ocv_0_2_0_5,
                ocv_1_0_0_5,
                ocv_2_0_1_0,
                ocv_3_0_2_0,
                ocv_5_0_3_0,
            ]
        ),
        axis=0,
    )

    scatter = go.Scatter(x=depth_of_discharge_ref, y=ocv, name="Average OCV", mode="lines+markers")
    fig.add_trace(scatter)

    # fig.show()
    # Let's try to fit with the shape we want
    where_valid = np.where(depth_of_discharge_ref < 95.0)

    x0 = (94.50, -0.01292712, -91.349, -0.01362893, 1.472e-4)
    bnds = (
        (-1000.0, -10.0, -1000.0, -10.0, -10.0),
        (1000.0, 10.0, 1000.0, 10.0, 10.0),
    )

    p_opt, p_cov, _, _, _ = opt.curve_fit(
        ocv_function_curve_fit,
        depth_of_discharge_ref[where_valid],
        ocv[where_valid],
        x0,
        bounds=bnds,
        maxfev=2000,
        full_output=True,
    )

    # np.set_printoptions(suppress=True)
    # print("Optimal parameter", p_opt)

    fig4 = go.Figure()

    scatter = go.Scatter(x=depth_of_discharge_ref, y=ocv, name="Average OCV", mode="lines+markers")
    fig4.add_trace(scatter)
    scatter = go.Scatter(
        x=depth_of_discharge_ref,
        y=ocv_function(depth_of_discharge_ref),
        name="Interpolated OCV",
        mode="lines+markers",
    )
    fig4.add_trace(scatter)

    # fig4.show()

    # Let's simply try with a polyfit of order 6
    poly_ocv = np.polyfit(depth_of_discharge_ref[where_valid], ocv[where_valid], 5)
    print(poly_ocv)

    fig5 = go.Figure()

    scatter = go.Scatter(x=depth_of_discharge_ref, y=ocv, name="Average OCV", mode="lines+markers")
    fig5.add_trace(scatter)

    ocv_ref = np.polyval(poly_ocv, depth_of_discharge_ref)

    scatter = go.Scatter(
        x=depth_of_discharge_ref,
        y=ocv_ref,
        name="Interpolated OCV",
        mode="lines+markers",
    )
    fig5.add_trace(scatter)

    # fig5.show()
    # The first proposed model, even if it does not replicate the small hunch at the beginning
    # does not go back up like the 6th order polynomial interpolation does.

    ocv_ref = np.polyval(poly_ocv, depth_of_discharge_ref)
    # ocv_ref = ocv_function(depth_of_discharge_ref)

    # Now that we have the OCV we'll do an average of the internal resistance and fo another_fit
    r_int_0_2 = -(voltage_0_2_A - ocv_ref) / 0.2
    r_int_0_5 = -(voltage_0_5_A - ocv_ref) / 0.5
    r_int_1_0 = -(voltage_1_0_A - ocv_ref)
    r_int_2_0 = -(voltage_2_0_A - ocv_ref) / 2.0
    r_int_3_0 = -(voltage_3_0_A - ocv_ref) / 3.0
    r_int_5_0 = -(voltage_5_0_A - ocv_ref) / 5.0

    # They should more or less look the same
    x_list = [
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
        depth_of_discharge_ref,
    ]
    y_list = [
        r_int_0_2,
        r_int_0_5,
        r_int_1_0,
        r_int_2_0,
        r_int_3_0,
        r_int_5_0,
    ]
    names = ["0.2A", "0.5A", "1.0A", "2.0A", "3.0A", "5.0A"]

    r_int = np.mean(
        np.array(
            [
                r_int_1_0,
                r_int_2_0,
                r_int_3_0,
                r_int_5_0,
            ]
        ),
        axis=0,
    )

    fig6 = go.Figure()

    for x, y, name in zip(x_list, y_list, names):
        scatter = go.Scatter(x=x[where_valid], y=y[where_valid], name=name, mode="lines+markers")
        fig6.add_trace(scatter)

    scatter = go.Scatter(
        x=depth_of_discharge_ref[where_valid],
        y=r_int[where_valid],
        name="R_int ?",
        mode="lines+markers",
    )
    fig6.add_trace(scatter)

    poly_r_int = np.polyfit(depth_of_discharge_ref[where_valid], r_int[where_valid], 5)
    scatter = go.Scatter(
        x=depth_of_discharge_ref[where_valid],
        y=np.polyval(poly_r_int, depth_of_discharge_ref[where_valid]),
        name="R_int interpolated?",
        mode="lines+markers",
    )
    fig6.add_trace(scatter)

    # fig6.show()
    print(poly_r_int)

    fig7 = go.Figure()

    for current_plot in [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]:
        scatter = go.Scatter(
            x=depth_of_discharge_ref[where_valid],
            y=discharge_curve_func(depth_of_discharge_ref[where_valid], current_plot),
            name=str(current_plot) + " A",
            mode="lines+markers",
        )
        fig.add_trace(scatter)

    fig.show()
