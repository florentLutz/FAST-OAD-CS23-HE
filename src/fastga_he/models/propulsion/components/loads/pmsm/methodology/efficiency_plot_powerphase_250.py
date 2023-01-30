# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import numpy as np
import plotly.graph_objects as go
import scipy.optimize as opt

from utils.curve_reading import curve_reading

PLOT = True
MOTOR = "250"
MODEL = "SCALING"  # "SCALING", "MCDONALD", "POLITO", "POLITO_mod", "AUCKLAND", "VRATNY"
THRESHOLD = 0.0


def efficiency_for_curve_fit(parameter, alpha_test, beta_test):

    speed, torque = parameter

    power = speed * torque
    power_losses = alpha_test * torque ** 2.0 + beta_test * speed ** 1.5

    computed_efficiency = power / (power + power_losses)

    return computed_efficiency


def efficiency_for_curve_fit_mac_donalds(parameter, c_0_test, c_1_test, c_2_test, c_3_test):

    speed, torque = parameter

    power = speed * torque
    power_losses = c_0_test + c_1_test * speed + c_2_test * speed ** 3.0 + c_3_test * torque ** 2.0

    computed_efficiency = power / (power + power_losses)

    return computed_efficiency


def efficiency_for_curve_fit_polito(
    parameter, alpha_test, beta_test, gamma_test, delta_test, epsilon_test
):
    speed, torque = parameter

    power = speed * torque
    power_losses = (
        alpha_test
        + beta_test * torque ** 2.0
        + gamma_test * speed ** 2.0
        + delta_test * power ** 2.0
        + epsilon_test * speed ** 3.0
    )

    computed_efficiency = power / (power + power_losses)

    return computed_efficiency


def efficiency_for_curve_fit_polito_mod(parameter, alpha_test, beta_test, gamma_test, delta_test):
    speed, torque = parameter

    power = speed * torque
    power_losses = (
        alpha_test
        + beta_test * torque ** 2.0
        + gamma_test * speed ** 2.0
        + delta_test * power ** 2.0
    )

    computed_efficiency = power / (power + power_losses)

    return computed_efficiency


def efficiency_for_curve_fit_auckland(parameter, alpha_test, beta_test, gamma_test, delta_test):
    speed, torque = parameter

    power = speed * torque
    power_losses = (
        alpha_test + beta_test * torque ** 2.0 + gamma_test * speed + delta_test * speed ** 1.5
    )

    computed_efficiency = power / (power + power_losses)

    return computed_efficiency


def efficiency_for_curve_fit_vratny(parameter, param_1, param_2, param_3, param_4, param_5):

    speed, torque = parameter

    power = speed * torque
    power_losses = (
        param_1 * torque ** (1.0 / 1.5) * speed
        + param_2 * torque
        + param_3 * torque * speed ** 2.0
        + param_4 * speed ** 3.0
        + param_5 * torque * speed ** 1.25
    )

    computed_efficiency = power / (power + power_losses)

    return computed_efficiency


if __name__ == "__main__":

    if MOTOR == "250":
        data_file = pth.join(pth.dirname(__file__), "data/powerphase_250_efficiency_plot.csv")
    elif MOTOR == "160":
        data_file = pth.join(pth.dirname(__file__), "data/powerphase_160_efficiency_plot.csv")
    else:
        data_file = pth.join(pth.dirname(__file__), "data/powerphase_125_efficiency_plot.csv")

    speed_array, torque_array, efficiency_array, fig = curve_reading(data_file, threshold=THRESHOLD)
    speed_array_p = np.linspace(50.0, 1.1 * np.max(speed_array), 50)
    torque_array_p = np.linspace(10.0, 1.1 * np.max(torque_array), 50)
    [speed_array_plot, torque_array_plot] = np.meshgrid(speed_array_p, torque_array_p)

    parameter_curve_fit = np.vstack((speed_array, torque_array))

    if MODEL == "SCALING":
        x0 = (7e-2, 7e-2)
        bnds = ((0, 0.0), (1.0, 1.0))

        p_opt, p_cov, _, _, _ = opt.curve_fit(
            efficiency_for_curve_fit,
            parameter_curve_fit,
            efficiency_array,
            x0,
            bounds=bnds,
            full_output=True,
        )
        print("Optimal parameter", p_opt)
        alpha, beta = p_opt

        power_loss = alpha * torque_array_plot ** 2.0 + beta * speed_array_plot ** 1.5
        efficiency_grid = (
            torque_array_plot
            * speed_array_plot
            / (power_loss + torque_array_plot * speed_array_plot)
        ).reshape((50, 50))

        power_loss_orig_data = alpha * torque_array ** 2.0 + beta * speed_array ** 1.5
        efficiency_grid_orig_data = (
            torque_array * speed_array / (power_loss_orig_data + torque_array * speed_array)
        )
        print("RMS:", np.sqrt(np.mean((efficiency_grid_orig_data - efficiency_array) ** 2.0)))

    elif MODEL == "MCDONALD":

        x0 = (1000.0, 3.0, 4.0e-6, 1e-1)
        bounds_mc = ((0.0, 0.0, 0.0, 0.0), np.inf)
        p_opt_mc, p_cov_mc, _, _, _ = opt.curve_fit(
            efficiency_for_curve_fit_mac_donalds,
            parameter_curve_fit,
            efficiency_array,
            x0,
            full_output=True,
        )
        print("Optimal parameter for McDonald formula", p_opt_mc)
        c_0_opt, c_1_opt, c_2_opt, c_3_opt = p_opt_mc

        loss_mac_donalds_curve_fit = (
            c_0_opt
            + c_1_opt * speed_array_plot
            + c_2_opt * speed_array_plot ** 3.0
            + c_3_opt * torque_array_plot ** 2.0
        )
        efficiency_grid = (
            torque_array_plot
            * speed_array_plot
            / (loss_mac_donalds_curve_fit + torque_array_plot * speed_array_plot)
        )
        loss_mac_cv_donalds_orig_data = (
            c_0_opt
            + c_1_opt * speed_array
            + c_2_opt * speed_array ** 3.0
            + c_3_opt * torque_array ** 2.0
        )
        efficiency_grid_cv_mac_donalds_orig_data = (
            torque_array
            * speed_array
            / (loss_mac_cv_donalds_orig_data + torque_array * speed_array)
        )
        print(
            "RMS McDonald curve_fit:",
            np.sqrt(np.mean((efficiency_grid_cv_mac_donalds_orig_data - efficiency_array) ** 2.0)),
        )

    elif MODEL == "POLITO":

        x0_polito = (1.0, 1.0, 1.0, 1.0, 1.0)
        bounds_polito = ((0.0, 0.0, 0.0, 0.0, 0.0), np.inf)
        p_opt_polito, p_cov_polito, _, _, _ = opt.curve_fit(
            efficiency_for_curve_fit_polito,
            parameter_curve_fit,
            efficiency_array,
            x0_polito,
            bounds=bounds_polito,
            full_output=True,
        )
        print("Optimal parameter for Polito formula", p_opt_polito)
        a, b, c, d, e = p_opt_polito

        loss_polito = (
            a
            + b * torque_array_plot ** 2.0
            + c * speed_array_plot ** 2.0
            + d * (torque_array_plot * speed_array_plot) ** 2.0
            + e * speed_array_plot ** 3.0
        )
        efficiency_grid = (
            torque_array_plot
            * speed_array_plot
            / (loss_polito + torque_array_plot * speed_array_plot)
        )

        loss_cv_polito_orig_data = (
            a
            + b * torque_array ** 2.0
            + c * speed_array ** 2.0
            + d * (torque_array * speed_array) ** 2.0
            + e * speed_array ** 3.0
        )
        efficiency_grid_cv_polito_orig_data = (
            torque_array * speed_array / (loss_cv_polito_orig_data + torque_array * speed_array)
        )
        print(
            "RMS Polito curve_fit:",
            np.sqrt(np.mean((efficiency_grid_cv_polito_orig_data - efficiency_array) ** 2.0)),
        )

    elif MODEL == "POLITO_mod":

        x0_polito = (1.0, 1.0, 1.0, 1.0)
        bounds_polito = ((0.0, 0.0, 0.0, 0.0), np.inf)
        p_opt_polito_mod, p_cov_polito_mod, _, _, _ = opt.curve_fit(
            efficiency_for_curve_fit_polito_mod,
            parameter_curve_fit,
            efficiency_array,
            x0_polito,
            bounds=bounds_polito,
            full_output=True,
        )
        print("Optimal parameter for Polito formula", p_opt_polito_mod)
        a, b, c, d = p_opt_polito_mod

        loss_polito_mod = (
            a
            + b * torque_array_plot ** 2.0
            + c * speed_array_plot ** 2.0
            + d * (torque_array_plot * speed_array_plot) ** 2.0
        )
        efficiency_grid = (
            torque_array_plot
            * speed_array_plot
            / (loss_polito_mod + torque_array_plot * speed_array_plot)
        )

        loss_cv_polito_mod_orig_data = (
            a
            + b * torque_array ** 2.0
            + c * speed_array ** 2.0
            + d * (torque_array * speed_array) ** 2.0
        )
        efficiency_grid_cv_polito_mod_orig_data = (
            torque_array * speed_array / (loss_cv_polito_mod_orig_data + torque_array * speed_array)
        )
        print(
            "RMS Polito curve_fit:",
            np.sqrt(np.mean((efficiency_grid_cv_polito_mod_orig_data - efficiency_array) ** 2.0)),
        )

    elif MODEL == "AUCKLAND":

        x0_auckland = (1.0, 1.0, 1.0, 1.0)
        bounds_polito = ((0.0, 0.0, 0.0, 0.0), np.inf)
        p_opt_auckland, p_cov_auckland, _, _, _ = opt.curve_fit(
            efficiency_for_curve_fit_auckland,
            parameter_curve_fit,
            efficiency_array,
            x0_auckland,
            bounds=bounds_polito,
            full_output=True,
        )
        print("Optimal parameter for Auckland formula", p_opt_auckland)
        a, b, c, d = p_opt_auckland

        loss_auckland = (
            a + b * torque_array_plot ** 2.0 + c * speed_array_plot + d * speed_array_plot ** 1.5
        )
        efficiency_grid = (
            torque_array_plot
            * speed_array_plot
            / (loss_auckland + torque_array_plot * speed_array_plot)
        )

        loss_cv_auckland_orig_data = (
            a + b * torque_array ** 2.0 + c * speed_array + d * speed_array ** 1.5
        )
        efficiency_grid_cv_auckland_orig_data = (
            torque_array * speed_array / (loss_cv_auckland_orig_data + torque_array * speed_array)
        )
        print(
            "RMS Auckland curve_fit:",
            np.sqrt(np.mean((efficiency_grid_cv_auckland_orig_data - efficiency_array) ** 2.0)),
        )

    else:

        x0_vratny = (1.0, 1.0, 1.0, 1.0, 1.0)
        bounds_vratny = ((0.0, 0.0, 0.0, 0.0, 0.0), np.inf)
        p_opt_vratny, p_cov_vratny, _, _, _ = opt.curve_fit(
            efficiency_for_curve_fit_vratny,
            parameter_curve_fit,
            efficiency_array,
            x0_vratny,
            bounds=bounds_vratny,
            full_output=True,
        )
        print("Optimal parameter for Vratny formula", p_opt_vratny)
        a, b, c, d, e = p_opt_vratny

        loss_vratny = (
            a * torque_array_plot ** (1.0 / 1.5) * speed_array_plot
            + b * torque_array_plot
            + c * torque_array_plot * speed_array_plot ** 2.0
            + d * speed_array_plot ** 3.0
            + e * torque_array_plot * speed_array_plot ** 1.25
        )
        efficiency_grid = (
            torque_array_plot
            * speed_array_plot
            / (loss_vratny + torque_array_plot * speed_array_plot)
        )

        loss_cv_vratny_orig_data = (
            a * torque_array ** (1.0 / 1.5) * speed_array
            + b * torque_array
            + c * torque_array * speed_array ** 2.0
            + d * speed_array ** 3.0
            + e * torque_array * speed_array ** 1.25
        )
        efficiency_grid_cv_vratny_orig_data = (
            torque_array * speed_array / (loss_cv_vratny_orig_data + torque_array * speed_array)
        )
        print(
            "RMS Vratny curve_fit:",
            np.sqrt(np.mean((efficiency_grid_cv_vratny_orig_data - efficiency_array) ** 2.0)),
        )

    efficiency_contour = go.Contour(
        x=speed_array_p * 60 / 2 / np.pi,
        y=torque_array_p,
        z=efficiency_grid,
        ncontours=20,
        contours=dict(
            coloring="heatmap",
            showlabels=True,  # show labels on contours
            labelfont=dict(  # label font properties
                size=12,
                color="white",
            ),
        ),
        zmax=np.max(efficiency_array),
        zmin=np.min(efficiency_array),
    )
    fig.add_trace(efficiency_contour)

    if PLOT:
        fig.update_layout(
            title_text="Sampled efficiency map for the PowerPhase "
            + MOTOR
            + " with "
            + MODEL
            + " model",
            title_x=0.5,
            xaxis_title="Speed [rpm]",
            yaxis_title="Torque [Nm]",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )
        fig.show()
