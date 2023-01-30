# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import numpy as np
import plotly.graph_objects as go
import scipy.optimize as opt

from utils.curve_reading import curve_reading

PLOT = False
MOTOR = "268"  # "188", "208", "228", "268", "348"
THRESHOLD = 0.89
MODEL = "FULL"  # , "SIGNIFICANT"


def efficiency_for_curve_fit(
    parameter,
    alpha_test,
    beta_test,
    gamma_test,
    delta_test,
    epsilon_test,
):

    speed, torque = parameter

    power = speed * torque
    power_losses = (
        alpha_test * torque ** 2.0
        + beta_test * speed
        + gamma_test * speed ** 1.5
        + delta_test * speed ** 3.0
        + epsilon_test
    )

    computed_efficiency = power / (power + power_losses)

    return computed_efficiency


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    data_file = pth.join(pth.dirname(__file__), "data/emrax_" + MOTOR + "_efficiency_plot.csv")

    speed_array, torque_array, efficiency_array, fig = curve_reading(data_file, threshold=THRESHOLD)
    if MOTOR == "188":
        max_torque_orig = 50.0
        max_speed_orig = 6500.0 * 2 * np.pi / 60
    elif MOTOR == "208":
        max_torque_orig = 80.0
        max_speed_orig = 6000.0 * 2 * np.pi / 60
    elif MOTOR == "228":
        max_torque_orig = 120.0
        max_speed_orig = 5500.0 * 2 * np.pi / 60
    elif MOTOR == "268":
        max_torque_orig = 250.0
        max_speed_orig = 5000.0 * 2 * np.pi / 60
    else:
        max_torque_orig = 500.0
        max_speed_orig = 4500.0 * 2 * np.pi / 60

    proper_idx = np.where(torque_array < max_torque_orig)[0]
    speed_array = speed_array[proper_idx]
    torque_array = torque_array[proper_idx]
    efficiency_array = efficiency_array[proper_idx]
    speed_array_p = np.linspace(50.0, 1.1 * np.max(speed_array), 50)
    torque_array_p = np.linspace(10.0, 1.1 * np.max(torque_array), 50)
    [speed_array_plot, torque_array_plot] = np.meshgrid(speed_array_p, torque_array_p)

    parameter_curve_fit = np.vstack(
        (
            speed_array,
            torque_array,
        )
    )

    x0 = (7e-2, 7e-2, 7e-2, 7e-2, 7e-2)
    bnds = ((-0.0, -0.0, -0.0, -0.0, -0.0), (100.0, 100.0, 100.0, 100.0, 1e6))

    p_opt, p_cov, _, _, _ = opt.curve_fit(
        efficiency_for_curve_fit,
        parameter_curve_fit,
        efficiency_array,
        x0,
        bounds=bnds,
        full_output=True,
    )
    print("Optimal parameter", p_opt)
    alpha, beta, gamma, delta, epsilon = p_opt

    power_loss = (
        alpha * torque_array_plot ** 2.0
        + beta * speed_array_plot
        + gamma * speed_array_plot ** 1.5
        + delta * speed_array_plot ** 3.0
        + epsilon
    )
    efficiency_grid = (
        torque_array_plot * speed_array_plot / (power_loss + torque_array_plot * speed_array_plot)
    ).reshape((50, 50))

    power_loss_orig_data = (
        alpha * torque_array ** 2.0
        + beta * speed_array
        + gamma * speed_array ** 1.5
        + delta * speed_array ** 3.0
        + epsilon
    )
    efficiency_grid_orig_data = (
        torque_array * speed_array / (power_loss_orig_data + torque_array * speed_array)
    )
    print("RMS:", np.sqrt(np.mean((efficiency_grid_orig_data - efficiency_array) ** 2.0)))

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
            title_text="Sampled efficiency map for the EMRAX " + MOTOR,
            title_x=0.5,
            xaxis_title="Speed [rpm]",
            yaxis_title="Torque [Nm]",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )
        fig.show()
