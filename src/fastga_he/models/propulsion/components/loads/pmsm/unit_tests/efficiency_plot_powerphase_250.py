# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as opt

PLOT = True


def diff_to_target_efficiency(design_var, speed, torque, efficiency_target):

    alpha_test = design_var[0]
    beta_test = design_var[1]

    power = speed * torque
    power_losses = alpha_test * torque ** 2.0 + beta_test * speed ** 1.5

    computed_efficiency = power / (power + power_losses)

    return computed_efficiency - efficiency_target


if __name__ == "__main__":

    data_file = pth.join(pth.dirname(__file__), "data/powerphase_250_efficiency_plot.csv")

    efficiency_data = pd.read_csv(data_file)

    fig = go.Figure()

    efficiency = 1.0
    x = np.array([])
    y = np.array([])

    speed_array = np.array([])
    torque_array = np.array([])
    efficiency_array = np.array([])

    for idx, name in enumerate(efficiency_data.columns):

        if idx % 2 == 0:
            efficiency = float(name) / 100.0
            x = np.array(efficiency_data[name][1:]).astype(float)
            x = x[np.logical_not(np.isnan(x))]

            speed_array = np.concatenate((speed_array, x * 2.0 * np.pi / 60))
            efficiency_array = np.concatenate((efficiency_array, np.full_like(x, efficiency)))

        else:
            y = np.array(efficiency_data[name][1:]).astype(float)
            y = y[np.logical_not(np.isnan(y))]

            torque_array = np.concatenate((torque_array, y))

            scatter = go.Scatter(x=x, y=y, mode="lines+markers", name=efficiency)
            fig.add_trace(scatter)

    x0 = (7e-4, 7e-4)
    bnds = ((0, 0.0), (1.0, 1.0))
    arguments = (speed_array, torque_array, efficiency_array)

    res = opt.least_squares(diff_to_target_efficiency, x0, bounds=bnds, args=arguments)

    alpha, beta = res.x

    print("alpha", alpha)
    print("beta", beta)
    print(np.mean(res.fun))

    speed_array_p = np.linspace(np.min(speed_array), np.max(speed_array), 50)
    torque_array_p = np.linspace(np.min(torque_array), np.max(torque_array), 50)
    [speed_array_plot, torque_array_plot] = np.meshgrid(speed_array_p, torque_array_p)
    power_loss = alpha * torque_array_plot ** 2.0 + beta * speed_array_plot ** 1.5
    efficiency_grid = (
        torque_array_plot * speed_array_plot / (power_loss + torque_array_plot * speed_array_plot)
    )
    efficiency_grid = efficiency_grid.reshape((50, 50))

    omega_hat = 4500 * 2 * np.pi / 60.0
    q_hat = 290.0
    eta_hat = 0.950

    c_0 = 0.5 * omega_hat * q_hat / 6 * (1 - eta_hat) / eta_hat
    c_1 = -3 * c_0 / 2 / omega_hat + q_hat * (1 - eta_hat) / 4 / eta_hat
    c_2 = c_0 / 2 / omega_hat ** 3.0 + q_hat * (1 - eta_hat) / 4 / eta_hat / omega_hat ** 2.0
    c_3 = omega_hat * (1 - eta_hat) / 2 / q_hat / eta_hat

    loss_mac_donalds = (
        c_0
        + c_1 * speed_array_plot
        + c_2 * speed_array_plot ** 3.0
        + c_3 * torque_array_plot ** 2.0
    )
    efficiency_grid_mac_donalds = (
        torque_array_plot
        * speed_array_plot
        / (loss_mac_donalds + torque_array_plot * speed_array_plot)
    )

    loss_mac_donalds_orig = (
        c_0 + c_1 * speed_array + c_2 * speed_array ** 3.0 + c_3 * torque_array ** 2.0
    )
    efficiency_grid_mac_donalds_orig = (
        torque_array * speed_array / (loss_mac_donalds_orig + torque_array * speed_array)
    )
    print(np.mean((efficiency_grid_mac_donalds_orig - efficiency_array) ** 2.0))

    efficiency_contour = go.Contour(
        x=speed_array_p * 60 / 2 / np.pi,
        y=torque_array_p,
        z=efficiency_grid_mac_donalds,
        ncontours=20,
        contours=dict(
            coloring="heatmap",
            showlabels=True,  # show labels on contours
            labelfont=dict(  # label font properties
                size=12,
                color="white",
            ),
        ),
        zmax=0.95,
        zmin=0.85,
    )
    fig.add_trace(efficiency_contour)

    if PLOT:
        fig.update_layout(
            title_text="Sampled efficiency map for the PowerPhase 250",
            title_x=0.5,
            xaxis_title="Speed [rpm]",
            yaxis_title="Torque [Nm]",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )
        fig.show()
