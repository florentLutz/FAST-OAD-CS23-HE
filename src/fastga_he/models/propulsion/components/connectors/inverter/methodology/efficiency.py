# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import pandas as pd

import numpy as np
import plotly.graph_objects as go

if __name__ == "__main__":

    # Obtaining the reference parameters
    data_file = pth.join(pth.dirname(__file__), "data/new_reference_IGBT_module.csv")

    energy_data = pd.read_csv(data_file)

    e_on_x = np.array(energy_data["E_on_X"])
    e_on_y = np.array(energy_data["E_on_Y"])

    e_rr_x = np.array(energy_data["E_rr_X"])
    e_rr_y = np.array(energy_data["E_rr_Y"])

    e_off_x = np.array(energy_data["E_off_X"])
    e_off_y = np.array(energy_data["E_off_Y"])

    a_on, b_on, c_on = np.polyfit(e_on_x, e_on_y, 2)
    print("ON Coefficient", a_on, b_on, c_on)
    a_rr, b_rr, c_rr = np.polyfit(e_rr_x, e_rr_y, 2)
    print("RR Coefficient", a_rr, b_rr, c_rr)
    a_off, b_off, c_off = np.polyfit(e_off_x, e_off_y, 2)
    print("OFF Coefficient", a_off, b_off, c_off)

    x_plot = np.linspace(np.amin(e_on_x), np.amax(e_on_x))

    fig = go.Figure()

    scatter_e_on_orig = go.Scatter(
        x=e_on_x,
        y=e_on_y,
        mode="lines+markers",
        name="E_on original data",
        legendgroup="e_on",
        legendgrouptitle_text="E_on",
    )
    fig.add_trace(scatter_e_on_orig)
    scatter_e_on_interp = go.Scatter(
        x=e_on_x,
        y=np.polyval([a_on, b_on, c_on], e_on_x),
        mode="lines+markers",
        name="E_on interpolated data",
        legendgroup="e_on",
    )
    fig.add_trace(scatter_e_on_interp)

    scatter_e_rr_orig = go.Scatter(
        x=e_rr_x,
        y=e_rr_y,
        mode="lines+markers",
        name="E_rr original data",
        legendgroup="e_rr",
        legendgrouptitle_text="E_rr",
    )
    fig.add_trace(scatter_e_rr_orig)
    scatter_e_rr_interp = go.Scatter(
        x=e_rr_x,
        y=np.polyval([a_rr, b_rr, c_rr], e_rr_x),
        mode="lines+markers",
        name="E_rr interpolated data",
        legendgroup="e_rr",
    )
    fig.add_trace(scatter_e_rr_interp)

    scatter_e_off_orig = go.Scatter(
        x=e_off_x,
        y=e_off_y,
        mode="lines+markers",
        name="E_off original data",
        legendgroup="e_off",
        legendgrouptitle_text="E_off",
    )
    fig.add_trace(scatter_e_off_orig)
    scatter_e_off_interp = go.Scatter(
        x=e_off_x,
        y=np.polyval([a_off, b_off, c_off], e_off_x),
        mode="lines+markers",
        name="E_off interpolated data",
        legendgroup="e_off",
    )
    fig.add_trace(scatter_e_off_interp)

    fig.show()
