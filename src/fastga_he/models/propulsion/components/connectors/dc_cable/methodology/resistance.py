# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import numpy as np
import plotly.graph_objects as go

import pandas as pd

ORDER = 1

if __name__ == "__main__":

    # Based on the data from
    # https://www.engineeringtoolbox.com/copper-aluminum-conductor-resistance-d_1877.html

    data_file = pth.join(pth.dirname(__file__), "data/electrical_resistance.csv")

    resistance_data = pd.read_csv(data_file)

    area = resistance_data["AREA_MM2"]
    log_area = np.log(area)
    copper_resistance = resistance_data["COPPER"]
    aluminium_resistance = resistance_data["ALUMINIUM"]

    polyfit_copper = np.polyfit(log_area, np.log(copper_resistance), ORDER)
    polyfit_aluminium = np.polyfit(log_area, np.log(aluminium_resistance), ORDER)

    fig = go.Figure()

    scatter_cu_orig = go.Scatter(
        x=area,
        y=copper_resistance,
        mode="lines+markers",
        name="Copper original data",
        legendgroup="copper",
        legendgrouptitle_text="Copper",
    )
    fig.add_trace(scatter_cu_orig)
    scatter_al_orig = go.Scatter(
        x=area,
        y=aluminium_resistance,
        mode="lines+markers",
        name="Aluminium original data",
        legendgroup="aluminium",
        legendgrouptitle_text="Aluminium",
    )
    fig.add_trace(scatter_al_orig)

    scatter_cu = go.Scatter(
        x=area,
        y=np.exp(np.polyval(polyfit_copper, log_area)),
        mode="lines+markers",
        name="Copper interpolated data",
        legendgroup="copper",
    )
    fig.add_trace(scatter_cu)
    scatter_al = go.Scatter(
        x=area,
        y=np.exp(np.polyval(polyfit_aluminium, log_area)),
        mode="lines+markers",
        name="Aluminium interpolated data",
        legendgroup="aluminium",
    )
    fig.add_trace(scatter_al)

    fig.update_layout(
        title_text="Polynomial fit on electrical resistance",
        title_x=0.5,
        xaxis_title="Area [mm2]",
        yaxis_title="Resistance-per-length [Ohm/km]",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.show()

    print(polyfit_copper)
    print(polyfit_aluminium)
