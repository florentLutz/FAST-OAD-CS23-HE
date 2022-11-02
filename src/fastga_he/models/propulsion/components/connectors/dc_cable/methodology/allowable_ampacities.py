# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import numpy as np
import plotly.graph_objects as go

import pandas as pd

ORDER = 2

if __name__ == "__main__":

    data_file = pth.join(pth.dirname(__file__), "data/allowable_ampacities.csv")

    ampacities_data = pd.read_csv(data_file)

    area_log = ampacities_data["LOG_AREA"]
    copper_ampacities = ampacities_data["COPPER"]
    aluminium_ampacities = ampacities_data["ALUMINIUM"]

    polyfit_copper = np.polyfit(area_log, copper_ampacities, ORDER)
    polyfit_aluminium = np.polyfit(area_log, aluminium_ampacities, ORDER)

    fig = go.Figure()

    scatter_cu_orig = go.Scatter(
        x=np.exp(area_log),
        y=copper_ampacities,
        mode="lines+markers",
        name="Copper original data",
        legendgroup="copper",
        legendgrouptitle_text="Copper",
    )
    fig.add_trace(scatter_cu_orig)
    scatter_al_orig = go.Scatter(
        x=np.exp(area_log),
        y=aluminium_ampacities,
        mode="lines+markers",
        name="Aluminium original data",
        legendgroup="aluminium",
        legendgrouptitle_text="Aluminium",
    )
    fig.add_trace(scatter_al_orig)

    scatter_cu = go.Scatter(
        x=np.exp(area_log),
        y=np.polyval(polyfit_copper, area_log),
        mode="lines+markers",
        name="Copper interpolated data",
        legendgroup="copper",
    )
    fig.add_trace(scatter_cu)
    scatter_al = go.Scatter(
        x=np.exp(area_log),
        y=np.polyval(polyfit_aluminium, area_log),
        mode="lines+markers",
        name="Aluminium interpolated data",
        legendgroup="aluminium",
    )
    fig.add_trace(scatter_al)

    fig.update_layout(
        title_text="Polynomial fit on allowable ampacities",
        title_x=0.5,
        xaxis_title="Log(Area) [Log(mm2)]",
        yaxis_title="Ampacities [A]",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(type="log")
    fig.show()

    print(polyfit_copper)
    print(polyfit_aluminium)

    mean_error_copper = (
        np.mean((copper_ampacities - np.polyval(polyfit_copper, area_log)) / copper_ampacities)
        * 100.0
    )
    max_error_copper = (
        np.max((copper_ampacities - np.polyval(polyfit_copper, area_log)) / copper_ampacities)
        * 100.0
    )
    mean_error_aluminium = (
        np.mean(
            (aluminium_ampacities - np.polyval(polyfit_aluminium, area_log)) / aluminium_ampacities
        )
        * 100.0
    )
    max_error_aluminium = (
        np.max(
            (aluminium_ampacities - np.polyval(polyfit_aluminium, area_log)) / aluminium_ampacities
        )
        * 100.0
    )

    print("Mean error copper: ", mean_error_copper)
    print("Max error copper: ", max_error_copper)
    print("Mean error aluminium: ", mean_error_aluminium)
    print("Max error aluminium: ", max_error_aluminium)

    print("========== Other way around ================")
    polyfit_copper_inv = np.polyfit(copper_ampacities, area_log, ORDER)

    print(polyfit_copper_inv)

    mean_error_copper_inv = (
        np.mean((area_log - np.polyval(polyfit_copper_inv, copper_ampacities)) / area_log) * 100.0
    )
    max_error_copper_inv = (
        np.max((area_log - np.polyval(polyfit_copper_inv, copper_ampacities)) / area_log) * 100.0
    )
    print("Mean error copper inv: ", mean_error_copper_inv)
    print("Max error copper inv: ", max_error_copper_inv)
