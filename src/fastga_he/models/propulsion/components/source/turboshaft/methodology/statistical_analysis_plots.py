# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

if __name__ == "__main__":

    path_to_current_file = pathlib.Path(__file__)
    parent_folder = path_to_current_file.parents[0]
    data_file_path = parent_folder / "data_pt6_reduced.csv"

    data = pd.read_csv(data_file_path)

    rated_power = data["Max. Cont. Shaft Power (kW)"].to_numpy()
    dry_weight = data["Dry Spec. Weight (kg)"].to_numpy()
    diameter = data["Overall Diameter (mm)"].to_numpy()
    length = data["Overall Length (mm)"].to_numpy()

    trend_x = np.linspace(min(rated_power), max(rated_power), 100)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Dry weight (kg)", "Overall diameter (mm)", "Overall length (mm)"),
    )

    # Weight
    data_scatter_weight = go.Scatter(
        x=rated_power,
        y=dry_weight,
        mode="markers",
        marker=dict(color="black", symbol="circle", size=8),
    )
    fig.add_trace(
        data_scatter_weight,
        row=1,
        col=1,
    )
    weight_trend_y = 105.04 + 0.1387 * trend_x
    trend_scatter_weight = go.Scatter(
        x=trend_x,
        y=weight_trend_y,
        mode="lines",
        line=dict(color="grey", dash="dash"),
    )
    fig.add_trace(
        trend_scatter_weight,
        row=1,
        col=1,
    )

    # Diameter
    data_scatter_diameter = go.Scatter(
        x=rated_power,
        y=diameter,
        mode="markers",
        marker=dict(color="black", symbol="circle", size=8),
    )
    fig.add_trace(
        data_scatter_diameter,
        row=1,
        col=2,
    )
    diameter_trend_y = 2961.3 * trend_x ** -0.272
    trend_scatter_diameter = go.Scatter(
        x=trend_x,
        y=diameter_trend_y,
        mode="lines",
        line=dict(color="grey", dash="dash"),
    )
    fig.add_trace(
        trend_scatter_diameter,
        row=1,
        col=2,
    )

    # Length
    data_scatter_length = go.Scatter(
        x=rated_power,
        y=length,
        mode="markers",
        marker=dict(color="black", symbol="circle", size=8),
    )
    fig.add_trace(
        data_scatter_length,
        row=1,
        col=3,
    )
    length_trend_y = 0.6119 * trend_x + 1314.9
    trend_scatter_length = go.Scatter(
        x=trend_x,
        y=length_trend_y,
        mode="lines",
        line=dict(color="grey", dash="dash"),
    )
    fig.add_trace(
        trend_scatter_length,
        row=1,
        col=3,
    )

    fig.update_layout(
        height=600,
        width=1900,
        title_text="Statistical analysis PT6A Family",
        title_x=0.5,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Rated Power (kW)", row=1, col=1)
    fig.update_xaxes(title_text="Rated Power (kW)", row=1, col=2)
    fig.update_xaxes(title_text="Rated Power (kW)", row=1, col=3)

    fig.show()
