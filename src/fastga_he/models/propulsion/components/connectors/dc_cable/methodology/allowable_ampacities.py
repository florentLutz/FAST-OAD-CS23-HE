# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import numpy as np
import plotly.graph_objects as go

import pandas as pd

ORDER = 5

if __name__ == "__main__":
    data_file = pth.join(pth.dirname(__file__), "data/allowable_ampacities.csv")

    ampacities_data = pd.read_csv(data_file)

    area_log = np.log(ampacities_data["AREA_MM2"])
    copper_ampacities = ampacities_data["COPPER"]
    aluminium_ampacities = ampacities_data["ALUMINIUM"]

    fig0 = go.Figure()
    scatter_cu_orig = go.Scatter(
        x=copper_ampacities,
        y=np.exp(area_log),
        mode="lines+markers",
        name="Copper original data",
    )
    fig0.add_trace(scatter_cu_orig)
    scatter_al_orig = go.Scatter(
        x=aluminium_ampacities,
        y=np.exp(area_log),
        mode="lines+markers",
        name="Aluminium original data",
    )
    fig0.add_trace(scatter_al_orig)
    fig0.update_layout(
        title_text="Cross section",
        title_x=0.5,
        xaxis_title="Ampacities [A]",
        yaxis_title="Area [mm2]",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    # fig0.show()

    polyfit_copper = np.polyfit(area_log, copper_ampacities, ORDER)
    polyfit_aluminium = np.polyfit(area_log, aluminium_ampacities, ORDER)

    fig = go.Figure()

    scatter_cu_orig = go.Scatter(
        x=np.exp(area_log),
        y=copper_ampacities,
        mode="markers",
        name="Copper original data",
        legendgroup="copper",
        legendgrouptitle_text="Copper",
    )
    fig.add_trace(scatter_cu_orig)
    scatter_al_orig = go.Scatter(
        x=np.exp(area_log),
        y=aluminium_ampacities,
        mode="markers",
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
    # fig.show()

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
    polyfit_alu_inv = np.polyfit(aluminium_ampacities, area_log, ORDER)

    print(polyfit_copper_inv)
    print(polyfit_alu_inv)

    mean_error_copper_inv = (
        np.mean(np.abs(area_log - np.polyval(polyfit_copper_inv, copper_ampacities)) / area_log)
        * 100.0
    )
    max_error_copper_inv = (
        np.max(np.abs(area_log - np.polyval(polyfit_copper_inv, copper_ampacities)) / area_log)
        * 100.0
    )

    mean_error_alu_inv = (
        np.mean(np.abs(area_log - np.polyval(polyfit_alu_inv, aluminium_ampacities)) / area_log)
        * 100.0
    )
    max_error_alu_inv = (
        np.max(np.abs(area_log - np.polyval(polyfit_alu_inv, aluminium_ampacities)) / area_log)
        * 100.0
    )

    print("Mean error copper inv: ", mean_error_copper_inv)
    print("Max error copper inv: ", max_error_copper_inv)
    print("Max error alu inv: ", mean_error_alu_inv)
    print("Max error alu inv: ", max_error_alu_inv)

    print("700 Amp log", np.polyval(polyfit_copper_inv, np.array([700])))
    print("700 Amp solution", np.exp(np.polyval(polyfit_copper_inv, np.array([700]))))

    fig2 = go.Figure()

    scatter_cu_orig = go.Scatter(
        y=np.exp(area_log),
        x=copper_ampacities,
        mode="markers",
        name="Copper original data",
        legendgroup="copper",
        legendgrouptitle_text="Copper",
        marker_size=15,
        marker_symbol="circle",
        marker=dict(color="red"),
    )
    fig2.add_trace(scatter_cu_orig)
    scatter_al_orig = go.Scatter(
        y=np.exp(area_log),
        x=aluminium_ampacities,
        mode="markers",
        name="Aluminium original data",
        legendgroup="aluminium",
        legendgrouptitle_text="Aluminium",
        marker_size=15,
        marker_symbol="circle",
        marker=dict(color="grey"),
    )
    fig2.add_trace(scatter_al_orig)

    x_values_plot_copper = np.linspace(min(copper_ampacities), max(copper_ampacities), 1000)
    scatter_cu = go.Scatter(
        y=np.exp(np.polyval(polyfit_copper_inv, x_values_plot_copper)),
        x=x_values_plot_copper,
        mode="lines",
        name="Copper interpolated data",
        legendgroup="copper",
        line=dict(color="red"),
    )
    fig2.add_trace(scatter_cu)
    x_values_plot_alu = np.linspace(min(aluminium_ampacities), max(aluminium_ampacities), 1000)
    scatter_al = go.Scatter(
        y=np.exp(np.polyval(polyfit_alu_inv, x_values_plot_alu)),
        x=x_values_plot_alu,
        mode="lines",
        name="Aluminium interpolated data",
        legendgroup="aluminium",
        line=dict(color="grey"),
    )
    fig2.add_trace(scatter_al)

    fig2.update_layout(
        title_text="Polynomial fit on conductor surface",
        title_x=0.5,
        yaxis_title="Area [mm2]",
        xaxis_title="Ampacities [A]",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=800,
        width=1600,
    )
    fig2.show()
    fig2.write_image("conductor_scaling.pdf")
    # Works well but only on the values used for the extrapolation

    print("========== With power law ================")

    B = np.log(ampacities_data["AREA_MM2"])
    A = np.column_stack([np.ones_like(copper_ampacities), np.log(copper_ampacities)])
    x = np.linalg.lstsq(A, B, rcond=None)
    a_copper, b_copper = x[0]
    print(a_copper, b_copper)

    A = np.column_stack([np.ones_like(aluminium_ampacities), np.log(aluminium_ampacities)])
    x = np.linalg.lstsq(A, B, rcond=None)
    a_alu, b_alu = x[0]
    print(a_alu, b_alu)

    fig3 = go.Figure()
    orig_data_copper = go.Scatter(
        x=copper_ampacities,
        y=ampacities_data["AREA_MM2"],
        mode="lines+markers",
        name="Copper original data",
        legendgroup="copper",
        legendgrouptitle_text="Copper",
    )
    fig3.add_trace(orig_data_copper)
    scatter_cu = go.Scatter(
        x=np.linspace(25, 1000.0),
        y=np.exp(a_copper) * np.linspace(25, 1000.0) ** b_copper,
        mode="lines+markers",
        name="Copper interpolated data",
        legendgroup="copper",
    )
    fig3.add_trace(scatter_cu)

    orig_data_aluminium = go.Scatter(
        x=aluminium_ampacities,
        y=ampacities_data["AREA_MM2"],
        mode="lines+markers",
        name="Aluminium original data",
        legendgroup="aluminium",
        legendgrouptitle_text="Aluminium",
    )
    fig3.add_trace(orig_data_aluminium)
    scatter_alu = go.Scatter(
        x=np.linspace(25, 1000.0),
        y=np.exp(a_alu) * np.linspace(25, 1000.0) ** b_alu,
        mode="lines+markers",
        name="Aluminium interpolated data",
        legendgroup="aluminium",
    )
    fig3.add_trace(scatter_alu)

    fig3.update_layout(
        title_text="Cable cross-section",
        title_x=0.5,
        xaxis_title="Ampacities [A]",
        yaxis_title="Area [mm2]",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    # fig3.show()
