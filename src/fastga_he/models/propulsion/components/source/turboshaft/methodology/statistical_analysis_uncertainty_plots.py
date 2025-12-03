# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import time
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots

if __name__ == "__main__":
    path_to_current_file = pathlib.Path(__file__)
    parent_folder = path_to_current_file.parents[0]
    data_file_path = parent_folder / "data_pt6_reduced_without_outlier.csv"

    data = pd.read_csv(data_file_path)

    rated_power = data["Max. Cont. Shaft Power (kW)"].to_numpy()
    dry_weight = data["Dry Spec. Weight (kg)"].to_numpy()
    diameter = data["Overall Diameter (mm)"].to_numpy()
    length = data["Overall Length (mm)"].to_numpy()

    trend_x = np.linspace(min(rated_power), max(rated_power), 100)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Dry weight", "Overall diameter", "Overall length"),
    )

    count_category_width_weight = 1.25
    count_category_width_diameter = 1.25
    count_category_width_length = 1.25

    # Weight
    # Relative error between predicted and actual value
    uncertainties_weight = ((105.04 + 0.1387 * rated_power) - dry_weight) / dry_weight * 100.0
    # Take the largest error to normalize rest and begin counting how many point per range of errors
    weight_max_relative_error = (np.max(np.abs(uncertainties_weight)) // 5) * 5 + 5
    weight_error_categories = np.arange(
        -weight_max_relative_error,
        weight_max_relative_error + count_category_width_weight,
        count_category_width_weight,
    )
    number_of_point_in_weight_error_category = np.zeros_like(weight_error_categories)
    for idx, weight_error_category in enumerate(weight_error_categories):
        number_of_point_in_weight_error_category[idx] = np.count_nonzero(
            np.abs(uncertainties_weight - weight_error_category)
            <= count_category_width_weight / 2.0
        )

    probability_weight_uncertainty = (
        number_of_point_in_weight_error_category
        / np.sum(number_of_point_in_weight_error_category)
        * 100.0
    )
    mean_weight_uncertainty = np.mean(uncertainties_weight)
    std_weight_uncertainty = np.std(uncertainties_weight)

    bar_chart_weight = go.Bar(
        x=weight_error_categories,
        y=probability_weight_uncertainty,
        marker=dict(
            color=probability_weight_uncertainty,
            colorscale="Viridis",
        ),
    )
    fig.add_trace(
        bar_chart_weight,
        row=1,
        col=1,
    )

    # Diameter
    diameter_trend_y = 2961.3 * trend_x**-0.272
    # Relative error between predicted and actual value
    uncertainties_diameter = ((2961.3 * rated_power**-0.272) - diameter) / diameter * 100.0
    # Take the largest error to normalize rest and begin counting how many point per range of errors
    diameter_max_relative_error = (np.max(np.abs(uncertainties_diameter)) // 5) * 5 + 5
    diameter_error_categories = np.arange(
        -diameter_max_relative_error,
        diameter_max_relative_error + count_category_width_diameter,
        count_category_width_diameter,
    )
    number_of_point_in_diameter_error_category = np.zeros_like(diameter_error_categories)
    for idx, diameter_error_category in enumerate(diameter_error_categories):
        number_of_point_in_diameter_error_category[idx] = np.count_nonzero(
            np.abs(uncertainties_diameter - diameter_error_category)
            <= count_category_width_diameter / 2.0
        )

    probability_diameter_uncertainty = (
        number_of_point_in_diameter_error_category
        / np.sum(number_of_point_in_diameter_error_category)
        * 100.0
    )
    mean_diameter_uncertainty = np.mean(uncertainties_diameter)
    std_diameter_uncertainty = np.std(uncertainties_diameter)

    bar_chart_diamater = go.Bar(
        x=diameter_error_categories,
        y=probability_diameter_uncertainty,
        marker=dict(
            color=probability_diameter_uncertainty,
            colorscale="Viridis",
        ),
    )
    fig.add_trace(
        bar_chart_diamater,
        row=1,
        col=2,
    )

    # Length
    # Relative error between predicted and actual value
    uncertainties_length = ((0.6119 * rated_power + 1314.9) - length) / length * 100.0
    # Take the largest error to normalize rest and begin counting how many point per range of errors
    length_max_relative_error = (np.max(np.abs(uncertainties_length)) // 5) * 5 + 5
    length_error_categories = np.arange(
        -length_max_relative_error,
        length_max_relative_error + count_category_width_length,
        count_category_width_length,
    )
    number_of_point_in_length_error_category = np.zeros_like(length_error_categories)
    for idx, length_error_category in enumerate(length_error_categories):
        number_of_point_in_length_error_category[idx] = np.count_nonzero(
            np.abs(uncertainties_length - length_error_category)
            <= count_category_width_length / 2.0
        )

    probability_length_uncertainty = (
        number_of_point_in_length_error_category
        / np.sum(number_of_point_in_length_error_category)
        * 100.0
    )
    mean_length_uncertainty = np.mean(uncertainties_length)
    std_length_uncertainty = np.std(uncertainties_length)

    bar_chart_length = go.Bar(
        x=length_error_categories,
        y=probability_length_uncertainty,
        marker=dict(
            color=probability_length_uncertainty,
            colorscale="Viridis",
        ),
    )
    fig.add_trace(
        bar_chart_length,
        row=1,
        col=3,
    )

    fig.update_layout(
        height=600,
        width=1900,
        showlegend=False,
        font=dict(size=25),
        bargap=0,
        bargroupgap=0,
        margin=dict(l=5, r=5, t=60, b=5),
        barmode="stack",
        plot_bgcolor="white",
    )
    fig.update_xaxes(
        ticks="outside",
        tickfont=dict(size=25),
        range=[-27, 27],
        row=1,
        col=1,
        dtick=10.0,
        linecolor="black",
        gridcolor="lightgrey",
        showgrid=True,
        showline=True,
        linewidth=3,
    )
    fig.update_yaxes(
        ticks="outside",
        showgrid=True,
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        linewidth=3,
        tickfont=dict(size=25),
        row=1,
        col=1,
        title="Probability",
        range=[0, 35],
    )

    fig.update_xaxes(
        ticks="outside",
        tickfont=dict(size=25),
        range=[-27, 27],
        row=1,
        col=2,
        dtick=10.0,
        linecolor="black",
        gridcolor="lightgrey",
        showgrid=True,
        showline=True,
        linewidth=3,
    )
    fig.update_yaxes(
        ticks="outside",
        showgrid=True,
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        linewidth=3,
        tickfont=dict(size=25),
        row=1,
        col=2,
        range=[0, 35],
    )

    fig.update_xaxes(
        ticks="outside",
        tickfont=dict(size=25),
        range=[-27, 27],
        row=1,
        col=3,
        dtick=10.0,
        linecolor="black",
        gridcolor="lightgrey",
        showgrid=True,
        showline=True,
        linewidth=3,
    )
    fig.update_yaxes(
        ticks="outside",
        showgrid=True,
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        linewidth=3,
        tickfont=dict(size=25),
        row=1,
        col=3,
        range=[0, 35],
    )

    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["yaxis2"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis2"]["tickfont"]["size"] = 20
    fig["layout"]["yaxis3"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis3"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 30
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.update_annotations(font_size=30)

    # Looks tiny but will be cleaned up for the thesis elsewhere
    annotation_wright_graph = (
        "$\overline\epsilon="
        + str(np.round(mean_weight_uncertainty, 2))
        + "\%, \sigma_\epsilon="
        + str(np.round(std_weight_uncertainty, 2))
        + "\%$"
    )
    fig.update_xaxes(title_text=annotation_wright_graph, row=1, col=1)
    annotation_diameter_graph = (
        "$\overline\epsilon="
        + str(np.round(mean_diameter_uncertainty, 2))
        + "\%, \sigma_\epsilon="
        + str(np.round(std_diameter_uncertainty, 2))
        + "\%$"
    )
    fig.update_xaxes(title_text=annotation_diameter_graph, row=1, col=2)
    annotation_length_graph = (
        "$\overline\epsilon="
        + str(np.round(mean_length_uncertainty, 2))
        + "\%, \sigma_\epsilon="
        + str(np.round(std_length_uncertainty, 2))
        + "\%$"
    )
    fig.update_xaxes(title_text=annotation_length_graph, row=1, col=3)

    fig.show()

    svg_path = "results/pt6a_family_uncertainties.svg"

    fig.update_layout(title=None)
    pio.write_image(fig, svg_path, width=1600, height=900)
    time.sleep(3)
    pio.write_image(fig, svg_path, width=1600, height=900)
