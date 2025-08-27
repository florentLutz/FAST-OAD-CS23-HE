import time
import pathlib

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio

if __name__ == "__main__":
    turboshaft_rated_power = 321.0
    gearbox_efficiency = 0.98

    path_to_current_file = pathlib.Path(__file__)
    parent_folder = path_to_current_file.parents[0]
    results_folder_path = parent_folder / "results"
    pt_watcher_results_file_path = results_folder_path / "hybridized_kodiak_100.csv"

    pt_watcher_data_pd = pd.read_csv(pt_watcher_results_file_path)
    power_profile = pt_watcher_data_pd["propeller_1 shaft_power_in [kW]"].to_numpy()[1:-1]

    mission_results_file_path = results_folder_path / "hybrid_propulsion.csv"

    mission_data_pd = pd.read_csv(mission_results_file_path)
    time_array = mission_data_pd["time"].to_numpy()

    # For a clean plot, we need to add a point where the power profile intersects the turboshaft rated power. Let's
    # do it in a generic manner which imposes that we mathematically find the intersections knowing the last climb
    # point (point of index 29) and first cruise point (point of index 30)
    last_climb_time = time_array[29]
    last_climb_power = power_profile[29]

    first_cruise_time = time_array[30]
    first_cruise_power = power_profile[30]

    a = (first_cruise_power - last_climb_power) / (first_cruise_time - last_climb_time)
    b = last_climb_power - a * last_climb_time

    x_intersection = (turboshaft_rated_power * gearbox_efficiency - b) / a

    time_array = np.insert(time_array, 30, x_intersection)
    power_profile = np.insert(power_profile, 30, turboshaft_rated_power * gearbox_efficiency)

    end_elec_index = np.argwhere(time_array == x_intersection)[0][0]
    elec_power_profile = power_profile[: end_elec_index + 1]

    fig = go.Figure()

    elec_power_color = "rgba(65,105,225,0.5)"

    scatter_turboshaft_rated_power_elec_part = go.Scatter(
        x=[min(time_array / 3600), x_intersection / 3600],
        y=[
            turboshaft_rated_power * gearbox_efficiency,
            turboshaft_rated_power * gearbox_efficiency,
        ],
        mode="lines",
        showlegend=False,
        line=dict(color="gray", dash="dash"),
    )
    fig.add_trace(scatter_turboshaft_rated_power_elec_part)

    scatter = go.Scatter(
        x=time_array / 3600,
        y=elec_power_profile,
        mode="lines",
        showlegend=True,
        line=dict(color=elec_power_color),
        marker=dict(color=elec_power_color),
        fill="tonexty",
        fillcolor=elec_power_color,
        fillpattern=dict(shape="\\"),
        name="Electrical power",
    )
    fig.add_trace(scatter)

    scatter_turboshaft_power = go.Scatter(
        x=[x_intersection / 3600, max(time_array / 3600)],
        y=[
            turboshaft_rated_power * gearbox_efficiency,
            turboshaft_rated_power * gearbox_efficiency,
        ],
        mode="lines",
        showlegend=False,
        line=dict(color="gray", dash="dash"),
    )
    fig.add_trace(scatter_turboshaft_power)

    thermal_power_color = "rgba(255,69,0,0.666)"
    turboshaft_power_fill = go.Scatter(
        x=time_array / 3600,
        y=np.minimum(power_profile, turboshaft_rated_power * gearbox_efficiency),
        fill="tozeroy",
        mode="lines",
        showlegend=True,
        line=dict(color=thermal_power_color),
        marker=dict(color=thermal_power_color),
        fillcolor=thermal_power_color,
        fillpattern=dict(shape="/"),
        name="Thermal power",
    )
    fig.add_trace(turboshaft_power_fill)

    fig.update_layout(
        xaxis_title="Flight time [h]",
        yaxis_title="Propeller shaft power [kW]",
        height=600,
        width=1900,
        font=dict(size=20),
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    fig.update_xaxes(
        ticks="outside",
        title_font=dict(size=15),
        tickfont=dict(size=15),
        showline=True,
        linecolor="black",
        linewidth=3,
        gridcolor="lightgrey",
    )

    # Add 10% to the range for proper display of the topmost points
    fig.update_yaxes(
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        linewidth=3,
        tickfont=dict(size=15),
        title_font=dict(size=15),
    )

    fig.add_annotation(
        x=1,
        y=turboshaft_rated_power,
        xref="paper",
        text="Turboshaft rated power",
        bgcolor="rgba(255,255,255,0.75)",
        font=dict(color="grey", size=20),
        xanchor="right",
        yanchor="bottom",
        align="right",
        showarrow=False
    )

    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    fig.show()

    pdf_path = "results/hybrid_kodiak_power_profile.pdf"

    fig.update_layout(title=None)
    pio.write_image(fig, pdf_path, width=1900, height=600)
    time.sleep(3)
    pio.write_image(fig, pdf_path, width=1900, height=600)
