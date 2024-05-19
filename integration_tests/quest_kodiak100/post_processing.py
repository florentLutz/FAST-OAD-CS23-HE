import pathlib

import numpy as np
import pandas as pd

import plotly.graph_objects as go

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
    time = mission_data_pd["time"].to_numpy()

    # For a clean plot, we need to add a point where the power profile intersects the turboshaft rated power. Let's
    # do it in a generic manner which imposes that we mathematically find the intersections knowing the last climb
    # point (point of index 29) and first cruise point (point of index 30)
    last_climb_time = time[29]
    last_climb_power = power_profile[29]

    first_cruise_time = time[30]
    first_cruise_power = power_profile[30]

    a = (first_cruise_power - last_climb_power) / (first_cruise_time - last_climb_time)
    b = last_climb_power - a * last_climb_time

    x_intersection = (turboshaft_rated_power * gearbox_efficiency - b) / a

    time = np.insert(time, 30, x_intersection)
    power_profile = np.insert(power_profile, 30, turboshaft_rated_power * gearbox_efficiency)

    fig = go.Figure()
    scatter = go.Scatter(
        x=time,
        y=power_profile,
        mode="lines+markers",
        showlegend=False,
        line=dict(color="rgba(0,128,0,0.7)"),
        marker=dict(color="rgba(0,128,0,0.7)"),
        fill="tozeroy",
        fillcolor='rgba(0,128,0,0.7)'
    )
    fig.add_trace(scatter)

    scatter_turboshaft_power = go.Scatter(
        x=[min(time), max(time)],
        y=[
            turboshaft_rated_power * gearbox_efficiency,
            turboshaft_rated_power * gearbox_efficiency,
        ],
        mode="lines",
        showlegend=False,
        line=dict(color="gray", dash="dash"),
    )
    fig.add_trace(scatter_turboshaft_power)

    turboshaft_power_fill = go.Scatter(
        x=time,
        y=np.minimum(power_profile, turboshaft_rated_power * gearbox_efficiency),
        fill="tozeroy",
        mode="lines+markers",
        showlegend=False,
        line=dict(color="rgba(255,127,80,1.0)"),
        marker=dict(color="rgba(255,127,80,1.0)"),
        fillcolor='rgba(255,127,80,1.0)'
    )
    fig.add_trace(turboshaft_power_fill)

    fig.update_layout(
        xaxis_title="Time [s]",
        yaxis_title="Propeller shaft power [kW]",
        height=600,
        width=1900,
        font=dict(size=20),
    )

    fig.show()
