# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pandas as pd

import plotly.graph_objects as go


def curve_reading(file_path, threshold=0.0):
    fig = go.Figure()

    efficiency_data = pd.read_csv(file_path)

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

            if efficiency > threshold:
                speed_array = np.concatenate((speed_array, x * 2.0 * np.pi / 60))
                efficiency_array = np.concatenate((efficiency_array, np.full_like(x, efficiency)))

        else:
            y = np.array(efficiency_data[name][1:]).astype(float)
            y = y[np.logical_not(np.isnan(y))]

            if efficiency > threshold:
                torque_array = np.concatenate((torque_array, y))

            scatter = go.Scatter(x=x, y=y, mode="lines+markers", name=efficiency)
            fig.add_trace(scatter)

    return speed_array, torque_array, efficiency_array, fig
