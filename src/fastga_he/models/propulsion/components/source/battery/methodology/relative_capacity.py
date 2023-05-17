# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO
"""
This module is to study the effect of current on the apparent capacity of the battery, which is
translated by a relative capacity smaller than 1.0. This is something which is suggested in
:cite:`vratny:2013` and the data used for th interpolation comes from the datasheet of the
reference cell :cite:`samsung:2015`.
"""

import numpy as np
import plotly.graph_objects as go


if __name__ == "__main__":

    current = np.array([680, 3400, 6800, 8000])
    relative_capacity = np.array([1, 0.97, 0.95, 0.92])

    fig = go.Figure()
    scatter = go.Scatter(
        x=current,
        y=relative_capacity,
        mode="markers",
        name="Manufacturers data",
        marker=dict(symbol="diamond", size=10),
    )
    fig.add_trace(scatter)

    poly = np.polyfit(current, relative_capacity, 3)
    print(poly)

    scatter_inter = go.Scatter(
        x=np.linspace(680, 8000, 50),
        y=np.polyval(poly, np.linspace(680, 8000, 50)),
        mode="lines",
        name="Battery cell relative capacity interpolation",
    )
    fig.add_trace(scatter_inter)

    fig.update_layout(
        title_text="Battery relative capacity",
        title_x=0.5,
        xaxis_title="Cell current [mA]",
        yaxis_title="Relative capacity [-]",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=800,
        width=1600,
    )

    fig.show()
    fig.write_image("relative_cap.pdf")
