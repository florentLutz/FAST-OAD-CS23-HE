#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import plotly.graph_objects as go
import numpy as np

if __name__ == "__main__":
    torque_cont = np.linspace(100, 1000, 100)
    weight = 2.8 + 9.54e-3 * torque_cont + 0.1632 * torque_cont ** (3.0 / 3.5)

    fig = go.Figure()

    scatter_data = go.Scatter(
        x=torque_cont,
        y=weight,
    )
    fig.add_trace(scatter_data)

    fig.show()
