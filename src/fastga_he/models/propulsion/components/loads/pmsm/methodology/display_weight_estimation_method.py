# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import plotly.graph_objects as go
import numpy as np

if __name__ == "__main__":
    d_star = np.array(
        [
            1,
            1.106382979,
            1.212765957,
            1.425531915,
            1.85106383,
        ]
    )
    l_star = np.array(
        [
            1.0,
            1.103896104,
            1.116883117,
            1.18181818,
            1.38961039,
        ]
    )
    m_star_data = np.array(
        [
            1.0,
            1.3,
            1.714285714,
            2.857142857,
            5.857142857,
        ]
    )
    cont_torque = np.array([40, 64, 96, 200, 400])
    torque_star = cont_torque / cont_torque[0]

    m_from_formula = 2.8 + 9.54e-3 * cont_torque + 0.1632 * cont_torque ** (3.0 / 3.5)
    m_star_from_formula = m_from_formula / m_from_formula[0]
    m_star_1 = torque_star ** (3.0 / 3.5)
    m_star_2 = torque_star

    fig = go.Figure()

    discrete_x = ["EMRAX 188", "EMRAX 208", "EMRAX 228", "EMRAX 268", "EMRAX 348"]

    scatter_data = go.Scatter(
        x=discrete_x,
        y=m_star_data,
        mode="markers",
        name=r"Reference values",
        marker_size=15,
        marker_symbol="diamond",
        showlegend=True,
    )
    fig.add_trace(scatter_data)

    scatter_formula = go.Scatter(
        x=discrete_x,
        y=m_star_from_formula,
        mode="markers",
        name=r"From regression",
        marker_size=15,
        marker_symbol="star",
        showlegend=True,
    )
    fig.add_trace(scatter_formula)

    scatter_1 = go.Scatter(
        x=discrete_x,
        y=m_star_1,
        mode="markers",
        name=r"$m^{*} = {T_{em,nom}^{*}}^{3/3.5}$",
        marker_size=15,
        marker_symbol="circle",
        showlegend=True,
    )
    fig.add_trace(scatter_1)

    scatter_2 = go.Scatter(
        x=discrete_x,
        y=m_star_2,
        mode="markers",
        name=r"$m^{*} = {T_{em,nom}^{*}}$",
        marker_size=15,
        marker_symbol="x",
        showlegend=True,
    )
    fig.add_trace(scatter_2)

    fig.update_layout(
        title_text="Comparison of the methodologies to estimate the scaling of the motor mass",
        title_x=0.5,
        xaxis_title="Reference motors",
        yaxis_title=r"$m^{*}$",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=800,
        width=1600,
    )
    fig.update_layout(legend=dict(font=dict(size=20)))
    fig.show()

    fig.write_image("mass_scaling.pdf")
