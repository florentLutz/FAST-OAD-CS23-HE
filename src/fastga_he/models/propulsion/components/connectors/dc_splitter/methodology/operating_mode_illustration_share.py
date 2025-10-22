#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import time

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

true = False
false = True

if __name__ == "__main__":
    """
    Just illustrates the different operating mode of the splitter.
    """

    fig = go.Figure()

    time_array = np.linspace(0, 2, 50)  # In hours, illustrative
    power_array = np.concatenate(
        (np.linspace(200, 170, 10), np.full(20, 140), np.linspace(20, 10, 5), np.full(15, 130))
    )

    scatter_first_source = go.Scatter(
        x=time_array,
        y=np.minimum(power_array, 135.0),
        mode="lines",
        name="Primary input",
        showlegend=True,
        line={"color": "rgba(255,69,0,0.666)", "width": 2},
        fill="tozeroy",
        fillcolor="rgba(255,69,0,0.666)",
    )
    fig.add_trace(scatter_first_source)

    scatter_second_source = go.Scatter(
        x=time_array,
        y=power_array,
        mode="lines",
        name="Secondary input",
        showlegend=True,
        line={"color": "rgba(65,105,225,0.5)", "width": 2},
        fill="tonexty",
        fillcolor="rgba(65,105,225,0.5)",
    )
    fig.add_trace(scatter_second_source)

    scatter_output_power_profile = go.Scatter(
        x=time_array,
        y=power_array,
        mode="lines",
        name="Output power profile",
        showlegend=True,
        line={"color": "black", "width": 3},
    )
    fig.add_trace(scatter_output_power_profile)

    fig.add_annotation(
        ax=0.5,
        ay=135 * 0.3,
        x=0.5,
        y=135 * 0.95,
        showarrow=True,
        arrowhead=3,
        arrowside="end",
        arrowcolor="black",
        arrowwidth=2,
        axref="x",
        ayref="y",
    )
    fig.add_annotation(
        ax=0.5,
        ay=135 * 0.7,
        x=0.5,
        y=135 * 0.05,
        showarrow=True,
        arrowhead=3,
        arrowside="end",
        arrowcolor="black",
        arrowwidth=2,
        axref="x",
        ayref="y",
    )
    fig.add_annotation(
        x=0.51,
        y=135 * 0.5,
        text=r"$P_{1} =\begin{cases} P_{th} & \text{if } P_{th} < P_{out} \\ P_{out} & \text{otherwise} \end{cases}$",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=50, color="black"),
        align="left",
    )

    # fig.add_annotation(
    #     x=4, y=3,
    #     ax=2, ay=3,
    #     showarrow=True,
    #     arrowhead=3,
    #     arrowside="end"
    # )

    fig.update_layout(
        plot_bgcolor="white",
        legend_font=dict(size=20),
        height=800,
        width=1600,
        margin=dict(l=5, r=5, t=60, b=5),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        title_font=dict(size=20),
        tickfont=dict(size=20),
        dtick=10,
        title="Time",
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        title_font=dict(size=20),
        tickfont=dict(size=20),
        title="Power profile",
        range=[0, 210],
    )

    fig["layout"]["yaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["yaxis"]["tickfont"]["size"] = 20
    fig["layout"]["xaxis"]["title"]["font"]["size"] = 20
    fig["layout"]["xaxis"]["tickfont"]["size"] = 20

    pdf_path = "results/power_share_mode.pdf"

    fig.show()

    write = false

    if write:
        fig.update_layout(title=None)
        pio.write_image(fig, pdf_path, width=1600, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1600, height=900)
