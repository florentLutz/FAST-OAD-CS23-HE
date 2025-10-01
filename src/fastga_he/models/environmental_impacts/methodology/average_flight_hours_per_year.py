# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import time
import pathlib

import pandas as pd
import numpy as np

from plotly.colors import DEFAULT_PLOTLY_COLORS as COLS
import plotly.graph_objects as go
import plotly.io as pio

RESULTS_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "results"
MARKER_DICTIONARY = ["circle-open", "square", "diamond", "cross"]


if __name__ == "__main__":
    # Data will be provided in the following order:
    # - Piston 1 engine 1-3 seats
    # - Piston 1 engine 4+ seats
    # - Piston 1 engine total
    # - Piston 2 engine 1-6 seats
    # - Piston 2 engine 7+ seats
    # - Piston 2 engine total
    # - TP 1 engine total
    # - TP 2 engines 1-12 seats
    # - TP 2 engines 13+ seats
    # - TP 2 engines total

    # /!\ We will start with 2004 which does not include Commercial Commuter Activity !!!
    # All data are taken from the General Aviation Survey from the FAA

    column_names = [
        "Piston 1 engine 1-3 seats",
        "Piston 1 engine 4+ seats",
        "Piston 1 engine total",
        "Piston 2 engine 1-6 seats",
        "Piston 2 engine 7+ seats",
        "Piston 2 engine total",
        "Turboprop 1 engine total",
        "Turboprop 2 engines 1-12 seats",
        "Turboprop 2 engines 13+ seats",
        "Turboprop 2 engines total",
    ]

    flt_hrs_2004 = [093.5, 108.9, 104.8, 131.3, 193.6, 149.7, 307.7, 225.3, 314.6, 238.0]
    flt_hrs_2005 = [078.5, 098.0, 092.8, 117.1, 181.9, 137.9, 326.0, 223.2, 300.2, 235.9]
    flt_hrs_2006 = [078.6, 102.6, 096.4, 117.8, 177.6, 136.3, 331.0, 228.7, 302.3, 238.7]
    flt_hrs_2007 = [075.8, 097.3, 092.0, 121.7, 188.4, 138.9, 275.4, 266.0, 370.0, 282.9]
    flt_hrs_2008 = [071.9, 093.1, 087.6, 120.1, 163.6, 132.9, 310.2, 250.7, 272.4, 254.1]
    # flt_hrs_2009 = [] # Data missing for this year
    flt_hrs_2010 = [073.0, 092.4, 087.2, 104.6, 137.5, 114.3, 257.8, 231.6, 281.6, 240.2]
    flt_hrs_2012 = [067.3, 096.6, 088.8, 110.7, 153.8, 123.3, 269.5, 243.1, 309.3, 261.1]
    flt_hrs_2013 = [075.3, 089.8, 086.1, 115.2, 146.1, 124.2, 292.6, 233.7, 294.3, 248.4]
    flt_hrs_2014 = [069.3, 087.2, 082.5, 104.8, 157.6, 119.6, 278.8, 233.4, 311.9, 257.0]
    flt_hrs_2015 = [069.3, 094.5, 087.7, 115.4, 136.3, 121.3, 281.7, 215.0, 317.7, 244.5]
    flt_hrs_2016 = [065.7, 100.0, 091.5, 119.3, 158.1, 129.6, 301.3, 199.1, 415.7, 255.4]
    flt_hrs_2017 = [069.0, 101.0, 092.8, 112.3, 131.0, 117.4, 301.7, 186.1, 341.7, 228.5]
    flt_hrs_2018 = [071.0, 100.2, 092.9, 126.7, 144.4, 131.7, 283.6, 222.5, 374.4, 267.9]
    flt_hrs_2019 = [068.4, 108.5, 098.5, 138.3, 140.4, 138.8, 281.9, 179.6, 333.1, 229.6]
    flt_hrs_2020 = [064.7, 103.2, 093.5, 110.2, 116.6, 111.9, 247.1, 166.9, 292.6, 206.4]
    flt_hrs_2021 = [073.6, 110.1, 101.1, 126.6, 123.3, 125.7, 278.2, 168.5, 374.8, 242.9]
    flt_hrs_2022 = [068.7, 113.8, 103.1, 124.8, 117.4, 122.9, 279.8, 197.1, 351.2, 248.6]
    flt_hrs_2023 = [073.7, 127.0, 114.5, 131.0, 116.5, 127.2, 281.3, 177.1, 351.7, 231.7]

    data = [
        dict(zip(column_names, flt_hrs_2004)),
        dict(zip(column_names, flt_hrs_2005)),
        dict(zip(column_names, flt_hrs_2006)),
        dict(zip(column_names, flt_hrs_2007)),
        dict(zip(column_names, flt_hrs_2008)),
        dict(zip(column_names, flt_hrs_2010)),
        dict(zip(column_names, flt_hrs_2012)),
        dict(zip(column_names, flt_hrs_2013)),
        dict(zip(column_names, flt_hrs_2014)),
        dict(zip(column_names, flt_hrs_2015)),
        dict(zip(column_names, flt_hrs_2016)),
        dict(zip(column_names, flt_hrs_2017)),
        dict(zip(column_names, flt_hrs_2018)),
        dict(zip(column_names, flt_hrs_2019)),
        dict(zip(column_names, flt_hrs_2020)),
        dict(zip(column_names, flt_hrs_2021)),
        dict(zip(column_names, flt_hrs_2022)),
        dict(zip(column_names, flt_hrs_2023)),
    ]

    df = pd.DataFrame(
        data,
        index=[
            "2004",
            "2005",
            "2006",
            "2007",
            "2008",
            "2010",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "2022",
            "2023",
        ],
    )

    data_to_show = [
        "Piston 1 engine total",
        "Piston 2 engine total",
        "Turboprop 1 engine total",
        "Turboprop 2 engines total",
    ]

    fig = go.Figure()

    abscissa = list(df.index)
    abscissa = [int(x) for x in abscissa]

    count = 0
    for idx, column_name in enumerate(list(df.columns)):
        if column_name in data_to_show:
            group_name = " ".join(column_name.split(" ")[0:2])

            local_data = df[column_name].values

            fig.add_trace(
                go.Scatter(
                    x=abscissa,
                    y=local_data,
                    mode="lines+markers",
                    name=column_name,
                    showlegend=True,
                    legendgroup=group_name,
                    line=dict(color=COLS[count]),
                    marker=dict(symbol=MARKER_DICTIONARY[count], color=COLS[count], size=10),
                )
            )

            index_2010 = abscissa.index(2010)
            index_2020 = abscissa.index(2020)

            x_for_regression = abscissa[index_2010:index_2020]
            y_for_regression = local_data[index_2010:index_2020]

            mean = np.mean(np.array(y_for_regression))
            std = np.std(np.array(y_for_regression)) / mean * 100.0
            fig.add_trace(
                go.Scatter(
                    x=[abscissa[0], abscissa[-1]],
                    y=[mean, mean],
                    mode="lines",
                    name="Mean: "
                    + str(np.round(mean, 1))
                    + " hours, std: "
                    + str(np.round(std, 1))
                    + " %",
                    showlegend=False,
                    legendgroup=group_name,
                    line=dict(color=COLS[idx], dash="dot"),
                )
            )
            count += 1

    fig.update_layout(
        xaxis_title="Year [-]",
        yaxis_title="Average flight hours per year [h]",
        height=800.0,
        width=1600.0,
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=5, r=5, t=60, b=5),
    )
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        title_font=dict(size=20),
        tickfont=dict(size=20),
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        title_font=dict(size=20),
        tickfont=dict(size=20),
        side="right",
    )
    fig.show()

    pdf_path = RESULTS_FOLDER_PATH / "average_yearly_hours_used.pdf"

    write = True

    if write:
        fig.update_layout(title=None)
        pio.write_image(fig, pdf_path, width=1600, height=900)
        time.sleep(3)
        pio.write_image(fig, pdf_path, width=1600, height=900)
