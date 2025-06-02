# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO

import pandas as pd
import numpy as np

from plotly.colors import DEFAULT_PLOTLY_COLORS as COLS
import plotly.graph_objects as go

MARKER_DICTIONARY = ["circle-open",  "square",  "diamond",  "cross"]

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
        "TP 1 engine total",
        "TP 2 engines 1-12 seats",
        "TP 2 engines 13+ seats",
        "TP 2 engines total",
    ]

    af_hours_2004 = [2796.8, 3371.0, 3178.6, 3752.5, 7615.0, 4098.1, 3278.5, 5285.4, 6653.7, 5501.9]
    af_hours_2005 = [2769.3, 3290.5, 3115.9, 3773.4, 5696.2, 4381.1, 3343.3, 5746.3, 5667.3, 5732.9]
    af_hours_2006 = [3041.5, 3680.5, 3470.6, 3774.6, 5257.0, 4212.2, 3259.6, 5517.3, 7119.2, 5748.3]
    af_hours_2007 = [2360.4, 3021.7, 2820.4, 3785.9, 6149.7, 4370.2, 3627.9, 5915.3, 7149.6, 6133.1]
    af_hours_2008 = [2422.1, 3421.2, 3105.9, 4108.9, 5493.4, 4511.8, 4141.3, 5806.7, 6103.4, 5855.7]
    af_hours_2009 = [2150.5, 3301.1, 2934.6, 3525.4, 4810.4, 3920.0, 3729.2, 5921.1, 6394.3, 6003.0]
    af_hours_2010 = [2280.6, 3168.5, 2872.3, 3536.1, 4682.3, 3870.8, 3747.3, 5923.7, 6622.3, 6049.9]
    af_hours_2012 = [2215.5, 3087.9, 2799.7, 3642.2, 4820.9, 3984.2, 3135.5, 6326.6, 4742.1, 5908.5]
    af_hours_2013 = [2229.8, 3206.2, 2888.9, 3728.3, 4925.8, 4066.5, 3711.9, 6371.9, 4331.4, 5877.6]
    af_hours_2014 = [2353.0, 3205.4, 2928.9, 3637.6, 5431.0, 4125.9, 3814.8, 6899.0, 5926.2, 6612.5]
    af_hours_2015 = [2351.8, 3181.6, 2912.8, 3673.6, 5246.1, 4105.6, 3476.9, 5813.2, 7986.8, 6431.5]
    af_hours_2016 = [2283.5, 3298.9, 2992.0, 3884.5, 5434.8, 4258.5, 3081.1, 6192.7, 7225.2, 6461.7]
    af_hours_2017 = [2425.6, 3335.2, 3055.6, 3628.4, 5529.5, 4131.1, 3342.9, 6183.4, 6282.8, 6212.1]
    af_hours_2018 = [2413.7, 3335.9, 3064.1, 3794.1, 5104.3, 4150.7, 3377.6, 5691.6, 5984.0, 5777.1]
    af_hours_2019 = [2469.7, 3389.7, 3116.8, 4178.1, 4947.2, 4367.7, 3831.6, 6290.5, 6926.6, 6494.2]
    af_hours_2020 = [2299.5, 3319.9, 3004.2, 3758.7, 5195.2, 4116.5, 3583.0, 6685.0, 8488.1, 7243.9]
    af_hours_2021 = [2172.6, 3409.6, 3036.2, 3993.8, 5268.9, 4322.4, 3275.0, 5688.4, 4489.1, 5277.2]
    af_hours_2022 = [2499.4, 3373.7, 3122.5, 3802.8, 5279.7, 4170.2, 3108.3, 6486.3, 8753.7, 7231.7]
    af_hours_2023 = [2226.6, 3512.8, 3132.1, 3859.6, 5445.9, 4259.6, 3267.2, 6208.1, 7092.3, 6490.5]

    # Create a list of dict to populate the dataframe

    data = [
        dict(zip(column_names, af_hours_2004)),
        dict(zip(column_names, af_hours_2005)),
        dict(zip(column_names, af_hours_2006)),
        dict(zip(column_names, af_hours_2007)),
        dict(zip(column_names, af_hours_2008)),
        dict(zip(column_names, af_hours_2009)),
        dict(zip(column_names, af_hours_2010)),
        dict(zip(column_names, af_hours_2012)),
        dict(zip(column_names, af_hours_2013)),
        dict(zip(column_names, af_hours_2014)),
        dict(zip(column_names, af_hours_2015)),
        dict(zip(column_names, af_hours_2016)),
        dict(zip(column_names, af_hours_2017)),
        dict(zip(column_names, af_hours_2018)),
        dict(zip(column_names, af_hours_2019)),
        dict(zip(column_names, af_hours_2020)),
        dict(zip(column_names, af_hours_2021)),
        dict(zip(column_names, af_hours_2022)),
        dict(zip(column_names, af_hours_2023)),
    ]

    df = pd.DataFrame(
        data,
        index=[
            "2004",
            "2005",
            "2006",
            "2007",
            "2008",
            "2009",
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

    fig = go.Figure()

    abscissa = list(df.index)
    abscissa = [int(x) for x in abscissa]

    data_to_show = [
        "Piston 1 engine total",
        "Piston 2 engine total",
        "TP 1 engine total",
        "TP 2 engines total",
    ]

    count = 0
    for _, column_name in enumerate(list(df.columns)):
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

            index_2009 = abscissa.index(2009)
            index_2020 = abscissa.index(2020)

            x_for_regression = abscissa[index_2009:index_2020]
            y_for_regression = local_data[index_2009:index_2020]

            mean = np.mean(np.array(y_for_regression))
            std = np.std(np.array(y_for_regression))
            fig.add_trace(
                go.Scatter(
                    x=[abscissa[0], abscissa[-1]],
                    y=[mean, mean],
                    mode="lines",
                    name="Mean: "
                    + str(np.round(mean, 1))
                    + " hours, std: "
                    + str(np.round(std, 1)),
                    showlegend=True,
                    legendgroup=group_name,
                    line=dict(color=COLS[count], dash="dot"),
                )
            )
            count +=1

    fig.update_layout(
        title_text="Average airframe hours per category",
        xaxis_title="Year [-]",
        yaxis_title="Average airframe hours [h]",
        title_x=0.5,
        height=800.0,
        width=1600.0,
        plot_bgcolor="white",
        title_font=dict(size=20),
        legend_font=dict(size=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
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
