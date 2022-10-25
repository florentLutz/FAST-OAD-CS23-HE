# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import plotly.express as px

if __name__ == "__main__":

    I_cal = np.array(
        [
            200,
            150,
            300,
            200,
            400,
            450,
            600,
            400,
            450,
            450,
            600,
            600,
            75,
            150,
            100,
            200,
            200,
            150,
            150,
            150,
            300,
            300,
            300,
            200,
            400,
            400,
            300,
            300,
            600,
            600,
            400,
        ]
    )
    V_cal = np.array(
        [
            600,
            1200,
            600,
            1700,
            600,
            650,
            600,
            1700,
            1700,
            1200,
            1700,
            1200,
            1700,
            600,
            1700,
            650,
            600,
            1200,
            1200,
            1700,
            600,
            650,
            600,
            1200,
            650,
            600,
            1200,
            1700,
            650,
            600,
            1200,
        ]
    )
    R_th_cs = np.array(
        [
            0.045,
            0.045,
            0.045,
            0.045,
            0.045,
            0.013,
            0.04,
            0.03,
            0.04,
            0.04,
            0.03,
            0.03,
            0.05,
            0.05,
            0.05,
            0.017,
            0.05,
            0.05,
            0.038,
            0.038,
            0.05,
            0.01074,
            0.038,
            0.038,
            0.0101,
            0.038,
            0.038,
            0.038,
            0.0088,
            0.038,
            0.038,
        ]
    )

    interesting_idx = np.where(R_th_cs > 0.02)

    B = R_th_cs[interesting_idx]
    A = np.column_stack(
        [
            np.ones_like(R_th_cs[interesting_idx]),
            I_cal[interesting_idx],
            I_cal[interesting_idx] * V_cal[interesting_idx],
        ]
    )
    x = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = x[0]
    print(a, b, c)
    computed = a + I_cal[interesting_idx] * b + I_cal[interesting_idx] * V_cal[interesting_idx] * c
    print((computed - R_th_cs[interesting_idx]) / R_th_cs[interesting_idx] * 100)

    fig = px.scatter(
        x=I_cal[interesting_idx] * V_cal[interesting_idx],
        y=computed,
        trendline="ols",
        trendline_options=dict(log_x=True),
    )
    fig.show()
