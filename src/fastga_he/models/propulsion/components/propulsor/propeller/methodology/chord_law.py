# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import plotly.graph_objects as go
import scipy.optimize as opt


def generic_chord_law(
    radius_distribution,
    radius_mid,
    p_parameter,
    root_chord,
    tip_chord,
    hub_radius,
    tip_radius,
):
    c_linear_mid = (tip_chord - root_chord) / (tip_radius - hub_radius) * (
        radius_mid - hub_radius
    ) + root_chord

    matrix_to_inv = np.array(
        [
            [hub_radius**2.0, hub_radius, 1.0, 0.0, 0.0, 0.0],
            [radius_mid**2.0, radius_mid, 1.0, 0.0, 0.0, 0.0],
            [2.0 * radius_mid, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0 * radius_mid, 1.0, 0.0],
            [0.0, 0.0, 0.0, radius_mid**2.0, radius_mid, 1.0],
            [0.0, 0.0, 0.0, tip_radius**2.0, tip_radius, 1.0],
        ]
    )
    result_matrix = np.array(
        [
            [root_chord],
            [(p_parameter + 1.0) * c_linear_mid],
            [(tip_chord - root_chord) / (tip_radius - hub_radius)],
            [(tip_chord - root_chord) / (tip_radius - hub_radius)],
            [(p_parameter + 1.0) * c_linear_mid],
            [tip_chord],
        ]
    )

    k12, k11, k10, k22, k21, k20 = np.dot(np.linalg.inv(matrix_to_inv), result_matrix).transpose()[
        0
    ]
    chord_distribution = np.where(
        radius_distribution < radius_mid,
        k12 * radius_distribution**2.0 + k11 * radius_distribution + k10,
        k22 * radius_distribution**2.0 + k21 * radius_distribution + k20,
    )

    return chord_distribution


if __name__ == "__main__":
    radius_tbm900 = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]) * 2.31 / 2.0
    chord_tbm900 = np.array([0.1, 0.17, 0.23, 0.26, 0.275, 0.28, 0.265, 0.22, 0.158])

    radius_naca = (
        np.array([0.2, 0.259, 0.312, 0.393, 0.472, 0.545, 0.684, 0.786, 0.890, 0.949]) * 1.524
    )
    chord_naca = np.array([0.072, 0.091, 0.107, 0.137, 0.150, 0.150, 0.134, 0.114, 0.091, 0.076])

    x0 = (0.5775, 0.1, chord_tbm900[0], chord_tbm900[-1], chord_tbm900[0], chord_tbm900[-1])
    bnds = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
    )

    # TBM900 ---------------------------------------------------------------------------------------

    p_opt, _, _, _, _ = opt.curve_fit(
        generic_chord_law, radius_tbm900, chord_tbm900, x0, bounds=bnds, full_output=True
    )
    fit_coeff_1, fit_coeff_2, fit_coeff_3, fit_coeff_4, fit_coeff_5, fit_coeff_6 = p_opt

    fig = go.Figure()
    orig_chord_tbm900 = go.Scatter(
        x=radius_tbm900,
        y=chord_tbm900,
        mode="lines+markers",
        name="Chord law TBM900 propeller in FAST-OAD-CS23",
        legendgroup="TBM900",
        legendgrouptitle_text="TBM900",
    )
    fig.add_trace(orig_chord_tbm900)

    fitted_chord_tbm900 = go.Scatter(
        x=radius_tbm900,
        y=generic_chord_law(
            radius_tbm900,
            fit_coeff_1,
            fit_coeff_2,
            fit_coeff_3,
            fit_coeff_4,
            fit_coeff_5,
            fit_coeff_6,
        ),
        mode="lines+markers",
        name="Chord law TBM900 propeller with 4 parameters distribution",
        legendgroup="TBM900",
        legendgrouptitle_text="TBM900",
    )
    fig.add_trace(fitted_chord_tbm900)

    # NACA report ----------------------------------------------------------------------------------

    p_opt, p_cov, _, _, _ = opt.curve_fit(
        generic_chord_law, radius_naca, chord_naca, x0, bounds=bnds, full_output=True
    )
    fit_coeff_1, fit_coeff_2, fit_coeff_3, fit_coeff_4, fit_coeff_5, fit_coeff_6 = p_opt

    orig_chord_naca = go.Scatter(
        x=radius_naca,
        y=chord_naca,
        mode="lines+markers",
        name="Chord law NACA report propeller",
        legendgroup="NACA",
        legendgrouptitle_text="NACA report",
    )
    fig.add_trace(orig_chord_naca)

    fitted_chord_naca = go.Scatter(
        x=radius_naca,
        y=generic_chord_law(
            radius_naca,
            fit_coeff_1,
            fit_coeff_2,
            fit_coeff_3,
            fit_coeff_4,
            fit_coeff_5,
            fit_coeff_6,
        ),
        mode="lines+markers",
        name="Chord law NACA report propeller with 4 parameters distribution",
        legendgroup="NACA",
    )
    fig.add_trace(fitted_chord_naca)

    fig.update_layout(
        title_text="Evaluation of chord law on existing propeller",
        title_x=0.5,
        xaxis_title="Radius [m]",
        yaxis_title="Chord [m]",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.show()
