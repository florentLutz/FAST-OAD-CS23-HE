# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

# TODO: In this file we will check the results compare to NACA N640 report considering the
#  2-blades, 5868-9 propeller

import numpy as np
import plotly.graph_objects as go
from stdatm import Atmosphere

ELEMENTS_NUMBER = 100


def cp_from_ct(j, tip_mach, re_d, solidity, ct, activity_factor, aspect_ratio):
    cp = (
        10 ** -0.79766
        * j
        ** (
            -0.39622 * np.log10(ct) ** 2
            - 0.02834 * np.log10(re_d)
            - 0.02257 * np.log10(j) ** 2
            + 0.18445
            + 0.00240 * np.log10(re_d) ** 2
            - 0.00606 * np.log10(re_d) * np.log10(ct)
            + 0.08364 * np.log10(j) * np.log10(solidity)
            - 1.05617 * np.log10(ct)
            - 0.00997 * np.log10(solidity) ** 2
            + 0.04019 * np.log10(re_d) * np.log10(solidity)
            - 0.28942 * np.log10(j) * np.log10(ct)
            - 0.01073 * np.log10(tip_mach) * np.log10(re_d)
            + 0.21099 * np.log10(solidity) * np.log10(ct)
        )
        * tip_mach ** (+0.00514 * np.log10(aspect_ratio) ** 2 - 0.07179 * np.log10(tip_mach) ** 2)
        * re_d
        ** (
            -0.03651 * np.log10(ct)
            + 0.00755 * np.log10(re_d) * np.log10(ct)
            + 0.39978 * np.log10(solidity)
            + 0.02960
            - 0.02570 * np.log10(re_d) * np.log10(solidity)
            + 0.00868 * np.log10(ct) ** 2
            + 0.05693 * np.log10(solidity) ** 2
        )
        * solidity
        ** (
            -1.71876
            + 0.03820 * np.log10(solidity) ** 2
            - 0.83599 * np.log10(ct) * np.log10(activity_factor)
            - 0.11067 * np.log10(solidity) * np.log10(ct)
            - 0.08756 * np.log10(solidity) * np.log10(activity_factor)
            - 0.87973 * np.log10(ct) * np.log10(aspect_ratio)
            - 0.08332 * np.log10(solidity) * np.log10(aspect_ratio)
            + 1.89700 * np.log10(ct)
        )
        * ct
        ** (
            5.00687
            - 0.42958 * np.log10(activity_factor) ** 2
            - 2.62528 * np.log10(aspect_ratio)
            + 0.46196 * np.log10(aspect_ratio) ** 2
            + 0.23352 * np.log10(ct)
        )
        * activity_factor
        ** (-0.12652 * np.log10(activity_factor) ** 2 - 0.48768 * np.log10(aspect_ratio) + 1.38909)
        * aspect_ratio ** (+0.14316 * np.log10(aspect_ratio) ** 2)
    )
    return cp


if __name__ == "__main__":
    radius_ratio_chord = np.array(
        [
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.427,
            0.457,
            0.487,
            0.527,
            0.581,
            0.637,
            0.681,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ]
    )
    chord_to_diameter_ratio = np.array(
        [
            0.0377,
            0.0451,
            0.0532,
            0.0611,
            0.0687,
            0.0725,
            0.0747,
            0.0760,
            0.0763,
            0.0744,
            0.0710,
            0.0671,
            0.0620,
            0.0565,
            0.0514,
            0.0448,
            0.0380,
        ]
    )

    prop_diameter = 3.048
    hub_diameter = 0.2 * prop_diameter
    n_blades = 4.0

    radius_max = prop_diameter / 2.0
    radius_min = hub_diameter / 2.0

    length = radius_max - radius_min
    elements_number = np.arange(ELEMENTS_NUMBER)
    element_length = length / ELEMENTS_NUMBER
    radius_ratio = (radius_min + (elements_number + 0.5) * element_length) / radius_max
    radius = radius_ratio * radius_max

    chord_array = (
        np.interp(radius_ratio, radius_ratio_chord, chord_to_diameter_ratio) * prop_diameter
    )
    solidity_naca = n_blades / np.pi / radius_max ** 2.0 * np.sum(chord_array * element_length)
    activity_factor_naca = (
        100000 / 32 / radius_max ** 5.0 * np.sum(chord_array * radius ** 3.0 * element_length)
    )
    c_star = np.sum(chord_array * radius ** 2.0 * element_length) / np.sum(
        radius ** 2.0 * element_length
    )
    aspect_ratio_naca = radius_max / c_star

    atm = Atmosphere(0.0, altitude_in_feet=False)

    ct_list = np.array(
        [
            0.19,
            0.18,
            0.17,
            0.16,
            0.15,
            0.14,
            0.13,
            0.12,
            0.11,
            0.10,
            0.09,
            0.08,
            0.07,
            0.06,
            0.05,
            0.04,
            0.03,
            0.02,
            0.01,
        ]
    )
    j_list_25_deg = np.array(
        [
            0.39375,
            0.54375,
            0.60000,
            0.65000,
            0.68750,
            0.73750,
            0.78125,
            0.82500,
            0.86250,
            0.90625,
            0.95000,
            0.98750,
            1.0250,
            1.0625,
            1.1000,
            1.1375,
            1.1687,
            1.2125,
            1.2500,
        ]
    )
    cp_list_verif_25_deg = np.array(
        [
            0.16353,
            0.15668,
            0.15356,
            0.14920,
            0.14484,
            0.13924,
            0.13363,
            0.12678,
            0.11993,
            0.11183,
            0.10374,
            0.095640,
            0.086298,
            0.076955,
            0.067612,
            0.056401,
            0.045813,
            0.035848,
            0.025260,
        ]
    )
    cp_list = np.zeros_like(ct_list)
    cp_list_bemt_25_deg = np.array(
        [
            0.14408674,
            0.15456696,
            0.15226541,
            0.14787878,
            0.14120706,
            0.13543482,
            0.12976568,
            0.12312898,
            0.11632607,
            0.10876137,
            0.10070174,
            0.09261825,
            0.08360107,
            0.07383256,
            0.06383584,
            0.05425261,
            0.04305809,
            0.03168417,
            0.02032022,
        ]
    )
    for idx, (j, ct) in enumerate(zip(j_list_25_deg, ct_list)):
        cp_list[idx] = cp_from_ct(
            j,
            (j ** 2.0 + np.pi ** 2.0) * (900 / 60 * 3.048) ** 2.0 / atm.speed_of_sound ** 2.0,
            j * 900 / 60 * 3.048 ** 2.0 / atm.kinematic_viscosity,
            solidity_naca,
            ct,
            activity_factor_naca,
            aspect_ratio_naca,
        )

    fig = go.Figure()
    scatter_verif = go.Scatter(
        x=j_list_25_deg, y=cp_list_verif_25_deg, mode="lines+markers", name="NACA report"
    )
    fig.add_trace(scatter_verif)
    scatter_computed = go.Scatter(
        x=j_list_25_deg, y=cp_list, mode="lines+markers", name="VPLM repgression on BEMT"
    )
    fig.add_trace(scatter_computed)
    scatter_pure_bemt = go.Scatter(
        x=j_list_25_deg, y=cp_list_bemt_25_deg, mode="lines+markers", name="BEMT"
    )
    fig.add_trace(scatter_pure_bemt)
    fig.show()
    print((cp_list - cp_list_verif_25_deg) / cp_list_verif_25_deg * 100.0)
