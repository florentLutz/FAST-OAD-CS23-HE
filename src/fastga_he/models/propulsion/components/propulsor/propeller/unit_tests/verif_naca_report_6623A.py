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
        10 ** 0.91202
        * j
        ** (
            -0.05831 * np.log10(re_d) * np.log10(ct)
            - 0.03839 * np.log10(re_d) ** 2
            + 0.16329 * np.log10(j)
            + 0.61371 * np.log10(re_d)
            + 0.08990 * np.log10(j) * np.log10(tip_mach)
            - 0.36310 * np.log10(solidity) * np.log10(activity_factor)
            - 2.64511
            + 0.17732 * np.log10(re_d) * np.log10(solidity)
            - 0.48567 * np.log10(solidity) * np.log10(aspect_ratio)
            - 0.15692 * np.log10(j) * np.log10(solidity)
            + 0.16157 * np.log10(ct) * np.log10(aspect_ratio)
            + 0.03057 * np.log10(tip_mach) ** 2
            + 0.30703 * np.log10(activity_factor) * np.log10(aspect_ratio)
            + 0.08397 * np.log10(activity_factor) ** 2
        )
        * tip_mach
        ** (
            +0.13037 * np.log10(ct) ** 2
            + 0.02184 * np.log10(activity_factor) ** 2
            + 0.05922 * np.log10(ct) * np.log10(activity_factor)
            - 0.08255 * np.log10(solidity) * np.log10(ct)
            + 0.01411 * np.log10(solidity) * np.log10(activity_factor)
            - 0.08674 * np.log10(tip_mach) * np.log10(ct)
            + 0.10718 * np.log10(tip_mach) * np.log10(solidity)
        )
        * re_d
        ** (
            +0.02134 * np.log10(re_d) ** 2
            - 0.08052 * np.log10(ct) ** 2
            - 0.33138 * np.log10(re_d)
            - 0.93130 * np.log10(ct)
            - 0.04308 * np.log10(solidity) * np.log10(ct)
            + 0.04528 * np.log10(re_d) * np.log10(ct)
            + 0.01031 * np.log10(activity_factor) ** 2
            - 0.08840 * np.log10(aspect_ratio) ** 2
            + 0.21141 * np.log10(aspect_ratio)
            + 1.12924
            + 0.00338 * np.log10(solidity) ** 2
        )
        * solidity
        ** (0.03744 * np.log10(ct) ** 2 + 0.06902 * np.log10(solidity) * np.log10(activity_factor))
        * ct
        ** (
            5.75508
            + 0.31945 * np.log10(ct) * np.log10(activity_factor)
            + 0.35294 * np.log10(ct) * np.log10(aspect_ratio)
            + 0.01597 * np.log10(ct) ** 2
        )
        * activity_factor ** (-0.19051 * np.log10(activity_factor) * np.log10(aspect_ratio))
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
    n_blades = 3.0

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

    # ct = 0.08 at J = 1.02 for pitch at 35 deg, at 35deg, rpm = 800 meaning v is
    # 41.452799999999996 m/s
    ct_list = np.array(
        [
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

    j_list_40_deg = np.array(
        [
            1.180803571,
            1.289285714,
            1.357589286,
            1.429910714,
            1.490178571,
            1.558482143,
            1.614732143,
            1.675,
            1.739285714,
            1.803571429,
            1.855803571,
            1.920089286,
            1.976339286,
            2.040625,
            2.092857143,
            2.153125,
            2.217410714,
        ]
    )
    cp_list_verif_40_deg = np.array(
        [
            0.249459459,
            0.245,
            0.237702703,
            0.228783784,
            0.219054054,
            0.206891892,
            0.195540541,
            0.182567568,
            0.168783784,
            0.152972973,
            0.137162162,
            0.120540541,
            0.103918919,
            0.086891892,
            0.068648649,
            0.050810811,
        ]
    )
    cp_list_bemt_40_deg = np.array(
        [
            0.23409944,
            0.23307469,
            0.22491868,
            0.21641029,
            0.20542531,
            0.19456652,
            0.18164156,
            0.16844193,
            0.15472414,
            0.14012653,
            0.12389048,
            0.10768391,
            0.09017963,
            0.07209448,
            0.05268179,
            0.03269478,
        ]
    )

    rpm = 700.0

    cp_list = np.zeros_like(ct_list)
    for idx, (j_loop, ct_loop) in enumerate(zip(j_list_40_deg, ct_list)):
        cp_list[idx] = cp_from_ct(
            j_loop,
            (j_loop ** 2.0 + np.pi ** 2.0) * (rpm / 60 * 3.048) ** 2.0 / atm.speed_of_sound ** 2.0,
            j_loop * rpm / 60 * 3.048 ** 2.0 / atm.kinematic_viscosity,
            solidity_naca,
            ct_loop,
            activity_factor_naca,
            aspect_ratio_naca,
        )
        print(
            (
                j_loop,
                (j_loop ** 2.0 + np.pi ** 2.0)
                * (rpm / 60 * 3.048) ** 2.0
                / atm.speed_of_sound ** 2.0,
                j_loop * rpm / 60 * 3.048 ** 2.0 / atm.kinematic_viscosity,
                solidity_naca,
                ct_loop,
                activity_factor_naca,
                aspect_ratio_naca,
            )
        )

    print((cp_list_bemt_40_deg - cp_list_verif_40_deg) / cp_list_verif_40_deg * 100.0)

    fig = go.Figure()
    scatter_verif = go.Scatter(
        x=j_list_40_deg, y=cp_list_verif_40_deg, mode="lines+markers", name="NACA report"
    )
    fig.add_trace(scatter_verif)
    scatter_computed = go.Scatter(
        x=j_list_40_deg, y=cp_list, mode="lines+markers", name="VPLM repgression on BEMT"
    )
    fig.add_trace(scatter_computed)
    scatter_pure_bemt = go.Scatter(
        x=j_list_40_deg, y=cp_list_bemt_40_deg, mode="lines+markers", name="BEMT"
    )
    fig.add_trace(scatter_pure_bemt)
    fig.show()
