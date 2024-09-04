# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from stdatm import Atmosphere


def cp_from_ct_new_new(j, tip_mach, re_d, solidity, ct, activity_factor, twist_blade):
    # Careful, new definition of the tip mach which should now be computed using the absolute
    # speed on the blades

    cp = (
        10**2.43553
        * j
        ** (
            0.61554
            + 0.06980 * np.log10(j) * np.log10(ct) * np.log10(twist_blade)
            - 0.01794 * np.log10(re_d) * np.log10(ct) * np.log10(activity_factor)
            + 0.02595 * np.log10(tip_mach) * np.log10(solidity) * np.log10(ct)
            + 0.00430 * np.log10(j) * np.log10(re_d) ** 2.0
            + 0.09827 * np.log10(ct) * np.log10(twist_blade)
            + 0.03663 * np.log10(solidity) * np.log10(activity_factor)
        )
        * tip_mach ** (-0.00097 * np.log10(re_d) ** 2.0 * np.log10(tip_mach))
        * re_d
        ** (
            -0.35804
            + 0.00018 * np.log10(re_d) ** 3.0
            - 0.01879 * np.log10(ct) ** 3.0
            + 0.00119 * np.log10(re_d) * np.log10(solidity) * np.log10(activity_factor)
            - 0.00886 * np.log10(ct) ** 2.0 * np.log10(twist_blade)
        )
        * solidity
        ** (
            0.08015 * np.log10(ct) ** 2.0 * np.log10(activity_factor)
            + 0.04562 * np.log10(solidity) * np.log10(activity_factor) ** 2.0
            - 0.04121 * np.log10(solidity) * np.log10(ct) * np.log10(twist_blade)
        )
        * ct
        ** (
            1.33164
            + 0.06989 * np.log10(ct) * np.log10(activity_factor)
            + 0.00206 * np.log10(ct) * np.log10(twist_blade) ** 2.0
            + 0.03617 * np.log10(ct) ** 2.0 * np.log10(activity_factor)
        )
    )

    return cp


def cp_from_ct_new(j, tip_mach, re_d, solidity, ct, activity_factor, twist_blade):
    # Careful, new definition of the tip mach which should now be computed using the absolute
    # speed on the blades

    cp = (
        10**2.31538
        * j
        ** (
            +0.05414 * np.log10(re_d)
            + 0.06795 * np.log10(j) * np.log10(activity_factor) ** 2
            - 0.47030 * np.log10(ct)
            - 0.05973 * np.log10(j) ** 2
            + 0.17931 * np.log10(ct) * np.log10(activity_factor) * np.log10(twist_blade)
            - 0.19079 * np.log10(twist_blade) ** 2
            - 0.07782 * np.log10(j) * np.log10(activity_factor) * np.log10(twist_blade)
            - 0.07683 * np.log10(ct) ** 2 * np.log10(activity_factor)
        )
        * tip_mach ** (+0.00011 * np.log10(re_d) ** 3)
        * re_d
        ** (
            +0.02100 * np.log10(solidity) * np.log10(ct) ** 2
            - 0.32454
            - 0.00078 * np.log10(ct) ** 2 * np.log10(twist_blade)
            + 0.00371 * np.log10(re_d) * np.log10(ct) ** 2
            + 0.00015 * np.log10(re_d) ** 3
        )
        * solidity ** (-0.02698 * np.log10(solidity) ** 2 * np.log10(activity_factor))
        * ct
        ** (
            +1.48237
            + 0.01548 * np.log10(ct) ** 2 * np.log10(activity_factor)
            + 0.10762 * np.log10(ct)
            - 0.05141 * np.log10(ct) ** 2
        )
        * activity_factor ** (-0.00292 * np.log10(activity_factor) ** 2 * np.log10(twist_blade))
    )

    return cp


if __name__ == "__main__":
    file_path = pth.join(pth.dirname(__file__), "data/mtv1a_180_51.csv")
    orig_data = pd.read_csv(file_path)

    eff_orig = orig_data["ETA"].to_numpy()
    cp_orig = orig_data["CP"].to_numpy()
    j_orig = orig_data["ADVANCE_RATIO"].to_numpy()

    eff_screening = np.where(eff_orig > 0.5)

    ct_orig = cp_orig * eff_orig / j_orig

    n_rot = 2500.0 / 60.0  # rps
    diameter = 1.8

    atm = Atmosphere(altitude=0.0)

    airspeed = j_orig * n_rot * diameter
    m_tip = (
        airspeed**2.0 + (n_rot * 2.0 * np.pi) ** 2.0 * (diameter / 2.0) ** 2.0
    ) / atm.speed_of_sound**2.0
    reynolds_d = (
        np.sqrt(airspeed**2.0 + (n_rot * 2.0 * np.pi) ** 2.0 * (diameter / 2.0) ** 2.0)
        * diameter
        / atm.kinematic_viscosity
    )

    # Hereon starts the speculation
    solidity_orig = 0.0896
    activity_factor_orig = 112
    twist_blade_orig = 30.0 / 180.0 * np.pi

    cp_computed = cp_from_ct_new_new(
        j=j_orig[eff_screening],
        tip_mach=m_tip[eff_screening],
        re_d=reynolds_d[eff_screening],
        solidity=solidity_orig,
        ct=ct_orig[eff_screening],
        activity_factor=activity_factor_orig,
        twist_blade=twist_blade_orig,
    )

    error = np.abs(cp_orig[eff_screening] - cp_computed) / cp_orig[eff_screening] * 100.0

    fig1 = go.Figure()

    efficiency_orig_contour = go.Contour(
        x=j_orig[eff_screening],
        y=cp_orig[eff_screening],
        z=eff_orig[eff_screening] - ct_orig[eff_screening] / cp_computed * j_orig[eff_screening],
        ncontours=20,
        contours=dict(
            coloring="heatmap",
            showlabels=True,  # show labels on contours
            labelfont=dict(  # label font properties
                size=12,
                color="white",
            ),
        ),
        zmax=0.1,
        zmin=0.0,
    )
    fig1.add_trace(efficiency_orig_contour)
    fig1.update_layout(
        title_text=r"$\text{Error on the efficiency map of the MTV-1-A/180-51 propeller for }"
        r"\eta > 0.5$",
        title_x=0.5,
        xaxis_title="Advance ratio [-]",
        yaxis_title="Power coefficient [-]",
    )
    fig1.show()
