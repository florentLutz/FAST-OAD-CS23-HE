# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

# TODO: In this file we will check the results compare to NACA N640 report considering the
#  2-blades, 5868-9 propeller

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stdatm import Atmosphere

ELEMENTS_NUMBER = 100

TWIST = 20


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
            1.196863740662988,
            1.301875701801494,
            1.374490064931894,
            1.442976126543963,
            1.507374896255431,
            1.571718986476590,
            1.632016794414881,
            1.692273592735439,
            1.756535663721135,
            1.812664160523361,
            1.876885221891325,
            1.937018991358687,
            1.993106478543182,
            2.053212908265391,
            2.109259385832154,
            2.169352145681785,
        ]
    )
    cp_list_verif_40_deg = np.array(
        [
            0.25020514573060587,
            0.24455987892398576,
            0.23809441976273008,
            0.22919533271493436,
            0.2190788458721867,
            0.2073407215739882,
            0.19560142557242594,
            0.1826459014792755,
            0.1684753209979007,
            0.15308616901821023,
            0.13769936044524725,
            0.12109515207733243,
            0.1044897720060538,
            0.08707474491041353,
            0.06925313674754674,
            0.05143270028804375,
        ]
    )
    cp_list_bemt_40_deg = np.array(
        [
            0.23674812,
            0.23507897,
            0.22743483,
            0.21828829,
            0.20770463,
            0.1962428,
            0.18365363,
            0.17032552,
            0.15651235,
            0.14118648,
            0.12567844,
            0.10909631,
            0.09143312,
            0.07309928,
            0.05369756,
            0.03356892,
        ]
    )

    j_list_35_deg = np.array(
        [
            0.9013698630136988,
            1.0164383561643837,
            1.0904109589041098,
            1.1561643835616440,
            1.2219178082191782,
            1.2753424657534247,
            1.3369863013698633,
            1.3863013698630140,
            1.4438356164383563,
            1.4972602739726029,
            1.5506849315068494,
            1.6041095890410961,
            1.6534246575342466,
            1.7150684931506850,
            1.7602739726027399,
            1.8178082191780824,
        ]
    )
    cp_list_verif_35_deg = np.array(
        [
            0.2,
            0.19595141700404858,
            0.19149797570850202,
            0.18582995951417006,
            0.1793522267206478,
            0.1700404858299595,
            0.15870445344129552,
            0.14817813765182186,
            0.13562753036437247,
            0.12307692307692308,
            0.11052631578947367,
            0.09635627530364374,
            0.08299595141700405,
            0.06761133603238867,
            0.052226720647773284,
            0.035627530364372495,
        ]
    )
    cp_list_bemt_35_deg = np.array(
        [
            0.19091221,
            0.19295575,
            0.18794868,
            0.18094397,
            0.17303403,
            0.16289146,
            0.15315517,
            0.14150774,
            0.1299607,
            0.1173649,
            0.10413456,
            0.09025926,
            0.07545485,
            0.06035288,
            0.04390672,
            0.02708258,
        ]
    )

    j_list_30_deg = np.array(
        [
            0.6630136986301373,
            0.7616438356164387,
            0.8397260273972604,
            0.9095890410958904,
            0.967123287671233,
            1.0328767123287672,
            1.0863013698630137,
            1.1315068493150686,
            1.1808219178082193,
            1.2342465753424658,
            1.2753424657534247,
            1.3246575342465756,
            1.3780821917808221,
            1.4273972602739728,
            1.4602739726027398,
            1.5219178082191782,
        ]
    )
    cp_list_verif_30_deg = np.array(
        [
            0.15951417004048582,
            0.15708502024291496,
            0.15465587044534412,
            0.15020242914979756,
            0.14453441295546557,
            0.13724696356275304,
            0.1311740890688259,
            0.12267206477732792,
            0.11214574898785426,
            0.10161943319838057,
            0.09068825910931178,
            0.07935222672064779,
            0.0668016194331984,
            0.05425101214574901,
            0.04129554655870449,
            0.02753036437246964,
        ]
    )
    cp_list_bemt_30_deg = np.array(
        [
            0.15576752,
            0.15657474,
            0.15401672,
            0.14956286,
            0.14271056,
            0.13608228,
            0.12766287,
            0.11796162,
            0.1080794,
            0.09794708,
            0.08642378,
            0.07493363,
            0.06292725,
            0.05001256,
            0.03601573,
            0.02215445,
        ]
    )

    j_list_25_deg = np.array(
        [
            0.3506849315068496,
            0.4575342465753427,
            0.56027397260274,
            0.6342465753424661,
            0.7041095890410962,
            0.7616438356164387,
            0.8150684931506853,
            0.8726027397260276,
            0.9260273972602742,
            0.9630136986301372,
            1.0123287671232877,
            1.0575342465753426,
            1.1027397260273972,
            1.152054794520548,
            1.193150684931507,
            1.2424657534246575,
        ]
    )
    cp_list_verif_25_deg = np.array(
        [
            0.11740890688259112,
            0.11781376518218623,
            0.11781376518218623,
            0.11578947368421055,
            0.11214574898785426,
            0.10688259109311743,
            0.10242914979757084,
            0.09595141700404858,
            0.08866396761133602,
            0.08056680161943319,
            0.07206477732793526,
            0.06275303643724695,
            0.05344129554655874,
            0.04331983805668016,
            0.03238866396761131,
            0.021457489878542513,
        ]
    )
    cp_list_bemt_25_deg = np.array(
        [
            0.12917972,
            0.11624116,
            0.11796743,
            0.11582404,
            0.11243503,
            0.1068532,
            0.10071728,
            0.0944364,
            0.08722613,
            0.07829211,
            0.06984561,
            0.060524,
            0.05060284,
            0.04029366,
            0.0291526,
            0.01765749,
        ]
    )

    j_list_20_deg = np.array(
        [
            0.1130434782608694,
            0.2388405797101448,
            0.3281159420289853,
            0.4011594202898549,
            0.4660869565217390,
            0.5391304347826086,
            0.5878260869565217,
            0.6527536231884058,
            0.6933333333333334,
            0.7501449275362319,
            0.7866666666666666,
            0.8353623188405797,
            0.8881159420289855,
            0.9246376811594201,
            0.9692753623188404,
            1.009855072463768,
        ]
    )
    cp_list_verif_20_deg = np.array(
        [
            0.08938775510204086,
            0.0885714285714286,
            0.08775510204081638,
            0.08734693877551022,
            0.08571428571428574,
            0.08285714285714291,
            0.07877551020408169,
            0.07387755102040822,
            0.06857142857142862,
            0.06285714285714289,
            0.055918367346938835,
            0.04979591836734698,
            0.04122448979591842,
            0.03306122448979598,
            0.024081632653061236,
            0.015510204081632673,
        ]
    )
    cp_list_bemt_20_deg = np.array(
        [
            0.09146229,
            0.09146229,
            0.09086901,
            0.08929317,
            0.08662091,
            0.08404534,
            0.07904708,
            0.07514493,
            0.06871005,
            0.06326294,
            0.05601372,
            0.04885499,
            0.0412965,
            0.03254319,
            0.02364223,
            0.01415908,
        ]
    )

    j_list_15_deg = np.array(
        [
            0.019710144927536116,
            0.019710144927536116,
            0.019710144927536116,
            0.10898550724637668,
            0.19420289855072453,
            0.26318840579710134,
            0.3321739130434781,
            0.3849275362318839,
            0.4457971014492752,
            0.5026086956521738,
            0.551304347826087,
            0.6,
            0.648695652173913,
            0.697391304347826,
            0.7339130434782608,
            0.7866666666666666,
        ]
    )
    cp_list_verif_15_deg = np.array(
        [
            0.057551020408163345,
            0.057551020408163345,
            0.057551020408163345,
            0.05795918367346942,
            0.057551020408163345,
            0.05673469387755109,
            0.05551020408163271,
            0.05346938775510207,
            0.05020408163265311,
            0.04612244897959189,
            0.041632653061224545,
            0.0371428571428572,
            0.03142857142857142,
            0.025306122448979618,
            0.01877551020408169,
            0.011428571428571455,
        ]
    )
    cp_list_bemt_15_deg = np.array(
        [
            0.06469607,
            0.06469607,
            0.06469607,
            0.0619596,
            0.06112431,
            0.05903839,
            0.05687954,
            0.05341644,
            0.05032339,
            0.04663819,
            0.04210182,
            0.03698745,
            0.03137016,
            0.02523418,
            0.01825663,
            0.01113809,
        ]
    )

    if TWIST == 40:
        cp_list_bemt = cp_list_bemt_40_deg
        j_list = j_list_40_deg
        cp_list_verif = cp_list_verif_40_deg
        rpm = 700.0
    elif TWIST == 35:
        cp_list_bemt = cp_list_bemt_35_deg
        j_list = j_list_35_deg
        cp_list_verif = cp_list_verif_35_deg
        rpm = 800.0
    elif TWIST == 30:
        cp_list_bemt = cp_list_bemt_30_deg
        j_list = j_list_30_deg
        cp_list_verif = cp_list_verif_30_deg
        rpm = 800.0
    elif TWIST == 25:
        cp_list_bemt = cp_list_bemt_25_deg
        j_list = j_list_25_deg
        cp_list_verif = cp_list_verif_25_deg
        rpm = 800.0
    elif TWIST == 20:
        cp_list_bemt = cp_list_bemt_20_deg
        j_list = j_list_20_deg
        cp_list_verif = cp_list_verif_20_deg
        rpm = 1000.0
    elif TWIST == 15:
        cp_list_bemt = cp_list_bemt_15_deg
        j_list = j_list_15_deg
        cp_list_verif = cp_list_verif_15_deg
        rpm = 1000.0
    else:
        cp_list_bemt = cp_list_bemt_40_deg
        j_list = j_list_40_deg
        cp_list_verif = cp_list_verif_40_deg
        rpm = 700.0

    cp_list = np.zeros_like(ct_list)
    for idx, (j_loop, ct_loop) in enumerate(zip(j_list, ct_list)):
        cp_list[idx] = cp_from_ct(
            j_loop,
            (j_loop ** 2.0 + np.pi ** 2.0) * (rpm / 60 * 3.048) ** 2.0 / atm.speed_of_sound ** 2.0,
            j_loop * rpm / 60 * 3.048 ** 2.0 / atm.kinematic_viscosity,
            solidity_naca,
            ct_loop,
            activity_factor_naca,
            aspect_ratio_naca,
        )

    error_margin = (cp_list_bemt - cp_list_verif) / cp_list_verif * 100.0
    print(error_margin)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    scatter_verif = go.Scatter(
        x=j_list,
        y=cp_list_verif,
        mode="lines+markers",
        name="NACA " "report",
        legendgroup="ct",
        legendgrouptitle_text="Thrust coefficient",
    )
    fig.add_trace(scatter_verif)
    scatter_computed = go.Scatter(
        x=j_list,
        y=cp_list,
        mode="lines+markers",
        name="VPLM repgression on BEMT",
        legendgroup="ct",
    )
    fig.add_trace(scatter_computed)
    scatter_pure_bemt = go.Scatter(
        x=j_list,
        y=cp_list_bemt,
        mode="lines+markers",
        name="BEMT",
        legendgroup="ct",
    )
    fig.add_trace(scatter_pure_bemt)
    scatter_effy = go.Scatter(
        x=j_list,
        y=j_list * ct_list / cp_list_verif,
        mode="lines+markers",
        name="efficiency NACA report",
        legendgroup="effy",
        legendgrouptitle_text="Efficiency",
    )
    fig.add_trace(scatter_effy, secondary_y=True)

    scatter_error = go.Scatter(
        x=j_list,
        y=abs(error_margin) / 100,
        mode="lines+markers",
        name="Error margin",
    )
    fig.add_trace(scatter_error, secondary_y=True)

    fig.update_layout(title="Twist = " + str(TWIST) + " deg", title_x=0.5)
    fig.update_yaxes(title_text=r"$\text{Thrust coefficient } C_T \text{ [-]}$", secondary_y=False)
    fig.update_yaxes(
        title_text=r"$\text{Efficiency } \eta \text{, Error margin [-]}$", secondary_y=True
    )
    fig.update_xaxes(title_text=r"$\text{Advance ratio } J \text{ [-]}$")

    fig.show()
