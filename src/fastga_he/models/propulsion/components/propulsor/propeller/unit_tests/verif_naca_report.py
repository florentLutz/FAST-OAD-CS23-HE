# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

# TODO: In this file we will check the results compare to NACA N640 report considering the
#  2-blades, 5868-9 propeller

import numpy as np
import matplotlib.pyplot as plt
from stdatm import Atmosphere

ELEMENTS_NUMBER = 100


def cp_from_ct(j, tip_mach, re_d, solidity, ct, activity_factor, aspect_ratio):

    cp = (
        10 ** 0.36560
        * j
        ** (
            -4.61201 * np.log10(ct)
            - 0.02274 * np.log10(j) * np.log10(activity_factor)
            + 0.01441 * np.log10(re_d) ** 2
            - 0.27259 * np.log10(ct) ** 2
            - 0.69104 * np.log10(tip_mach) * np.log10(solidity)
            - 0.29060 * np.log10(j) * np.log10(ct)
            - 0.50680 * np.log10(j) ** 2
            + 0.15905 * np.log10(solidity) ** 2
            - 2.13139 * np.log10(solidity) * np.log10(activity_factor)
            - 0.00610 * np.log10(re_d) * np.log10(aspect_ratio)
            + 6.36947 * np.log10(solidity)
            + 1.22123 * np.log10(ct) * np.log10(aspect_ratio)
            + 1.23502 * np.log10(ct) * np.log10(activity_factor)
            - 2.09630 * np.log10(solidity) * np.log10(aspect_ratio)
        )
        * tip_mach
        ** (
            -0.14528 * np.log10(tip_mach) * np.log10(activity_factor)
            - 0.45985 * np.log10(solidity) * np.log10(ct)
            - 0.77646 * np.log10(tip_mach) * np.log10(solidity)
            + 1.54416 * np.log10(solidity)
            + 0.38145 * np.log10(solidity) ** 2
            - 0.03329 * np.log10(re_d) * np.log10(ct)
            - 0.21297 * np.log10(re_d) * np.log10(solidity)
            - 0.20773 * np.log10(solidity) * np.log10(aspect_ratio)
            - 0.20054 * np.log10(solidity) * np.log10(activity_factor)
            + 0.02692 * np.log10(ct) ** 2
        )
        * re_d
        ** (
            +0.01092 * np.log10(ct) ** 2
            + 0.00145 * np.log10(re_d) ** 2
            - 0.02079 * np.log10(aspect_ratio)
            - 0.04205 * np.log10(solidity) * np.log10(ct)
        )
        * solidity
        ** (
            -0.13955 * np.log10(ct) * np.log10(activity_factor)
            + 0.12848 * np.log10(solidity) * np.log10(activity_factor)
            + 0.05448 * np.log10(solidity) * np.log10(ct)
            - 0.00950 * np.log10(activity_factor) ** 2
            - 0.07964 * np.log10(solidity) ** 2
        )
        * ct
        ** (
            2.70875
            + 0.12148 * np.log10(ct) * np.log10(activity_factor)
            - 0.64370 * np.log10(aspect_ratio)
            - 0.41614 * np.log10(activity_factor)
        )
        * activity_factor
        ** (-0.00180 * np.log10(activity_factor) ** 2 - 0.06746 * np.log10(aspect_ratio))
    )
    return cp


def ct_computation(j, tip_mach, re_d, solidity, pitch_ratio, activity_factor, aspect_ratio):

    ct = (
        10 ** -0.36356
        * j
        ** (
            -10.93795 * np.log10(pitch_ratio) * np.log10(activity_factor)
            + 7.68967 * np.log10(j) * np.log10(tip_mach)
            - 7.59555 * np.log10(j) * np.log10(aspect_ratio)
            + 6.53862 * np.log10(aspect_ratio)
            + 0.01403 * np.log10(j) * np.log10(solidity)
            - 4.28265 * np.log10(tip_mach) * np.log10(aspect_ratio)
            - 2.13630 * np.log10(re_d) * np.log10(aspect_ratio)
            - 0.02564 * np.log10(re_d) * np.log10(activity_factor)
            + 4.61797 * np.log10(re_d) * np.log10(pitch_ratio)
            + 3.52641 * np.log10(tip_mach) * np.log10(activity_factor)
            + 18.35857 * np.log10(j) * np.log10(pitch_ratio)
            - 4.36732 * np.log10(pitch_ratio) ** 2
            + 0.08796 * np.log10(solidity) ** 2
            - 0.38494 * np.log10(re_d)
        )
        * tip_mach
        ** (
            -7.33393 * np.log10(tip_mach) * np.log10(pitch_ratio)
            + 0.76901 * np.log10(tip_mach) ** 2
            + 0.16486 * np.log10(tip_mach) * np.log10(solidity)
            + 2.15327 * np.log10(tip_mach) * np.log10(re_d)
            - 14.83501 * np.log10(pitch_ratio) * np.log10(activity_factor)
            + 1.60807 * np.log10(pitch_ratio) ** 2
            - 13.36682 * np.log10(tip_mach)
            + 2.50216
        )
        * re_d
        ** (
            -0.46968
            - 4.66321 * np.log10(pitch_ratio) * np.log10(activity_factor)
            + 4.65643 * np.log10(pitch_ratio)
            + 0.16856 * np.log10(re_d) * np.log10(solidity)
            - 0.01069 * np.log10(activity_factor) * np.log10(aspect_ratio)
            - 2.06165 * np.log10(solidity)
        )
        * solidity
        ** (
            0.15090 * np.log10(pitch_ratio) * np.log10(activity_factor)
            - 0.00435 * np.log10(pitch_ratio) ** 2
            + 0.06730 * np.log10(solidity) ** 2
            + 6.45572
            + 0.02611 * np.log10(solidity) * np.log10(aspect_ratio)
        )
        * pitch_ratio
        ** (
            -0.22285 * np.log10(pitch_ratio) * np.log10(activity_factor)
            - 0.48000 * np.log10(pitch_ratio) ** 2
            + 4.69475 * np.log10(activity_factor) ** 2
            + 3.09433 * np.log10(activity_factor) * np.log10(aspect_ratio)
            - 2.55947 * np.log10(aspect_ratio) ** 2
            - 2.24554 * np.log10(pitch_ratio)
        )
    )

    return ct


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
radius_ratio_pitch_15 = np.array(
    [
        0.19792899408284026,
        0.22455621301775147,
        0.2467455621301775,
        0.27189349112426037,
        0.29852071005917163,
        0.32514792899408285,
        0.3473372781065089,
        0.3710059171597633,
        0.39615384615384613,
        0.4272189349112426,
        0.44940828402366867,
        0.4760355029585799,
        0.49822485207100586,
        0.520414201183432,
        0.5470414201183432,
        0.5736686390532544,
        0.5973372781065089,
        0.6210059171597633,
        0.6461538461538462,
        0.671301775147929,
        0.693491124260355,
        0.7201183431952662,
        0.7482248520710059,
        0.7763313609467456,
        0.7955621301775148,
        0.8236686390532544,
        0.8473372781065088,
        0.8710059171597632,
        0.8991124260355029,
        0.9272189349112425,
        0.9494082840236685,
    ]
)
pitch_15 = np.array(
    [
        0.4620253164556962,
        0.50042194092827,
        0.5358649789029533,
        0.559493670886076,
        0.5860759493670884,
        0.6126582278481012,
        0.6244725738396624,
        0.6333333333333333,
        0.6392405063291138,
        0.6392405063291138,
        0.6362869198312235,
        0.6333333333333333,
        0.6333333333333333,
        0.6333333333333333,
        0.6333333333333333,
        0.6303797468354428,
        0.6333333333333333,
        0.6303797468354428,
        0.6303797468354428,
        0.6303797468354428,
        0.6303797468354428,
        0.6303797468354428,
        0.6303797468354428,
        0.6303797468354428,
        0.6333333333333333,
        0.6333333333333333,
        0.6392405063291138,
        0.6362869198312235,
        0.6392405063291138,
        0.6362869198312235,
        0.6392405063291138,
    ]
)
radius_ratio_pitch_35 = np.array(
    [
        0.19792899408284026,
        0.21568047337278107,
        0.2467455621301775,
        0.27041420118343196,
        0.29556213017751476,
        0.322189349112426,
        0.35029585798816565,
        0.3754437869822485,
        0.39911242603550295,
        0.42278106508875735,
        0.4479289940828402,
        0.47307692307692306,
        0.4967455621301775,
        0.527810650887574,
        0.5485207100591716,
        0.5751479289940828,
        0.5988165680473372,
        0.6224852071005916,
        0.6491124260355029,
        0.6727810650887573,
        0.6994082840236686,
        0.7275147928994083,
        0.7497041420118342,
        0.777810650887574,
        0.7999999999999999,
        0.8281065088757397,
        0.8502958579881656,
        0.8724852071005916,
        0.9020710059171597,
        0.9257396449704143,
        0.9479289940828401,
    ]
)
pitch_35 = np.array(
    [
        0.9464135021097044,
        1.00253164556962,
        1.091139240506329,
        1.1472573839662445,
        1.2063291139240506,
        1.250632911392405,
        1.2860759493670886,
        1.3274261603375528,
        1.342194092827004,
        1.3540084388185654,
        1.3658227848101263,
        1.380590717299578,
        1.3983122362869196,
        1.421940928270042,
        1.4426160337552743,
        1.4691983122362866,
        1.4928270042194092,
        1.5223628691983122,
        1.5459915611814345,
        1.5637130801687764,
        1.5991561181434597,
        1.631645569620253,
        1.6493670886075948,
        1.6759493670886076,
        1.711392405063291,
        1.7350210970464133,
        1.7556962025316456,
        1.782278481012658,
        1.8118143459915612,
        1.841350210970464,
        1.8679324894514766,
    ]
)
prop_diameter = 3.048
hub_diameter = 0.2 * prop_diameter
n_blades = 2.0

radius_max = prop_diameter / 2.0
radius_min = hub_diameter / 2.0

length = radius_max - radius_min
elements_number = np.arange(ELEMENTS_NUMBER)
element_length = length / ELEMENTS_NUMBER
radius_ratio = (radius_min + (elements_number + 0.5) * element_length) / radius_max
radius = radius_ratio * radius_max

chord_array = np.interp(radius_ratio, radius_ratio_chord, chord_to_diameter_ratio) * prop_diameter
pitch_15_array = np.interp(radius_ratio, radius_ratio_pitch_15, pitch_15)
pitch_15_angle = np.arctan2(pitch_15_array * 2.0, radius_ratio * np.pi)
pitch_35_array = np.interp(radius_ratio, radius_ratio_pitch_35, pitch_35)
pitch_35_angle = np.arctan2(pitch_35_array * 2.0, radius_ratio * np.pi)
solidity_naca = n_blades / np.pi / radius_max ** 2.0 * np.sum(chord_array * element_length)
activity_factor_naca = (
    100000 / 32 / radius_max ** 5.0 * np.sum(chord_array * radius ** 3.0 * element_length)
)
c_star = np.sum(chord_array * radius ** 2.0 * element_length) / np.sum(
    radius ** 2.0 * element_length
)
aspect_ratio_naca = radius_max / c_star

atm = Atmosphere(0.0, altitude_in_feet=False)

# print(
#     ct_computation(
#         np.array([0.183, 0.183, 0.553, 0.183, 0.183]),
#         np.sqrt(np.array([0.783, 0.879, 0.285, 0.814, 0.814])),
#         np.array([1668736.192, 1120142.261, 2778670.745, 1460652.116, 1460652.116]),
#         np.array([0.059, 0.055, 0.106, 0.106, 0.076]),
#         np.array([2.292, 1.514, 2.292, 2.292, 1.514]),
#         np.array([100.287, 94.882, 164.328, 164.328, 117.711]),
#         np.array([7.927, 8.431, 4.668, 4.668, 6.586]),
#     )
# )

# ct = 0.08 at J = 1.02 for pitch at 35 deg, at 35deg, rpm = 800 meaning v is 41.452799999999996 m/s
j_list_45_deg = np.array(
    [
        1.1745,
        1.4473,
        1.6582,
        1.7927,
        1.8982,
        2.0000,
        2.0945,
        2.1891,
        2.2764,
        2.3673,
        2.4618,
        2.5527,
    ]
)
ct_list = np.array([0.11, 0.105, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
cp_list_verif_45_deg = np.array(
    [
        0.23133,
        0.21662,
        0.20800,
        0.19292,
        0.17821,
        0.16205,
        0.14626,
        0.12938,
        0.11323,
        0.096718,
        0.079487,
        0.062615,
    ]
)
cp_list_bemt_45_deg = np.array(
    [
        0.15242,
        0.17231065,
        0.18475006,
        0.17853666,
        0.16445737,
        0.15192319,
        0.13739132,
        0.12015745,
        0.10010211,
        0.08148222,
        0.05877337,
        0.03446293,
    ]
)
j_list_35_deg = np.array(
    [
        0.60727,
        0.86545,
        1.0582,
        1.1782,
        1.2655,
        1.3382,
        1.4109,
        1.4836,
        1.5564,
        1.6291,
        1.6982,
        1.7636,
    ]
)
cp_list_verif_35_deg = np.array(
    [
        0.15703,
        0.14410,
        0.13621,
        0.12759,
        0.11862,
        0.10821,
        0.097077,
        0.084513,
        0.072308,
        0.057949,
        0.043231,
        0.028154,
    ]
)
j_list_25_deg = np.array(
    [
        0.11636,
        0.43273,
        0.57818,
        0.68000,
        0.77091,
        0.84364,
        0.90909,
        0.97455,
        1.0436,
        1.1055,
        1.1673,
        1.2327,
    ]
)
cp_list_verif_25_deg = np.array(
    [
        0.10605,
        0.089897,
        0.083077,
        0.080205,
        0.075538,
        0.069795,
        0.062974,
        0.055077,
        0.046103,
        0.037487,
        0.027436,
        0.016667,
    ]
)
cp_list = np.zeros_like(ct_list)
for idx, (j, ct) in enumerate(zip(j_list_45_deg, ct_list)):
    cp_list[idx] = cp_from_ct(
        j,
        (j ** 2.0 + np.pi ** 2.0) * (800 / 60 * 3.048) ** 2.0 / atm.speed_of_sound ** 2.0,
        j * 800 / 60 * 3.048 ** 2.0 / atm.kinematic_viscosity,
        solidity_naca,
        ct,
        activity_factor_naca,
        aspect_ratio_naca,
    )

# plt.plot(j_list_25_deg, cp_list, label="VPLM repgression on BEMT")
# plt.plot(j_list_25_deg, cp_list_verif_25_deg, label="NACA report")

# plt.plot(j_list_35_deg, cp_list, label="VPLM repgression on BEMT")
# plt.plot(j_list_35_deg, cp_list_verif_35_deg, label="NACA report")

plt.plot(j_list_45_deg, cp_list, label="VPLM repgression on BEMT")
plt.plot(j_list_45_deg, cp_list_bemt_45_deg, label="BEMT")
plt.plot(j_list_45_deg, cp_list_verif_45_deg, label="NACA report")
print((cp_list - cp_list_verif_45_deg) / cp_list_verif_45_deg * 100.0)
plt.legend()
plt.show()
