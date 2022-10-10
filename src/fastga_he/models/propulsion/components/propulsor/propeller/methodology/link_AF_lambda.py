# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import copy
from pyDOE2 import *
import matplotlib.pyplot as plt
import numpy as np


def compute_coeff_not_adim():

    diameter = 2.5
    radius_max = diameter / 2
    radius_mid = 0.5 * radius_max
    chord_ratio_mid = 1.5
    chord_root = 0.0380 * diameter
    root_radius = 0.2 * radius_max
    chord_tip = 0.0380 * diameter
    tip_radius = 0.999 * radius_max
    chord_mid = chord_ratio_mid * chord_root

    matrix_to_inv = np.array(
        [
            [root_radius ** 2.0, root_radius, 1.0, 0.0, 0.0, 0.0],
            [radius_mid ** 2.0, radius_mid, 1.0, 0.0, 0.0, 0.0],
            [2.0 * radius_mid, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0 * radius_mid, 1.0, 0.0],
            [0.0, 0.0, 0.0, radius_mid ** 2.0, radius_mid, 1.0],
            [0.0, 0.0, 0.0, tip_radius ** 2.0, tip_radius, 1.0],
        ]
    )
    result_matrix = np.array(
        [
            [chord_root],
            [chord_mid],
            [(chord_tip - chord_root) / (tip_radius - root_radius)],
            [(chord_tip - chord_root) / (tip_radius - root_radius)],
            [chord_mid],
            [chord_tip],
        ]
    )

    k12, k11, k10, k22, k21, k20 = np.dot(np.linalg.inv(matrix_to_inv), result_matrix).transpose()[
        0
    ]

    return k12, k11, k10, k22, k21, k20


def compute_coeff_adim():

    radius_ratio_mid = 0.5
    chord_ratio_mid = 1.5
    chord_to_diameter_root = 0.0380
    root_radius_ratio = 0.2
    chord_to_diameter_tip = 0.0380
    tip_radius_ratio = 0.999

    chord_to_diameter_ratio_mid = chord_ratio_mid * chord_to_diameter_root

    matrix_to_inv = np.array(
        [
            [root_radius_ratio ** 2.0, root_radius_ratio, 1.0, 0.0, 0.0, 0.0],
            [radius_ratio_mid ** 2.0, radius_ratio_mid, 1.0, 0.0, 0.0, 0.0],
            [2.0 * radius_ratio_mid, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0 * radius_ratio_mid, 1.0, 0.0],
            [0.0, 0.0, 0.0, radius_ratio_mid ** 2.0, radius_ratio_mid, 1.0],
            [0.0, 0.0, 0.0, tip_radius_ratio ** 2.0, tip_radius_ratio, 1.0],
        ]
    )
    result_matrix = np.array(
        [
            [chord_to_diameter_root],
            [chord_to_diameter_ratio_mid],
            [
                (chord_to_diameter_tip - chord_to_diameter_root)
                / (tip_radius_ratio - root_radius_ratio)
            ],
            [
                (chord_to_diameter_tip - chord_to_diameter_root)
                / (tip_radius_ratio - root_radius_ratio)
            ],
            [chord_to_diameter_ratio_mid],
            [chord_to_diameter_tip],
        ]
    )

    k12, k11, k10, k22, k21, k20 = np.dot(np.linalg.inv(matrix_to_inv), result_matrix).transpose()[
        0
    ]

    return k12, k11, k10, k22, k21, k20


def compute_lambda_af(radius_ratio_mid, chord_ratio_mid, diameter, chord_to_diameter_root):

    root_radius_ratio = 0.2
    chord_to_diameter_tip = 0.0380
    tip_radius_ratio = 0.999

    chord_to_diameter_ratio_mid = chord_ratio_mid * chord_to_diameter_root

    radius_ratio_vect = np.array([0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    radius_max = diameter / 2.0

    matrix_to_inv = np.array(
        [
            [root_radius_ratio ** 2.0, root_radius_ratio, 1.0, 0.0, 0.0, 0.0],
            [radius_ratio_mid ** 2.0, radius_ratio_mid, 1.0, 0.0, 0.0, 0.0],
            [2.0 * radius_ratio_mid, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0 * radius_ratio_mid, 1.0, 0.0],
            [0.0, 0.0, 0.0, radius_ratio_mid ** 2.0, radius_ratio_mid, 1.0],
            [0.0, 0.0, 0.0, tip_radius_ratio ** 2.0, tip_radius_ratio, 1.0],
        ]
    )
    result_matrix = np.array(
        [
            [chord_to_diameter_root],
            [chord_to_diameter_ratio_mid],
            [
                (chord_to_diameter_tip - chord_to_diameter_root)
                / (tip_radius_ratio - root_radius_ratio)
            ],
            [
                (chord_to_diameter_tip - chord_to_diameter_root)
                / (tip_radius_ratio - root_radius_ratio)
            ],
            [chord_to_diameter_ratio_mid],
            [chord_to_diameter_tip],
        ]
    )

    k12, k11, k10, k22, k21, k20 = np.dot(np.linalg.inv(matrix_to_inv), result_matrix).transpose()[
        0
    ]
    chord_distribution = np.where(
        radius_ratio_vect < radius_ratio_mid,
        k12 * radius_ratio_vect ** 2.0 + k11 * radius_ratio_vect + k10,
        k22 * radius_ratio_vect ** 2.0 + k21 * radius_ratio_vect + k20,
    )

    chord_array = chord_distribution * diameter

    length = radius_max - 0.2 * radius_max
    elements_number = np.arange(100)
    element_length = length / 100
    radius = 0.2 * radius_max + (elements_number + 0.5) * element_length

    chord_array = np.interp(radius, radius_ratio_vect * radius_max, chord_array)

    radius_min = 0.2 * radius_max
    radius_mid = radius_ratio_mid * radius_max
    a1 = k12 * 2 / radius_max
    a2 = k22 * 2 / radius_max
    b1 = 2 * k11
    b2 = 2 * k21
    c1 = k10 * 2 * radius_max
    c2 = k20 * 2 * radius_max
    analytical_integral_af = (
        a1 / 6 * (radius_mid ** 6 - radius_min ** 6)
        + b1 / 5 * (radius_mid ** 5 - radius_min ** 5)
        + c1 / 4 * (radius_mid ** 4 - radius_min ** 4)
        + a2 / 6 * (radius_max ** 6 - radius_mid ** 6)
        + b2 / 5 * (radius_max ** 5 - radius_mid ** 5)
        + c2 / 4 * (radius_max ** 4 - radius_mid ** 4)
    )
    analytical_integral_c_star = (
        a1 / 5 * (radius_mid ** 5 - radius_min ** 5)
        + b1 / 4 * (radius_mid ** 4 - radius_min ** 4)
        + c1 / 3 * (radius_mid ** 3 - radius_min ** 3)
        + a2 / 5 * (radius_max ** 5 - radius_mid ** 5)
        + b2 / 4 * (radius_max ** 4 - radius_mid ** 4)
        + c2 / 3 * (radius_max ** 3 - radius_mid ** 3)
    )

    activity_factor = (
        100000 / 32 / radius_max ** 5.0 * np.sum(chord_array * radius ** 3.0 * element_length)
    )
    c_star = np.sum(chord_array * radius ** 2.0 * element_length) / np.sum(
        radius ** 2.0 * element_length
    )
    aspect_ratio = radius_max / c_star

    product = 100000 / 96 / radius_max * analytical_integral_af / analytical_integral_c_star

    return activity_factor, aspect_ratio, product


if __name__ == "__main__":

    doe_temp = lhs(4, samples=2000, criterion="correlation")

    D_min, D_max = 0.5, 3.0
    c_mid_radius_val_min, c_mid_radius_val_max = 1.2, 5.0
    c_mid_radius_ratio_val_min, c_mid_radius_ratio_val_max = 0.4, 0.8
    chord_to_diameter_root_min, chord_to_diameter_root_max = 0.025, 0.050

    doeX = copy.deepcopy(doe_temp)

    diameter_array = doe_temp[:, 0] * (D_max - D_min) + D_min
    c_mid_ratio_array = (
        doe_temp[:, 1] * (c_mid_radius_val_max - c_mid_radius_val_min) + c_mid_radius_val_min
    )
    c_mid_radius_ratio_array = (
        doe_temp[:, 2] * (c_mid_radius_ratio_val_max - c_mid_radius_ratio_val_min)
        + c_mid_radius_ratio_val_min
    )
    chord_to_diameter_root_array = (
        doe_temp[:, 3] * (chord_to_diameter_root_max - chord_to_diameter_root_min)
        + chord_to_diameter_root_min
    )

    activity_factor_array = []
    lambda_array = []
    product_array = []

    for diameter_loop, c_mid_radius, c_mid_ratio, chord_to_diameter_root_loop in zip(
        diameter_array, c_mid_radius_ratio_array, c_mid_ratio_array, chord_to_diameter_root_array
    ):

        af, aspect_ratio_loop, product = compute_lambda_af(
            c_mid_radius, c_mid_ratio, diameter_loop, chord_to_diameter_root_loop
        )

        activity_factor_array.append(af)
        lambda_array.append(aspect_ratio_loop)
        product_array.append(product)

    # cst = np.array(lambda_array) * np.array(activity_factor_array)
    #
    plt.plot(activity_factor_array, lambda_array, "o")
    plt.plot(activity_factor_array, np.array(product_array) / np.array(activity_factor_array), "o")
    plt.show()
