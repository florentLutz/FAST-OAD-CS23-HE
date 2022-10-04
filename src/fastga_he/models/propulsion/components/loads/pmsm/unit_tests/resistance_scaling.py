# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np


def best_fit(parameter, k, a, b, c):

    voltage_to_fit, diameter_to_fit, length_to_fit = parameter

    return k * voltage_to_fit ** a * diameter_to_fit ** b * length_to_fit ** c


if __name__ == "__main__":

    resistances = np.array([12.0, 5, 0.8, 12, 5, 0.9, 16.7, 7, 1.1, 22.9, 10.5, 1.8])
    resistance_star = resistances / resistances[0]

    voltages = np.array([430, 300, 110, 550, 350, 120, 680, 500, 160, 800, 650, 250])
    voltage_star = voltages / voltages[0]

    currents = np.array([100, 150, 400, 100, 160, 400, 115, 160, 450, 125, 190, 500])

    torques = np.array([40, 40, 40, 64, 64, 64, 96, 96, 96, 200, 200, 200])

    diameter = np.array([188, 188, 188, 208, 208, 208, 228, 228, 228, 268, 268, 268])
    diameter_star = diameter / diameter[0]

    length = np.array([77, 77, 77, 85, 85, 85, 86, 86, 86, 91, 91, 91])
    length_star = length / length[0]

    B = np.log(resistance_star) - 2.0 * np.log(voltage_star)

    A = np.column_stack([np.log(diameter_star), np.log(length_star)])
    # A = np.column_stack([np.ones_like(np.log(currents)), np.log(currents), np.log(torques)])

    x = np.linalg.lstsq(A, B)
    print(x[0])
    print(x[1])

    b_sol, c_sol = x[0]
    print(voltage_star ** 2.0 * diameter_star ** b_sol * length_star ** c_sol)
    print(resistance_star)

    # log_k, a_sol, b_sol = x[0]
    # print(np.exp(log_k) * currents ** a_sol * torques ** b_sol)
