# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np


if __name__ == "__main__":
    diameter = np.array([188, 208, 228, 268, 348])
    diameter_star = diameter / diameter[0]

    length = np.array([77, 85, 86, 91, 107])
    length_star = length / length[0]

    torque = np.array([40, 64, 96, 200, 400])

    weight = np.array([7, 9.1, 12, 20, 41])

    torque_star = torque / torque[0]

    print("===== Torque =====")
    # We exclude the last motor to ensure the exponent are positive
    B = np.log(torque_star[:-1])
    A = np.column_stack([np.log(diameter_star[:-1]), np.log(length_star[:-1])])
    x = np.linalg.lstsq(A, B, rcond=None)
    print(x[0])
    print(x[1])

    a_sol, b_sol = x[0]
    print(diameter_star**a_sol * length_star**b_sol)
    print(torque_star)

    # log_k, a_sol, b_sol = x[0]
    # print(np.exp(log_k) * currents ** a_sol * torques ** b_sol)

    print("===== Weight =====")
    # We exclude the last motor to ensure the coefficients are positive
    B = weight[:-1] - 2.8
    A = np.column_stack([torque[:-1], torque[:-1] ** (3 / 3.5)])
    x = np.linalg.lstsq(A, B, rcond=None)
    print(x[0])
    print(x[1])
    print(2.8 + x[0][0] * torque + x[0][1] * torque ** (3 / 3.5))
    print(weight)

    print("===== D/L ratio =====")
    B = diameter[:-1] / length[:-1]
    A = np.column_stack([torque[:-1]])
    x = np.linalg.lstsq(A, B, rcond=None)
    print(x[0])
    print(x[1])
    print(torque[:-1] ** x[0])
    print(diameter[:-1] / length[:-1])
