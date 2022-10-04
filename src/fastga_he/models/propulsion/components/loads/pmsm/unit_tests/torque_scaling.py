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

    B = np.log(torque_star)

    A = np.column_stack([np.log(diameter_star), np.log(length_star)])
    # A = np.column_stack([np.ones_like(np.log(currents)), np.log(currents), np.log(torques)])

    print("===== Torque =====")
    x = np.linalg.lstsq(A, B, rcond=None)
    print(x[0])
    print(x[1])

    a_sol, b_sol = x[0]
    print(diameter_star ** a_sol * length_star ** b_sol)

    # log_k, a_sol, b_sol = x[0]
    # print(np.exp(log_k) * currents ** a_sol * torques ** b_sol)

    print("===== Weight =====")
    B = weight - 2.8
    A = np.column_stack([torque, torque ** (3 / 3.5)])
    x = np.linalg.lstsq(A, B, rcond=None)
    print(x[0])
    print(x[1])
    print(2.8 + x[0][0] * torque + x[0][1] * torque ** (3 / 3.5))
