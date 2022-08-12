# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np


if __name__ == "__main__":

    diameter = np.array([188, 188, 188, 208, 208, 208, 228, 228, 228, 268, 268, 268])
    diameter_star = diameter / diameter[0]

    length = np.array([77, 77, 77, 85, 85, 85, 86, 86, 86, 91, 91, 91])
    length_star = length / length[0]

    torque = np.array([40, 40, 40, 64, 64, 64, 96, 96, 96, 200, 200, 200])
    torque_star = torque / torque[0]

    x0 = (1.0, 1.0)
    bnds = ((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf))

    B = np.log(torque_star)

    A = np.column_stack([np.log(diameter_star), np.log(length_star)])
    # A = np.column_stack([np.ones_like(np.log(currents)), np.log(currents), np.log(torques)])

    x = np.linalg.lstsq(A, B, rcond=None)
    print(x[0])
    print(x[1])

    a_sol, b_sol = x[0]
    print(diameter_star ** a_sol * length_star ** b_sol)

    # log_k, a_sol, b_sol = x[0]
    # print(np.exp(log_k) * currents ** a_sol * torques ** b_sol)
