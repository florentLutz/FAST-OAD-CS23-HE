# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np


if __name__ == "__main__":
    conductor_area = np.array(
        [
            1.02e-5,
            1.52e-5,
            3.8e-5,
            1.14e-5,
            1.70e-5,
            4.25e-5,
            1.14e-5,
            1.70e-5,
            4.25e-5,
            1.14e-5,
            1.70e-5,
            4.25e-5,
        ]
    )
    conductor_area_star = conductor_area / conductor_area[0]

    voltages = np.array([430, 300, 110, 550, 350, 120, 680, 500, 160, 800, 650, 250])
    voltage_star = voltages / voltages[0]

    currents = np.array([100, 150, 400, 100, 160, 400, 115, 160, 450, 125, 190, 500])
    current_star = currents / currents[0]

    diameter = np.array([188, 188, 188, 208, 208, 208, 228, 228, 228, 268, 268, 268])
    diameter_star = diameter / diameter[0]

    length = np.array([77, 77, 77, 85, 85, 85, 86, 86, 86, 91, 91, 91])
    length_star = length / length[0]

    x0 = (1.0, 1.0, 1.0)
    bnds = ((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf))

    B = np.log(conductor_area_star)

    A = np.column_stack([np.log(current_star), np.log(diameter_star), np.log(length_star)])
    # A = np.column_stack([np.ones_like(np.log(currents)), np.log(currents), np.log(torques)])

    x = np.linalg.lstsq(A, B, rcond=None)
    print(x[0])
    print(x[1])

    a_sol, b_sol, c_sol = x[0]
    print(current_star**a_sol * diameter_star**b_sol * length_star**c_sol)

    # log_k, a_sol, b_sol = x[0]
    # print(np.exp(log_k) * currents ** a_sol * torques ** b_sol)
