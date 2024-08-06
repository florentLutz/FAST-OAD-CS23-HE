# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pandas as pd
import os.path as pth

if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=np.inf)

    for model_number in ["188", "208", "228", "268", "348"]:
        file_path = pth.join(pth.dirname(__file__), "data/free_run_losses_" + model_number + ".csv")
        data = pd.read_csv(file_path)

        speed = data["SPEED"].to_numpy() * 2 * np.pi / 60
        losses = data["LOSSES"].to_numpy()

        try:
            B = losses
            # A = np.column_stack([speed, speed ** 1.5, speed ** 2])
            A = np.column_stack([speed**1.5, speed**3])
            # A = np.column_stack([speed ** 1.5])

            x = np.linalg.lstsq(A, B, rcond=None)
            # alpha, beta, gamma = x[0]
            alpha, beta = x[0]
            # alpha = x[0]

            print("======== EMRAX " + model_number + " ========")

            # print(alpha, beta, gamma)
            print(alpha, beta)
            # print(alpha)

            # print(
            #     (losses - alpha * speed - beta * speed ** 1.5 - gamma * speed ** 2.0) / losses * 100
            # )
            print((losses - alpha * speed**1.5 - beta * speed**3) / losses * 100)
            print(np.mean(np.abs((losses - alpha * speed**1.5 - beta * speed**3) / losses * 100)))
            # print((losses - alpha * speed ** 1.5) / losses * 100)
            # print(np.mean(np.abs((losses - alpha * speed ** 1.5) / losses * 100)))

        except:
            print("EMRAX " + model_number + " did not converge")

    print("===== Coefficients read on map =====")

    beta_star = np.array([1, 1.65313832, 3.04546602, 8.29828454, 13.0894464])
    gamma_star = np.array([1, 1.68075497, 2.29491343, 7.09017588, 14.7347612])

    diameter = np.array([188, 208, 228, 268, 348])
    diameter_star = diameter / diameter[0]

    length = np.array([77, 85, 86, 91, 107])
    length_star = length / length[0]

    B = np.log(beta_star[:-1])
    A = np.column_stack([np.log(diameter_star[:-1]), np.log(length_star[:-1])])

    print("===== Beta scaling =====")
    x = np.linalg.lstsq(A, B, rcond=None)
    c_1, c_2 = x[0]
    print(c_1, c_2)
    print((beta_star - diameter_star**c_1 * length_star**c_2) / beta_star * 100.0)

    print("===== Gamma scaling =====")
    B = np.log(gamma_star[:-1])
    x = np.linalg.lstsq(A, B, rcond=None)
    c_1, c_2 = x[0]
    print(c_1, c_2)
    print((beta_star - diameter_star**c_1 * length_star**c_2) / beta_star * 100.0)
