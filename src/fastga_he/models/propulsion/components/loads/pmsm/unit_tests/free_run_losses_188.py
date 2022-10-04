# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

if __name__ == "__main__":

    speed = np.array(
        [
            7.142857142857338,
            564.2857142857147,
            1007.142857142857,
            1485.714285714286,
            2000,
            2571.428571428572,
            2992.857142857142,
            3528.571428571429,
            4007.142857142858,
            4607.142857142857,
            5007.142857142857,
            5500.0,
            6014.285714285716,
            6500,
            7000,
            7400,
            8000,
        ]
    )
    losses = np.array(
        [
            11.387900355871807,
            64.76868327402144,
            103.9145907473312,
            146.61921708185082,
            207.11743772242016,
            267.6156583629895,
            317.4377224199293,
            399.2882562277582,
            481.1387900355876,
            595.0177935943063,
            687.5444839857655,
            797.8647686832742,
            929.5373665480431,
            1061.2099644128118,
            1200.0000000000005,
            1306.7615658362993,
            1498.9323843416375,
        ]
    )

    losses_star = losses / losses[-1]
    speed_star = speed / speed[-1]

    B = losses

    A = np.column_stack(
        [
            speed,
            speed ** 2.0,
        ]
    )

    x = np.linalg.lstsq(A, B, rcond=None)
    alpha, beta = x[0]
    print(x[1])
    plt.plot(speed, losses)
    plt.show()
    print(((alpha * speed + beta * speed ** 2.0) - losses) / losses * 100.0)
