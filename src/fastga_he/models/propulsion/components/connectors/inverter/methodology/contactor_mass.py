# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    rms_current = np.array(
        [
            23.183391003460187,
            48.442906574394456,
            74.0484429065744,
            99.0795847750866,
            124.22145328719725,
            149.82698961937714,
            174.74048442906573,
            200.00000000000003,
            224.5674740484429,
            250.1730103806228,
            275.0865051903114,
            300,
        ]
    )
    weight = np.array(
        [
            0.2129032258064516,
            0.3548387096774195,
            0.46881720430107543,
            0.5806451612903226,
            0.6795698924731183,
            0.7655913978494624,
            0.8580645161290322,
            0.9440860215053763,
            1.0279569892473117,
            1.10752688172043,
            1.1849462365591399,
            1.2559139784946236,
        ]
    )

    B = np.log(weight)

    A = np.column_stack([np.ones_like(rms_current), np.log(rms_current)])

    x = np.linalg.lstsq(A, B, rcond=-1)
    a, b = x[0]

    print(np.exp(a), b)
    print(np.exp(a) * rms_current**b)
    print(weight)

    plt.plot(weight, weight, "o", color="red")
    plt.plot(weight, np.exp(a) * rms_current**b)
    plt.show()
