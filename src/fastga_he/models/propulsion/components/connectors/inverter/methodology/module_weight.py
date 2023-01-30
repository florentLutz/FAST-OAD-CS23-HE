# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    I_cal_star = np.array([1, 0.75, 1.5, 1, 2, 3, 2, 2.25, 2.25, 3, 3, 0.75, 1, 1.5, 2])
    V_cal_star = np.array(
        [1, 2, 1, 2.833333333, 1, 1, 2.833333333, 2.833333333, 2, 2.833333333, 2, 2, 2, 2, 2]
    )
    W_star = np.array([1, 1, 1, 1, 1, 1.2, 1.6, 1.2, 1.2, 1.6, 1.6, 1.3, 1.3, 1.3, 1.3])

    B = np.log(W_star)

    A = np.column_stack([np.ones_like(I_cal_star), np.log(I_cal_star), np.log(V_cal_star)])

    x = np.linalg.lstsq(A, B)
    log_k, a, b = x[0]

    print(np.exp(log_k) * I_cal_star ** a * V_cal_star ** b)
    print(W_star)

    plt.plot(W_star)
    plt.plot(np.exp(log_k) * I_cal_star ** a * V_cal_star ** b)
    plt.show()
