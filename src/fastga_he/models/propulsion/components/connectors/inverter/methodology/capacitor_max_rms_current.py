# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

if __name__ == "__main__":

    modulation_idx = np.linspace(0.0, 1.0, 10001)
    factor = np.sqrt(
        np.sqrt(3.0) * modulation_idx / 4.0 / np.pi
        + np.sqrt(3.0) * modulation_idx / np.pi
        - 9.0 * modulation_idx ** 2.0 / 16.0
    )
    print(np.max(factor))
    print(modulation_idx[np.argmax(factor)])

    factor_kappa = np.sqrt(-0.27 * modulation_idx ** 2.0 + 0.3643 * modulation_idx + 9e-5)
    print(np.max(factor_kappa))
    print(modulation_idx[np.argmax(factor_kappa)])
