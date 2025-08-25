#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np


def capacity_loss(dod, temperature, n_cycle):
    """dod in %, temperature in degK, n_cycle has no units."""
    f_dod = -0.002315 * dod**3.0 + 1.071 * dod**2.0 - 27.49 * dod + 8473
    q_loss = f_dod * np.exp(-4345 / temperature) * n_cycle**0.5
    return q_loss


if __name__ == "__main__":
    """
    This file will be used both to check the implementation of the formula from :cite:`chen:2019`,
    but also to find the default value of the "correction factor" which gives the value of 40% 
    capacity loss in 500 cycles in default conditions as can be found in :cite:`samsung:2015`.
    """

    # Figure 5)a) at 900 cycles with a DOD of 60% the capacity loss for 10, 25 and 45 degC should be
    # around 0.056, 0.12 and 0.31

    print(capacity_loss(60.0, 273.15 + np.array([10, 25, 45]), 900.0))

    # Figure 5)b) at 900 cycles, with a temperature of 25 degC the capacity loss for 10, 20, 40, 60,
    # 80 and 100% should be 0.111, 0.115, 0.123, 0.136, 0.171 and 0.194

    print(capacity_loss(np.array([10, 20, 40, 60, 80, 100]), 298.15, 900.0))

    # Now we will look for the corrective factor that gives exactly 40% discharge for 500 cycle,
    # with a DOD of 100% (assumed), and a temperature of 23 degC (seems to be nominal condition from
    # the datasheet).

    capacity_loss_nominal = capacity_loss(100.0, 296.15, 500)
    print(capacity_loss_nominal)
    corrective_factor = 0.4 / capacity_loss_nominal
    print(corrective_factor)
