# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesInternalResistance(om.ExplicitComponent):
    """
    Computation of the internal resistance of the battery. The shape is inherited from
    :cite:`chen:2006`, fitted on the 20Ah pouch data available in :cite:`cicconi:2017`. Fit works
    somewhat well for 1C and 2C discharge rate but starts to greatly underestimate battery
    voltage at 3C and more (on the conservative side).
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("state_of_charge", units="percent", val=np.full(number_of_points, np.nan))

        self.add_output("internal_resistance", units="ohm", val=np.full(number_of_points, 1e-3))

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dod = 100.0 - inputs["state_of_charge"]

        internal_resistance = (
            7.94693564e-05 * dod
            - 1.18383130e-06 * dod ** 2.0
            + 5.75440812e-09 * dod ** 3.0
            + 2.96477143e-03
        )

        outputs["internal_resistance"] = internal_resistance

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dod = 100.0 - inputs["state_of_charge"]

        d_ri_d_dod = 7.94693564e-05 - 2.0 * 1.18383130e-06 * dod + 3.0 * 5.75440812e-09 * dod ** 2.0

        partials["internal_resistance", "state_of_charge"] = -np.diag(d_ri_d_dod)
