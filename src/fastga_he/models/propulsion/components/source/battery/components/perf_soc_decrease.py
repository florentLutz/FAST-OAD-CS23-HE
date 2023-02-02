# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesSOCDecrease(om.ExplicitComponent):
    """
    Computation of the decrease of the state of charge of the battery caused by the required
    C-rate. :cite:`vratny:2013` seems to suggest that a high c-rate causes an apparent c-rate
    lower than the nominal one. We won't reproduce this effect here for now.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("c_rate", units="h**-1", val=np.full(number_of_points, np.nan))
        self.add_input("time_step", units="h", val=np.full(number_of_points, np.nan))
        self.add_input("relative_capacity", val=np.full(number_of_points, np.nan))

        self.add_output(
            "state_of_charge_decrease", units="percent", val=np.full(number_of_points, 1.0)
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["state_of_charge_decrease"] = (
            inputs["c_rate"] * inputs["time_step"] * 100.0 / inputs["relative_capacity"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["state_of_charge_decrease", "c_rate"] = 100.0 * np.diag(
            inputs["time_step"] / inputs["relative_capacity"]
        )
        partials["state_of_charge_decrease", "time_step"] = 100.0 * np.diag(
            inputs["c_rate"] / inputs["relative_capacity"]
        )
        partials["state_of_charge_decrease", "relative_capacity"] = -100.0 * np.diag(
            inputs["c_rate"] * inputs["time_step"] / inputs["relative_capacity"] ** 2.0
        )
