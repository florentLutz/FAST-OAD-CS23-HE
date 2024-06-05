# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesEntropicHeatCoefficient(om.ExplicitComponent):
    """
    Computation of the entropic heat coefficient, used for the estimation losses. Based on
    :cite:`lai:2019`
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("state_of_charge", units="percent", val=np.full(number_of_points, np.nan))

        self.add_output(
            "entropic_heat_coefficient", units="V/degK", val=np.full(number_of_points, 1e-3)
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10 - 1e-3),
            np.full_like(inputs["state_of_charge"], 100 + 1e-3),
        )

        outputs["entropic_heat_coefficient"] = (
            -0.355 + 2.154e-2 * soc - 2.869e-4 * soc ** 2.0 + 1.028e-6 * soc ** 3.0
        ) * 1e-3

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10 - 1e-3),
            np.full_like(inputs["state_of_charge"], 100 + 1e-3),
        )

        partials["entropic_heat_coefficient", "state_of_charge"] = (
            2.154e-2 - 2.0 * 2.869e-4 * soc + 3.0 * 1.028e-6 * soc ** 2.0
        ) * 1e-3
