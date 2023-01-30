# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesEntropicHeatCoefficient(om.ExplicitComponent):
    """
    Computation of the entropic heat coefficient, used for the estimation losses. Based on a
    regression that can be found in ..methodology.entropic_heat_coefficient.py on data from
    :cite:`geng:2020`
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

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        soc = inputs["state_of_charge"]

        outputs["entropic_heat_coefficient"] = (
            -8.95880680e-09 * soc ** 5.0
            + 2.67279229e-06 * soc ** 4.0
            - 2.94955067e-04 * soc ** 3.0
            + 1.46460304e-02 * soc ** 2.0
            - 3.13899180e-01 * soc
            + 2.18357694e00
        ) * 1e-3

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        soc = inputs["state_of_charge"]

        partials["entropic_heat_coefficient", "state_of_charge"] = (
            np.diag(
                -5.0 * 8.95880680e-09 * soc ** 4.0
                + 4.0 * 2.67279229e-06 * soc ** 3.0
                - 3.0 * 2.94955067e-04 * soc ** 2.0
                + 2.0 * 1.46460304e-02 * soc
                - 3.13899180e-01
            )
            * 1e-3
        )
