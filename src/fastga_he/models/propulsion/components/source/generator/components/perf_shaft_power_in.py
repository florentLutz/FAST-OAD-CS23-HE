# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesShaftPowerIn(om.ExplicitComponent):
    """Computation of the shaft power at the input of the generator."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "active_power", units="W", val=np.full(number_of_points, np.nan), shape=number_of_points
        )
        self.add_input("efficiency", val=np.nan, shape=number_of_points)

        self.add_output("shaft_power_in", units="W", val=500.0e3, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["shaft_power_in"] = inputs["active_power"] / inputs["efficiency"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["shaft_power_in", "active_power"] = np.diag(1.0 / inputs["efficiency"])
        partials["shaft_power_in", "efficiency"] = -np.diag(
            inputs["active_power"] / inputs["efficiency"] ** 2.0
        )
