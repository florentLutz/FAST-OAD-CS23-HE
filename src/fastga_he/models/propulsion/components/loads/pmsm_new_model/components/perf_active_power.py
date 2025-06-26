# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesActivePower(om.ExplicitComponent):
    """Computation of the electric active power required to run the motor."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)
        self.add_input("efficiency", val=np.nan, shape=number_of_points)

        self.add_output(
            "active_power", units="W", val=np.full(number_of_points, 50e3), shape=number_of_points
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["active_power"] = inputs["shaft_power_out"] / inputs["efficiency"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["active_power", "shaft_power_out"] = 1.0 / inputs["efficiency"]
        partials["active_power", "efficiency"] = -(
            inputs["shaft_power_out"] / inputs["efficiency"] ** 2.0
        )
