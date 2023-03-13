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

        self.add_input("torque_in", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output("shaft_power_in", units="W", val=np.nan, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        omega = inputs["rpm"] * 2.0 * np.pi / 60

        outputs["shaft_power_in"] = inputs["torque_in"] * omega

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["shaft_power_in", "torque_in"] = np.diag(inputs["rpm"] * 2.0 * np.pi / 60)
        partials["shaft_power_in", "rpm"] = np.diag(inputs["torque_in"] * 2.0 * np.pi / 60)
