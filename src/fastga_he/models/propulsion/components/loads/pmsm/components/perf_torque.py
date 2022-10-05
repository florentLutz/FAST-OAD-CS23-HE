# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTorque(om.ExplicitComponent):
    """Computation of the torque from power and rpm."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("power", units="W", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output("torque", units="N*m", val=0.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        power = inputs["power"]
        rpm = inputs["rpm"]

        torque = power / rpm

        outputs["torque"] = torque

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        power = inputs["power"]
        rpm = inputs["rpm"]

        partials["torque", "power"] = np.diag(1.0 / rpm)
        partials["torque", "rpm"] = -np.diag(power / rpm ** 2.0)
