# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesTorque(om.ExplicitComponent):
    """Computation of the torque required on the shaft of the propeller."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("rpm", units="min**-1", val=np.full(number_of_points, np.nan))
        self.add_input("shaft_power_in", units="W", val=np.full(number_of_points, np.nan))

        self.add_output("torque_in", units="N*m", val=np.full(number_of_points, 800.0))

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        omega = inputs["rpm"] * 2.0 * np.pi / 60

        outputs["torque_in"] = inputs["shaft_power_in"] / omega

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        power = inputs["shaft_power_in"]
        rpm = inputs["rpm"]

        omega = rpm * 2.0 * np.pi / 60

        partials["torque_in", "shaft_power_in"] = 1.0 / omega
        partials["torque_in", "rpm"] = -power / omega**2.0 * 2.0 * np.pi / 60.0
