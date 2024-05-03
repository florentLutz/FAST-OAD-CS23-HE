# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesTorqueOut(om.ExplicitComponent):
    """
    Component which computes the output torque, will be useful for the sizing since the output
    torque is one of the sizing parameters for the mass of the gearbox
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("shaft_power_out", units="kW", val=np.nan, shape=number_of_points)
        self.add_input("rpm_out", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output("torque_out", units="N*m", val=200.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        power = inputs["shaft_power_out"] * 1000.0
        rpm = inputs["rpm_out"]
        omega = rpm * 2.0 * np.pi / 60

        outputs["torque_out"] = power / omega

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        power = inputs["shaft_power_out"]
        rpm = inputs["rpm_out"]
        omega = rpm * 2.0 * np.pi / 60

        partials["torque_out", "shaft_power_out"] = np.diag(1000.0 / omega)
        partials["torque_out", "rpm_out"] = (
            -np.diag(power * 1000.0 / omega ** 2.0) * 2.0 * np.pi / 60
        )
