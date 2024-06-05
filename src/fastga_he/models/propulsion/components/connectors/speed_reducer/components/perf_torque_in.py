# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesTorqueIn(om.ExplicitComponent):
    """
    Component which computes the input torque, will be useful for the sizing since the input
    torque is one of the sizing parameters for the dimensions of the gearbox
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("shaft_power_in", units="W", val=np.nan, shape=number_of_points)
        self.add_input("rpm_in", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output("torque_in", units="N*m", val=200.0, shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        power = inputs["shaft_power_in"]
        rpm = inputs["rpm_in"]
        omega = rpm * 2.0 * np.pi / 60

        outputs["torque_in"] = power / omega

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        power = inputs["shaft_power_in"]
        rpm = inputs["rpm_in"]
        omega = rpm * 2.0 * np.pi / 60

        partials["torque_in", "shaft_power_in"] = 1.0 / omega
        partials["torque_in", "rpm_in"] = -power / omega ** 2.0 * 2.0 * np.pi / 60
