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

        for input_number in ("1", "2"):
            self.add_input(
                "shaft_power_in_" + input_number, units="W", val=np.nan, shape=number_of_points
            )
            self.add_input(
                "rpm_in_" + input_number, units="min**-1", val=np.nan, shape=number_of_points
            )

            self.add_output(
                "torque_in_" + input_number, units="N*m", val=200.0, shape=number_of_points
            )

            self.declare_partials(
                of="torque_in_" + input_number,
                wrt=["rpm_in_" + input_number, "shaft_power_in_" + input_number],
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        for input_number in ("1", "2"):
            power = inputs["shaft_power_in_" + input_number]
            rpm = inputs["rpm_in_" + input_number]
            omega = rpm * 2.0 * np.pi / 60

            outputs["torque_in_" + input_number] = power / omega

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        for input_number in ("1", "2"):
            power = inputs["shaft_power_in_" + input_number]
            rpm = inputs["rpm_in_" + input_number]
            omega = rpm * 2.0 * np.pi / 60

            partials["torque_in_" + input_number, "shaft_power_in_" + input_number] = np.diag(
                1.0 / omega
            )
            partials["torque_in_" + input_number, "rpm_in_" + input_number] = (
                -np.diag(power / omega ** 2.0) * 2.0 * np.pi / 60
            )
