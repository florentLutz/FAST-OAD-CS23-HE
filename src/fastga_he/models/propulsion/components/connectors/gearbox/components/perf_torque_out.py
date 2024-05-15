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

        for output_number in ("1", "2"):
            self.add_input(
                "shaft_power_out_" + output_number, units="kW", val=np.nan, shape=number_of_points
            )
            self.add_input(
                "rpm_out_" + output_number, units="min**-1", val=np.nan, shape=number_of_points
            )

            self.add_output(
                "torque_out_" + output_number, units="N*m", val=200.0, shape=number_of_points
            )

            self.declare_partials(
                of="torque_out_" + output_number,
                wrt=["shaft_power_out_" + output_number, "rpm_out_" + output_number],
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        for output_number in ("1", "2"):
            power = inputs["shaft_power_out_" + output_number] * 1000.0
            rpm = inputs["rpm_out_" + output_number]
            omega = rpm * 2.0 * np.pi / 60

            outputs["torque_out_" + output_number] = power / omega

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        for output_number in ("1", "2"):
            power = inputs["shaft_power_out_" + output_number]
            rpm = inputs["rpm_out_" + output_number]
            omega = rpm * 2.0 * np.pi / 60

            partials["torque_out_" + output_number, "shaft_power_out_" + output_number] = np.diag(
                1000.0 / omega
            )
            partials["torque_out_" + output_number, "rpm_out_" + output_number] = (
                -np.diag(power * 1000.0 / omega ** 2.0) * 2.0 * np.pi / 60
            )
