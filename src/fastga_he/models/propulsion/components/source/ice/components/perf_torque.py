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

        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output("shaft_power_for_power_rate", units="W", val=50e3, shape=number_of_points)
        self.add_output("torque_out", units="N*m", val=500.0, shape=number_of_points, lower=0.0)

        self.declare_partials(of="torque_out", wrt="*", method="exact")
        self.declare_partials(
            of="shaft_power_for_power_rate", wrt="shaft_power_out", method="exact"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        power = inputs["shaft_power_out"]
        rpm = inputs["rpm"]
        omega = rpm * 2.0 * np.pi / 60

        torque = power / omega

        outputs["torque_out"] = torque
        outputs["shaft_power_for_power_rate"] = power

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        power = inputs["shaft_power_out"]
        rpm = inputs["rpm"]

        omega = rpm * 2.0 * np.pi / 60

        partials["shaft_power_for_power_rate", "shaft_power_out"] = np.eye(number_of_points)

        partials["torque_out", "shaft_power_out"] = np.diag(1.0 / omega)
        partials["torque_out", "rpm"] = -np.diag(power / omega ** 2.0) * 2.0 * np.pi / 60
