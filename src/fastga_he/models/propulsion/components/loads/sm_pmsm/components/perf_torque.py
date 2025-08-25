# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

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
        self.add_output("torque_out", units="N*m", val=0.0, shape=number_of_points)
        # Creating a duplicate of shaft_power_out (with a new name) to provide the required
        # "power rate" input, since direct connection to the propulsive load would cause conflicts.

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="torque_out",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="shaft_power_for_power_rate",
            wrt="shaft_power_out",
            method="exact",
            val=np.ones(number_of_points),
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        power = inputs["shaft_power_out"]
        rpm = inputs["rpm"]

        omega = rpm * 2.0 * np.pi / 60
        torque = power / omega

        outputs["torque_out"] = torque
        outputs["shaft_power_for_power_rate"] = power

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        power = inputs["shaft_power_out"]
        rpm = inputs["rpm"]

        omega = rpm * 2.0 * np.pi / 60

        partials["torque_out", "shaft_power_out"] = 1.0 / omega

        partials["torque_out", "rpm"] = -power / omega**2.0 * 2.0 * np.pi / 60
