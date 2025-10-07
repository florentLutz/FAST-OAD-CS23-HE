# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesElectromagneticTorque(om.ExplicitComponent):
    """
    Computation of the electromagnetic torque of the SM PMSM.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input("mechanical_power_losses", units="W", val=np.nan, shape=number_of_points)

        self.add_output(
            "electromagnetic_torque",
            units="N*m",
            val=200.0,
            shape=number_of_points,
            desc="Total electromechanical torque from the motor",
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="electromagnetic_torque",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rpm = inputs["rpm"]
        power = inputs["shaft_power_out"] + inputs["mechanical_power_losses"]
        omega = rpm * 2.0 * np.pi / 60

        outputs["electromagnetic_torque"] = power / omega

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rpm = inputs["rpm"]
        power = inputs["shaft_power_out"] + inputs["mechanical_power_losses"]
        omega = rpm * 2.0 * np.pi / 60

        partials["electromagnetic_torque", "shaft_power_out"] = 1.0 / omega

        partials["electromagnetic_torque", "mechanical_power_losses"] = 1.0 / omega

        partials["electromagnetic_torque", "rpm"] = -power / omega**2.0 * 2.0 * np.pi / 60
