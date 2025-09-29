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

        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input("active_power", units="W", val=np.nan, shape=number_of_points)

        self.add_output(
            "electromagnetic_torque",
            units="N*m",
            val=0.0,
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
        outputs["electromagnetic_torque"] = 30.0 * inputs["active_power"] * np.pi / inputs["rpm"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["electromagnetic_torque", "active_power"] = 30.0 * np.pi / inputs["rpm"]

        partials["electromagnetic_torque", "rpm"] = (
            -30.0 * inputs["active_power"] * np.pi / inputs["rpm"] ** 2.0
        )
