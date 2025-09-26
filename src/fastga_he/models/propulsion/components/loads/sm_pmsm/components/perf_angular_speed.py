# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAngularSpeed(om.ExplicitComponent):
    """
    Computation of the conversion between RPM and rads/s.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output(
            "angular_speed",
            units="rad/s",
            val=0.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="rpm",
            wrt="angular_speed",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=2.0 * np.pi / 60.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["angular_speed"] = 2.0 * np.pi * inputs["rpm"] / 60.0
