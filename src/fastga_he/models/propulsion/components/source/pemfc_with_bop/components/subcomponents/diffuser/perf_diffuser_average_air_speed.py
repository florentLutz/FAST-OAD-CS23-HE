# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDiffuserAverageAirSpeed(om.ExplicitComponent):
    """
    Computation of the average air speed in the diffuser.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "throat_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            "exit_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )

        self.add_output("average_air_speed", val=0.3, units="m/s")

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=0.5,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["average_air_speed"] = 0.5 * (inputs["throat_air_speed"] + inputs["exit_air_speed"])
