# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesNozzleDrag(om.ExplicitComponent):
    """
    Computation of the nozzle drag.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "exit_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            "true_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            name="air_mass_flow_rate",
            val=np.nan,
            units="kg/s",
            shape=number_of_points,
        )

        self.add_output("drag", val=0.3, units="N", shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]

        outputs["drag"] = inputs["air_mass_flow_rate"] * (
            inputs["exit_air_speed"] - inputs["true_air_speed"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]

        partials["drag", "air_mass_flow_rate"] = inputs["exit_air_speed"] - inputs["true_air_speed"]
        partials["drag", "exit_air_speed"] = inputs["air_mass_flow_rate"]
        partials["drag", "true_air_speed"] = -inputs["air_mass_flow_rate"]
