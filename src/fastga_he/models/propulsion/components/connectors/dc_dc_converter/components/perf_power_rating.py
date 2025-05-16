# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPowerRating(om.ExplicitComponent):
    """
    Component of the power rating of the DC-DC converter.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))
        self.add_input("dc_voltage_out", units="V", val=np.full(number_of_points, np.nan))

        self.add_output("power_rating", units="W", val=200.0, shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["power_rating"] = inputs["dc_current_out"] * inputs["dc_voltage_out"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["power_rating", "dc_current_out"] = inputs["dc_voltage_out"]
        partials["power_rating", "dc_voltage_out"] = inputs["dc_current_out"]
