# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesSFC(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("power_required", units="kW", val=np.nan, shape=number_of_points)
        self.add_input("fuel_consumption", units="g/h", val=np.nan, shape=number_of_points)

        self.add_output(
            "specific_fuel_consumption", units="g/kW/h", val=200.0, shape=number_of_points
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["specific_fuel_consumption"] = inputs["fuel_consumption"] / inputs["power_required"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["specific_fuel_consumption", "fuel_consumption"] = 1.0 / inputs["power_required"]
        partials["specific_fuel_consumption", "power_required"] = -(
            inputs["fuel_consumption"] / inputs["power_required"] ** 2.0
        )
