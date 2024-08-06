# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesPowerForPowerRate(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("power_required", units="kW", val=np.nan, shape=number_of_points)

        self.add_output("shaft_power_for_power_rate", units="kW", val=500.0, shape=number_of_points)
        self.declare_partials(
            of="shaft_power_for_power_rate",
            wrt="*",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["shaft_power_for_power_rate"] = inputs["power_required"]
