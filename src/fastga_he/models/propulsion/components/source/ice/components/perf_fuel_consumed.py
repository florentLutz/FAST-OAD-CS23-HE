# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesICEFuelConsumed(om.ExplicitComponent):
    """
    Computation of the ICE at each flight point for the required power and rpm. Simply based on the
    results of the fuel consumption and time
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("fuel_consumption", units="kg/h", val=np.nan, shape=number_of_points)
        self.add_input("time_step", units="h", val=np.full(number_of_points, np.nan))

        self.add_output("fuel_consumed_t", np.full(number_of_points, 1.0), units="kg")

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["fuel_consumed_t"] = inputs["time_step"] * inputs["fuel_consumption"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["fuel_consumed_t", "time_step"] = inputs["fuel_consumption"]
        partials["fuel_consumed_t", "fuel_consumption"] = inputs["time_step"]
