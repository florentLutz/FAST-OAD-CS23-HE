# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesICEFuelConsumption(om.ExplicitComponent):
    """
    Computation of the ICE fuel consumption for the required power and rpm. Simply based on the
    results of the sfc consumption
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "specific_fuel_consumption", units="g/W/h", val=np.nan, shape=number_of_points
        )
        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)

        self.add_output("fuel_consumption", units="kg/h", val=30.0, shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["fuel_consumption"] = (
            inputs["shaft_power_out"] * inputs["specific_fuel_consumption"]
        ) / 1000.0  # To convert to kg/h

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["fuel_consumption", "shaft_power_out"] = (
            inputs["specific_fuel_consumption"] / 1000.0
        )
        partials["fuel_consumption", "specific_fuel_consumption"] = (
            inputs["shaft_power_out"] / 1000.0
        )
