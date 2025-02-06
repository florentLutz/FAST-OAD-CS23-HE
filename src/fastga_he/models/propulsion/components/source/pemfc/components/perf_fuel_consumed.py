# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesPEMFCFuelConsumed(om.ExplicitComponent):
    """
    Computation of the hydrogen at each flight point for the required power. Simply based on the
    results of the hydrogen consumption and time
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(name="fuel_consumption", units="kg/h", val=np.full(number_of_points, np.nan))
        self.add_input("time_step", units="h", val=np.full(number_of_points, np.nan))

        self.add_output(name="fuel_consumed_t", val=np.full(number_of_points, 2.0), units="kg")

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        clipped_fuel_consumption = np.clip(inputs["fuel_consumption"], 1e-6, 1e6)
        outputs["fuel_consumed_t"] = inputs["time_step"] * clipped_fuel_consumption

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        clipped_fuel_consumption = np.clip(inputs["fuel_consumption"], 1e-6, 1e6)

        partials["fuel_consumed_t", "time_step"] = clipped_fuel_consumption
        partials["fuel_consumed_t", "fuel_consumption"] = np.where(
            inputs["fuel_consumption"] == clipped_fuel_consumption,
            inputs["time_step"],
            1e-6,
        )
