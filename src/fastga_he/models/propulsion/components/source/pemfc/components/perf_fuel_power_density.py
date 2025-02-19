# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPEMFCStackHydrogenPowerDensity(om.ExplicitComponent):
    """
    Computation of the hydrogen power density of PEMFC, which only considered in post-processing.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(name="fuel_consumed_t", units="kg", val=np.full(number_of_points, np.nan))

        self.add_input("power_out", units="kW", val=np.full(number_of_points, np.nan))

        self.add_output("specific_power", units="kW/kg", val=np.full(number_of_points, 5.0))

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["specific_power"] = inputs["power_out"] / inputs["fuel_consumed_t"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["specific_power", "power_out"] = 1 / inputs["fuel_consumed_t"]
        partials["specific_power", "fuel_consumed_t"] = (
            -inputs["power_out"] / inputs["fuel_consumed_t"] ** 2
        )
