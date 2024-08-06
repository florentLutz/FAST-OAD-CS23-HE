# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTemperatureIncrease(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "cable_temperature_time_derivative",
            val=np.full(number_of_points, np.nan),
            units="degK/s",
            desc="temperature derivative inside of the cable",
            shape=number_of_points,
        )
        self.add_input("time_step", shape=number_of_points, units="s", val=np.nan)

        self.add_output(
            "cable_temperature_increase",
            val=np.full(number_of_points, 0.0),
            units="degK",
            desc="temperature increase inside of the cable",
            shape=number_of_points,
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cable_temperature_increase = (
            inputs["cable_temperature_time_derivative"] * inputs["time_step"]
        )

        outputs["cable_temperature_increase"] = cable_temperature_increase

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["cable_temperature_increase", "cable_temperature_time_derivative"] = inputs[
            "time_step"
        ]
        partials["cable_temperature_increase", "time_step"] = inputs[
            "cable_temperature_time_derivative"
        ]
