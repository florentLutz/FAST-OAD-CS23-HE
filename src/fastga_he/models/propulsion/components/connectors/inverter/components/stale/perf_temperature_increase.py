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
            "inverter_temperature_time_derivative",
            val=np.full(number_of_points, np.nan),
            units="degK/s",
            desc="temperature derivative inside of the inverter module",
            shape=number_of_points,
        )
        self.add_input("time_step", shape=number_of_points, units="s", val=np.nan)

        self.add_output(
            "inverter_temperature_increase",
            val=np.full(number_of_points, 0.0),
            units="degK",
            desc="temperature increase inside of the inverter",
            shape=number_of_points,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["inverter_temperature_increase"] = (
            inputs["inverter_temperature_time_derivative"] * inputs["time_step"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["inverter_temperature_increase", "inverter_temperature_time_derivative"] = np.diag(
            inputs["time_step"]
        )
        partials["inverter_temperature_increase", "time_step"] = np.diag(
            inputs["inverter_temperature_time_derivative"]
        )
