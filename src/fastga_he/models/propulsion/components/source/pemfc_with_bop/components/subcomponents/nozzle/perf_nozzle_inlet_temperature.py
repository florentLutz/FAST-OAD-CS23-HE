# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesNozzleInletTemperature(om.ExplicitComponent):
    """
    Computation of the exit air temperature from the heat exchanger.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="heat_exchanger_exit_temperature",
            units="K",
            val=np.nan,
        )
        self.add_input(
            "exterior_temperature",
            val=np.nan,
            units="K",
            shape=number_of_points,
        )

        self.add_output("nozzle_inlet_temperature", val=320.0, units="K", shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="exterior_temperature",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=0.5,
        )
        self.declare_partials(
            of="*",
            wrt="heat_exchanger_exit_temperature",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
            val=0.5,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["nozzle_inlet_temperature"] = 0.5 * (
            inputs["heat_exchanger_exit_temperature"] + inputs["exterior_temperature"]
        )
