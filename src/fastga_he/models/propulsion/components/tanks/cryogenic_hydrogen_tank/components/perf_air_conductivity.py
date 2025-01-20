# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import POSSIBLE_POSITION


class PerformancesAirThermalConductivity(om.ExplicitComponent):
    """
    Computation of the air thermal conductivity based on free stream temperature
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="exterior_temperature",
            units="K",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(
            "air_thermal_conductivity",
            units="W/m/K",
            val=np.full(number_of_points, 0.024),
            desc="Tank Nusselt number at each time step",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]

        T = inputs["exterior_temperature"]

        outputs["air_thermal_conductivity"] = (
            1.5207e-11 * T ** 3
            - 4.8574e-08 * T ** 2
            + 1.0184e-04 * T
            - 3.9333e-04 * np.ones(number_of_points)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]

        T = inputs["exterior_temperature"]

        partials["air_thermal_conductivity", "exterior_temperature"] = (
            3 * 1.5207e-11 * T ** 2 - 2 * 4.8574e-08 * T + 1.0184e-04 * np.ones(number_of_points)
        )
