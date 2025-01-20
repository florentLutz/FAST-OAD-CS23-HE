# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from stdatm import AtmosphereWithPartials

DEFAULT_TEMPERATURE = 300.0


class PerformancesExteriorTemperature(om.ExplicitComponent):
    """
    Computation of the free stream temperature of the exterior surface of the tank
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("altitude", units="m", val=np.zeros(number_of_points))

        self.add_output(
            name="exterior_temperature",
            units="K",
            val=np.full(number_of_points, DEFAULT_TEMPERATURE),
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["exterior_temperature"] = AtmosphereWithPartials(inputs["altitude"]).temperature

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["exterior_temperature", "altitude"] = AtmosphereWithPartials(
            inputs["altitude"]
        ).partial_temperature_altitude
