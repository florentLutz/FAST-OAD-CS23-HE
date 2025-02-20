# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials
from ..constants import DEFAULT_PRESSURE


class PerformancesPEMFCStackAmbientPressure(om.ExplicitComponent):
    """
    Computation of the ambient pressure that PEMFC is working under. The ambient pressure varies
    based on the change of operating altitude of PEMFC.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("altitude", units="m", val=np.zeros(number_of_points))

        self.add_output(
            name="ambient_pressure",
            units="Pa",
            val=np.full(number_of_points, DEFAULT_PRESSURE),
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["ambient_pressure"] = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).pressure

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["ambient_pressure", "altitude"] = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).partial_pressure_altitude
