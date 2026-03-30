# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from stdatm import AtmosphereWithPartials


class PerformancesAirSpeedOfSound(om.ExplicitComponent):
    """
    Compute the speed of sound in air.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("altitude", units="m", val=np.zeros(number_of_points))

        self.add_output(name="speed_of_sound", units="m/s", val=340.0, shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            "*",
            "*",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["speed_of_sound"] = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).speed_of_sound

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["speed_of_sound", "altitude"] = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).partial_speed_of_sound_altitude
