# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials


class InitializeTemperature(om.ExplicitComponent):
    """Initializes the temperature at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points_climb", default=1, desc="number of equilibrium to be treated in climb"
        )
        self.options.declare(
            "number_of_points_cruise",
            default=1,
            desc="number of equilibrium to be treated in cruise",
        )
        self.options.declare(
            "number_of_points_descent",
            default=1,
            desc="number of equilibrium to be treated in descent",
        )
        self.options.declare(
            "number_of_points_reserve",
            default=1,
            desc="number of equilibrium to be treated in reserve",
        )

    def setup(self):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        self.add_input("altitude", shape=number_of_points, units="m", val=np.nan)

        self.add_output("exterior_temperature", shape=number_of_points, units="degK", val=297.15)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["exterior_temperature"] = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).temperature

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["exterior_temperature", "altitude"] = np.diag(
            AtmosphereWithPartials(
                inputs["altitude"], altitude_in_feet=False
            ).partial_temperature_altitude
        )
