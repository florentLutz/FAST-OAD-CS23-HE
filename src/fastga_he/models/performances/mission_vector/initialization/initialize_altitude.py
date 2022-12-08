# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class InitializeAltitude(om.ExplicitComponent):
    """Initializes the altitude at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points_climb", default=1, desc="number of equilibrium to be treated in climb"
        )
        self.options.declare(
            "number_of_points_cruise",
            default=1,
            desc="number of equilibrium to be treated in " "cruise",
        )
        self.options.declare(
            "number_of_points_descent",
            default=1,
            desc="number of equilibrium to be treated in descent",
        )

    def setup(self):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]

        number_of_points = (
            number_of_points_climb + number_of_points_cruise + number_of_points_descent
        )

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        self.add_output("altitude", shape=number_of_points, units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]

        altitude_climb = np.linspace(0, cruise_altitude, number_of_points_climb)[:, 0]
        altitude_cruise = np.full(number_of_points_cruise, cruise_altitude)
        altitude_descent = np.linspace(cruise_altitude, 0.0, number_of_points_descent)[:, 0]

        altitude = np.concatenate((altitude_climb, altitude_cruise, altitude_descent))

        outputs["altitude"] = altitude
