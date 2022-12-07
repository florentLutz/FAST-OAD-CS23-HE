# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CLIMB,
    POINTS_NB_CRUISE,
    POINTS_NB_DESCENT,
)


class InitializeAltitude(om.ExplicitComponent):
    """Intializes the altitude at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        self.add_output("altitude", shape=number_of_points, units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]

        altitude_climb = np.linspace(0, cruise_altitude, POINTS_NB_CLIMB)[:, 0]
        altitude_cruise = np.full(POINTS_NB_CRUISE, cruise_altitude)
        altitude_descent = np.linspace(cruise_altitude, 0.0, POINTS_NB_DESCENT)[:, 0]

        altitude = np.concatenate((altitude_climb, altitude_cruise, altitude_descent))

        outputs["altitude"] = altitude
