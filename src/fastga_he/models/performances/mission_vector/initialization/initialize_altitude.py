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

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:reserve:altitude", val=np.nan, units="m")

        self.add_output("altitude", shape=number_of_points, units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        reserve_altitude = inputs["data:mission:sizing:main_route:reserve:altitude"]

        altitude_climb = np.linspace(0, cruise_altitude, number_of_points_climb)[:, 0]
        altitude_cruise = np.full(number_of_points_cruise, cruise_altitude)
        altitude_descent = np.linspace(cruise_altitude, 0.0, number_of_points_descent)[:, 0]
        altitude_reserve = np.full(number_of_points_reserve, reserve_altitude)

        altitude = np.concatenate(
            (altitude_climb, altitude_cruise, altitude_descent, altitude_reserve)
        )

        outputs["altitude"] = altitude

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        flat_partials_cruise_alt = np.concatenate(
            (
                np.arange(number_of_points_climb) / (number_of_points_climb - 1),
                np.ones(number_of_points_cruise),
                np.flip(np.arange(number_of_points_descent)) / (number_of_points_descent - 1),
                np.zeros(number_of_points_reserve),
            )
        )
        flat_partials_reserve_alt = np.concatenate(
            (
                np.zeros(number_of_points_climb),
                np.zeros(number_of_points_cruise),
                np.zeros(number_of_points_descent),
                np.ones(number_of_points_reserve),
            )
        )

        partials[
            "altitude", "data:mission:sizing:main_route:cruise:altitude"
        ] = flat_partials_cruise_alt
        partials[
            "altitude", "data:mission:sizing:main_route:reserve:altitude"
        ] = flat_partials_reserve_alt
