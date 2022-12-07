# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.
import numpy as np
import openmdao.api as om

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CLIMB,
    POINTS_NB_CRUISE,
)


class InitializeGamma(om.ExplicitComponent):
    """Initializes the climb angle at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:sea_level", val=np.nan, units="m/s"
        )
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:cruise_level", val=np.nan, units="m/s"
        )
        self.add_input("data:mission:sizing:main_route:descent:descent_rate", np.nan, units="m/s")
        self.add_input(
            "altitude", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "true_airspeed",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="m/s",
        )

        self.add_output("vertical_speed", val=np.full(number_of_points, 0.0), units="m/s")
        self.add_output("gamma", val=np.full(number_of_points, 0.0), units="deg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        climb_rate_sl = float(inputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"])
        climb_rate_cl = float(
            inputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"]
        )
        descent_rate = -abs(inputs["data:mission:sizing:main_route:descent:descent_rate"])
        altitude = inputs["altitude"]
        true_airspeed = inputs["true_airspeed"]

        altitude_climb = altitude[0:POINTS_NB_CLIMB]
        altitude_cruise = altitude[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE]
        altitude_descent = altitude[POINTS_NB_CLIMB + POINTS_NB_CRUISE :]

        vertical_speed_climb = np.interp(
            altitude_climb, [0.0, cruise_altitude], [climb_rate_sl, climb_rate_cl]
        )
        vertical_speed_cruise = np.full_like(altitude_cruise, 0.0)
        vertical_speed_descent = np.full_like(altitude_descent, descent_rate)

        vertical_speed = np.concatenate(
            (vertical_speed_climb, vertical_speed_cruise, vertical_speed_descent)
        )

        outputs["vertical_speed"] = vertical_speed

        gamma = np.arcsin(vertical_speed / true_airspeed) * 180.0 / np.pi

        outputs["gamma"] = gamma
