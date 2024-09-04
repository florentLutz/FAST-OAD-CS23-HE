# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import logging

import openmdao.api as om
import numpy as np

_LOGGER = logging.getLogger(__name__)


class InitializeVerticalAirspeed(om.ExplicitComponent):
    """Initializes the vertical airspeed from mission requirement."""

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

        self.add_output("vertical_speed", val=np.full(number_of_points, 0.0), units="m/s")

        climb_idx = np.linspace(
            0,
            number_of_points_climb - 1,
            number_of_points_climb,
        ).astype(int)
        descent_idx = np.linspace(
            number_of_points_climb + number_of_points_cruise,
            number_of_points_climb + number_of_points_cruise + number_of_points_descent - 1,
            number_of_points_descent,
        ).astype(int)

        self.declare_partials(
            of="vertical_speed",
            wrt=[
                "altitude",
            ],
            method="exact",
            rows=climb_idx,
            cols=climb_idx,
        )
        self.declare_partials(
            of="vertical_speed",
            wrt=[
                "data:mission:sizing:main_route:climb:climb_rate:cruise_level",
                "data:mission:sizing:main_route:climb:climb_rate:sea_level",
                "data:mission:sizing:main_route:cruise:altitude",
            ],
            rows=climb_idx,
            cols=np.zeros_like(climb_idx),
        )

        self.declare_partials(
            of="vertical_speed",
            wrt="data:mission:sizing:main_route:descent:descent_rate",
            val=1.0,
            rows=descent_idx,
            cols=np.zeros_like(descent_idx),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        climb_rate_sl = inputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"]
        climb_rate_cl = inputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"]
        descent_rate = inputs["data:mission:sizing:main_route:descent:descent_rate"]

        if descent_rate > 0.0:
            _LOGGER.warning("Descent rate greater than 0.0, please check the inputs")

        altitude = inputs["altitude"]

        altitude_climb = altitude[:number_of_points_climb]
        altitude_cruise = altitude[
            number_of_points_climb : number_of_points_climb + number_of_points_cruise
        ]
        altitude_descent = altitude[
            number_of_points_climb + number_of_points_cruise : number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
        ]
        altitude_reserve = altitude[
            number_of_points_climb + number_of_points_cruise + number_of_points_descent :
        ]

        vertical_speed_climb = climb_rate_sl + (climb_rate_cl - climb_rate_sl) / cruise_altitude * (
            altitude_climb
        )

        vertical_speed = np.concatenate(
            (
                vertical_speed_climb,
                np.zeros_like(altitude_cruise),
                np.full_like(altitude_descent, descent_rate),
                np.zeros_like(altitude_reserve),
            )
        )

        outputs["vertical_speed"] = vertical_speed

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points_climb = self.options["number_of_points_climb"]

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        climb_rate_sl = inputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"]
        climb_rate_cl = inputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"]

        altitude = inputs["altitude"]

        altitude_climb = altitude[:number_of_points_climb]

        d_vz_climb_d_cruise_altitude = (
            -(climb_rate_cl - climb_rate_sl) / cruise_altitude**2.0 * altitude_climb
        )
        d_vz_climb_d_climb_rate_sl = 1.0 - altitude_climb / cruise_altitude
        d_vz_climb_d_climb_rate_cl = altitude_climb / cruise_altitude

        partials["vertical_speed", "data:mission:sizing:main_route:cruise:altitude"] = np.full_like(
            altitude_climb, d_vz_climb_d_cruise_altitude
        )
        partials["vertical_speed", "data:mission:sizing:main_route:climb:climb_rate:sea_level"] = (
            np.full_like(altitude_climb, d_vz_climb_d_climb_rate_sl)
        )
        partials[
            "vertical_speed", "data:mission:sizing:main_route:climb:climb_rate:cruise_level"
        ] = np.full_like(altitude_climb, d_vz_climb_d_climb_rate_cl)
        partials["vertical_speed", "altitude"] = np.full_like(
            altitude_climb, (climb_rate_cl - climb_rate_sl) / cruise_altitude
        )
