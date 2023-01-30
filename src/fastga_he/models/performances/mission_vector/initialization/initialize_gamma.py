# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import logging

import numpy as np
import openmdao.api as om

_LOGGER = logging.getLogger(__name__)


class InitializeGamma(om.ExplicitComponent):
    """Initializes the climb angle at each time step."""

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
        non_nul_pd = np.concatenate((climb_idx, descent_idx))

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
        self.add_output("gamma", val=np.full(number_of_points, 0.0), units="rad")

        # We can't really define the pd for this component because of the use of the Atmosphere
        # class, however, since cruise and reserve are constant speed phase, we can use sparse
        # derivative
        self.declare_partials(
            of=["vertical_speed", "gamma"],
            wrt=["altitude"],
            method="fd",
            rows=non_nul_pd,
            cols=non_nul_pd,
        )

        self.declare_partials(
            of="vertical_speed",
            wrt=[
                "data:mission:sizing:main_route:descent:descent_rate",
                "data:mission:sizing:main_route:climb:climb_rate:cruise_level",
                "data:mission:sizing:main_route:climb:climb_rate:sea_level",
                "data:mission:sizing:main_route:cruise:altitude",
            ],
            method="exact",
        )
        self.declare_partials(
            of="gamma",
            wrt=[
                "true_airspeed",
                "data:mission:sizing:main_route:descent:descent_rate",
                "data:mission:sizing:main_route:climb:climb_rate:cruise_level",
                "data:mission:sizing:main_route:climb:climb_rate:sea_level",
                "data:mission:sizing:main_route:cruise:altitude",
            ],
            method="exact",
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
        true_airspeed = inputs["true_airspeed"]

        altitude_climb = altitude[:number_of_points_climb]
        altitude_cruise = altitude[
            number_of_points_climb : number_of_points_climb + number_of_points_cruise
        ]
        altitude_descent = altitude[
            number_of_points_climb
            + number_of_points_cruise : number_of_points_climb
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

        gamma = np.arcsin(vertical_speed / true_airspeed)

        outputs["gamma"] = gamma

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        climb_rate_sl = inputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"]
        climb_rate_cl = inputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"]

        descent_rate = inputs["data:mission:sizing:main_route:descent:descent_rate"]

        altitude = inputs["altitude"]
        true_airspeed = inputs["true_airspeed"]

        altitude_climb = altitude[:number_of_points_climb]

        vertical_speed_climb = climb_rate_sl + (climb_rate_cl - climb_rate_sl) / cruise_altitude * (
            altitude_climb
        )

        vertical_speed = np.concatenate(
            (
                vertical_speed_climb,
                np.zeros(number_of_points_cruise),
                np.full(number_of_points_descent, descent_rate),
                np.zeros(number_of_points_reserve),
            )
        )

        d_vz_climb_d_cruise_altitude = (
            -(climb_rate_cl - climb_rate_sl) / cruise_altitude ** 2.0 * altitude_climb
        )
        d_vz_climb_d_climb_rate_sl = 1.0 - cruise_altitude / altitude_climb
        d_vz_climb_d_climb_rate_cl = cruise_altitude / altitude_climb

        d_vz_d_cruise_altitude = np.concatenate(
            (
                d_vz_climb_d_cruise_altitude,
                np.zeros(
                    number_of_points_cruise + number_of_points_descent + number_of_points_reserve
                ),
            )
        )
        d_vz_d_climb_rate_sl = np.concatenate(
            (
                d_vz_climb_d_climb_rate_sl,
                np.zeros(
                    number_of_points_cruise + number_of_points_descent + number_of_points_reserve
                ),
            )
        )
        d_vz_d_climb_rate_cl = np.concatenate(
            (
                d_vz_climb_d_climb_rate_cl,
                np.zeros(
                    number_of_points_cruise + number_of_points_descent + number_of_points_reserve
                ),
            )
        )
        d_vz_d_descent_rate = np.concatenate(
            (
                np.zeros(number_of_points_climb + number_of_points_cruise),
                np.ones(number_of_points_descent),
                np.zeros(number_of_points_reserve),
            )
        )

        partials[
            "vertical_speed", "data:mission:sizing:main_route:cruise:altitude"
        ] = d_vz_d_cruise_altitude
        partials[
            "vertical_speed", "data:mission:sizing:main_route:climb:climb_rate:sea_level"
        ] = d_vz_d_climb_rate_sl
        partials[
            "vertical_speed", "data:mission:sizing:main_route:climb:climb_rate:cruise_level"
        ] = d_vz_d_climb_rate_cl
        partials[
            "vertical_speed", "data:mission:sizing:main_route:descent:descent_rate"
        ] = d_vz_d_descent_rate

        partials["gamma", "data:mission:sizing:main_route:cruise:altitude"] = (
            (1.0 - (vertical_speed / true_airspeed) ** 2.0) ** (-1.0 / 2.0)
            * d_vz_d_cruise_altitude
            / true_airspeed
        )
        partials["gamma", "data:mission:sizing:main_route:climb:climb_rate:sea_level"] = (
            (1.0 - (vertical_speed / true_airspeed) ** 2.0) ** (-1.0 / 2.0)
            * d_vz_d_climb_rate_sl
            / true_airspeed
        )
        partials["gamma", "data:mission:sizing:main_route:climb:climb_rate:cruise_level"] = (
            (1.0 - (vertical_speed / true_airspeed) ** 2.0) ** (-1.0 / 2.0)
            * d_vz_d_climb_rate_cl
            / true_airspeed
        )
        partials["gamma", "data:mission:sizing:main_route:descent:descent_rate"] = (
            (1.0 - (vertical_speed / true_airspeed) ** 2.0) ** (-1.0 / 2.0)
            * d_vz_d_descent_rate
            / true_airspeed
        )
        partials["gamma", "true_airspeed"] = -np.diag(
            (1.0 - (vertical_speed / true_airspeed) ** 2.0) ** (-1.0 / 2.0)
            * vertical_speed
            / true_airspeed ** 2.0
        )
