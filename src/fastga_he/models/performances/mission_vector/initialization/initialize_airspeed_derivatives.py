# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om
from stdatm import Atmosphere


class InitializeAirspeedDerivatives(om.ExplicitComponent):
    """Computes the d_vx_dt at each time step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Contains the index of the different flight phases
        self.climb_idx = None
        self.descent_idx = None

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

        self.climb_idx = np.linspace(
            0,
            number_of_points_climb - 1,
            number_of_points_climb,
        ).astype(int)
        self.descent_idx = np.linspace(
            number_of_points_climb + number_of_points_cruise,
            number_of_points_climb + number_of_points_cruise + number_of_points_descent - 1,
            number_of_points_descent,
        ).astype(int)
        non_nul_pd = np.concatenate((self.climb_idx, self.descent_idx))

        self.add_input(
            "true_airspeed",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "equivalent_airspeed",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "altitude", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "gamma", shape=number_of_points, val=np.full(number_of_points, np.nan), units="rad"
        )

        self.add_output(
            "d_vx_dt", shape=number_of_points, val=np.full(number_of_points, 0.0), units="m/s**2"
        )

        # We can't really define the pd for this component because of the use of the Atmosphere
        # class, however, since cruise and reserve are constant speed phase, we can use sparse
        # derivative
        self.declare_partials(
            of="d_vx_dt",
            wrt=["altitude", "equivalent_airspeed", "true_airspeed"],
            method="fd",
            rows=non_nul_pd,
            cols=non_nul_pd,
        )
        self.declare_partials(
            of="d_vx_dt",
            wrt="gamma",
            method="exact",
            rows=non_nul_pd,
            cols=non_nul_pd,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        true_airspeed = inputs["true_airspeed"]
        equivalent_airspeed = inputs["equivalent_airspeed"]
        altitude = inputs["altitude"]
        gamma = inputs["gamma"]

        altitude_climb = altitude[self.climb_idx]
        gamma_climb = gamma[self.climb_idx]
        equivalent_airspeed_climb = equivalent_airspeed[self.climb_idx]
        true_airspeed_climb = true_airspeed[self.climb_idx]

        altitude_descent = altitude[self.descent_idx]
        gamma_descent = gamma[self.descent_idx]
        equivalent_airspeed_descent = equivalent_airspeed[self.descent_idx]
        true_airspeed_descent = true_airspeed[self.descent_idx]

        atm_climb_plus_1 = Atmosphere(altitude_climb + 1.0, altitude_in_feet=False)
        atm_climb_plus_1.equivalent_airspeed = equivalent_airspeed_climb
        d_v_tas_dh_climb = atm_climb_plus_1.true_airspeed - true_airspeed_climb
        d_vx_dt_climb = d_v_tas_dh_climb * true_airspeed_climb * np.sin(gamma_climb)

        atm_descent_plus_1 = Atmosphere(altitude_descent + 1.0, altitude_in_feet=False)
        atm_descent_plus_1.equivalent_airspeed = equivalent_airspeed_descent
        d_v_tas_dh_descent = atm_descent_plus_1.true_airspeed - true_airspeed_descent
        d_vx_dt_descent = d_v_tas_dh_descent * true_airspeed_descent * np.sin(gamma_descent)

        d_vx_dt = np.concatenate(
            (
                d_vx_dt_climb,
                np.zeros(number_of_points_cruise),
                d_vx_dt_descent,
                np.zeros(number_of_points_reserve),
            )
        )

        outputs["d_vx_dt"] = d_vx_dt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        true_airspeed = inputs["true_airspeed"]
        equivalent_airspeed = inputs["equivalent_airspeed"]
        altitude = inputs["altitude"]
        gamma = inputs["gamma"]

        altitude_climb = altitude[self.climb_idx]
        gamma_climb = gamma[self.climb_idx]
        equivalent_airspeed_climb = equivalent_airspeed[self.climb_idx]
        true_airspeed_climb = true_airspeed[self.climb_idx]

        altitude_descent = altitude[self.descent_idx]
        gamma_descent = gamma[self.descent_idx]
        equivalent_airspeed_descent = equivalent_airspeed[self.descent_idx]
        true_airspeed_descent = true_airspeed[self.descent_idx]

        atm_climb_plus_1 = Atmosphere(altitude_climb + 1.0, altitude_in_feet=False)
        atm_climb_plus_1.equivalent_airspeed = equivalent_airspeed_climb
        d_v_tas_dh_climb = atm_climb_plus_1.true_airspeed - true_airspeed_climb
        partials_wrt_gamma_climb = d_v_tas_dh_climb * true_airspeed_climb * np.cos(gamma_climb)

        atm_descent_plus_1 = Atmosphere(altitude_descent + 1.0, altitude_in_feet=False)
        atm_descent_plus_1.equivalent_airspeed = equivalent_airspeed_descent
        d_v_tas_dh_descent = atm_descent_plus_1.true_airspeed - true_airspeed_descent
        partials_wrt_gamma_descent = (
            d_v_tas_dh_descent * true_airspeed_descent * np.cos(gamma_descent)
        )

        partials_wrt_gamma = np.concatenate((partials_wrt_gamma_climb, partials_wrt_gamma_descent))

        partials["d_vx_dt", "gamma"] = partials_wrt_gamma
