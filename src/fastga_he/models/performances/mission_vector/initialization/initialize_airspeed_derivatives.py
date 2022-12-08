# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om
from stdatm import Atmosphere


class InitializeAirspeedDerivatives(om.ExplicitComponent):
    """Computes the d_vx_dt at each time step."""

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
            "gamma", shape=number_of_points, val=np.full(number_of_points, np.nan), units="deg"
        )

        self.add_output(
            "d_vx_dt", shape=number_of_points, val=np.full(number_of_points, 0.0), units="m/s**2"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]

        true_airspeed = inputs["true_airspeed"]
        equivalent_airspeed = inputs["equivalent_airspeed"]
        altitude = inputs["altitude"]
        gamma = inputs["gamma"] * np.pi / 180.0

        altitude_climb = altitude[0:number_of_points_climb]
        gamma_climb = gamma[0:number_of_points_climb]
        equivalent_airspeed_climb = equivalent_airspeed[0:number_of_points_climb]
        true_airspeed_climb = true_airspeed[0:number_of_points_climb]

        altitude_descent = altitude[number_of_points_climb + number_of_points_cruise :]
        gamma_descent = gamma[number_of_points_climb + number_of_points_cruise :]
        equivalent_airspeed_descent = equivalent_airspeed[
            number_of_points_climb + number_of_points_cruise :
        ]
        true_airspeed_descent = true_airspeed[number_of_points_climb + number_of_points_cruise :]

        atm_climb_plus_1 = Atmosphere(altitude_climb + 1.0, altitude_in_feet=False)
        atm_climb_plus_1.equivalent_airspeed = equivalent_airspeed_climb
        d_v_tas_dh_climb = atm_climb_plus_1.true_airspeed - true_airspeed_climb
        d_vx_dt_climb = d_v_tas_dh_climb * true_airspeed_climb * np.sin(gamma_climb)

        atm_descent_plus_1 = Atmosphere(altitude_descent + 1.0, altitude_in_feet=False)
        atm_descent_plus_1.equivalent_airspeed = equivalent_airspeed_descent
        d_v_tas_dh_descent = atm_descent_plus_1.true_airspeed - true_airspeed_descent
        d_vx_dt_descent = d_v_tas_dh_descent * true_airspeed_descent * np.sin(gamma_descent)

        d_vx_dt = np.concatenate(
            (d_vx_dt_climb, np.zeros(number_of_points_cruise), d_vx_dt_descent)
        )

        outputs["d_vx_dt"] = d_vx_dt
