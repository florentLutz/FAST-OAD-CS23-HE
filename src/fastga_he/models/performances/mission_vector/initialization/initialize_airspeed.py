# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om

from stdatm import Atmosphere


class InitializeAirspeed(om.ExplicitComponent):
    """Initializes the airspeeds at each time step."""

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

        self.add_input(
            "altitude", val=np.full(number_of_points, np.nan), shape=number_of_points, units="m"
        )

        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:reserve:v_tas", val=np.nan, units="m/s")

        self.add_input("data:mission:sizing:main_route:climb:v_eas", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:descent:v_eas", val=np.nan, units="m/s")

        self.add_output("true_airspeed", val=np.full(number_of_points, 50.0), units="m/s")
        self.add_output("equivalent_airspeed", val=np.full(number_of_points, 50.0), units="m/s")

        # Because of how we do the conversion between TAS and EAS, the partials can't be written
        # explicitly but we can use sparse partials to save computation time
        cruise_idx = np.linspace(
            number_of_points_climb,
            number_of_points_climb + number_of_points_cruise - 1,
            number_of_points_cruise,
        ).astype(int)
        reserve_idx = np.linspace(
            number_of_points_climb + number_of_points_cruise + number_of_points_descent,
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
            - 1,
            number_of_points_reserve,
        ).astype(int)
        eas_via_tas = np.concatenate((cruise_idx, reserve_idx))

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
        tas_via_eas = np.concatenate((climb_idx, descent_idx))
        self.declare_partials(
            of="equivalent_airspeed",
            wrt="altitude",
            method="fd",
            cols=eas_via_tas,
            rows=eas_via_tas,
        )
        self.declare_partials(
            of="true_airspeed",
            wrt="altitude",
            method="fd",
            cols=tas_via_eas,
            rows=tas_via_eas,
        )

        self.declare_partials(
            of="equivalent_airspeed",
            wrt="data:TLAR:v_cruise",
            method="fd",
            rows=cruise_idx,
            cols=np.zeros_like(cruise_idx),
        )
        self.declare_partials(
            of="true_airspeed",
            wrt="data:TLAR:v_cruise",
            method="exact",
        )

        self.declare_partials(
            of="equivalent_airspeed",
            wrt="data:mission:sizing:main_route:reserve:v_tas",
            method="fd",
            rows=reserve_idx,
            cols=np.zeros_like(reserve_idx),
        )
        self.declare_partials(
            of="true_airspeed",
            wrt="data:mission:sizing:main_route:reserve:v_tas",
            method="exact",
        )

        self.declare_partials(
            of="equivalent_airspeed",
            wrt="data:mission:sizing:main_route:climb:v_eas",
            method="exact",
        )
        self.declare_partials(
            of="true_airspeed",
            wrt="data:mission:sizing:main_route:climb:v_eas",
            method="fd",
            rows=climb_idx,
            cols=np.zeros_like(climb_idx),
        )

        self.declare_partials(
            of="equivalent_airspeed",
            wrt="data:mission:sizing:main_route:descent:v_eas",
            method="exact",
        )
        self.declare_partials(
            of="true_airspeed",
            wrt="data:mission:sizing:main_route:descent:v_eas",
            method="fd",
            rows=descent_idx,
            cols=np.zeros_like(descent_idx),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        v_tas_cruise = inputs["data:TLAR:v_cruise"]
        v_tas_reserve = inputs["data:mission:sizing:main_route:reserve:v_tas"]

        altitude = inputs["altitude"]

        altitude_climb = altitude[0:number_of_points_climb]
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
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent : number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        ]

        v_eas_climb = inputs["data:mission:sizing:main_route:climb:v_eas"]
        atm_climb = Atmosphere(altitude_climb, altitude_in_feet=False)

        atm_climb.equivalent_airspeed = np.full_like(altitude_climb, v_eas_climb)
        true_airspeed_climb = atm_climb.true_airspeed

        atm_cruise = Atmosphere(altitude_cruise, altitude_in_feet=False)
        true_airspeed_cruise = np.full_like(altitude_cruise, v_tas_cruise)
        atm_cruise.true_airspeed = true_airspeed_cruise
        equivalent_airspeed_cruise = atm_cruise.equivalent_airspeed

        atm_reserve = Atmosphere(altitude_reserve, altitude_in_feet=False)
        true_airspeed_reserve = np.full_like(altitude_reserve, v_tas_reserve)
        atm_reserve.true_airspeed = true_airspeed_reserve
        equivalent_airspeed_reserve = atm_reserve.equivalent_airspeed

        v_eas_descent = inputs["data:mission:sizing:main_route:descent:v_eas"]
        atm_descent = Atmosphere(altitude_descent, altitude_in_feet=False)

        atm_descent.equivalent_airspeed = np.full_like(altitude_descent, v_eas_descent)
        true_airspeed_descent = atm_descent.true_airspeed

        outputs["true_airspeed"] = np.concatenate(
            (
                true_airspeed_climb,
                true_airspeed_cruise,
                true_airspeed_descent,
                true_airspeed_reserve,
            )
        )
        outputs["equivalent_airspeed"] = np.concatenate(
            (
                atm_climb.equivalent_airspeed,
                equivalent_airspeed_cruise,
                atm_descent.equivalent_airspeed,
                equivalent_airspeed_reserve,
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

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

        cruise_idx = np.linspace(
            number_of_points_climb,
            number_of_points_climb + number_of_points_cruise - 1,
            number_of_points_cruise,
        ).astype(int)
        reserve_idx = np.linspace(
            number_of_points_climb + number_of_points_cruise + number_of_points_descent,
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
            - 1,
            number_of_points_reserve,
        ).astype(int)
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

        partials["true_airspeed", "data:TLAR:v_cruise"] = np.put(
            np.zeros(number_of_points), cruise_idx, np.ones_like(cruise_idx)
        )
        partials["true_airspeed", "data:mission:sizing:main_route:reserve:v_tas"] = np.put(
            np.zeros(number_of_points), reserve_idx, np.ones_like(reserve_idx)
        )

        partials["equivalent_airspeed", "data:mission:sizing:main_route:climb:v_eas"] = np.put(
            np.zeros(number_of_points), climb_idx, np.ones_like(climb_idx)
        )
        partials["equivalent_airspeed", "data:mission:sizing:main_route:descent:v_eas"] = np.put(
            np.zeros(number_of_points), descent_idx, np.ones_like(descent_idx)
        )
