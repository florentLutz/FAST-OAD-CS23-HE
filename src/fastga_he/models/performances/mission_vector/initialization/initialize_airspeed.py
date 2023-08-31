# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials

RHO_SL = AtmosphereWithPartials(0.0).density


class InitializeAirspeed(om.ExplicitComponent):
    """Initializes the airspeeds at each time step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Contains the value of equivalent airspeed (eas) and true airspeed (tas), used to avoid
        # recomputing them in compute_partials
        self.eas = None
        self.tas = None

        # Contains the index where the tas is computed from the eas and the eas from the tas
        # respectively
        self.tas_via_eas = None
        self.eas_via_tas = None

        # Contains the index of the different flight phases
        self.climb_idx = None
        self.cruise_idx = None
        self.descent_idx = None
        self.reserve_idx = None

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
        self.cruise_idx = np.linspace(
            number_of_points_climb,
            number_of_points_climb + number_of_points_cruise - 1,
            number_of_points_cruise,
        ).astype(int)
        self.reserve_idx = np.linspace(
            number_of_points_climb + number_of_points_cruise + number_of_points_descent,
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
            - 1,
            number_of_points_reserve,
        ).astype(int)
        self.eas_via_tas = np.concatenate((self.cruise_idx, self.reserve_idx))

        self.declare_partials(
            of="equivalent_airspeed",
            wrt="altitude",
            method="exact",
            cols=self.eas_via_tas,
            rows=self.eas_via_tas,
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

        self.tas_via_eas = np.concatenate((self.climb_idx, self.descent_idx))

        self.declare_partials(
            of="true_airspeed",
            wrt="altitude",
            method="exact",
            cols=self.tas_via_eas,
            rows=self.tas_via_eas,
        )

        self.declare_partials(
            of=["true_airspeed", "equivalent_airspeed"],
            wrt="data:TLAR:v_cruise",
            method="exact",
            rows=self.cruise_idx,
            cols=np.zeros_like(self.cruise_idx),
        )

        self.declare_partials(
            of=["true_airspeed", "equivalent_airspeed"],
            wrt="data:mission:sizing:main_route:reserve:v_tas",
            method="exact",
            rows=self.reserve_idx,
            cols=np.zeros_like(self.reserve_idx),
        )

        self.declare_partials(
            of=["true_airspeed", "equivalent_airspeed"],
            wrt="data:mission:sizing:main_route:climb:v_eas",
            method="exact",
            rows=self.climb_idx,
            cols=np.zeros_like(self.climb_idx),
        )

        self.declare_partials(
            of=["equivalent_airspeed", "true_airspeed"],
            wrt="data:mission:sizing:main_route:descent:v_eas",
            method="exact",
            rows=self.descent_idx,
            cols=np.zeros_like(self.descent_idx),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        v_tas_cruise = inputs["data:TLAR:v_cruise"]
        v_tas_reserve = inputs["data:mission:sizing:main_route:reserve:v_tas"]

        altitude = inputs["altitude"]

        altitude_climb = altitude[self.climb_idx]
        altitude_cruise = altitude[self.cruise_idx]
        altitude_descent = altitude[self.descent_idx]
        altitude_reserve = altitude[self.reserve_idx]

        v_eas_climb = inputs["data:mission:sizing:main_route:climb:v_eas"]
        atm_climb = AtmosphereWithPartials(altitude_climb, altitude_in_feet=False)
        atm_climb.equivalent_airspeed = np.full_like(altitude_climb, v_eas_climb)
        true_airspeed_climb = atm_climb.true_airspeed

        atm_cruise = AtmosphereWithPartials(altitude_cruise, altitude_in_feet=False)
        true_airspeed_cruise = np.full_like(altitude_cruise, v_tas_cruise)
        atm_cruise.true_airspeed = true_airspeed_cruise
        equivalent_airspeed_cruise = atm_cruise.equivalent_airspeed

        atm_reserve = AtmosphereWithPartials(altitude_reserve, altitude_in_feet=False)
        true_airspeed_reserve = np.full_like(altitude_reserve, v_tas_reserve)
        atm_reserve.true_airspeed = true_airspeed_reserve
        equivalent_airspeed_reserve = atm_reserve.equivalent_airspeed

        v_eas_descent = inputs["data:mission:sizing:main_route:descent:v_eas"]
        atm_descent = AtmosphereWithPartials(altitude_descent, altitude_in_feet=False)
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
        self.tas = outputs["true_airspeed"]
        outputs["equivalent_airspeed"] = np.concatenate(
            (
                atm_climb.equivalent_airspeed,
                equivalent_airspeed_cruise,
                atm_descent.equivalent_airspeed,
                equivalent_airspeed_reserve,
            )
        )
        self.eas = outputs["equivalent_airspeed"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        atm = AtmosphereWithPartials(altitude=inputs["altitude"], altitude_in_feet=False)
        eas_over_tas = np.sqrt(atm.density / RHO_SL)
        factor_tas_alt = -0.5 / eas_over_tas * 1.0 / atm.density * atm.partial_density_altitude
        factor_eas_alt = 0.5 / np.sqrt(RHO_SL * atm.density) * atm.partial_density_altitude

        partials["true_airspeed", "data:TLAR:v_cruise"] = np.ones_like(self.cruise_idx)
        partials["equivalent_airspeed", "data:TLAR:v_cruise"] = eas_over_tas[self.cruise_idx]

        partials["true_airspeed", "data:mission:sizing:main_route:reserve:v_tas"] = np.ones_like(
            self.reserve_idx
        )
        partials[
            "equivalent_airspeed", "data:mission:sizing:main_route:reserve:v_tas"
        ] = eas_over_tas[self.reserve_idx]

        partials[
            "equivalent_airspeed", "data:mission:sizing:main_route:climb:v_eas"
        ] = np.ones_like(self.climb_idx)
        partials["true_airspeed", "data:mission:sizing:main_route:climb:v_eas"] = (
            1.0 / eas_over_tas[self.climb_idx]
        )

        partials[
            "equivalent_airspeed", "data:mission:sizing:main_route:descent:v_eas"
        ] = np.ones_like(self.descent_idx)
        partials["true_airspeed", "data:mission:sizing:main_route:descent:v_eas"] = (
            1.0 / eas_over_tas[self.descent_idx]
        )

        partials["true_airspeed", "altitude"] = (
            self.eas[self.tas_via_eas] * factor_tas_alt[self.tas_via_eas]
        )

        partials["equivalent_airspeed", "altitude"] = (
            self.tas[self.eas_via_tas] * factor_eas_alt[self.eas_via_tas]
        )
