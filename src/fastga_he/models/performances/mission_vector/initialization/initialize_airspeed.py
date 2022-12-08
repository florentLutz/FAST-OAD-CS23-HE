# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om
from scipy.constants import g
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

    def setup(self):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]

        number_of_points = (
            number_of_points_climb + number_of_points_cruise + number_of_points_descent
        )

        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL", np.nan)

        self.add_input(
            "mass", val=np.full(number_of_points, np.nan), shape=number_of_points, units="kg"
        )
        self.add_input(
            "altitude", val=np.full(number_of_points, np.nan), shape=number_of_points, units="m"
        )

        self.add_output("true_airspeed", val=np.full(number_of_points, 50.0), units="m/s")
        self.add_output("equivalent_airspeed", val=np.full(number_of_points, 50.0), units="m/s")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]

        v_tas_cruise = inputs["data:TLAR:v_cruise"]

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]

        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]

        mass = inputs["mass"]
        altitude = inputs["altitude"]

        altitude_climb = altitude[0:number_of_points_climb]
        altitude_cruise = altitude[
            number_of_points_climb : number_of_points_climb + number_of_points_cruise
        ]
        altitude_descent = altitude[number_of_points_climb + number_of_points_cruise :]

        # Computes the airspeed that gives the best climb rate
        # FIXME: VCAS constant-speed strategy is specific to ICE-propeller configuration,
        # FIXME: could be an input!
        c_l = np.sqrt(3 * cd0 / coeff_k_wing)
        atm_climb = Atmosphere(altitude_climb, altitude_in_feet=False)

        vs1 = np.sqrt((mass[0] * g) / (0.5 * atm_climb.density[0] * wing_area * cl_max_clean))
        # Using the denomination in Gudmundsson
        v_y = np.sqrt((mass[0] * g) / (0.5 * atm_climb.density[0] * wing_area * c_l))
        v_eas_climb = max(v_y, 1.3 * vs1)
        atm_climb.equivalent_airspeed = np.full_like(altitude_climb, v_eas_climb)

        true_airspeed_climb = atm_climb.true_airspeed

        atm_cruise = Atmosphere(altitude_cruise[0], altitude_in_feet=False)
        atm_cruise.true_airspeed = v_tas_cruise
        true_airspeed_cruise = np.full_like(altitude_cruise, v_tas_cruise)
        equivalent_airspeed_cruise = np.full_like(altitude_cruise, atm_cruise.equivalent_airspeed)

        cl_opt = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]

        mass_descent = mass[number_of_points_climb + number_of_points_cruise + 1]
        atm_descent = Atmosphere(altitude_descent, altitude_in_feet=False)
        vs1 = np.sqrt(
            (mass_descent * g) / (0.5 * atm_descent.density[0] * wing_area * cl_max_clean)
        )
        v_eas_descent = max(
            np.sqrt((mass_descent * g) / (0.5 * atm_descent.density[0] * wing_area * cl_opt)),
            1.3 * vs1,
        )

        atm_descent.equivalent_airspeed = np.full_like(altitude_descent, v_eas_descent)
        true_airspeed_descent = atm_descent.true_airspeed

        outputs["true_airspeed"] = np.concatenate(
            (true_airspeed_climb, true_airspeed_cruise, true_airspeed_descent)
        )
        outputs["equivalent_airspeed"] = np.concatenate(
            (
                atm_climb.equivalent_airspeed,
                equivalent_airspeed_cruise,
                atm_descent.equivalent_airspeed,
            )
        )
