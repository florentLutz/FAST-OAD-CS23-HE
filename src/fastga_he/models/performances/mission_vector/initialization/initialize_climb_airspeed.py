# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om

from scipy.constants import g
from stdatm import Atmosphere

import fastoad.api as oad

from .constants import SUBMODEL_CLIMB_SPEED_VECT


@oad.RegisterSubmodel(
    SUBMODEL_CLIMB_SPEED_VECT, "fastga_he.submodel.performances.mission_vector.climb_speed.legacy"
)
class InitializeClimbAirspeed(om.ExplicitComponent):
    """Computes the equivalent airspeed at which the climb will be done."""

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

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL", np.nan)

        self.add_input(
            "mass", val=np.full(number_of_points, np.nan), shape=number_of_points, units="kg"
        )

        self.add_output("data:mission:sizing:main_route:climb:v_eas", val=50.0, units="m/s")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]

        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]

        mass = inputs["mass"]

        density_sl = Atmosphere(0, altitude_in_feet=False).density

        # Computes the airspeed that gives the best climb rate
        c_l = np.sqrt(3 * cd0 / coeff_k_wing)

        vs1 = np.sqrt((mass[0] * g) / (0.5 * density_sl * wing_area * cl_max_clean))
        # Using the denomination in Gudmundsson
        v_y = np.sqrt((mass[0] * g) / (0.5 * density_sl * wing_area * c_l))
        v_eas_climb = max(v_y, 1.3 * vs1)

        outputs["data:mission:sizing:main_route:climb:v_eas"] = v_eas_climb

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]

        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]

        mass = inputs["mass"]

        density_sl = Atmosphere(0, altitude_in_feet=False).density

        # Computes the airspeed that gives the best climb rate
        c_l = np.sqrt(3 * cd0 / coeff_k_wing)

        vs1 = np.sqrt((mass[0] * g) / (0.5 * density_sl * wing_area * cl_max_clean))
        # Using the denomination in Gudmundsson
        v_y = np.sqrt((mass[0] * g) / (0.5 * density_sl * wing_area * c_l))

        mass_partials = np.full_like(mass, np.inf)
        mass_partials[0] = mass[0]

        if v_y > 1.3 * vs1:

            d_c_l_d_k = -0.5 * np.sqrt(3.0 * cd0) * coeff_k_wing ** (-3.0 / 2.0)
            d_c_l_d_cd_0 = np.sqrt(3 / (coeff_k_wing * cd0))

            partials["data:mission:sizing:main_route:climb:v_eas", "mass"] = 0.5 * np.sqrt(
                g / (0.5 * density_sl * wing_area * c_l * mass_partials)
            )
            partials[
                "data:mission:sizing:main_route:climb:v_eas", "data:geometry:wing:area"
            ] = -0.5 * np.sqrt(mass[0] * g / (0.5 * density_sl * c_l) * wing_area ** (-3.0 / 2.0))
            partials[
                "data:mission:sizing:main_route:climb:v_eas",
                "data:aerodynamics:wing:cruise:induced_drag_coefficient",
            ] = (
                -0.5
                * np.sqrt(mass[0] * g / (0.5 * density_sl * wing_area))
                * c_l ** (-3.0 / 2.0)
                * d_c_l_d_k
            )
            partials[
                "data:mission:sizing:main_route:climb:v_eas",
                "data:aerodynamics:aircraft:cruise:CD0",
            ] = (
                -0.5
                * (mass[0] * g / (0.5 * density_sl * wing_area * c_l ** 2.0))
                * c_l ** (-3.0 / 2.0)
                * d_c_l_d_cd_0
            )
            partials[
                "data:mission:sizing:main_route:climb:v_eas",
                "data:aerodynamics:wing:low_speed:CL_max_clean",
            ] = 0

        else:

            partials["data:mission:sizing:main_route:climb:v_eas", "mass"] = 0.65 * np.sqrt(
                g / (0.5 * density_sl * wing_area * cl_max_clean * mass_partials)
            )
            partials["data:mission:sizing:main_route:climb:v_eas", "data:geometry:wing:area"] = (
                -0.65
                * np.sqrt(mass[0] * g / (0.5 * density_sl * cl_max_clean))
                * wing_area ** (-3.0 / 2.0)
            )
            partials[
                "data:mission:sizing:main_route:climb:v_eas",
                "data:aerodynamics:wing:cruise:induced_drag_coefficient",
            ] = 0.0
            partials[
                "data:mission:sizing:main_route:climb:v_eas",
                "data:aerodynamics:aircraft:cruise:CD0",
            ] = 0.0
            partials[
                "data:mission:sizing:main_route:climb:v_eas",
                "data:aerodynamics:wing:low_speed:CL_max_clean",
            ] = (
                -0.65
                * np.sqrt(mass[0] * g / (0.5 * density_sl * wing_area))
                * cl_max_clean ** (-3.0 / 2.0)
            )
