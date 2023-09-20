# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om

from scipy.constants import g
from stdatm import AtmosphereWithPartials

import fastoad.api as oad

from .constants import SUBMODEL_DESCENT_SPEED_VECT


@oad.RegisterSubmodel(
    SUBMODEL_DESCENT_SPEED_VECT,
    "fastga_he.submodel.performances.mission_vector.descent_speed.legacy",
)
class InitializeDescentAirspeed(om.ExplicitComponent):
    """Initializes the descent airspeed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Contains the index of the different flight phases
        self.vs1 = None
        self.v_descent = None

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

        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL", np.nan)

        self.add_input(
            "mass", val=np.full(number_of_points, np.nan), shape=number_of_points, units="kg"
        )
        self.add_input(
            "altitude", val=np.full(number_of_points, np.nan), shape=number_of_points, units="m"
        )

        self.add_output("data:mission:sizing:main_route:descent:v_eas", val=50.0, units="m/s")

        self.declare_partials(
            of="*",
            wrt=[
                "mass",
                "data:aerodynamics:wing:low_speed:CL_max_clean",
                "data:aerodynamics:aircraft:cruise:optimal_CL",
                "data:geometry:wing:area",
            ],
            method="exact",
        )
        self.declare_partials(
            of="*",
            wrt="altitude",
            method="exact",
            cols=np.array([number_of_points_climb + number_of_points_cruise]),
            rows=np.array([0]),
        )
        # Only depends on the cruise altitude which is stored at the number_of_points_climb +
        # number_of_points_cruise-th position

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]

        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]

        mass = inputs["mass"]
        altitude = inputs["altitude"]

        altitude_cruise = altitude[number_of_points_climb + number_of_points_cruise]

        cl_opt = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]

        mass_descent = mass[number_of_points_climb + number_of_points_cruise]
        atm = AtmosphereWithPartials(altitude_cruise, altitude_in_feet=False)
        self.v_descent = np.sqrt((mass_descent * g) / (0.5 * atm.density * wing_area * cl_opt))
        self.vs1 = np.sqrt((mass_descent * g) / (0.5 * atm.density * wing_area * cl_max_clean))

        v_eas_descent = max(self.v_descent, 1.3 * self.vs1)

        outputs["data:mission:sizing:main_route:descent:v_eas"] = v_eas_descent

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]

        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]

        mass = inputs["mass"]
        altitude = inputs["altitude"]

        altitude_cruise = altitude[number_of_points_climb + number_of_points_cruise]

        cl_opt = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]

        mass_descent = mass[number_of_points_climb + number_of_points_cruise]
        atm = AtmosphereWithPartials(altitude_cruise, altitude_in_feet=False)
        density_cruise = atm.density

        mass_partials = np.full_like(mass, np.inf)
        mass_partials[number_of_points_climb + number_of_points_cruise] = mass_descent

        if self.v_descent > 1.3 * self.vs1:

            partials["data:mission:sizing:main_route:descent:v_eas", "mass"] = 0.5 * np.sqrt(
                g / (0.5 * density_cruise * wing_area * cl_opt * mass_partials)
            )
            partials["data:mission:sizing:main_route:descent:v_eas", "data:geometry:wing:area"] = (
                -0.5
                * np.sqrt(mass_descent * g / (0.5 * density_cruise * cl_opt))
                * wing_area ** (-3.0 / 2.0)
            )
            partials[
                "data:mission:sizing:main_route:descent:v_eas",
                "data:aerodynamics:aircraft:cruise:optimal_CL",
            ] = (
                -0.5
                * np.sqrt(mass_descent * g / (0.5 * density_cruise * wing_area))
                * cl_opt ** (-3.0 / 2.0)
            )
            partials[
                "data:mission:sizing:main_route:descent:v_eas",
                "data:aerodynamics:wing:low_speed:CL_max_clean",
            ] = 0.0

            partials["data:mission:sizing:main_route:descent:v_eas", "altitude"] = (
                -0.5
                * np.sqrt((mass_descent * g) / (0.5 * density_cruise ** 3.0 * wing_area * cl_opt))
                * atm.partial_density_altitude
            )

        else:

            partials["data:mission:sizing:main_route:descent:v_eas", "mass"] = 0.65 * np.sqrt(
                g / (0.5 * density_cruise * wing_area * cl_max_clean * mass_partials)
            )
            partials["data:mission:sizing:main_route:descent:v_eas", "data:geometry:wing:area"] = (
                -0.65
                * np.sqrt(mass_descent * g / (0.5 * density_cruise * cl_max_clean))
                * wing_area ** (-3.0 / 2.0)
            )
            partials[
                "data:mission:sizing:main_route:descent:v_eas",
                "data:aerodynamics:aircraft:cruise:optimal_CL",
            ] = 0.0
            partials[
                "data:mission:sizing:main_route:descent:v_eas",
                "data:aerodynamics:wing:low_speed:CL_max_clean",
            ] = (
                -0.65
                * np.sqrt(mass_descent * g / (0.5 * density_cruise * wing_area))
                * cl_max_clean ** (-3.0 / 2.0)
            )
            partials["data:mission:sizing:main_route:descent:v_eas", "altitude"] = -0.65 * np.sqrt(
                (mass_descent * g)
                / (0.5 * density_cruise ** 3.0 * wing_area * cl_max_clean)
                * atm.partial_density_altitude
            )
