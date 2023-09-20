# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om

from scipy.constants import g
from stdatm import AtmosphereWithPartials

import fastoad.api as oad

from .constants import SUBMODEL_RESERVE_SPEED_VECT


@oad.RegisterSubmodel(
    SUBMODEL_RESERVE_SPEED_VECT,
    "fastga_he.submodel.performances.mission_vector.reserve_speed.legacy",
)
class InitializeReserveAirspeed(om.ExplicitComponent):
    """Computes the true airspeed at which the reserve will be done."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Contains the index of the different flight phases
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

        self.reserve_idx = (
            number_of_points_climb + number_of_points_cruise + number_of_points_descent
        )

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:mission:sizing:main_route:reserve:altitude", val=np.nan, units="m")

        self.add_input(
            "settings:mission:sizing:main_route:reserve:speed:k_factor",
            val=1.3,
            desc="Ration between the speed during the reserve segment and stall speed",
        )

        self.add_input(
            "mass", val=np.full(number_of_points, np.nan), shape=number_of_points, units="kg"
        )

        self.add_output("data:mission:sizing:main_route:reserve:v_tas", val=60.0, units="m/s")

        self.declare_partials(
            of="*",
            wrt=[
                "data:geometry:wing:area",
                "data:aerodynamics:wing:low_speed:CL_max_clean",
                "settings:mission:sizing:main_route:reserve:speed:k_factor",
                "data:mission:sizing:main_route:reserve:altitude",
            ],
            method="exact",
        )
        self.declare_partials(
            of="*",
            wrt="mass",
            method="exact",
            cols=np.array([self.reserve_idx]),
            rows=np.array([0]),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]
        reserve_altitude = inputs["data:mission:sizing:main_route:reserve:altitude"]

        stall_speed_margin = inputs["settings:mission:sizing:main_route:reserve:speed:k_factor"]

        mass_reserve = inputs["mass"][self.reserve_idx]

        density_reserve = AtmosphereWithPartials(reserve_altitude, altitude_in_feet=False).density

        vs_1 = np.sqrt((mass_reserve * g) / (0.5 * density_reserve * wing_area * cl_max_clean))

        outputs["data:mission:sizing:main_route:reserve:v_tas"] = stall_speed_margin * vs_1

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]
        reserve_altitude = inputs["data:mission:sizing:main_route:reserve:altitude"]

        stall_speed_margin = inputs["settings:mission:sizing:main_route:reserve:speed:k_factor"]

        mass_reserve = inputs["mass"][self.reserve_idx]

        atm = AtmosphereWithPartials(reserve_altitude, altitude_in_feet=False)
        density_reserve = atm.density

        partials[
            "data:mission:sizing:main_route:reserve:v_tas",
            "data:aerodynamics:wing:low_speed:CL_max_clean",
        ] = (
            -0.5
            * stall_speed_margin
            * np.sqrt((mass_reserve * g) / (0.5 * density_reserve * wing_area))
            * cl_max_clean ** (-3.0 / 2.0)
        )
        partials["data:mission:sizing:main_route:reserve:v_tas", "data:geometry:wing:area",] = (
            -0.5
            * stall_speed_margin
            * np.sqrt((mass_reserve * g) / (0.5 * density_reserve * cl_max_clean))
            * wing_area ** (-3.0 / 2.0)
        )
        partials[
            "data:mission:sizing:main_route:reserve:v_tas",
            "settings:mission:sizing:main_route:reserve:speed:k_factor",
        ] = np.sqrt((mass_reserve * g) / (0.5 * density_reserve * wing_area * cl_max_clean))

        partials["data:mission:sizing:main_route:reserve:v_tas", "mass"] = (
            0.5
            * stall_speed_margin
            * np.sqrt(g / (0.5 * density_reserve * wing_area * cl_max_clean * mass_reserve))
        )

        partials[
            "data:mission:sizing:main_route:reserve:v_tas",
            "data:mission:sizing:main_route:reserve:altitude",
        ] = (
            -0.5
            * stall_speed_margin
            * np.sqrt(
                (mass_reserve * g) / (0.5 * density_reserve ** 3.0 * wing_area * cl_max_clean)
            )
        ) * atm.partial_density_altitude
