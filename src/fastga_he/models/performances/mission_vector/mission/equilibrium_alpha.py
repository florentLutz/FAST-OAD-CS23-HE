# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om
from scipy.constants import g


class EquilibriumAlpha(om.ImplicitComponent):
    """Find the conditions necessary for the aircraft equilibrium."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be " "treated"
        )
        self.options.declare(
            "flaps_position",
            default="cruise",
            desc="position of the flaps for the computation of the equilibrium",
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("mass", val=np.full(number_of_points, 1500.0), units="kg")
        self.add_input("gamma", val=np.full(number_of_points, 0.0), units="deg")
        self.add_input("density", val=np.full(number_of_points, 1.225), units="kg/m**3")
        self.add_input("true_airspeed", val=np.full(number_of_points, 50.0), units="m/s")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")

        if self.options["flaps_position"] == "takeoff":
            self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)
        if self.options["flaps_position"] == "landing":
            self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)

        self.add_input("delta_Cl", val=np.full(number_of_points, 0.0))

        self.add_input("thrust", val=np.full(number_of_points, np.nan), units="N")
        self.add_input("delta_m", val=np.full(number_of_points, np.nan), units="deg")

        self.add_output("alpha", val=np.full(number_of_points, 5.0), units="deg")

        self.declare_partials(
            of="alpha",
            wrt=[
                "mass",
                "gamma",
                "density",
                "true_airspeed",
                "delta_Cl",
                "thrust",
                "alpha",
                "delta_m",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="alpha",
            wrt=[
                "data:geometry:wing:area",
                "data:aerodynamics:wing:cruise:CL0_clean",
                "data:aerodynamics:wing:cruise:CL_alpha",
                "data:aerodynamics:horizontal_tail:cruise:CL0",
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha",
                "data:aerodynamics:elevator:low_speed:CL_delta",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        if self.options["flaps_position"] == "takeoff":
            self.declare_partials(
                of="alpha",
                wrt="data:aerodynamics:flaps:takeoff:CL",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )
        if self.options["flaps_position"] == "landing":
            self.declare_partials(
                of="alpha",
                wrt="data:aerodynamics:flaps:landing:CL",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]

        mass = inputs["mass"]
        true_airspeed = inputs["true_airspeed"]
        gamma = inputs["gamma"] * np.pi / 180.0

        wing_area = inputs["data:geometry:wing:area"]

        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        cl_delta_m = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]

        alpha = outputs["alpha"] * np.pi / 180.0
        thrust = inputs["thrust"]
        delta_m = inputs["delta_m"] * np.pi / 180.0

        rho = inputs["density"]

        dynamic_pressure = 0.5 * rho * true_airspeed ** 2.0

        # ------------------ Derivatives wrt alpha residuals ------------------ #

        jacobian["alpha", "data:aerodynamics:wing:cruise:CL0_clean"] = np.ones(number_of_points)
        jacobian["alpha", "data:aerodynamics:wing:cruise:CL_alpha"] = alpha
        jacobian["alpha", "data:aerodynamics:horizontal_tail:cruise:CL0"] = np.ones(
            number_of_points
        )
        jacobian["alpha", "data:aerodynamics:horizontal_tail:cruise:CL_alpha"] = alpha
        jacobian["alpha", "delta_Cl"] = np.ones(number_of_points)
        jacobian["alpha", "data:aerodynamics:elevator:low_speed:CL_delta"] = delta_m
        d_alpha_d_mass_vector = -g * np.cos(gamma) / (dynamic_pressure * wing_area)
        jacobian["alpha", "mass"] = d_alpha_d_mass_vector
        d_alpha_d_thrust_vector = np.sin(alpha) / (dynamic_pressure * wing_area)
        jacobian["alpha", "thrust"] = d_alpha_d_thrust_vector
        d_alpha_d_gamma_vector = (
            mass * g * np.sin(gamma) / (dynamic_pressure * wing_area) / 180.0 * np.pi
        )
        jacobian["alpha", "gamma"] = d_alpha_d_gamma_vector
        d_alpha_d_q_vector = -(thrust * np.sin(alpha) - mass * g * np.cos(gamma)) / (
            wing_area * dynamic_pressure ** 2.0
        )
        jacobian["alpha", "true_airspeed"] = d_alpha_d_q_vector * rho * true_airspeed
        d_alpha_d_s_vector = -(thrust * np.sin(alpha) - mass * g * np.cos(gamma)) / (
            dynamic_pressure * wing_area ** 2.0
        )
        jacobian["alpha", "data:geometry:wing:area"] = d_alpha_d_s_vector
        d_alpha_d_alpha_vector = (
            cl_alpha_wing + cl_alpha_htp + thrust * np.cos(alpha) / (dynamic_pressure * wing_area)
        )
        jacobian["alpha", "density"] = d_alpha_d_q_vector * 0.5 * true_airspeed ** 2.0
        jacobian["alpha", "alpha"] = d_alpha_d_alpha_vector * np.pi / 180.0
        jacobian["alpha", "delta_m"] = np.ones(number_of_points) * cl_delta_m * np.pi / 180.0

        if self.options["flaps_position"] == "takeoff":
            jacobian["alpha", "data:aerodynamics:flaps:takeoff:CL"] = np.ones(number_of_points)
        if self.options["flaps_position"] == "landing":
            jacobian["alpha", "data:aerodynamics:flaps:landing:CL"] = np.ones(number_of_points)

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        mass = inputs["mass"]
        gamma = inputs["gamma"] * np.pi / 180.0
        true_airspeed = inputs["true_airspeed"]

        wing_area = inputs["data:geometry:wing:area"]

        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        cl_delta_m = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]

        delta_cl = inputs["delta_Cl"]

        if self.options["flaps_position"] == "takeoff":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:takeoff:CL"]
        elif self.options["flaps_position"] == "landing":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
        else:  # Cruise conditions
            delta_cl_flaps = 0.0

        alpha = outputs["alpha"] * np.pi / 180.0
        thrust = inputs["thrust"]
        delta_m = inputs["delta_m"] * np.pi / 180.0

        rho = inputs["density"]

        dynamic_pressure = 0.5 * rho * true_airspeed ** 2.0

        cl_wing = cl0_wing + cl_alpha_wing * alpha + delta_cl_flaps + delta_cl
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m

        residuals["alpha"] = (
            cl_wing
            + cl_htp
            + (thrust * np.sin(alpha) - mass * g * np.cos(gamma)) / (dynamic_pressure * wing_area)
        )
