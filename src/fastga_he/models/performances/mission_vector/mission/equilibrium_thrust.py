# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om
from scipy.constants import g
from stdatm import Atmosphere


class EquilibriumThrust(om.ImplicitComponent):
    """Find the conditions necessary for the aircraft equilibrium along the X-axis only."""

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

        self.add_input("d_vx_dt", val=np.full(number_of_points, 0.0), units="m/s**2")
        self.add_input("mass", val=np.full(number_of_points, 1500.0), units="kg")
        self.add_input("gamma", val=np.full(number_of_points, 0.0), units="deg")
        self.add_input("altitude", val=np.full(number_of_points, 0.0), units="m")
        self.add_input("true_airspeed", val=np.full(number_of_points, 50.0), units="m/s")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:elevator:low_speed:CD_delta", val=np.nan, units="rad**-2")
        if self.options["flaps_position"] == "takeoff":
            self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)
            self.add_input("data:aerodynamics:flaps:takeoff:CD", val=np.nan)
        if self.options["flaps_position"] == "landing":
            self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)
            self.add_input("data:aerodynamics:flaps:landing:CD", val=np.nan)

        self.add_input("delta_Cl", val=np.full(number_of_points, 0.0))
        self.add_input("delta_Cd", val=np.full(number_of_points, 0.0))

        self.add_input("alpha", val=np.full(number_of_points, np.nan), units="deg")
        self.add_input("delta_m", val=np.full(number_of_points, np.nan), units="deg")

        self.add_output("thrust", val=np.full(number_of_points, 1000.0), units="N")

        self.declare_partials(
            of="thrust",
            wrt=[
                "altitude",
            ],
            method="fd",
            form="central",
            step=1.0e2,
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="thrust",
            wrt=[
                "gamma",
                "d_vx_dt",
                "mass",
                "true_airspeed",
                "data:geometry:wing:area",
                "data:aerodynamics:aircraft:cruise:CD0",
                "data:aerodynamics:wing:cruise:CL0_clean",
                "data:aerodynamics:wing:cruise:CL_alpha",
                "data:aerodynamics:wing:cruise:induced_drag_coefficient",
                "data:aerodynamics:horizontal_tail:cruise:CL0",
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha",
                "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient",
                "data:aerodynamics:elevator:low_speed:CD_delta",
                "data:aerodynamics:elevator:low_speed:CL_delta",
                "delta_Cd",
                "alpha",
                "thrust",
                "delta_m",
            ],
            method="exact",
        )
        if self.options["flaps_position"] == "takeoff":
            self.declare_partials(
                of="thrust", wrt="data:aerodynamics:flaps:takeoff:CD", method="exact"
            )
        if self.options["flaps_position"] == "landing":
            self.declare_partials(
                of="thrust", wrt="data:aerodynamics:flaps:landing:CD", method="exact"
            )

    def linearize(self, inputs, outputs, partials):

        mass = inputs["mass"]
        d_vx_dt = inputs["d_vx_dt"]
        true_airspeed = inputs["true_airspeed"]
        altitude = inputs["altitude"]
        gamma = inputs["gamma"] * np.pi / 180.0

        wing_area = inputs["data:geometry:wing:area"]

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        coeff_k_htp = inputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        cl_delta_m = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cd_delta_m = inputs["data:aerodynamics:elevator:low_speed:CD_delta"]

        alpha = inputs["alpha"] * np.pi / 180.0
        thrust = outputs["thrust"]
        delta_m = inputs["delta_m"] * np.pi / 180.0

        if self.options["flaps_position"] == "takeoff":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:takeoff:CL"]
            delta_cd_flaps = inputs["data:aerodynamics:flaps:takeoff:CD"]
        elif self.options["flaps_position"] == "landing":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
            delta_cd_flaps = inputs["data:aerodynamics:flaps:landing:CD"]
        else:  # Cruise conditions
            delta_cl_flaps = 0.0
            delta_cd_flaps = 0.0

        delta_cl = inputs["delta_Cl"]
        delta_cd = inputs["delta_Cd"]

        rho = Atmosphere(altitude, altitude_in_feet=False).density

        dynamic_pressure = 1.0 / 2.0 * rho * np.square(true_airspeed)

        cl_wing = cl0_wing + cl_alpha_wing * alpha + delta_cl + delta_cl_flaps
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m

        cd_tot = (
            cd0
            + delta_cd
            + delta_cd_flaps
            + coeff_k_wing * cl_wing ** 2
            + coeff_k_htp * cl_htp ** 2
            + (cd_delta_m * delta_m ** 2.0)
        )

        d_q_d_airspeed = rho * true_airspeed

        # ------------------ Derivatives wrt thrust residuals ------------------ #

        d_thrust_d_cl_w = -2.0 * dynamic_pressure * wing_area * coeff_k_wing * cl_wing
        d_thrust_d_cl_h = -2.0 * dynamic_pressure * wing_area * coeff_k_htp * cl_htp

        d_cl_w_d_cl_alpha_w = alpha
        d_cl_h_d_cl_alpha_h = alpha
        d_cl_h_d_cl_delta = delta_m

        partials["thrust", "d_vx_dt"] = np.diag(-mass)
        partials["thrust", "gamma"] = np.diag(-mass * g * np.cos(gamma) * np.pi / 180.0)
        partials["thrust", "mass"] = np.diag(-d_vx_dt - g * np.sin(gamma))
        partials["thrust", "true_airspeed"] = -np.diag(wing_area * cd_tot * d_q_d_airspeed)
        partials["thrust", "data:geometry:wing:area"] = -dynamic_pressure * cd_tot
        partials["thrust", "data:aerodynamics:aircraft:cruise:CD0"] = -dynamic_pressure * wing_area
        partials["thrust", "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"] = (
            -dynamic_pressure * wing_area * cl_htp ** 2.0
        )
        partials["thrust", "data:aerodynamics:wing:cruise:induced_drag_coefficient"] = (
            -dynamic_pressure * wing_area * cl_wing ** 2.0
        )
        partials["thrust", "delta_Cd"] = -np.diag(dynamic_pressure * wing_area)
        partials["thrust", "data:aerodynamics:elevator:low_speed:CD_delta"] = (
            -dynamic_pressure * wing_area * delta_m ** 2.0
        )
        partials["thrust", "data:aerodynamics:elevator:low_speed:CL_delta"] = (
            d_thrust_d_cl_h * d_cl_h_d_cl_delta
        )
        partials["thrust", "data:aerodynamics:wing:cruise:CL0_clean"] = d_thrust_d_cl_w
        partials["thrust", "data:aerodynamics:wing:cruise:CL_alpha"] = (
            d_thrust_d_cl_w * d_cl_w_d_cl_alpha_w
        )
        partials["thrust", "data:aerodynamics:horizontal_tail:cruise:CL0"] = d_thrust_d_cl_h
        partials["thrust", "data:aerodynamics:horizontal_tail:cruise:CL_alpha"] = (
            d_thrust_d_cl_h * d_cl_h_d_cl_alpha_h
        )
        partials["thrust", "thrust"] = np.diag(np.cos(alpha))
        d_thrust_d_alpha_vector = (
            (
                -thrust * np.sin(alpha)
                + d_thrust_d_cl_w * cl_alpha_wing
                + d_thrust_d_cl_h * cl_alpha_htp
            )
            * np.pi
            / 180.0
        )
        partials["thrust", "alpha"] = np.diag(d_thrust_d_alpha_vector)
        d_thrust_d_delta_m_vector = (
            (
                d_thrust_d_cl_h * cl_delta_m
                - 2.0 * dynamic_pressure * wing_area * cd_delta_m * delta_m
            )
            * np.pi
            / 180.0
        )
        partials["thrust", "delta_m"] = np.diag(d_thrust_d_delta_m_vector)
        if self.options["flaps_position"] == "takeoff":
            partials["thrust", "data:aerodynamics:flaps:takeoff:CD"] = -dynamic_pressure * wing_area
        if self.options["flaps_position"] == "landing":
            partials["thrust", "data:aerodynamics:flaps:landing:CD"] = -dynamic_pressure * wing_area

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        d_vx_dt = inputs["d_vx_dt"]
        mass = inputs["mass"]
        gamma = inputs["gamma"] * np.pi / 180.0
        true_airspeed = inputs["true_airspeed"]
        altitude = inputs["altitude"]

        wing_area = inputs["data:geometry:wing:area"]

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        coeff_k_htp = inputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        cl_delta_m = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cd_delta_m = inputs["data:aerodynamics:elevator:low_speed:CD_delta"]

        delta_cd = inputs["delta_Cd"]

        if self.options["flaps_position"] == "takeoff":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:takeoff:CL"]
            delta_cd_flaps = inputs["data:aerodynamics:flaps:takeoff:CD"]
        elif self.options["flaps_position"] == "landing":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
            delta_cd_flaps = inputs["data:aerodynamics:flaps:landing:CD"]
        else:  # Cruise conditions
            delta_cl_flaps = 0.0
            delta_cd_flaps = 0.0

        alpha = inputs["alpha"] * np.pi / 180.0
        thrust = outputs["thrust"]
        delta_m = inputs["delta_m"] * np.pi / 180.0

        rho = Atmosphere(altitude, altitude_in_feet=False).density

        dynamic_pressure = 1.0 / 2.0 * rho * np.square(true_airspeed)

        cl_wing_clean = cl0_wing + cl_alpha_wing * alpha
        cl_wing_flaps = cl_wing_clean + delta_cl_flaps
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m

        cd_tot = (
            cd0
            + delta_cd
            + delta_cd_flaps
            + coeff_k_wing * cl_wing_flaps ** 2.0
            + coeff_k_htp * cl_htp ** 2.0
            + (cd_delta_m * delta_m ** 2.0)
        )

        residuals["thrust"] = (
            thrust * np.cos(alpha)
            - dynamic_pressure * wing_area * cd_tot
            - mass * g * np.sin(gamma)
            - mass * d_vx_dt
        )
