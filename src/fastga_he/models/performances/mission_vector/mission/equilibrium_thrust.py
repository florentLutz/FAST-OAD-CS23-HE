# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om
from scipy.constants import g


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
        self.options.declare(
            "low_speed_aero",
            default=False,
            desc="Boolean to consider low speed aerodynamics",
            types=bool,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("d_vx_dt", val=np.full(number_of_points, 0.0), units="m/s**2")
        self.add_input("mass", val=np.full(number_of_points, 1500.0), units="kg")
        self.add_input("gamma", val=np.full(number_of_points, 0.0), units="deg")
        self.add_input("density", val=np.full(number_of_points, 1.225), units="kg/m**3")
        self.add_input("true_airspeed", val=np.full(number_of_points, 50.0), units="m/s")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:aerodynamics:aircraft:" + ls_tag + ":CD0", np.nan)
        self.add_input(
            "data:aerodynamics:wing:" + ls_tag + ":CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:aerodynamics:wing:" + ls_tag + ":CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:" + ls_tag + ":induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:" + ls_tag + ":CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:" + ls_tag + ":CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:" + ls_tag + ":induced_drag_coefficient", np.nan
        )
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:elevator:low_speed:CD_delta", val=np.nan, units="rad**-2")
        if self.options["flaps_position"] == "takeoff":
            self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)
            self.add_input("data:aerodynamics:flaps:takeoff:CD", val=np.nan)
        if self.options["flaps_position"] == "landing":
            self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)
            self.add_input("data:aerodynamics:flaps:landing:CD", val=np.nan)

        self.add_input("delta_Cd", val=np.full(number_of_points, 0.0))

        self.add_input("alpha", val=np.full(number_of_points, np.nan), units="deg")
        self.add_input("delta_m", val=np.full(number_of_points, np.nan), units="deg")

        self.add_output("thrust", val=np.full(number_of_points, 1000.0), units="N")

        self.declare_partials(
            of="thrust",
            wrt=[
                "gamma",
                "density",
                "d_vx_dt",
                "mass",
                "true_airspeed",
                "delta_Cd",
                "alpha",
                "thrust",
                "delta_m",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="thrust",
            wrt=[
                "data:geometry:wing:area",
                "data:aerodynamics:aircraft:" + ls_tag + ":CD0",
                "data:aerodynamics:wing:" + ls_tag + ":CL0_clean",
                "data:aerodynamics:wing:" + ls_tag + ":CL_alpha",
                "data:aerodynamics:wing:" + ls_tag + ":induced_drag_coefficient",
                "data:aerodynamics:horizontal_tail:" + ls_tag + ":CL0",
                "data:aerodynamics:horizontal_tail:" + ls_tag + ":CL_alpha",
                "data:aerodynamics:horizontal_tail:" + ls_tag + ":induced_drag_coefficient",
                "data:aerodynamics:elevator:low_speed:CD_delta",
                "data:aerodynamics:elevator:low_speed:CL_delta",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        if self.options["flaps_position"] == "takeoff":
            self.declare_partials(
                of="thrust",
                wrt="data:aerodynamics:flaps:takeoff:CD",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )
        if self.options["flaps_position"] == "landing":
            self.declare_partials(
                of="thrust",
                wrt="data:aerodynamics:flaps:landing:CD",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
            )

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        mass = inputs["mass"]
        d_vx_dt = inputs["d_vx_dt"]
        true_airspeed = inputs["true_airspeed"]
        gamma = inputs["gamma"] * np.pi / 180.0
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        wing_area = inputs["data:geometry:wing:area"]

        cd0 = inputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0"]
        cl0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CL_alpha"]
        coeff_k_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":induced_drag_coefficient"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:" + ls_tag + ":CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:" + ls_tag + ":CL_alpha"]
        coeff_k_htp = inputs[
            "data:aerodynamics:horizontal_tail:" + ls_tag + ":induced_drag_coefficient"
        ]
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

        delta_cd = inputs["delta_Cd"]

        rho = inputs["density"]

        dynamic_pressure = 1.0 / 2.0 * rho * np.square(true_airspeed)

        cl_wing = cl0_wing + cl_alpha_wing * alpha + delta_cl_flaps
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m

        cd_tot = (
            cd0
            + delta_cd
            + delta_cd_flaps
            + coeff_k_wing * cl_wing**2
            + coeff_k_htp * cl_htp**2
            + (cd_delta_m * delta_m**2.0)
        )

        # ------------------ Derivatives wrt thrust residuals ------------------ #

        d_thrust_d_cl_w = -2.0 * dynamic_pressure * wing_area * coeff_k_wing * cl_wing
        d_thrust_d_cl_h = -2.0 * dynamic_pressure * wing_area * coeff_k_htp * cl_htp

        jacobian["thrust", "d_vx_dt"] = -mass
        jacobian["thrust", "gamma"] = -mass * g * np.cos(gamma) * np.pi / 180.0
        jacobian["thrust", "density"] = -wing_area * cd_tot * 0.5 * true_airspeed**2.0
        jacobian["thrust", "mass"] = -d_vx_dt - g * np.sin(gamma)
        jacobian["thrust", "true_airspeed"] = -wing_area * cd_tot * rho * true_airspeed
        jacobian["thrust", "data:geometry:wing:area"] = -dynamic_pressure * cd_tot
        jacobian["thrust", "data:aerodynamics:aircraft:" + ls_tag + ":CD0"] = (
            -dynamic_pressure * wing_area
        )
        jacobian[
            "thrust", "data:aerodynamics:horizontal_tail:" + ls_tag + ":induced_drag_coefficient"
        ] = -dynamic_pressure * wing_area * cl_htp**2.0
        jacobian["thrust", "data:aerodynamics:wing:" + ls_tag + ":induced_drag_coefficient"] = (
            -dynamic_pressure * wing_area * cl_wing**2.0
        )
        jacobian["thrust", "delta_Cd"] = -dynamic_pressure * wing_area
        jacobian["thrust", "data:aerodynamics:elevator:low_speed:CD_delta"] = (
            -dynamic_pressure * wing_area * delta_m**2.0
        )
        jacobian["thrust", "data:aerodynamics:elevator:low_speed:CL_delta"] = (
            d_thrust_d_cl_h * delta_m
        )
        jacobian["thrust", "data:aerodynamics:wing:" + ls_tag + ":CL0_clean"] = d_thrust_d_cl_w
        jacobian["thrust", "data:aerodynamics:wing:" + ls_tag + ":CL_alpha"] = (
            d_thrust_d_cl_w * alpha
        )
        jacobian["thrust", "data:aerodynamics:horizontal_tail:" + ls_tag + ":CL0"] = d_thrust_d_cl_h
        jacobian["thrust", "data:aerodynamics:horizontal_tail:" + ls_tag + ":CL_alpha"] = (
            d_thrust_d_cl_h * alpha
        )
        jacobian["thrust", "thrust"] = np.cos(alpha)
        d_thrust_d_alpha_vector = (
            (
                -thrust * np.sin(alpha)
                + d_thrust_d_cl_w * cl_alpha_wing
                + d_thrust_d_cl_h * cl_alpha_htp
            )
            * np.pi
            / 180.0
        )
        jacobian["thrust", "alpha"] = d_thrust_d_alpha_vector
        d_thrust_d_delta_m_vector = (
            (
                d_thrust_d_cl_h * cl_delta_m
                - 2.0 * dynamic_pressure * wing_area * cd_delta_m * delta_m
            )
            * np.pi
            / 180.0
        )
        jacobian["thrust", "delta_m"] = d_thrust_d_delta_m_vector
        if self.options["flaps_position"] == "takeoff":
            jacobian["thrust", "data:aerodynamics:flaps:takeoff:CD"] = -dynamic_pressure * wing_area
        if self.options["flaps_position"] == "landing":
            jacobian["thrust", "data:aerodynamics:flaps:landing:CD"] = -dynamic_pressure * wing_area

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        d_vx_dt = inputs["d_vx_dt"]
        mass = inputs["mass"]
        gamma = inputs["gamma"] * np.pi / 180.0
        true_airspeed = inputs["true_airspeed"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        wing_area = inputs["data:geometry:wing:area"]

        cd0 = inputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0"]
        cl0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CL_alpha"]
        coeff_k_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":induced_drag_coefficient"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:" + ls_tag + ":CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:" + ls_tag + ":CL_alpha"]
        coeff_k_htp = inputs[
            "data:aerodynamics:horizontal_tail:" + ls_tag + ":induced_drag_coefficient"
        ]
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

        dynamic_pressure = 1.0 / 2.0 * inputs["density"] * np.square(true_airspeed)

        cl_wing_flaps = cl0_wing + cl_alpha_wing * alpha + delta_cl_flaps
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m

        cd_tot = (
            cd0
            + delta_cd
            + delta_cd_flaps
            + coeff_k_wing * cl_wing_flaps**2.0
            + coeff_k_htp * cl_htp**2.0
            + cd_delta_m * delta_m**2.0
        )

        residuals["thrust"] = (
            thrust * np.cos(alpha)
            - dynamic_pressure * wing_area * cd_tot
            - mass * g * np.sin(gamma)
            - mass * d_vx_dt
        )
