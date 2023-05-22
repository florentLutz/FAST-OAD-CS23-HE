# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om
from scipy.constants import g
from stdatm import Atmosphere


class EquilibriumDeltaM(om.ImplicitComponent):
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

        self.add_input("x_cg", val=np.full(number_of_points, 5.0), units="m")

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )

        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:fuselage:cm_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")

        if self.options["flaps_position"] == "takeoff":
            self.add_input("data:aerodynamics:flaps:takeoff:CM", val=np.nan)
        if self.options["flaps_position"] == "landing":
            self.add_input("data:aerodynamics:flaps:landing:CM", val=np.nan)

        self.add_input("delta_Cl", val=np.full(number_of_points, 0.0))
        self.add_input("delta_Cm", val=np.full(number_of_points, 0.0))

        self.add_input("alpha", val=np.full(number_of_points, np.nan), units="deg")
        self.add_output("delta_m", val=np.full(number_of_points, -5.0), units="deg")

        self.declare_partials(
            of="delta_m",
            wrt=[
                "x_cg",
                "data:geometry:wing:MAC:length",
                "data:geometry:wing:MAC:at25percent:x",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:aerodynamics:fuselage:cm_alpha",
                "data:aerodynamics:wing:cruise:CL0_clean",
                "data:aerodynamics:wing:cruise:CL_alpha",
                "data:aerodynamics:wing:cruise:CM0_clean",
                "data:aerodynamics:horizontal_tail:cruise:CL0",
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha",
                "data:aerodynamics:elevator:low_speed:CL_delta",
                "delta_Cl",
                "delta_Cm",
                "alpha",
                "delta_m",
            ],
            method="exact",
        )
        if self.options["flaps_position"] == "takeoff":
            self.declare_partials(
                of="delta_m", wrt="data:aerodynamics:flaps:takeoff:CM", method="exact"
            )
        if self.options["flaps_position"] == "landing":
            self.declare_partials(
                of="delta_m", wrt="data:aerodynamics:flaps:landing:CM", method="exact"
            )

    def linearize(self, inputs, outputs, partials):

        number_of_points = self.options["number_of_points"]

        x_cg = inputs["x_cg"]

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        x_htp = x_wing + inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        cm_alpha_fus = inputs["data:aerodynamics:fuselage:cm_alpha"]
        cm0_wing = inputs["data:aerodynamics:wing:cruise:CM0_clean"]
        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        cl_delta_m = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]

        alpha = inputs["alpha"] * np.pi / 180.0
        delta_m = outputs["delta_m"] * np.pi / 180.0

        if self.options["flaps_position"] == "takeoff":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:takeoff:CL"]
            delta_cm_flaps = inputs["data:aerodynamics:flaps:takeoff:CM"]
        elif self.options["flaps_position"] == "landing":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
            delta_cm_flaps = inputs["data:aerodynamics:flaps:landing:CM"]
        else:  # Cruise conditions
            delta_cl_flaps = 0.0
            delta_cm_flaps = 0.0

        delta_cl = inputs["delta_Cl"]
        delta_cm = inputs["delta_Cm"]

        cl_wing = cl0_wing + cl_alpha_wing * alpha + delta_cl + delta_cl_flaps
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m

        # ------------------ Derivatives wrt delta_m residuals ------------------ #

        partials["delta_m", "data:aerodynamics:wing:cruise:CM0_clean"] = l0_wing * np.ones(
            number_of_points
        )
        partials["delta_m", "data:aerodynamics:fuselage:cm_alpha"] = l0_wing * alpha
        partials["delta_m", "data:aerodynamics:wing:cruise:CL0_clean"] = x_cg - x_wing
        partials["delta_m", "data:aerodynamics:wing:cruise:CM0_clean"] = l0_wing * np.ones(
            number_of_points
        )
        partials["delta_m", "data:aerodynamics:horizontal_tail:cruise:CL0"] = (
            x_cg - x_htp
        ) * np.ones(number_of_points)
        partials["delta_m", "delta_Cl"] = np.diag(x_cg - x_wing)
        partials["delta_m", "delta_Cm"] = np.diag(l0_wing)
        d_delta_m_d_alpha = (
            (x_cg - x_wing) * cl_alpha_wing + (x_cg - x_htp) * cl_alpha_htp + cm_alpha_fus
        )
        partials["delta_m", "alpha"] = np.diag(d_delta_m_d_alpha) * np.pi / 180.0
        partials["delta_m", "data:aerodynamics:horizontal_tail:cruise:CL_alpha"] = alpha * (
            x_cg - x_htp
        )
        partials["delta_m", "data:aerodynamics:elevator:low_speed:CL_delta"] = (
            x_cg - x_htp
        ) * delta_m
        partials["delta_m", "delta_m"] = (
            np.eye(number_of_points) * (x_cg - x_htp) * cl_delta_m * np.pi / 180.0
        )
        partials["delta_m", "x_cg"] = (cl_wing + cl_htp) * np.eye(number_of_points)
        partials["delta_m", "data:geometry:wing:MAC:at25percent:x"] = -(cl_wing + cl_htp) * np.ones(
            number_of_points
        )
        partials[
            "delta_m", "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"
        ] = -cl_htp
        partials["delta_m", "data:geometry:wing:MAC:length"] = (
            cm_alpha_fus * alpha + cm0_wing + delta_cm + delta_cm_flaps
        )
        partials["delta_m", "data:aerodynamics:wing:cruise:CL_alpha"] = (x_cg - x_wing) * alpha
        if self.options["flaps_position"] == "takeoff":
            partials["delta_m", "data:aerodynamics:flaps:takeoff:CM"] = l0_wing * np.ones(
                number_of_points
            )
        if self.options["flaps_position"] == "landing":
            partials["delta_m", "data:aerodynamics:flaps:landing:CM"] = l0_wing * np.ones(
                number_of_points
            )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        x_cg = inputs["x_cg"]

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        x_htp = x_wing + inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        cm0_wing = inputs["data:aerodynamics:wing:cruise:CM0_clean"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        cm_alpha_fus = inputs["data:aerodynamics:fuselage:cm_alpha"]
        cl_delta_m = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]

        delta_cl = inputs["delta_Cl"]
        delta_cm = inputs["delta_Cm"]

        if self.options["flaps_position"] == "takeoff":
            delta_cm_flaps = inputs["data:aerodynamics:flaps:takeoff:CM"]
        elif self.options["flaps_position"] == "landing":
            delta_cm_flaps = inputs["data:aerodynamics:flaps:landing:CM"]
        else:  # Cruise conditions
            delta_cm_flaps = 0.0

        alpha = inputs["alpha"] * np.pi / 180.0
        delta_m = outputs["delta_m"] * np.pi / 180.0

        cl_wing_clean = cl0_wing + cl_alpha_wing * alpha
        cl_wing_slip = cl_wing_clean + delta_cl
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m

        residuals["delta_m"] = (
            (x_cg - x_wing) * cl_wing_slip
            + (x_cg - x_htp) * cl_htp
            + (cm0_wing + delta_cm + delta_cm_flaps + cm_alpha_fus * alpha) * l0_wing
        )
