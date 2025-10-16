# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class AeroApproximation(om.Group):
    """
    Computation of the CL_ref and CD_ind based on an elliptic distribution assumption.
    """

    def setup(self):
        self.add_subsystem(
            "cl_ref",
            ClRef(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "induced_drag_ratio",
            InducedDragCoefficient(),
            promotes=["data:*"],
        )


class ClRef(om.ExplicitComponent):
    """Computation of the cl_ref based on an elliptic distribution assumption."""

    def setup(self):
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:b_50", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:aerodynamics:wing:low_speed:CL_ref", val=0.67888)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        s_w = inputs["data:geometry:wing:area"]
        b = inputs["data:geometry:wing:b_50"]
        c_r = inputs["data:geometry:wing:root:chord"]
        c_t = inputs["data:geometry:wing:tip:chord"]

        outputs["data:aerodynamics:wing:low_speed:CL_ref"] = (
            b * (3.0 * c_t + 5.0 * c_r) / (12.0 * s_w)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        s_w = inputs["data:geometry:wing:area"]
        b = inputs["data:geometry:wing:b_50"]
        c_r = inputs["data:geometry:wing:root:chord"]
        c_t = inputs["data:geometry:wing:tip:chord"]

        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:area"] = (
            -b * (3.0 * c_t + 5.0 * c_r) / (12.0 * s_w**2.0)
        )

        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:root:chord"] = (
            5.0 * b / (12.0 * s_w)
        )

        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:tip:chord"] = b / (
            4.0 * s_w
        )

        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:b_50"] = (
            3.0 * c_t + 5.0 * c_r
        ) / (12.0 * s_w)


class InducedDragCoefficient(om.ExplicitComponent):
    """Computation of the induced drag coefficient in cruise."""

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:oswald_coefficient", val=np.nan)

        self.add_output(
            "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", val=0.08234
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"] = (
            np.pi
            * inputs["data:geometry:horizontal_tail:aspect_ratio"]
            * inputs["data:aerodynamics:aircraft:cruise:oswald_coefficient"]
        ) ** -1.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient",
            "data:geometry:horizontal_tail:aspect_ratio",
        ] = -(
            (
                np.pi
                * inputs["data:geometry:horizontal_tail:aspect_ratio"] ** 2.0
                * inputs["data:aerodynamics:aircraft:cruise:oswald_coefficient"]
            )
            ** -1.0
        )

        partials[
            "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient",
            "data:aerodynamics:aircraft:cruise:oswald_coefficient",
        ] = -(
            (
                np.pi
                * inputs["data:geometry:horizontal_tail:aspect_ratio"]
                * inputs["data:aerodynamics:aircraft:cruise:oswald_coefficient"] ** 2.0
            )
            ** -1.0
        )
