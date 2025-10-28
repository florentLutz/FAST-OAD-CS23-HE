# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastga.models.aerodynamics.constants import SPAN_MESH_POINT


class AeroApproximation(om.Group):
    """
    Computation of the CL_ref and CD_ind based on an elliptic distribution assumption.
    """

    def setup(self):
        self.add_subsystem(
            "cl_ref",
            ClRef(),
            promotes=["*"],
        )
        self.add_subsystem(
            "induced_drag_ratio",
            InducedDragCoefficient(),
            promotes=["*"],
        )
        self.add_subsystem(
            "wing_low_speed_vector",
            WingLowSpeedVectors(),
            promotes=["*"],
        )


class ClRef(om.ExplicitComponent):
    """Computation of the cl_ref based on an elliptic distribution assumption."""

    def setup(self):
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:b_50", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:aerodynamics:wing:low_speed:CL_ref", val=0.67888)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        s_w = inputs["data:geometry:wing:area"]
        b = inputs["data:geometry:wing:b_50"]
        c_r = inputs["data:geometry:wing:root:chord"]
        c_t = inputs["data:geometry:wing:tip:chord"]
        y_kink = inputs["data:geometry:wing:kink:y"]
        integral_before_kink = (3.0 * y_kink * b**2.0 - 4.0 * y_kink**3.0) * c_r / (3.0 * b**2.0)
        integral_after_kink = (
            (4.0 * y_kink**2.0 - 4.0 * y_kink * b + b**2.0)
            / (24.0 * b**3.0)
            * (
                (12.0 * y_kink**2.0 + 12.0 * y_kink * b + 3.0 * b**2.0) * c_t
                + (-12.0 * y_kink**2.0 - 4.0 * y_kink * b + 5.0 * b**2.0) * c_r
            )
        )

        outputs["data:aerodynamics:wing:low_speed:CL_ref"] = (
            2.0 / s_w * (integral_before_kink + integral_after_kink)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        s_w = inputs["data:geometry:wing:area"]
        b = inputs["data:geometry:wing:b_50"]
        c_r = inputs["data:geometry:wing:root:chord"]
        c_t = inputs["data:geometry:wing:tip:chord"]
        y_kink = inputs["data:geometry:wing:kink:y"]

        integral_before_kink = (3.0 * y_kink * b**2.0 - 4.0 * y_kink**3.0) * c_r / (3.0 * b**2.0)
        integral_after_kink = (
            (4.0 * y_kink**2.0 - 4.0 * y_kink * b + b**2.0)
            / (24.0 * b**3.0)
            * (
                (12.0 * y_kink**2.0 + 12.0 * y_kink * b + 3.0 * b**2.0) * c_t
                + (-12.0 * y_kink**2.0 - 4.0 * y_kink * b + 5.0 * b**2.0) * c_r
            )
        )

        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:area"] = (
            -2.0 / s_w**2.0 * (integral_before_kink + integral_after_kink)
        )

        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:root:chord"] = (
            2.0
            / s_w
            * (
                (3.0 * y_kink * b**2.0 - 4.0 * y_kink**3.0) / (3.0 * b**2.0)
                + (4.0 * y_kink**2.0 - 4.0 * y_kink * b + b**2.0)
                / (24.0 * b**3.0)
                * (-12.0 * y_kink**2.0 - 4.0 * y_kink * b + 5.0 * b**2.0)
            )
        )

        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:tip:chord"] = (
            2.0
            / s_w
            * (
                (4.0 * y_kink**2.0 - 4.0 * y_kink * b + b**2.0)
                / (24.0 * b**3.0)
                * (12.0 * y_kink**2.0 + 12.0 * y_kink * b + 3.0 * b**2.0)
            )
        )

        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:b_50"] = (
            2.0
            / s_w
            * (
                8.0 * c_r * y_kink**3.0 / (3.0 * b**3.0)
                + (
                    (
                        (-4.0 * y_kink + 2.0 * b)
                        * (
                            (12.0 * y_kink**2.0 + 12.0 * y_kink * b + 3.0 * b**2.0) * c_t
                            + (-12.0 * y_kink**2.0 - 4.0 * y_kink * b + 5.0 * b**2.0) * c_r
                        )
                        + (4.0 * y_kink**2.0 - 4.0 * y_kink * b + b**2.0)
                        * ((12.0 * y_kink + 6.0 * b) * c_t + (-4.0 * y_kink + 10.0 * b) * c_r)
                        - 3.0
                        * (4.0 * y_kink**2.0 - 4.0 * y_kink * b + b**2.0)
                        * (
                            (12.0 * y_kink**2.0 + 12.0 * y_kink * b + 3.0 * b**2.0) * c_t
                            + (-12.0 * y_kink**2.0 - 4.0 * y_kink * b + 5.0 * b**2.0) * c_r
                        )
                        / b
                    )
                    / (24.0 * b**3.0)
                )
            )
        )
        partials["data:aerodynamics:wing:low_speed:CL_ref", "data:geometry:wing:kink:y"] = (
            2.0
            / s_w
            * (
                (3.0 * b**2.0 - 12.0 * y_kink**2.0) * c_r / (3.0 * b**2.0)
                + (
                    (24.0 * y_kink**3.0 - 6.0 * y_kink * b**2.0) * c_t
                    + (
                        -24.0 * y_kink**3.0
                        + 12.0 * y_kink**2.0 * b
                        + 6.0 * y_kink * b**2.0
                        - 3.0 * b**3.0
                    )
                    * c_r
                )
                / (3.0 * b**3.0)
            )
        )


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


class WingLowSpeedVectors(om.ExplicitComponent):
    """
    Defining the low speed vectors for other computations based on an elliptic distribution
    assumption.
    """

    def setup(self):
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:span_ratio", val=np.nan)
        self.add_input("data:geometry:wing:b_50", val=np.nan, units="m")

        self.add_output(
            "data:aerodynamics:wing:low_speed:Y_vector",
            val=np.linspace(0.0, 5.0, SPAN_MESH_POINT),
            units="m",
        )
        self.add_output(
            "data:aerodynamics:wing:low_speed:chord_vector",
            val=np.linspace(2.0, 1.0, SPAN_MESH_POINT),
            units="m",
        )
        self.add_output(
            "data:aerodynamics:wing:low_speed:CL_vector",
            val=np.full(SPAN_MESH_POINT, 0.5),
        )

    def setup_partials(self):
        self.declare_partials(
            "data:aerodynamics:wing:low_speed:Y_vector",
            "data:geometry:wing:b_50",
            val=np.linspace(0, 0.5, SPAN_MESH_POINT),
        )
        self.declare_partials(
            "data:aerodynamics:wing:low_speed:chord_vector",
            ["data:geometry:wing:tip:chord", "data:geometry:wing:root:chord"],
            method="exact",
            rows=np.arange(SPAN_MESH_POINT),
            cols=np.zeros(SPAN_MESH_POINT),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b = inputs["data:geometry:wing:b_50"]
        c_r = inputs["data:geometry:wing:root:chord"]
        c_t = inputs["data:geometry:wing:tip:chord"]
        before_kink_point = int(inputs["data:geometry:wing:kink:span_ratio"] * 50)

        outputs["data:aerodynamics:wing:low_speed:Y_vector"] = np.linspace(
            0, b / 2.0, SPAN_MESH_POINT
        )
        outputs["data:aerodynamics:wing:low_speed:chord_vector"] = np.append(
            np.full(before_kink_point, c_r),
            np.linspace(c_r, c_t, SPAN_MESH_POINT - before_kink_point),
        )
        outputs["data:aerodynamics:wing:low_speed:CL_vector"] = (
            1.0 - np.linspace(0, 1.0, SPAN_MESH_POINT) ** 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        before_kink_point = int(inputs["data:geometry:wing:kink:span_ratio"] * 50)

        partials[
            "data:aerodynamics:wing:low_speed:chord_vector", "data:geometry:wing:tip:chord"
        ] = np.append(
            np.zeros(before_kink_point), np.linspace(0.0, 1.0, SPAN_MESH_POINT - before_kink_point)
        )

        partials[
            "data:aerodynamics:wing:low_speed:chord_vector", "data:geometry:wing:root:chord"
        ] = np.append(
            np.ones(before_kink_point), np.linspace(1.0, 0.0, SPAN_MESH_POINT - before_kink_point)
        )
