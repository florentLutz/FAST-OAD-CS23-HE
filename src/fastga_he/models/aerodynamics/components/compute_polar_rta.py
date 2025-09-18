"""Computation of CL and CD for whole aircraft."""
# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from fastoad.module_management.service_registry import RegisterSubmodel

from fastoad_cs25.models.aerodynamics.constants import PolarType, SERVICE_POLAR


@RegisterSubmodel(SERVICE_POLAR, "fastoad.submodel.aerodynamics.polar.rta")
class ComputePolar(om.Group):
    """Computation of CL and CD for whole aircraft."""

    def initialize(self):
        self.options.declare("polar_type", default=PolarType.HIGH_SPEED, types=PolarType)

    def setup(self):
        type_tag = polar_type_string(PolarType, self.options["polar_type"])
        mp_tag = "takeoff" if self.options["polar_type"] == PolarType.TAKEOFF else "landing"

        if (
            self.options["polar_type"] == PolarType.TAKEOFF
            or self.options["polar_type"] == PolarType.LANDING
        ):
            self.add_subsystem(
                "polar_cl_" + mp_tag,
                _ComputePolarCL(polar_type=self.options["polar_type"]),
                promotes=["data:*"],
            )

        self.add_subsystem(
            "polar_cd_" + type_tag,
            _ComputePolarCD(polar_type=self.options["polar_type"]),
            promotes=["data:*"],
        )

        if self.options["polar_type"] == PolarType.HIGH_SPEED:
            self.add_subsystem(
                "polar_optimal",
                _ComputePolarOptimal(),
                promotes=["data:*"],
            )


class _ComputePolarCL(om.ExplicitComponent):
    """
    Computation of CL for whole aircraft.
    """

    def initialize(self):
        self.options.declare("polar_type", default=PolarType.HIGH_SPEED, types=PolarType)

    def setup(self):
        mp_tag = "takeoff" if self.options["polar_type"] == PolarType.TAKEOFF else "landing"

        self.add_input(
            "data:aerodynamics:aircraft:low_speed:CL",
            shape_by_conn=True,
            val=np.nan,
        )
        self.add_input("data:aerodynamics:high_lift_devices:" + mp_tag + ":CL", val=np.nan)

        self.add_output(
            "data:aerodynamics:aircraft:" + mp_tag + ":CL",
            copy_shape="data:aerodynamics:aircraft:low_speed:CL",
        )

    def setup_partials(self):
        mp_tag = "takeoff" if self.options["polar_type"] == PolarType.TAKEOFF else "landing"

        self.declare_partials("data:aerodynamics:aircraft:" + mp_tag + ":CL", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mp_tag = "takeoff" if self.options["polar_type"] == PolarType.TAKEOFF else "landing"

        outputs["data:aerodynamics:aircraft:" + mp_tag + ":CL"] = inputs[
            "data:aerodynamics:aircraft:low_speed:CL"
        ] + np.full_like(
            inputs["data:aerodynamics:aircraft:low_speed:CL"],
            inputs["data:aerodynamics:high_lift_devices:" + mp_tag + ":CL"],
        )


class _ComputePolarCD(om.ExplicitComponent):
    """Computation of CD for whole aircraft."""

    def initialize(self):
        self.options.declare("polar_type", default=PolarType.HIGH_SPEED, types=PolarType)

    def setup(self):
        type_tag = polar_type_string(PolarType, self.options["polar_type"])
        hs_tag = "cruise" if self.options["polar_type"] == PolarType.HIGH_SPEED else "low_speed"

        self.add_input("tuning:aerodynamics:aircraft:cruise:CD:k", val=1.0)
        self.add_input("tuning:aerodynamics:aircraft:cruise:CD:offset", val=0.0)
        self.add_input("tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k", val=1.0)
        self.add_input("tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset", val=0.0)
        self.add_input(
            "data:aerodynamics:aircraft:" + type_tag + ":CL",
            shape_by_conn=True,
            val=np.nan,
        )
        self.add_input("data:aerodynamics:aircraft:" + hs_tag + ":CD0", val=np.nan)
        self.add_input(
            "data:aerodynamics:aircraft:" + hs_tag + ":CD:trim",
            shape_by_conn=True,
            val=np.nan,
        )
        self.add_input(
            "data:aerodynamics:aircraft:" + hs_tag + ":induced_drag_coefficient",
            val=np.nan,
        )

        if (
            self.options["polar_type"] == PolarType.TAKEOFF
            or self.options["polar_type"] == PolarType.LANDING
        ):
            self.add_input("data:aerodynamics:high_lift_devices:" + type_tag + ":CD", val=np.nan)

        elif self.options["polar_type"] == PolarType.HIGH_SPEED:
            self.add_input(
                "data:aerodynamics:aircraft:cruise:CD:compressibility",
                shape_by_conn=True,
                val=np.nan,
            )

        self.add_output(
            "data:aerodynamics:aircraft:" + type_tag + ":CD",
            copy_shape="data:aerodynamics:aircraft:" + type_tag + ":CL",
        )

    def setup_partials(self):
        hs_tag = "cruise" if self.options["polar_type"] == PolarType.HIGH_SPEED else "low_speed"
        type_tag = polar_type_string(PolarType, self.options["polar_type"])

        self.declare_partials(
            "*",
            [
                "tuning:aerodynamics:aircraft:cruise:CD:k",
                "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k",
                "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset",
                "tuning:aerodynamics:aircraft:cruise:CD:offset",
                "data:aerodynamics:aircraft:" + type_tag + ":CL",
                "data:aerodynamics:aircraft:" + hs_tag + ":CD0",
                "data:aerodynamics:aircraft:" + hs_tag + ":CD:trim",
                "data:aerodynamics:aircraft:" + hs_tag + ":induced_drag_coefficient",
            ],
            method="exact",
        )

        if (
            self.options["polar_type"] == PolarType.TAKEOFF
            or self.options["polar_type"] == PolarType.LANDING
        ):
            self.declare_partials(
                "*", "data:aerodynamics:high_lift_devices:" + type_tag + ":CD", method="exact"
            )

        elif self.options["polar_type"] == PolarType.HIGH_SPEED:
            self.declare_partials(
                "*", "data:aerodynamics:aircraft:cruise:CD:compressibility", method="exact"
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        hs_tag = "cruise" if self.options["polar_type"] == PolarType.HIGH_SPEED else "low_speed"
        type_tag = polar_type_string(PolarType, self.options["polar_type"])

        cl = inputs["data:aerodynamics:aircraft:" + type_tag + ":CL"]
        k_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:k"]
        offset_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:offset"]
        k_winglet_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k"]
        offset_winglet_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset"]
        cd0 = inputs["data:aerodynamics:aircraft:" + hs_tag + ":CD0"]
        cd_trim = inputs["data:aerodynamics:aircraft:" + hs_tag + ":CD:trim"]
        cd_c = (
            inputs["data:aerodynamics:aircraft:cruise:CD:compressibility"]
            if self.options["polar_type"] == PolarType.HIGH_SPEED
            else np.zeros_like(cl)
        )
        coef_k = inputs["data:aerodynamics:aircraft:" + hs_tag + ":induced_drag_coefficient"]
        delta_cd_hl = 0.0

        if (
            self.options["polar_type"] == PolarType.TAKEOFF
            or self.options["polar_type"] == PolarType.LANDING
        ):
            delta_cd_hl = inputs["data:aerodynamics:high_lift_devices:" + type_tag + ":CD"]

        outputs["data:aerodynamics:aircraft:" + type_tag + ":CD"] = (
            np.full_like(cl, cd0)
            + cd_c
            + cd_trim
            + coef_k * cl**2.0 * k_winglet_cd
            + np.full_like(cl, offset_winglet_cd)
            + np.full_like(cl, delta_cd_hl)
        ) * k_cd + np.full_like(cl, offset_cd)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        hs_tag = "cruise" if self.options["polar_type"] == PolarType.HIGH_SPEED else "low_speed"
        type_tag = polar_type_string(PolarType, self.options["polar_type"])

        cl = inputs["data:aerodynamics:aircraft:" + type_tag + ":CL"]
        k_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:k"]
        k_winglet_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k"]
        offset_winglet_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset"]
        cd0 = inputs["data:aerodynamics:aircraft:" + hs_tag + ":CD0"]
        cd_trim = inputs["data:aerodynamics:aircraft:" + hs_tag + ":CD:trim"]
        cd_c = (
            inputs["data:aerodynamics:aircraft:cruise:CD:compressibility"]
            if self.options["polar_type"] == PolarType.HIGH_SPEED
            else np.zeros_like(cl)
        )
        coef_k = inputs["data:aerodynamics:aircraft:" + hs_tag + ":induced_drag_coefficient"]
        delta_cd_hl = (
            inputs["data:aerodynamics:high_lift_devices:" + type_tag + ":CD"]
            if (
                self.options["polar_type"] == PolarType.TAKEOFF
                or self.options["polar_type"] == PolarType.LANDING
            )
            else 0.0
        )

        if (
            self.options["polar_type"] == PolarType.TAKEOFF
            or self.options["polar_type"] == PolarType.LANDING
        ):
            partials[
                "data:aerodynamics:aircraft:" + type_tag + ":CD",
                "data:aerodynamics:high_lift_devices:" + type_tag + ":CD",
            ] = np.full_like(cl, k_cd)

        elif self.options["polar_type"] == PolarType.HIGH_SPEED:
            partials[
                "data:aerodynamics:aircraft:" + type_tag + ":CD",
                "data:aerodynamics:aircraft:cruise:CD:compressibility",
            ] = np.diag(np.full_like(cl, k_cd))

        partials[
            "data:aerodynamics:aircraft:" + type_tag + ":CD",
            "data:aerodynamics:aircraft:" + hs_tag + ":CD0",
        ] = np.full_like(cl, k_cd)

        partials[
            "data:aerodynamics:aircraft:" + type_tag + ":CD",
            "data:aerodynamics:aircraft:" + hs_tag + ":CD:trim",
        ] = np.diag(np.full_like(cl, k_cd))

        partials[
            "data:aerodynamics:aircraft:" + type_tag + ":CD",
            "data:aerodynamics:aircraft:" + hs_tag + ":induced_drag_coefficient",
        ] = k_cd * cl**2.0 * k_winglet_cd

        partials[
            "data:aerodynamics:aircraft:" + type_tag + ":CD",
            "data:aerodynamics:aircraft:" + type_tag + ":CL",
        ] = np.diag(2.0 * k_cd * coef_k * cl * k_winglet_cd)

        partials[
            "data:aerodynamics:aircraft:" + type_tag + ":CD",
            "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k",
        ] = k_cd * coef_k * cl**2.0

        partials[
            "data:aerodynamics:aircraft:" + type_tag + ":CD",
            "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset",
        ] = np.full_like(cl, k_cd)

        partials[
            "data:aerodynamics:aircraft:" + type_tag + ":CD",
            "tuning:aerodynamics:aircraft:cruise:CD:k",
        ] = (
            np.full_like(cl, cd0)
            + cd_c
            + cd_trim
            + coef_k * cl**2.0 * k_winglet_cd
            + np.full_like(cl, offset_winglet_cd)
            + np.full_like(cl, delta_cd_hl)
        )

        partials[
            "data:aerodynamics:aircraft:" + type_tag + ":CD",
            "tuning:aerodynamics:aircraft:cruise:CD:offset",
        ] = np.ones_like(cl)


class _ComputePolarOptimal(om.ExplicitComponent):
    """
    Computation of optimal CL, CD, L/D for whole aircraft during cruise.
    """

    def setup(self):
        self.add_input(
            "data:aerodynamics:aircraft:cruise:CL",
            shape_by_conn=True,
            val=np.nan,
        )
        self.add_input(
            "data:aerodynamics:aircraft:cruise:CD",
            shape_by_conn=True,
            val=np.nan,
        )

        self.add_output("data:aerodynamics:aircraft:cruise:L_D_max")
        self.add_output("data:aerodynamics:aircraft:cruise:optimal_CL")
        self.add_output("data:aerodynamics:aircraft:cruise:optimal_CD")

    def setup_partials(self):
        self.declare_partials("data:aerodynamics:aircraft:cruise:L_D_max", "*", method="exact")
        self.declare_partials(
            "data:aerodynamics:aircraft:cruise:optimal_CL",
            "data:aerodynamics:aircraft:cruise:CL",
            method="exact",
        )
        self.declare_partials(
            "data:aerodynamics:aircraft:cruise:optimal_CD",
            "data:aerodynamics:aircraft:cruise:CD",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cl = inputs["data:aerodynamics:aircraft:cruise:CL"]
        cd = inputs["data:aerodynamics:aircraft:cruise:CD"]

        lift_drag_ratio = cl / cd
        optimum_index = np.argmax(lift_drag_ratio)

        outputs["data:aerodynamics:aircraft:cruise:optimal_CL"] = cl[optimum_index]
        outputs["data:aerodynamics:aircraft:cruise:optimal_CD"] = cd[optimum_index]
        outputs["data:aerodynamics:aircraft:cruise:L_D_max"] = lift_drag_ratio[optimum_index]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cl = inputs["data:aerodynamics:aircraft:cruise:CL"]
        cd = inputs["data:aerodynamics:aircraft:cruise:CD"]

        lift_drag_ratio = cl / cd

        partials[
            "data:aerodynamics:aircraft:cruise:optimal_CL", "data:aerodynamics:aircraft:cruise:CL"
        ] = np.where(lift_drag_ratio == np.max(lift_drag_ratio), 1.0, 0.0)

        partials[
            "data:aerodynamics:aircraft:cruise:optimal_CD", "data:aerodynamics:aircraft:cruise:CD"
        ] = np.where(lift_drag_ratio == np.max(lift_drag_ratio), 1.0, 0.0)

        partials[
            "data:aerodynamics:aircraft:cruise:L_D_max", "data:aerodynamics:aircraft:cruise:CL"
        ] = np.where(lift_drag_ratio == np.max(lift_drag_ratio), 1.0 / cd, 0.0)

        partials[
            "data:aerodynamics:aircraft:cruise:L_D_max", "data:aerodynamics:aircraft:cruise:CD"
        ] = np.where(lift_drag_ratio == np.max(lift_drag_ratio), -cl / cd**2.0, 0.0)


def polar_type_string(PolarType, option):
    if option == PolarType.LANDING:
        return "landing"
    elif option == PolarType.TAKEOFF:
        return "takeoff"
    elif option == PolarType.HIGH_SPEED:
        return "cruise"
    elif option == PolarType.LOW_SPEED:
        return "low_speed"
    else:
        raise AttributeError(f"Unknown polar type: {option}")
