"""Computation of form drag for wing."""
# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastoad.module_management.service_registry import RegisterSubmodel
from fastoad_cs25.models.aerodynamics.constants import SERVICE_CD0_WING


@RegisterSubmodel(SERVICE_CD0_WING, "fastoad.submodel.aerodynamics.CD0.wing.rta")
class Cd0Wing(om.Group):
    """
    Computation of form drag for wing.

    See :meth:`~fastoad_cs25.models.aerodynamics.components.utils.cd0_lifting_surface` for
    used method.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_subsystem(
            "plate_friction_coeff_" + ls_tag,
            _FlatPlateFrictionDragCoefficient(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "wing_relative_thickness_" + ls_tag,
            _RelativeThicknessContribution(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "wing_camber_" + ls_tag,
            _CamberContribution(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "wing_sweep_25_" + ls_tag,
            _SweepCorrection(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "cd0_wing",
            _Cd0Wing(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )

        self.connect(
            "plate_friction_coeff_" + ls_tag + ".plate_drag_friction_coeff",
            "cd0_wing.plate_drag_friction_coeff",
        )
        self.connect(
            "wing_relative_thickness_" + ls_tag + ".thickness_contribution",
            "cd0_wing.thickness_contribution",
        )
        self.connect(
            "wing_camber_" + ls_tag + ".camber_contribution", "cd0_wing.camber_contribution"
        )
        self.connect("wing_sweep_25_" + ls_tag + ".sweep_correction", "cd0_wing.sweep_correction")


class _FlatPlateFrictionDragCoefficient(om.ExplicitComponent):
    """
    Computation of the flat plate friction drag coefficient.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"
        mach_variable = (
            "data:aerodynamics:aircraft:takeoff:mach"
            if self.options["low_speed_aero"]
            else "data:TLAR:cruise_mach"
        )

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:aerodynamics:wing:" + ls_tag + ":reynolds", val=np.nan)
        self.add_input(mach_variable, val=np.nan)

        self.add_output("plate_drag_friction_coeff")

    def setup_partials(self):
        self.declare_partials("plate_drag_friction_coeff", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"
        mach_variable = (
            "data:aerodynamics:aircraft:takeoff:mach"
            if self.options["low_speed_aero"]
            else "data:TLAR:cruise_mach"
        )

        length = inputs["data:geometry:wing:MAC:length"]
        mach = inputs[mach_variable]
        reynolds = inputs["data:aerodynamics:wing:" + ls_tag + ":reynolds"]

        outputs["plate_drag_friction_coeff"] = 0.455 / (
            (1.0 + 0.144 * mach**2.0) ** 0.65 * np.log10(reynolds * length) ** 2.58
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"
        mach_variable = (
            "data:aerodynamics:aircraft:takeoff:mach"
            if self.options["low_speed_aero"]
            else "data:TLAR:cruise_mach"
        )

        length = inputs["data:geometry:wing:MAC:length"]
        mach = inputs[mach_variable]
        reynolds = inputs["data:aerodynamics:wing:" + ls_tag + ":reynolds"]

        partials["plate_drag_friction_coeff", mach_variable] = (
            -0.085176
            * mach
            / ((1.0 + 0.144 * mach**2.0) ** 1.65 * np.log10(reynolds * length) ** 2.58)
        )

        partials["plate_drag_friction_coeff", "data:geometry:wing:MAC:length"] = -10.095959 / (
            (1.0 + 0.144 * mach**2.0) ** 0.65 * np.log(reynolds * length) ** 3.58 * length
        )

        partials["plate_drag_friction_coeff", "data:aerodynamics:wing:" + ls_tag + ":reynolds"] = (
            -10.095959
            / ((1.0 + 0.144 * mach**2.0) ** 0.65 * np.log(reynolds * length) ** 3.58 * reynolds)
        )


class _RelativeThicknessContribution(om.ExplicitComponent):
    """
    Computation of the contribution from the relative thickness.
    """

    def setup(self):
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)

        self.add_output("thickness_contribution")

    def setup_partials(self):
        self.declare_partials(
            "thickness_contribution", "data:geometry:wing:thickness_ratio", method="exact"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["thickness_contribution"] = (
            4.688 * inputs["data:geometry:wing:thickness_ratio"] ** 2.0
            + 3.146 * inputs["data:geometry:wing:thickness_ratio"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["thickness_contribution", "data:geometry:wing:thickness_ratio"] = (
            9.376 * inputs["data:geometry:wing:thickness_ratio"] + 3.146
        )


class _CamberContribution(om.ExplicitComponent):
    """
    Computation of the contribution from the wing camber.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input(
            "data:aerodynamics:aircraft:" + ls_tag + ":CL",
            shape_by_conn=True,
            val=np.nan,
        )

        self.add_output("camber_contribution")

    def setup_partials(self):
        self.declare_partials("camber_contribution", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cl_sw_frac = (
            inputs["data:aerodynamics:aircraft:" + ls_tag + ":CL"]
            / np.cos(inputs["data:geometry:wing:sweep_25"]) ** 2.0
        )

        outputs["camber_contribution"] = np.median(
            2.859 * cl_sw_frac**3.0 - 1.849 * cl_sw_frac**2.0 + 0.382 * cl_sw_frac + 0.06
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cl = inputs["data:aerodynamics:aircraft:" + ls_tag + ":CL"]
        sw_25 = inputs["data:geometry:wing:sweep_25"]
        denom = np.cos(sw_25) ** 2.0
        cl_sw_frac = (
            inputs["data:aerodynamics:aircraft:" + ls_tag + ":CL"]
            / np.cos(inputs["data:geometry:wing:sweep_25"]) ** 2.0
        )
        camber_contribution = (
            2.859 * cl_sw_frac**3.0 - 1.849 * cl_sw_frac**2.0 + 0.382 * cl_sw_frac + 0.06
        )
        contribution_median = np.median(camber_contribution)

        partials["camber_contribution", "data:aerodynamics:aircraft:" + ls_tag + ":CL"] = np.where(
            camber_contribution == contribution_median,
            (8.577 * cl**2.0 / denom**3.0 - 3.698 * cl / denom**2.0 + 0.382 / denom),
            0.0,
        )

        partials["camber_contribution", "data:geometry:wing:sweep_25"] = np.median(
            17.154 * cl_sw_frac**3.0 * np.tan(sw_25)
            - 7.396 * cl_sw_frac**2.0 * np.tan(sw_25)
            + 0.764 * cl_sw_frac * np.tan(sw_25)
        )


class _SweepCorrection(om.ExplicitComponent):
    """
    Computation of the correction factor from the wing sweep angle.
    """

    def setup(self):
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("sweep_correction")

    def setup_partials(self):
        self.declare_partials("sweep_correction", "data:geometry:wing:sweep_25", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["sweep_correction"] = (
            1.0
            - 0.000178 * inputs["data:geometry:wing:sweep_25"] ** 2.0
            - 0.0065 * inputs["data:geometry:wing:sweep_25"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["sweep_correction", "data:geometry:wing:sweep_25"] = (
            -0.000356 * inputs["data:geometry:wing:sweep_25"] - 0.0065
        )


class _Cd0Wing(om.ExplicitComponent):
    """
    Computation of form drag for wing.

    See :meth:`~fastoad_cs25.models.aerodynamics.components.utils.cd0_lifting_surface` for
    used method.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("plate_drag_friction_coeff", val=np.nan)
        self.add_input("thickness_contribution", val=np.nan)
        self.add_input("camber_contribution", val=np.nan)
        self.add_input("sweep_correction", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:wetted_area", val=np.nan, units="m**2")

        self.add_output("data:aerodynamics:wing:" + ls_tag + ":CD0")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        thickness_contribution = inputs["thickness_contribution"]
        camber_contribution = inputs["camber_contribution"]
        sweep_correction = inputs["sweep_correction"]
        cf = inputs["plate_drag_friction_coeff"]
        wet_area = inputs["data:geometry:wing:wetted_area"]
        wing_area = inputs["data:geometry:wing:area"]

        cd0_wing = (
            ((thickness_contribution + camber_contribution) * sweep_correction + 0.04 + 1.0)
            * cf
            * wet_area
            / wing_area
        )

        outputs["data:aerodynamics:wing:" + ls_tag + ":CD0"] = cd0_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        thickness_contribution = inputs["thickness_contribution"]
        camber_contribution = inputs["camber_contribution"]
        sweep_correction = inputs["sweep_correction"]
        cf = inputs["plate_drag_friction_coeff"]
        wet_area = inputs["data:geometry:wing:wetted_area"]
        wing_area = inputs["data:geometry:wing:area"]

        partials["data:aerodynamics:wing:" + ls_tag + ":CD0", "thickness_contribution"] = (
            sweep_correction * cf * wet_area / wing_area
        )

        partials["data:aerodynamics:wing:" + ls_tag + ":CD0", "camber_contribution"] = (
            sweep_correction * cf * wet_area / wing_area
        )

        partials["data:aerodynamics:wing:" + ls_tag + ":CD0", "sweep_correction"] = (
            (thickness_contribution + camber_contribution) * cf * wet_area / wing_area
        )

        partials["data:aerodynamics:wing:" + ls_tag + ":CD0", "plate_drag_friction_coeff"] = (
            ((thickness_contribution + camber_contribution) * sweep_correction + 0.04 + 1.0)
            * wet_area
            / wing_area
        )

        partials["data:aerodynamics:wing:" + ls_tag + ":CD0", "data:geometry:wing:wetted_area"] = (
            ((thickness_contribution + camber_contribution) * sweep_correction + 0.04 + 1.0)
            * cf
            / wing_area
        )

        partials["data:aerodynamics:wing:" + ls_tag + ":CD0", "data:geometry:wing:area"] = -(
            ((thickness_contribution + camber_contribution) * sweep_correction + 0.04 + 1.0)
            * cf
            * wet_area
            / wing_area**2.0
        )
