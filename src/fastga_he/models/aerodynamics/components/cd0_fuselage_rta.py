"""Computation of form drag for fuselage."""
# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastoad_cs25.models.aerodynamics.constants import SERVICE_CD0_FUSELAGE
from fastga_he.models.aerodynamics.components.flat_plate_friction_drag_coeff import (
    FlatPlateFrictionDragCoefficient,
)


@oad.RegisterSubmodel(SERVICE_CD0_FUSELAGE, "fastga_he.submodel.aerodynamics.fuselage.cd0.rta")
class Cd0Fuselage(om.Group):
    """
    Computation of form drag for fuselage.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_subsystem(
            "fuselage_plate_friction_coeff_" + ls_tag,
            FlatPlateFrictionDragCoefficient(),
            promotes=[
                "data:*",
                ("characteristic_length", "data:geometry:fuselage:length"),
                (
                    "characteristic_reynolds",
                    "data:aerodynamics:wing:" + ls_tag + ":reynolds",
                ),  # Yes, from the original model, the wing Reynolds is used.
            ],
        )
        self.add_subsystem(
            "fuselage_friction_drag_" + ls_tag,
            _FuselageFrictionDragContribution(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "fuselage_upsweep_" + ls_tag,
            _FuselageUpsweepContribution(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "cd0_fuselage",
            _Cd0Fuselage(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )

        self.connect(
            "fuselage_plate_friction_coeff_" + ls_tag + ".plate_drag_friction_coeff",
            "fuselage_friction_drag_" + ls_tag + ".plate_drag_friction_coeff",
        )
        self.connect(
            "fuselage_friction_drag_" + ls_tag + ".friction_drag_contribution",
            "cd0_fuselage.friction_drag_contribution",
        )
        self.connect(
            "fuselage_upsweep_" + ls_tag + ".upsweep_contribution",
            "cd0_fuselage.upsweep_contribution",
        )


class _FuselageFrictionDragContribution(om.ExplicitComponent):
    """
    Computation of the contribution from the fuselage friction drag.
    """

    def setup(self):
        self.add_input("plate_drag_friction_coeff", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("friction_drag_contribution", val=0.005)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        wet_area_fus = inputs["data:geometry:fuselage:wetted_area"]
        wing_area = inputs["data:geometry:wing:area"]
        fus_length = inputs["data:geometry:fuselage:length"]
        cf_fus = inputs["plate_drag_friction_coeff"]

        cd0_friction_fus = (
            (0.98 + 0.745 * np.sqrt(height_max * width_max) / fus_length)
            * cf_fus
            * wet_area_fus
            / wing_area
        )

        outputs["friction_drag_contribution"] = cd0_friction_fus

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        wet_area_fus = inputs["data:geometry:fuselage:wetted_area"]
        wing_area = inputs["data:geometry:wing:area"]
        fus_length = inputs["data:geometry:fuselage:length"]
        cf_fus = inputs["plate_drag_friction_coeff"]

        partials["friction_drag_contribution", "data:geometry:fuselage:maximum_height"] = (
            (0.745 * 0.5 * np.sqrt(width_max / height_max) / fus_length)
            * cf_fus
            * wet_area_fus
            / wing_area
        )
        partials["friction_drag_contribution", "data:geometry:fuselage:maximum_width"] = (
            (0.745 * 0.5 * np.sqrt(height_max / width_max) / fus_length)
            * cf_fus
            * wet_area_fus
            / wing_area
        )
        partials["friction_drag_contribution", "data:geometry:fuselage:length"] = (
            (-0.745 * np.sqrt(height_max * width_max) / fus_length**2)
            * cf_fus
            * wet_area_fus
            / wing_area
        )
        partials["friction_drag_contribution", "plate_drag_friction_coeff"] = (
            (0.98 + 0.745 * np.sqrt(height_max * width_max) / fus_length) * wet_area_fus / wing_area
        )
        partials["friction_drag_contribution", "data:geometry:fuselage:wetted_area"] = (
            (0.98 + 0.745 * np.sqrt(height_max * width_max) / fus_length) * cf_fus / wing_area
        )
        partials["friction_drag_contribution", "data:geometry:wing:area"] = -(
            (0.98 + 0.745 * np.sqrt(height_max * width_max) / fus_length)
            * cf_fus
            * wet_area_fus
            / wing_area**2
        )


class _FuselageUpsweepContribution(om.ExplicitComponent):
    """
    Computation of the contribution from the fuselage upsweep.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input(
            "data:aerodynamics:aircraft:" + ls_tag + ":CL",
            shape_by_conn=True,
            val=np.nan,
        )

        self.add_output("upsweep_contribution", val=0.005)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        height_max = inputs["data:geometry:fuselage:maximum_height"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        wing_area = inputs["data:geometry:wing:area"]
        cl = inputs["data:aerodynamics:aircraft:" + ls_tag + ":CL"]

        cd0_upsweep_fus = (
            (0.0029 * cl**2 - 0.0066 * cl + 0.0043)
            * (0.67 * 3.6 * height_max * width_max)
            / wing_area
        )

        outputs["upsweep_contribution"] = np.median(cd0_upsweep_fus)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        height_max = inputs["data:geometry:fuselage:maximum_height"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        wing_area = inputs["data:geometry:wing:area"]
        cl = inputs["data:aerodynamics:aircraft:" + ls_tag + ":CL"]
        cl_median = np.median(cl)

        partials["upsweep_contribution", "data:geometry:fuselage:maximum_height"] = np.median(
            (0.0029 * cl**2 - 0.0066 * cl + 0.0043) * (0.67 * 3.6 * width_max) / wing_area
        )
        partials["upsweep_contribution", "data:geometry:fuselage:maximum_width"] = np.median(
            (0.0029 * cl**2 - 0.0066 * cl + 0.0043) * (0.67 * 3.6 * height_max) / wing_area
        )
        partials["upsweep_contribution", "data:geometry:wing:area"] = -np.median(
            (0.0029 * cl**2 - 0.0066 * cl + 0.0043)
            * (0.67 * 3.6 * height_max * width_max)
            / wing_area**2.0
        )
        if len(cl) % 2 == 1:
            partials["upsweep_contribution", "data:aerodynamics:aircraft:" + ls_tag + ":CL"] = (
                np.where(
                    cl == cl_median,
                    (2.0 * 0.0029 * cl - 0.0066)
                    * (0.67 * 3.6 * height_max * width_max)
                    / wing_area,
                    0,
                )
            )
        else:
            partials["upsweep_contribution", "data:aerodynamics:aircraft:" + ls_tag + ":CL"] = (
                np.where(
                    np.abs(cl - cl_median) == np.min(np.abs(cl - cl_median)),
                    (2.0 * 0.0029 * cl - 0.0066)
                    * (0.67 * 3.6 * height_max * width_max)
                    / wing_area
                    / 2.0,
                    0,
                )
            )


class _Cd0Fuselage(om.ExplicitComponent):
    """
    Computation of form drag for fuselage.

    See :meth:`~fastoad_cs25.models.aerodynamics.components.utils.cd0_lifting_surface` for
    used method.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("friction_drag_contribution", val=np.nan)
        self.add_input("upsweep_contribution", val=np.nan)

        self.add_output("data:aerodynamics:fuselage:" + ls_tag + ":CD0")

    def setup_partials(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.declare_partials(of="data:aerodynamics:fuselage:" + ls_tag + ":CD0", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        outputs["data:aerodynamics:fuselage:" + ls_tag + ":CD0"] = (
            inputs["friction_drag_contribution"] + inputs["upsweep_contribution"]
        )
