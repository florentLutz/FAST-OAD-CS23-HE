# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class FlatPlateFrictionDragCoefficient(om.ExplicitComponent):
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
