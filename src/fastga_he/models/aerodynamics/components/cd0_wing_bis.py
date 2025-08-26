"""Computation of form drag for wing."""
#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2025 ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import openmdao.api as om
from fastoad.module_management.service_registry import RegisterSubmodel
from fastoad_cs25.models.aerodynamics.components.utils.cd0_lifting_surface import (
    LiftingSurfaceGeometry,
    compute_cd0_lifting_surface,
)
from fastoad_cs25.models.aerodynamics.constants import SERVICE_CD0_WING


@RegisterSubmodel(SERVICE_CD0_WING, "fastoad.submodel.aerodynamics.CD0.wing.as_float")
class Cd0Wing(om.ExplicitComponent):
    """
    Computation of form drag for wing.

    See :meth:`~fastoad_cs25.models.aerodynamics.components.utils.cd0_lifting_surface` for
    used method.
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

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:aerodynamics:wing:" + ls_tag + ":reynolds", val=np.nan)
        self.add_input(
            "data:aerodynamics:aircraft:" + ls_tag + ":CL",
            shape_by_conn=True,
            val=np.nan,
        )
        self.add_input(mach_variable, val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_output(
                "data:aerodynamics:wing:low_speed:CD0",
                copy_shape="data:aerodynamics:aircraft:low_speed:CL",
            )
        else:
            self.add_output("data:aerodynamics:wing:cruise:CD0")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"
        mach_variable = (
            "data:aerodynamics:aircraft:takeoff:mach"
            if self.options["low_speed_aero"]
            else "data:TLAR:cruise_mach"
        )

        wing_geometry = LiftingSurfaceGeometry(
            thickness_ratio=inputs["data:geometry:wing:thickness_ratio"],
            MAC_length=inputs["data:geometry:wing:MAC:length"],
            sweep_angle_25=inputs["data:geometry:wing:sweep_25"],
            wet_area=inputs["data:geometry:wing:wetted_area"],
            cambered=True,
            interaction_coeff=0.04,
        )

        wing_area = inputs["data:geometry:wing:area"]
        cl = inputs["data:aerodynamics:aircraft:" + ls_tag + ":CL"]
        mach = inputs[mach_variable]
        reynolds = inputs["data:aerodynamics:wing:" + ls_tag + ":reynolds"]

        cd0_wing = compute_cd0_lifting_surface(wing_geometry, mach, reynolds, wing_area, cl)

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:CD0"] = cd0_wing
        else:
            outputs["data:aerodynamics:wing:cruise:CD0"] = np.median(cd0_wing)
