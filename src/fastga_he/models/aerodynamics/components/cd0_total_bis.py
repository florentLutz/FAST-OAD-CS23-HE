"""Sum of form drags from aircraft components."""
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

from fastoad_cs25.models.aerodynamics.constants import SERVICE_CD0_SUM


@RegisterSubmodel(SERVICE_CD0_SUM, "fastoad.submodel.aerodynamics.CD0.sum.as_float")
class Cd0Total(om.ExplicitComponent):
    """Computes the sum of form drags from aircraft components."""

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"
        cd0_shape = (
            "data:aerodynamics:wing:low_speed:CD0"
            if self.options["low_speed_aero"]
            else "data:aerodynamics:fuselage:cruise:CD0"
        )

        self.add_input("data:geometry:aircraft:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:propulsion:L1_engine:hpc:hpc_pressure_ratio", val=np.nan)
        self.add_input("data:propulsion:L1_engine:lpc:lpc_pressure_ratio", val=np.nan)
        self.add_input(
            "data:propulsion:L1_engine:turbine_inlet_temperature", val=np.nan, units="degK"
        )
        self.add_input("data:aerodynamics:wing:" + ls_tag + ":CD0", shape_by_conn=True, val=np.nan)
        self.add_input(
            "data:aerodynamics:fuselage:" + ls_tag + ":CD0",
            shape_by_conn=True,
            val=np.nan,
        )
        self.add_input("data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0", val=np.nan)
        self.add_input("data:aerodynamics:vertical_tail:" + ls_tag + ":CD0", val=np.nan)
        self.add_input("data:aerodynamics:nacelles:" + ls_tag + ":CD0", val=np.nan)

        self.add_output("data:aerodynamics:aircraft:" + ls_tag + ":CD0")
        self.add_output(
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean",
            copy_shape=cd0_shape,
        )
        self.add_output(
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            copy_shape=cd0_shape,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        wet_area_total = inputs["data:geometry:aircraft:wetted_area"]
        cd0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CD0"]
        cd0_fus = inputs["data:aerodynamics:fuselage:" + ls_tag + ":CD0"]
        cd0_ht = inputs["data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0"]
        cd0_vt = inputs["data:aerodynamics:vertical_tail:" + ls_tag + ":CD0"]
        cd0_nac = inputs["data:aerodynamics:nacelles:" + ls_tag + ":CD0"]

        k_parasite = (
            -2.39 * pow(10, -12) * wet_area_total**3.0
            + 2.58 * pow(10, -8) * wet_area_total**2.0
            - 0.89 * pow(10, -4) * wet_area_total
            + 0.163
        )

        cd0_total_clean = cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac
        cd0_total = cd0_total_clean * (1.0 + k_parasite)

        outputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0"] = np.median(cd0_total)
        outputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean"] = cd0_total_clean
        outputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic"] = (
            cd0_total - cd0_total_clean
        )
