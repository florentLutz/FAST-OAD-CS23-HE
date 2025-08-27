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
class Cd0Total(om.Group):
    """Computes the sum of form drags from aircraft components."""

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_subsystem(
            "parasitic_factor_" + ls_tag,
            _TotalCd0ParasiticFactor(),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "cd0_aircraft",
            _AircraftCd0(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "total_cd0",
            _Cd0Median(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["data:*"],
        )

        self.connect("parasitic_factor_" + ls_tag + ".k_parasitic", "cd0_aircraft.k_parasitic")


class _TotalCd0ParasiticFactor(om.ExplicitComponent):
    """
    Computes the parasitic factor of the total aircraft form drag.
    """

    def setup(self):
        self.add_input("data:geometry:aircraft:wetted_area", val=np.nan, units="m**2")

        self.add_output("k_parasitic")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wet_area_total = inputs["data:geometry:aircraft:wetted_area"]

        k_parasitic = (
            -2.39 * pow(10, -12) * wet_area_total**3.0
            + 2.58 * pow(10, -8) * wet_area_total**2.0
            - 0.89 * pow(10, -4) * wet_area_total
            + 0.163
        )

        outputs["k_parasitic"] = k_parasitic

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wet_area_total = inputs["data:geometry:aircraft:wetted_area"]

        partials["k_parasitic", "data:geometry:aircraft:wetted_area"] = (
            -7.17 * pow(10, -12) * wet_area_total**2.0
            + 5.16 * pow(10, -8) * wet_area_total
            - 0.89 * pow(10, -4)
        )


class _AircraftCd0(om.ExplicitComponent):
    """
    Computes the clean and parasitic aircraft form drag coefficient.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cd0_shape = "data:aerodynamics:fuselage:" + ls_tag + ":CD0"

        self.add_input("k_parasitic")
        self.add_input("data:aerodynamics:wing:" + ls_tag + ":CD0", shape_by_conn=True, val=np.nan)
        self.add_input(
            "data:aerodynamics:fuselage:" + ls_tag + ":CD0",
            shape_by_conn=True,
            val=np.nan,
        )
        self.add_input("data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0", val=np.nan)
        self.add_input("data:aerodynamics:vertical_tail:" + ls_tag + ":CD0", val=np.nan)
        self.add_input("data:aerodynamics:nacelles:" + ls_tag + ":CD0", val=np.nan)

        self.add_output(
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean",
            copy_shape=cd0_shape,
        )
        self.add_output(
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            copy_shape=cd0_shape,
        )

    def setup_partials(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.declare_partials(
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean",
            [
                "data:aerodynamics:wing:" + ls_tag + ":CD0",
                "data:aerodynamics:fuselage:" + ls_tag + ":CD0",
                "data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0",
                "data:aerodynamics:vertical_tail:" + ls_tag + ":CD0",
                "data:aerodynamics:nacelles:" + ls_tag + ":CD0",
            ],
            val=1.0,
        )
        self.declare_partials(
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            [
                "k_parasitic",
                "data:aerodynamics:wing:" + ls_tag + ":CD0",
                "data:aerodynamics:fuselage:" + ls_tag + ":CD0",
                "data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0",
                "data:aerodynamics:vertical_tail:" + ls_tag + ":CD0",
                "data:aerodynamics:nacelles:" + ls_tag + ":CD0",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        k_parasitic = inputs["k_parasitic"]
        cd0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CD0"]
        cd0_fus = inputs["data:aerodynamics:fuselage:" + ls_tag + ":CD0"]
        cd0_ht = inputs["data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0"]
        cd0_vt = inputs["data:aerodynamics:vertical_tail:" + ls_tag + ":CD0"]
        cd0_nac = inputs["data:aerodynamics:nacelles:" + ls_tag + ":CD0"]

        outputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean"] = (
            cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac
        )
        outputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic"] = k_parasitic * (
            cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        k_parasitic = inputs["k_parasitic"]
        cd0_wing = inputs["data:aerodynamics:wing:" + ls_tag + ":CD0"]
        cd0_fus = inputs["data:aerodynamics:fuselage:" + ls_tag + ":CD0"]
        cd0_ht = inputs["data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0"]
        cd0_vt = inputs["data:aerodynamics:vertical_tail:" + ls_tag + ":CD0"]
        cd0_nac = inputs["data:aerodynamics:nacelles:" + ls_tag + ":CD0"]

        partials["data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic", "k_parasitic"] = (
            cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac
        )

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            "data:aerodynamics:wing:" + ls_tag + ":CD0",
        ] = k_parasitic

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            "data:aerodynamics:fuselage:" + ls_tag + ":CD0",
        ] = k_parasitic

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            "data:aerodynamics:horizontal_tail:" + ls_tag + ":CD0",
        ] = k_parasitic

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            "data:aerodynamics:vertical_tail:" + ls_tag + ":CD0",
        ] = k_parasitic

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            "data:aerodynamics:nacelles:" + ls_tag + ":CD0",
        ] = k_parasitic


class _Cd0Median(om.ExplicitComponent):
    """
    Computes the aircraft form drag obtaining from the median during the whole flight mission.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input(
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean",
            shape_by_conn=True,
            val=np.nan,
        )
        self.add_input(
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
            shape_by_conn=True,
            val=np.nan,
        )

        self.add_output("data:aerodynamics:aircraft:" + ls_tag + ":CD0")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        outputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0"] = np.median(
            inputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean"]
            + inputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        cd0_total = (
            inputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean"]
            + inputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic"]
        )
        cd0_total_median = np.median(
            inputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean"]
            + inputs["data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic"]
        )

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0",
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:clean",
        ] = np.where(cd0_total == cd0_total_median, 1.0, 0.0)

        partials[
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0",
            "data:aerodynamics:aircraft:" + ls_tag + ":CD0:parasitic",
        ] = np.where(cd0_total == cd0_total_median, 1.0, 0.0)
