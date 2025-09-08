"""
Python module for airframe mass calculation,
part of the Operating Empty Weight (OEW) estimation.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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
import fastoad.api as oad
import openmdao.api as om
from fastga.models.weight.mass_breakdown.a_airframe.constants import (
    SERVICE_WING_MASS,
    SERVICE_FUSELAGE_MASS,
    SERVICE_TAIL_MASS,
    SERVICE_FLIGHT_CONTROLS_MASS,
    SERVICE_LANDING_GEAR_MASS,
    SERVICE_PAINT_MASS,
)
from fastga.models.weight.mass_breakdown.constants import SERVICE_AIRFRAME_MASS


@oad.RegisterSubmodel(SERVICE_AIRFRAME_MASS, "fastga.submodel.weight.mass.airframe.weight_nan")
class AirframeWeight(om.Group):
    """Computes mass of airframe."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "wing_weight", oad.RegisterSubmodel.get_submodel(SERVICE_WING_MASS), promotes=["*"]
        )
        self.add_subsystem(
            "fuselage_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_FUSELAGE_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_TAIL_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "flight_controls_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_FLIGHT_CONTROLS_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "landing_gear_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_LANDING_GEAR_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "paint_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_PAINT_MASS),
            promotes=["*"],
        )

        self.add_subsystem("airframe_weight_sum", AirframeWeightSum(), promotes=["*"])


class AirframeWeightSum(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:airframe:wing:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:horizontal_tail:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:vertical_tail:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:flight_controls:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:main:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:front:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:paint:mass", val=np.nan, units="kg")

        self.add_output(
            "data:weight:airframe:mass", units="kg", desc="Mass of the airframe", val=0.0
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:weight:airframe:mass"] = np.sum(inputs.values())
