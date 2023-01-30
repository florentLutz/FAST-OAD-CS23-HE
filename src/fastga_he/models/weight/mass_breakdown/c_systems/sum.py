"""Computation of the systems mass."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

import fastoad.api as oad

from fastga.models.weight.mass_breakdown.c_systems.constants import (
    SUBMODEL_POWER_SYSTEM_MASS,
    SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS,
    SUBMODEL_AVIONICS_SYSTEM_MASS,
    SUBMODEL_RECORDING_SYSTEM_MASS,
)

from fastga.models.weight.mass_breakdown.constants import SUBMODEL_SYSTEMS_MASS


@oad.RegisterSubmodel(SUBMODEL_SYSTEMS_MASS, "fastga_he.submodel.weight.mass.systems.weight_nan")
class SystemsWeight(om.Group):
    """Computes mass of systems by summing the contribution of each systems, sets the default
    value at nan contrarily to the legacy submodel."""

    def setup(self):
        self.add_subsystem(
            "navigation_systems_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_AVIONICS_SYSTEM_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "power_systems_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_POWER_SYSTEM_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "life_support_systems_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "recording_systems_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_RECORDING_SYSTEM_MASS),
            promotes=["*"],
        )

        self.add_subsystem("systems_weight_sum", SystemsWeightSum(), promotes=["*"])


class SystemsWeightSum(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:weight:systems:power:electric_systems:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:power:hydraulic_systems:mass", val=np.nan, units="kg")
        self.add_input(
            "data:weight:systems:life_support:air_conditioning:mass", val=np.nan, units="kg"
        )
        self.add_input("data:weight:systems:life_support:insulation:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:life_support:de_icing:mass", val=np.nan, units="kg")
        self.add_input(
            "data:weight:systems:life_support:internal_lighting:mass", val=np.nan, units="kg"
        )
        self.add_input(
            "data:weight:systems:life_support:seat_installation:mass", val=np.nan, units="kg"
        )
        self.add_input("data:weight:systems:life_support:fixed_oxygen:mass", val=np.nan, units="kg")
        self.add_input(
            "data:weight:systems:life_support:security_kits:mass", val=np.nan, units="kg"
        )
        self.add_input("data:weight:systems:avionics:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:recording:mass", val=np.nan, units="kg")

        self.add_output(
            "data:weight:systems:mass", units="kg", desc="Mass of aircraft systems", val=0.0
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:weight:systems:mass"] = np.sum(inputs.values())
