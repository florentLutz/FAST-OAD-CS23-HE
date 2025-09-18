"""
Estimation of center of gravity ratio with aft
"""
#  This file is part of FAST-OAD_CS25
#  Copyright (C) 2022 ONERA & ISAE-SUPAERO
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

from fastoad_cs25.models.weight.cg.constants import SERVICE_EMPTY_AIRCRAFT_CG


@RegisterSubmodel(SERVICE_EMPTY_AIRCRAFT_CG, "fastoad.submodel.weight.cg.empty_aircraft.rta")
class ComputeCGRatioAft(om.Group):
    def setup(self):
        self.add_subsystem("cg_all", ComputeCG(), promotes=["*"])
        self.add_subsystem("cg_ratio", CGRatio(), promotes=["*"])


class ComputeCG(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "cg_names",
            default=[
                "data:weight:airframe:wing:CG:x",
                "data:weight:airframe:fuselage:CG:x",
                "data:weight:airframe:horizontal_tail:CG:x",
                "data:weight:airframe:vertical_tail:CG:x",
                "data:weight:airframe:landing_gear:main:CG:x",
                "data:weight:airframe:landing_gear:front:CG:x",  # paint
                "data:propulsion:he_power_train:CG:x",
                "data:weight:systems:auxiliary_power_unit:CG:x",
                "data:weight:systems:electric_systems:electric_generation:CG:x",  # x electric system
                "data:weight:systems:electric_systems:electric_common_installation:CG:x",  # x
                "data:weight:systems:hydraulic_systems:CG:x",
                "data:weight:systems:fire_protection:CG:x",
                "data:weight:systems:flight_furnishing:CG:x",
                "data:weight:systems:automatic_flight_system:CG:x",
                "data:weight:systems:communications:CG:x",
                "data:weight:systems:ECS:CG:x",
                "data:weight:systems:de-icing:CG:x",
                "data:weight:systems:navigation:CG:x",
                "data:weight:systems:flight_controls:CG:x",
                "data:weight:furniture:furnishing:CG:x",
                "data:weight:furniture:water:CG:x",
                "data:weight:furniture:interior_integration:CG:x",
                "data:weight:furniture:insulation:CG:x",
                "data:weight:furniture:cabin_lighting:CG:x",
                "data:weight:furniture:seats_crew_accommodation:CG:x",
                "data:weight:furniture:oxygen:CG:x",
                "data:weight:operational:items:passenger_seats:CG:x",
                "data:weight:operational:items:unusable_fuel:CG:x",
                "data:weight:operational:items:documents_toolkit:CG:x",
                "data:weight:operational:items:galley_structure:CG:x",
                "data:weight:operational:equipment:others:CG:x",
            ],
        )

        self.options.declare(
            "mass_names",
            [
                "data:weight:airframe:wing:mass",
                "data:weight:airframe:fuselage:mass",
                "data:weight:airframe:horizontal_tail:mass",
                "data:weight:airframe:vertical_tail:mass",
                "data:weight:airframe:landing_gear:main:mass",
                "data:weight:airframe:landing_gear:front:mass",
                "data:propulsion:he_power_train:mass",
                "data:weight:systems:auxiliary_power_unit:mass",
                "data:weight:systems:electric_systems:electric_generation:mass",
                "data:weight:systems:electric_systems:electric_common_installation:mass",
                "data:weight:systems:hydraulic_systems:mass",
                "data:weight:systems:fire_protection:mass",
                "data:weight:systems:flight_furnishing:mass",
                "data:weight:systems:automatic_flight_system:mass",
                "data:weight:systems:communications:mass",
                "data:weight:systems:ECS:mass",
                "data:weight:systems:de-icing:mass",
                "data:weight:systems:navigation:mass",
                "data:weight:systems:flight_controls:mass",
                "data:weight:furniture:furnishing:mass",
                "data:weight:furniture:water:mass",
                "data:weight:furniture:interior_integration:mass",
                "data:weight:furniture:insulation:mass",
                "data:weight:furniture:cabin_lighting:mass",
                "data:weight:furniture:seats_crew_accommodation:mass",
                "data:weight:furniture:oxygen:mass",
                "data:weight:operational:items:passenger_seats:mass",
                "data:weight:operational:items:unusable_fuel:mass",
                "data:weight:operational:items:documents_toolkit:mass",
                "data:weight:operational:items:galley_structure:mass",
                "data:weight:operational:equipment:others:mass",
            ],
        )

    def setup(self):
        for cg_name in self.options["cg_names"]:
            self.add_input(cg_name, val=np.nan, units="m")
        for mass_name in self.options["mass_names"]:
            self.add_input(mass_name, val=np.nan, units="kg")

        self.add_output("data:weight:aircraft_empty:mass", units="kg")
        self.add_output("data:weight:aircraft_empty:CG:x", units="m")

    def setup_partials(self):
        self.declare_partials("data:weight:aircraft_empty:mass", "*", method="fd")
        self.declare_partials("data:weight:aircraft_empty:CG:x", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cgs = [inputs[cg_name][0] for cg_name in self.options["cg_names"]]
        masses = [inputs[mass_name][0] for mass_name in self.options["mass_names"]]

        weight_moment = np.dot(cgs, masses)
        outputs["data:weight:aircraft_empty:mass"] = np.sum(masses)
        outputs["data:weight:aircraft_empty:CG:x"] = (
            weight_moment / outputs["data:weight:aircraft_empty:mass"]
        )


class CGRatio(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:weight:aircraft:empty:CG:MAC_position")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x_cg_all = inputs["data:weight:aircraft_empty:CG:x"]
        wing_position = inputs["data:geometry:wing:MAC:at25percent:x"]
        mac = inputs["data:geometry:wing:MAC:length"]

        outputs["data:weight:aircraft:empty:CG:MAC_position"] = (
            x_cg_all - wing_position + 0.25 * mac
        ) / mac
