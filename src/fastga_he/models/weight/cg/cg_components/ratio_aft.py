# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga.models.weight.cg.cg_components.constants import SUBMODEL_AIRCRAFT_X_CG


@oad.RegisterSubmodel(
    SUBMODEL_AIRCRAFT_X_CG, "fastga_he.submodel.weight.cg.aircraft_empty.x.with_propulsion_as_one"
)
class ComputeCG(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "cg_names",
            default=[
                "data:weight:airframe:wing:CG:x",
                "data:weight:airframe:fuselage:CG:x",
                "data:weight:airframe:horizontal_tail:CG:x",
                "data:weight:airframe:vertical_tail:CG:x",
                "data:weight:airframe:flight_controls:CG:x",
                "data:weight:airframe:landing_gear:main:CG:x",
                "data:weight:airframe:landing_gear:front:CG:x",
                "data:weight:propulsion:CG:x",
                "data:weight:systems:power:electric_systems:CG:x",
                "data:weight:systems:power:hydraulic_systems:CG:x",
                "data:weight:systems:life_support:air_conditioning:CG:x",
                "data:weight:systems:avionics:CG:x",
                "data:weight:systems:recording:CG:x",
                "data:weight:furniture:passenger_seats:CG:x",
            ],
        )

        self.options.declare(
            "mass_names",
            [
                "data:weight:airframe:wing:mass",
                "data:weight:airframe:fuselage:mass",
                "data:weight:airframe:horizontal_tail:mass",
                "data:weight:airframe:vertical_tail:mass",
                "data:weight:airframe:flight_controls:mass",
                "data:weight:airframe:landing_gear:main:mass",
                "data:weight:airframe:landing_gear:front:mass",
                "data:weight:propulsion:mass",
                "data:weight:systems:power:electric_systems:mass",
                "data:weight:systems:power:hydraulic_systems:mass",
                "data:weight:systems:life_support:air_conditioning:mass",
                "data:weight:systems:avionics:mass",
                "data:weight:systems:recording:mass",
                "data:weight:furniture:passenger_seats:mass",
            ],
        )

    def setup(self):
        for cg_name in self.options["cg_names"]:
            self.add_input(cg_name, val=np.nan, units="m")
        for mass_name in self.options["mass_names"]:
            self.add_input(mass_name, val=np.nan, units="kg")

        self.add_output("data:weight:aircraft_empty:mass", units="kg")
        self.add_output("data:weight:aircraft_empty:CG:x", units="m")

        self.declare_partials(
            of="data:weight:aircraft_empty:mass", wrt=self.options["mass_names"], method="exact"
        )
        self.declare_partials(of="data:weight:aircraft_empty:CG:x", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cgs = [inputs[cg_name][0] for cg_name in self.options["cg_names"]]
        masses = [inputs[mass_name][0] for mass_name in self.options["mass_names"]]

        weight_moment = np.dot(cgs, masses)
        outputs["data:weight:aircraft_empty:mass"] = np.sum(masses)
        x_cg_empty_aircraft = weight_moment / outputs["data:weight:aircraft_empty:mass"]
        outputs["data:weight:aircraft_empty:CG:x"] = x_cg_empty_aircraft

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cgs = [inputs[cg_name][0] for cg_name in self.options["cg_names"]]
        masses = [inputs[mass_name][0] for mass_name in self.options["mass_names"]]

        weight_moment = np.dot(cgs, masses)
        tot_mass = np.sum(masses)

        for cg_name, mass_name in zip(self.options["cg_names"], self.options["mass_names"]):

            partials["data:weight:aircraft_empty:mass", mass_name] = 1.0

            partials["data:weight:aircraft_empty:CG:x", cg_name] = inputs[mass_name] / tot_mass
            partials["data:weight:aircraft_empty:CG:x", mass_name] = (
                inputs[cg_name] * tot_mass - weight_moment
            ) / tot_mass ** 2.0
