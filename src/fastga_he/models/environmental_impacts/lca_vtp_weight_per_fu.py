# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCAVTPWeightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="airframe_material",
            default="aluminium",
            desc="Material used for the airframe which include wing, fuselage, HTP and VTP. LG will"
            " always be in aluminium and flight controls in steel",
            allow_none=False,
            values=["aluminium", "composite"],
        )

    def setup(self):
        self.add_input("data:weight:airframe:vertical_tail:mass", val=np.nan, units="kg")
        self.add_input("data:environmental_impact:aircraft_per_fu", val=np.nan)

        if self.options["airframe_material"] == "aluminium":
            self.add_input(
                "data:environmental_impact:buy_to_fly:metallic",
                val=1.0,
                desc="Ratio of the amount of material purchased to the one that actually flies. "
                "Typical value for metallic material is between 5 and 10",
            )
        else:
            self.add_input(
                "data:environmental_impact:buy_to_fly:composite",
                val=1.0,
                desc="Ratio of the amount of material purchased to the one that actually flies. "
                "Typical value for composite material is between 1 and 2",
            )

        self.add_output("data:weight:airframe:vertical_tail:mass_per_fu", val=1e-6, units="kg")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options["airframe_material"] == "aluminium":
            buy_to_fly = inputs["data:environmental_impact:buy_to_fly:metallic"]
        else:
            buy_to_fly = inputs["data:environmental_impact:buy_to_fly:composite"]

        outputs["data:weight:airframe:vertical_tail:mass_per_fu"] = (
            inputs["data:weight:airframe:vertical_tail:mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * buy_to_fly
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if self.options["airframe_material"] == "aluminium":
            buy_to_fly = inputs["data:environmental_impact:buy_to_fly:metallic"]

            partials[
                "data:weight:airframe:vertical_tail:mass_per_fu",
                "data:environmental_impact:buy_to_fly:metallic",
            ] = (
                inputs["data:weight:airframe:vertical_tail:mass"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )
        else:
            buy_to_fly = inputs["data:environmental_impact:buy_to_fly:composite"]

            partials[
                "data:weight:airframe:vertical_tail:mass_per_fu",
                "data:environmental_impact:buy_to_fly:composite",
            ] = (
                inputs["data:weight:airframe:vertical_tail:mass"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )

        partials[
            "data:weight:airframe:vertical_tail:mass_per_fu",
            "data:weight:airframe:vertical_tail:mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * buy_to_fly
        partials[
            "data:weight:airframe:vertical_tail:mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:weight:airframe:vertical_tail:mass"] * buy_to_fly
