# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCAPlanetaryGearProdWeightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="planetary_gear_id",
            default=None,
            desc="Identifier of the planetary gear",
            allow_none=False,
        )

    def setup(self):
        planetary_gear_id = self.options["planetary_gear_id"]

        self.add_input(
            name="data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Mass of the planetary gear",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Mass of planetary gear required for a functional unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        planetary_gear_id = self.options["planetary_gear_id"]

        outputs[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass_per_fu"
        ] = (
            inputs["data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        planetary_gear_id = self.options["planetary_gear_id"]

        partials[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass_per_fu",
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"]
        partials[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass"]
