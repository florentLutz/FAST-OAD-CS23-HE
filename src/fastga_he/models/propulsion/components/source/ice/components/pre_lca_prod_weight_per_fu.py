# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCAICEProdWeightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):
        ice_id = self.options["ice_id"]

        self.add_input(
            name="data:propulsion:he_power_train:ICE:" + ice_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Installed weight of the ICE engine",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ICE:" + ice_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Weight of the ICE required for a functional unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ice_id = self.options["ice_id"]

        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":mass_per_fu"] = (
            inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ice_id = self.options["ice_id"]

        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":mass_per_fu",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"]
        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":mass"]
