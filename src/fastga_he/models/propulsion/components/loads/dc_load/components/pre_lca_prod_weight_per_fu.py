#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCADCAuxLoadProdWeightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )

    def setup(self):
        aux_load_id = self.options["aux_load_id"]

        self.add_input(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass",
            val=10.0,
            units="kg",
            desc="Mass of the auxiliary load",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Weight of the auxiliary load required for a functional unit",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aux_load_id = self.options["aux_load_id"]

        outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass_per_fu"] = (
            inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aux_load_id = self.options["aux_load_id"]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass_per_fu",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass",
        ] = 1.0 * inputs["data:environmental_impact:aircraft_per_fu"]
        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = 1.0 * inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass"]
