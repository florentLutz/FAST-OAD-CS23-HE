# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCAHarnessProdLengthPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):
        harness_id = self.options["harness_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
            units="m",
            val=np.nan,
            desc="Length of the harness",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
            val=1.0,
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length_per_fu",
            units="m",
            val=1e-6,
            desc="Length of the cable required for a functional unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length_per_fu"
        ] = (
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length_per_fu",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
        )
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = (
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
        )
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length_per_fu",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
        ] = (
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )
