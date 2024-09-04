# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

POWER_COPPER = 2.85
POWER_ALU = 3.28


class SizingResistancePerLength(om.ExplicitComponent):
    """Computation of max current per cable ."""

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
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
            val=1.0,
            desc="1.0 for copper, 0.0 for aluminium",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:section",
            units="mm*mm",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:resistance_per_length",
            units="ohm/km",
            val=1.21,
        )

        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]
        area = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section"
        ]
        material = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material"
        ]

        resistance_per_length = np.exp((POWER_ALU + (POWER_COPPER - POWER_ALU) * material)) / area

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:resistance_per_length"
        ] = resistance_per_length

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        area = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section"
        ]
        material = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material"
        ]

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:resistance_per_length",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section",
        ] = -np.exp((POWER_ALU + (POWER_COPPER - POWER_ALU) * material)) / area**2.0

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:resistance_per_length",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        ] = (
            (POWER_COPPER - POWER_ALU)
            * np.exp((POWER_ALU + (POWER_COPPER - POWER_ALU) * material))
            / area
        )
