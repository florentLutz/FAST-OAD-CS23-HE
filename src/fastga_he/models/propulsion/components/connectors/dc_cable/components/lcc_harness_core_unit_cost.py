# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om

COST_COPPER = 92556.8
COST_ALU = 6775.0


class LCCHarnessCoreUnitCost(om.ExplicitComponent):
    """
    Computation of the core material price per unit volume.
    """

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

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cost_per_volume",
            units="USD/m**3",
            val=6775.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_volume",
            wrt="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        material = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material"
        ]

        # Linear variation between densities of copper and aluminum to allow for easy partial
        # derivatives computation
        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_volume"
        ] = COST_ALU + (COST_COPPER - COST_ALU) * material

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_volume",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        ] = COST_COPPER - COST_ALU
