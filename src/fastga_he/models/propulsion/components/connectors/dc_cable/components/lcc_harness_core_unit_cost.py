# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


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
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:density",
            val=8960.0,
            units="kg/m**3",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cost_per_volume",
            units="USD/m**3",
            val=6775.0,
        )

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        rho_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":properties:density"
        ]

        if rho_c < 8960.0:
            outputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_volume"
            ] = 6775.0  # USD/m^3
        else:
            outputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_volume"
            ] = 92556.8  # USD/m^3
