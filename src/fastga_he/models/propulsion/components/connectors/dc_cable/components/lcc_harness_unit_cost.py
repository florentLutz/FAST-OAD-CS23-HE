# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

GROSS_MARGIN = 0.45


class LCCHarnessUnitCost(om.ExplicitComponent):
    """
    Computation of cable purchase cost based on the raw material price and gross margin of the
    cable manufacturing industry.
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
            + ":conductor:section",
            units="m**2",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:section",
            units="m**2",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cost_per_volume",
            units="USD/m**3",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":shield:section",
            units="m**2",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":sheath:section",
            units="m**2",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
            units="m",
            val=np.nan,
            desc="Length of the harness",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":purchase_cost",
            units="USD",
            val=200.0,
            desc="Unit purchase cost of the DC cable harness",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        cs_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section"
        ]
        cs_in = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":insulation:section"
        ]
        cs_shield = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":shield:section"
        ]
        cs_sheath = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:section"
        ]
        cost_core = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_volume"
        ]
        length = inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
        cost_ins = 86000.0  # USD/m^3

        c_conductor = cost_core * cs_c
        c_i = cost_ins * cs_in
        c_shield = cost_core * cs_shield
        c_sheath = cost_ins * cs_sheath
        c_cable = (c_conductor + c_i + c_shield + c_sheath) * length

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":purchase_cost"
        ] = c_cable / (1.0 - GROSS_MARGIN)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        cs_c = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section"
        ]
        cs_in = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":insulation:section"
        ]
        cs_shield = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":shield:section"
        ]
        cs_sheath = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:section"
        ]
        length = inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"]
        cost_core = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_volume"
        ]
        cost_ins = 86000.0  # USD/m^3

        c_conductor = cs_c * cost_core
        c_i = cs_in * cost_ins
        c_shield = cs_shield * cost_core
        c_sheath = cs_sheath * cost_ins

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":purchase_cost",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
        ] = (c_conductor + c_i + c_shield + c_sheath) / (1.0 - GROSS_MARGIN)

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":purchase_cost",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section",
        ] = length * cost_core / (1.0 - GROSS_MARGIN)

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":purchase_cost",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":insulation:section",
        ] = length * cost_ins / (1.0 - GROSS_MARGIN)

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":purchase_cost",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":shield:section",
        ] = length * cost_core / (1.0 - GROSS_MARGIN)

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":purchase_cost",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":sheath:section",
        ] = length * cost_ins / (1.0 - GROSS_MARGIN)

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":purchase_cost",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cost_per_volume",
        ] = length * (cs_c + cs_shield) / (1.0 - GROSS_MARGIN)
