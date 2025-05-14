# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDCSSPCCost(om.ExplicitComponent):
    """
    Computation of the DC SSPC cost. Estimated from the MSRP of the IGBT modules from:
    https://www.semikron-danfoss.com/.
    """

    def initialize(self):
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )

    def setup(self):
        dc_sspc_id = self.options["dc_sspc_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":price_factor",
            val=2.0,
            desc="The factor of the SSPC compared to the IGBT module",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max",
            val=np.nan,
            units="A",
            desc="Maximum current flowing through the SSPC",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":cost_per_unit",
            val=1000.0,
            units="USD",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]
        current = inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max"]
        price_factor = inputs[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":price_factor"
        ]

        outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":cost_per_unit"] = (
            price_factor * (1.21 * current + 83.8)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]
        current = inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max"]
        price_factor = inputs[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":price_factor"
        ]

        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":cost_per_unit",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max",
        ] = 1.21 * price_factor

        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":cost_per_unit",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":price_factor",
        ] = 1.21 * current + 83.8
