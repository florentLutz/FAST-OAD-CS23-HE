# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDCLoadCost(om.ExplicitComponent):
    """
    Computation of the cost of the DC loads including the electronics of the powertrain.
    Load regression model obtained based on the price of the products
    from: https://emrax.com/e-motors/ and the reference electronic cost from :cite:`marciello:2024`.
    """

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
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
            units="kW",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":purchase_price",
            units="USD",
            val=0.0,
            desc="Cost of the DC load per unit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cost_per_unit",
            units="USD",
            val=1e4,
            desc="Cost of the DC load per unit including electronics",
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
            val=256.0,
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":purchase_price",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aux_load_id = self.options["aux_load_id"]

        outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cost_per_unit"] = (
            256.0 * inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max"]
            + inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":purchase_price"]
        )
