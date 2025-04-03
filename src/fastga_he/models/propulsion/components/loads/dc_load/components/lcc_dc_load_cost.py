# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDCLoadCost(om.ExplicitComponent):
    """
    Computation of the cost of the DC loads. Regression model obtained based on the price of the
    products from:https://emrax.com/e-motors/.
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

        self.add_output(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cost_per_load",
            units="USD",
            val=1e4,
            desc="Cost of the DC load per unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aux_load_id = self.options["aux_load_id"]

        outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cost_per_load"] = (
            1876.1
            * np.exp(
                0.0062
                * inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max"]
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aux_load_id = self.options["aux_load_id"]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cost_per_load",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
        ] = 11.632 * np.exp(
            0.0062 * inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max"]
        )
