# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCPMSMCost(om.ExplicitComponent):
    """
    Computation of the cost of the motor including the electronics of the powertrain. The PMSM
    regression model obtained based on the price of the products from: https://emrax.com/e-motors/.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_max",
            units="kW",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":purchase_cost",
            units="USD",
            val=1e4,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]
        power_max = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_max"]

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":purchase_cost"] = (
            893.51 * np.exp(0.0281 * power_max)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        power_max = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_max"]

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":purchase_cost",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_max",
        ] = 25.108 * np.exp(0.0281 * power_max)
