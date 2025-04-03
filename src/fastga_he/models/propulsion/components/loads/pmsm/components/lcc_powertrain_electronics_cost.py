# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCPTElectronicCost(om.ExplicitComponent):
    """
    Computation of the cost of the electronic consist with a branch of  electric or hybrid
    powertrain. Reference value obtained from :cite:`marciello:2024`.
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
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":electronic_cost_per_branch",
            units="USD",
            val=1e4,
            desc="Cost of the electronic cost per branch",
        )

        self.declare_partials(of="*", wrt="*", val=265.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":electronic_cost_per_branch"
        ] = 265.0 * inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_max"]
