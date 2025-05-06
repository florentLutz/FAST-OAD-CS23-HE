# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCPMSMOperationalCost(om.ExplicitComponent):
    """
    Computation of the maintenance cost of the PMSM. For the default value of the average lifespan
    of the motor, the value is taken from :cite:`thonemann:2024` for short term technologies.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":cost_per_unit",
            units="USD",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":lifespan",
            units="yr",
            val=15.0,
            desc="Expected lifetime of the PMSM, typically around 15 year",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":operational_cost",
            units="USD/yr",
            val=1.0e3,
            desc="Annual maintenance cost per unit",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":operational_cost"] = (
            inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":cost_per_unit"]
            / inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":lifespan"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":operational_cost",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":cost_per_unit",
        ] = 1 / inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":lifespan"]
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":operational_cost",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":lifespan",
        ] = (
            -inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":cost_per_unit"]
            / inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":lifespan"] ** 2.0
        )
