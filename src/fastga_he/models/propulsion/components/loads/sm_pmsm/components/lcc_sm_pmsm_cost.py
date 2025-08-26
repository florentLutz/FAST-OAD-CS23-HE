# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCSMPMSMCost(om.ExplicitComponent):
    """
    Computation of the cost of the motor including the electronics of the powertrain. The PMSM
    regression model obtained based on the price of the products from: https://emrax.com/e-motors/.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":torque_rating",
            val=np.nan,
            units="kN*m",
            desc="Max continuous torque of the motor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":rpm_rating",
            val=np.nan,
            units="min**-1",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":purchase_cost",
            units="USD",
            val=1e4,
            desc="Unit purchase cost of the PMS motor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        torque_rating = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":torque_rating"
        ]
        rpm_rating = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":rpm_rating"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":purchase_cost"] = (
            -17120 + 5730 * np.log(2.0 * np.pi * torque_rating * rpm_rating / 60.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        torque_rating = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":torque_rating"
        ]
        rpm_rating = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":rpm_rating"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":purchase_cost",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":rpm_rating",
        ] = 5730.0 / rpm_rating

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":purchase_cost",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":torque_rating",
        ] = 5730.0 / torque_rating
