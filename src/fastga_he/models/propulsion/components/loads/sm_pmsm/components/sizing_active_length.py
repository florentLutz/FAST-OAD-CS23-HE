# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingActiveLength(om.ExplicitComponent):
    """
    Computation of the length in the SM PMSM that is electromagnetically active. The formula is
    obtained from equation (II-44) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            val=np.nan,
            units="m",
            desc="Stator bore diameter of the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            units="m",
            desc="The length of electromagnetism active part of SM PMSM",
            val=0.169,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"]
            / inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = 1.0 / inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient",
        ] = (
            -inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"]
            / inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient"]
            ** 2.0
        )
