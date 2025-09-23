# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingSlotSection(om.ExplicitComponent):
    """
    Computation of single slot cross-section ares of the PMSM. The formula is obtained from
    equation (II-33) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
            val=np.nan,
            units="m",
            desc="Single stator slot height (radial)",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width",
            val=np.nan,
            units="m",
            desc="Single stator slot width (along the circumference)",
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section",
            units="m**2",
            val=0.0002,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"]
            * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
        ] = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width",
        ] = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"]
