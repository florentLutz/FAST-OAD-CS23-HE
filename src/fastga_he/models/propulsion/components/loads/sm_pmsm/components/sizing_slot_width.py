# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingSlotWidth(om.ExplicitComponent):
    """
    Computation of single slot width of the SM PMSM.The formula is obtained from
    equation (III-55) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
            val=np.nan,
            desc="Number of conductor slots",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            val=np.nan,
            units="m",
            desc="Stator bore diameter of the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            val=np.nan,
            desc="The fraction between overall tooth length and stator bore circumference",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width",
            units="m",
            desc="Single stator slot width (along the circumference)",
            val=0.007,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        tooth_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]
        num_conductor_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width"] = (
            (1.0 - tooth_ratio) * np.pi * bore_diameter / num_conductor_slot
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        tooth_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]
        num_conductor_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = (1.0 - tooth_ratio) * np.pi / num_conductor_slot

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
        ] = -np.pi * bore_diameter / num_conductor_slot

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_width",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
        ] = -(1.0 - tooth_ratio) * np.pi * bore_diameter / num_conductor_slot**2.0
