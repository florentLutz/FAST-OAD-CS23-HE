# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingSlotSectionArea(om.ExplicitComponent):
    """
    Computation of single slot cross-section area of the SM PMSM.
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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
            val=np.nan,
            desc="The fraction between overall tooth length and stator bore circumference",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
            desc="Number of conductor slots on the motor stator",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section_area",
            units="m**2",
            val=0.0006,
            desc="Single stator slot section area",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        tooth_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]
        slot_height = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"]
        num_slots = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section_area"] = (
            np.pi
            * (1.0 - tooth_ratio)
            * (slot_height**2.0 + slot_height * bore_diameter)
            / num_slots
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        tooth_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio"]
        slot_height = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height"]
        num_slots = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section_area",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_height",
        ] = np.pi * (1.0 - tooth_ratio) * (2.0 * slot_height + bore_diameter) / num_slots

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section_area",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = np.pi * (1.0 - tooth_ratio) * slot_height / num_slots

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section_area",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tooth_ratio",
        ] = -np.pi * (slot_height**2.0 + slot_height * bore_diameter) / num_slots

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section_area",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
        ] = -(
            np.pi
            * (1.0 - tooth_ratio)
            * (slot_height**2.0 + slot_height * bore_diameter)
            / num_slots**2.0
        )
