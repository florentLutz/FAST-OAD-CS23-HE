# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

COPPER_RESISTIVITY = 1.68e-8  # Copper resistivity at 293.15K [Ohm·m]


class SizingReferenceResistance(om.ExplicitComponent):
    """
    Computation of conductor reference resistance of the SM PMSM at 293.15K. The formula is
    obtained from equation (II-64) in :cite:`touhami:2020`.
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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_cable_length",
            val=np.nan,
            units="m",
            desc="Single Conductor cable length in one slot",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":conductor_section_area_per_slot",
            val=np.nan,
            units="m**2",
            desc="Conductor wires section area per stator slot",
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":reference_conductor_resistance",
            units="ohm",
            val=9e-4,
            desc="The conductor's reference electric resistance at 293.15K",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        wire_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_cable_length"
        ]
        conductor_section_area_per_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":conductor_section_area_per_slot"
        ]
        num_conductor_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]

        outputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":reference_conductor_resistance"
        ] = COPPER_RESISTIVITY * num_conductor_slot * wire_length / conductor_section_area_per_slot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        wire_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_cable_length"
        ]
        conductor_section_area_per_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":conductor_section_area_per_slot"
        ]
        num_conductor_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":reference_conductor_resistance",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
        ] = COPPER_RESISTIVITY * wire_length / conductor_section_area_per_slot

        partials[
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":reference_conductor_resistance",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_cable_length",
        ] = COPPER_RESISTIVITY * num_conductor_slot / conductor_section_area_per_slot

        partials[
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":reference_conductor_resistance",
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":conductor_section_area_per_slot",
        ] = (
            -COPPER_RESISTIVITY
            * num_conductor_slot
            * wire_length
            / conductor_section_area_per_slot**2.0
        )
