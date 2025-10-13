# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingConductorWireSectionArea(om.ExplicitComponent):
    """
    Computation of the circular cross-section area of a single conductor wire in the stator slot.
    The formula is obtained from equation (II-33) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":conductor_section_area_per_slot",
            units="m**2",
            val=np.nan,
            desc="Conductor wires section area per stator slot",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":series_per_slot",
            val=np.nan,
            desc="Number of wire in series per stator slot",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":parallel_per_slot",
            val=np.nan,
            desc="Number of series wire turns in parallel per stator slot",
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area",
            units="m**2",
            val=1.0e-07,
            desc="Single conductor circular wire cross-section area",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        conductor_area_per_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":conductor_section_area_per_slot"
        ]
        num_series = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":series_per_slot"
        ]
        num_parallel = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":parallel_per_slot"
        ]

        outputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area"
        ] = conductor_area_per_slot / (num_series * num_parallel)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        conductor_area_per_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":conductor_section_area_per_slot"
        ]
        num_series = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":series_per_slot"
        ]
        num_parallel = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":parallel_per_slot"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area",
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":conductor_section_area_per_slot",
        ] = 1.0 / (num_series * num_parallel)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":series_per_slot",
        ] = -conductor_area_per_slot / (num_series**2.0 * num_parallel)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_circular_section_area",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":parallel_per_slot",
        ] = -conductor_area_per_slot / (num_series * num_parallel**2.0)
