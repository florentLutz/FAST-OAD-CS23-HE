# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingPouilletGeometryFactor(om.ExplicitComponent):
    """
    Computation of conductor geometry factor for Pouillet's law for the SM PMSM. The formula is
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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":parallel_per_slot",
            val=np.nan,
            desc="Number of series wire turns in parallel per stator slot",
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
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            units="m**-1",
            val=44560.8,
            desc="Total length of the conductor wire divided by the wire cross-sectional area",
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
        num_parallel = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":parallel_per_slot"
        ]

        outputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor"
        ] = num_conductor_slot * wire_length / (conductor_section_area_per_slot * num_parallel)

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
        num_parallel = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":parallel_per_slot"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
        ] = wire_length / (conductor_section_area_per_slot * num_parallel)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_cable_length",
        ] = num_conductor_slot / (conductor_section_area_per_slot * num_parallel)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":conductor_section_area_per_slot",
        ] = (
            -num_conductor_slot
            * wire_length
            / (conductor_section_area_per_slot**2.0 * num_parallel)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":parallel_per_slot",
        ] = (
            -num_conductor_slot
            * wire_length
            / (conductor_section_area_per_slot * num_parallel**2.0)
        )
