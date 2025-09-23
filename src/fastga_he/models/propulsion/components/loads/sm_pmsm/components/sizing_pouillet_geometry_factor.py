# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingPouilletGeometryFactor(om.ExplicitComponent):
    """
    Computation of conductor geometry factor for Pouillet's law for the PMSM. The formula is
    obtained from equation (II-64) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductors_number",
            val=np.nan,
            desc="Number of conductor slots",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_length",
            val=np.nan,
            units="m",
            desc="Electrical conductor cable length",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_section",
            val=np.nan,
            units="m**2",
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            units="m**-1",
            val=5.3e4,
            desc="Total length of the conductor wire divided by the wire cross-sectional area",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        wire_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_length"
        ]
        conductor_section = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_section"
        ]
        num_conductor_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductors_number"
        ]

        outputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor"
        ] = num_conductor_slot * wire_length / conductor_section

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        wire_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_length"
        ]
        conductor_section = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_section"
        ]
        num_conductor_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductors_number"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductors_number",
        ] = wire_length / conductor_section

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_length",
        ] = num_conductor_slot / conductor_section

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pouillet_geometry_factor",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_section",
        ] = -num_conductor_slot * wire_length / conductor_section**2.0
