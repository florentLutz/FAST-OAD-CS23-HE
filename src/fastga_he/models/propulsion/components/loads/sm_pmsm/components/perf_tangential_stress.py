# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTangentialStree(om.ExplicitComponent):
    """
    Computation of the rotor surface tangential stress due to electromagnetism. The formula is
    obtained from equation (II-4) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            units="m",
            desc="The length of electromagnetism active part of SM PMSM",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":max_electromagnetic_torque",
            units="N*m",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress",
            units="Pa",
            desc="The length of electromagnetism active part of SM PMSM",
            val=0.169,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        d_rotor = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"]
        max_torque_em = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":max_electromagnetic_torque"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress"] = (
            2.0 * max_torque_em / (np.pi * d_rotor**2.0 * active_length)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        d_rotor = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"]
        max_torque_em = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":max_electromagnetic_torque"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":max_electromagnetic_torque",
        ] = 2.0 / (np.pi * d_rotor**2.0 * active_length)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = -2.0 * max_torque_em / (np.pi * d_rotor**2.0 * active_length**2.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = -4.0 * max_torque_em / (np.pi * d_rotor**3.0 * active_length)
