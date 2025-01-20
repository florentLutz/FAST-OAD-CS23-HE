# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMotorTorqueConstant(om.ExplicitComponent):
    """Computation of the torque constant of a cylindrical PMSM."""

    def initialize(self):
        # Reference motor : EMRAX 268

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "torque_constant_ref",
            default=1.9,
            desc="Torque constant of the reference motor in [Nm/1Aph rms]",
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:simple_PMSM:"
            + motor_id
            + ":scaling:torque_constant",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":torque_constant",
            val=self.options["torque_constant_ref"],
            units="N*m/A",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":torque_constant",
            wrt="data:propulsion:he_power_train:simple_PMSM:"
            + motor_id
            + ":scaling:torque_constant",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        k_t_scaling = inputs[
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:torque_constant"
        ]

        torque_constant_ref = self.options["torque_constant_ref"]

        outputs["data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":torque_constant"] = (
            torque_constant_ref * k_t_scaling
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        torque_constant_ref = self.options["torque_constant_ref"]

        partials[
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":torque_constant",
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:torque_constant",
        ] = torque_constant_ref
