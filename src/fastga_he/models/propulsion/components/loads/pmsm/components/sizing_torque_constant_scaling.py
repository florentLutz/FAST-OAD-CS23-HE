# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMotorTorqueConstantScaling(om.ExplicitComponent):
    """
    Computation of the torque constant of a cylindrical PMSM.

    Scaling law established based on the definition of the Joules losses and torque constant.
    Verified with the data from the EMRAX family.
    """

    def initialize(self):
        # Reference motor : EMRAX 268

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:torque_constant",
            val=1.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:torque_constant",
            wrt=[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        resistance_scaling = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance"
        ]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]
        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]

        k_t_scaling = resistance_scaling ** 0.5 * d_scaling ** 2.0 * l_scaling ** 0.5

        outputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:torque_constant"
        ] = k_t_scaling

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        resistance_scaling = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance"
        ]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]
        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:torque_constant",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance",
        ] = (
            0.5 * resistance_scaling ** -0.5 * d_scaling ** 2.0 * l_scaling ** 0.5
        )
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:torque_constant",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
        ] = (
            2.0 * resistance_scaling ** 0.5 * d_scaling * l_scaling ** 0.5
        )
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:torque_constant",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
        ] = (
            0.5 * resistance_scaling ** 0.5 * d_scaling ** 2.0 * l_scaling ** -0.5
        )
