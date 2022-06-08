# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class MotorScaling(om.ExplicitComponent):
    """Computation of scaling factor for cylindrical PMSM."""

    def initialize(self):
        # Reference motor : POWERPHASE HD 250

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "omega_max_ref",
            default=5500.0 / 60 * 2.0 * np.pi,
            desc="Max rotational speed of the reference motor in [rad/s]",
        )
        self.options.declare(
            "torque_peak_ref", default=900, desc="Max torque of the reference motor in [N*m]"
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":torque:peak",
            val=np.nan,
            units="N*m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":omega:peak",
            val=np.nan,
            units="rad/s",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            val=1.0,
        )
        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            val=1.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            wrt="data:propulsion:he_power_train:PMSM:" + motor_id + ":omega:peak",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            wrt=[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":omega:peak",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque:peak",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        torque_peak_ref = self.options["torque_peak_ref"]
        omega_max_ref = self.options["omega_max_ref"]

        torque_peak = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":torque:peak"]
        omega_max = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":omega:peak"]

        torque_peak_scaling = torque_peak / torque_peak_ref
        omega_peak_scaling = omega_max / omega_max_ref

        # Mechanical limit
        d_scaling = 1.0 / omega_peak_scaling

        # Demagnetization or iron saturation
        l_scaling = torque_peak_scaling * omega_peak_scaling ** 2.0

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"] = d_scaling
        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"] = l_scaling

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        torque_peak_ref = self.options["torque_peak_ref"]
        omega_max_ref = self.options["omega_max_ref"]

        torque_peak = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":torque:peak"]
        omega_max = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":omega:peak"]

        torque_peak_scaling = torque_peak / torque_peak_ref
        omega_peak_scaling = omega_max / omega_max_ref

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":omega:peak",
        ] = (
            -1.0 * omega_max_ref / omega_max ** 2.0
        )

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":omega:peak",
        ] = (
            2.0 * torque_peak_scaling * omega_peak_scaling / omega_max_ref
        )

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque:peak",
        ] = (
            omega_peak_scaling ** 2.0 / torque_peak_ref
        )
