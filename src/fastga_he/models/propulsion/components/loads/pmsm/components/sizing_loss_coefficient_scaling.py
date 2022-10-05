# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class MotorLossCoefficientScaling(om.ExplicitComponent):
    """Computation of scaling factor for cylindrical PMSM."""

    def initialize(self):
        # Reference motor : POWERPHASE HD 250

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha",
            val=1.0,
        )
        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta",
            val=1.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha",
            wrt="*",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]

        alpha_scaling = d_scaling ** -4 * l_scaling ** -1
        beta_scaling = d_scaling ** 2 * l_scaling

        outputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha"
        ] = alpha_scaling
        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta"] = beta_scaling

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
        ] = (
            -4.0 * d_scaling ** -5.0 * l_scaling ** -1.0
        )

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
        ] = (
            -(d_scaling ** -4.0) * l_scaling ** -2.0
        )

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
        ] = (
            2.0 * d_scaling * l_scaling
        )

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
        ] = (
            d_scaling ** 2.0
        )
