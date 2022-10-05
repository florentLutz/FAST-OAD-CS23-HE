# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class MotorLossCoefficient(om.ExplicitComponent):
    """Computation of scaling factor for cylindrical PMSM."""

    def initialize(self):
        # Reference motor : POWERPHASE HD 250

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "alpha_ref",
            default=0.042,
            desc="Joule loss coefficient for the reference motor",
        )
        self.options.declare(
            "beta_ref", default=0.48, desc="Iron loss coefficient for the reference motor"
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha",
            val=1.0,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta",
            val=1.0,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":alpha",
            val=1.0,
        )
        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":beta",
            val=1.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":alpha",
            wrt="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":beta",
            wrt="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        alpha_ref = self.options["alpha_ref"]
        beta_ref = self.options["beta_ref"]

        alpha_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha"]
        beta_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta"]

        alpha = alpha_ref * alpha_scaling
        beta = beta_ref * beta_scaling

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":alpha"] = alpha
        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":beta"] = beta

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        alpha_ref = self.options["alpha_ref"]
        beta_ref = self.options["beta_ref"]

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":alpha",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha",
        ] = alpha_ref

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":beta",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta",
        ] = beta_ref
