# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMotorLossCoefficientScaling(om.ExplicitComponent):
    """
    Computation of loss coefficients scaling factor for cylindrical PMSM.

    Main losses considered in this model are :
    - Joules losses (alpha * T^2)
    - Hysteresis losses (beta * omega)
    - Eddy current losses (gamma * omega^2)

    Scaling of alpha coefficient is taken from :cite:`thauvin:2018`. Scaling of beta and gamma is
    obtained based on a regression on the EMRAX family efficiency maps. Regression can be seen in
    ..methodology.free_run_losses_scaling.
    """

    def initialize(self):
        # Reference motor : EMRAX 268

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
        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:gamma",
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
        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:gamma",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]

        alpha_scaling = d_scaling**-4 * l_scaling**-1
        beta_scaling = d_scaling**6.89 * l_scaling**-1.96
        gamma_scaling = d_scaling**6.08 * l_scaling**-1.60

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha"] = (
            alpha_scaling
        )
        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta"] = beta_scaling
        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:gamma"] = (
            gamma_scaling
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]

        # Partials of alpha
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
        ] = -4.0 * d_scaling**-5.0 * l_scaling**-1.0
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:alpha",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
        ] = -(d_scaling**-4.0) * l_scaling**-2.0

        # Partials of beta
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
        ] = 6.89 * d_scaling**5.89 * l_scaling**-1.96
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:beta",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
        ] = -1.96 * d_scaling**6.89 * l_scaling**-2.96

        # Partials of gamma
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:gamma",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
        ] = 6.08 * d_scaling**5.08 * l_scaling**-1.60
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:gamma",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
        ] = -1.60 * d_scaling**6.08 * l_scaling**-2.60
