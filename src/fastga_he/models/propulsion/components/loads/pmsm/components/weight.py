# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class MotorWeight(om.ExplicitComponent):
    def initialize(self):
        # Reference motor : POWERPHASE HD 250

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare("mass_ref", default=85.0, desc="Mass of the reference motor in [kg]")

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
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":mass",
            val=self.options["mass_ref"],
            units="kg",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":mass",
            wrt=[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]

        m_ref = self.options["mass_ref"]

        m_scaling = d_scaling ** 2.0 * l_scaling

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":mass"] = m_ref * m_scaling

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]

        m_ref = self.options["mass_ref"]

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":mass",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
        ] = (
            m_ref * 2.0 * d_scaling * l_scaling
        )
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":mass",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
        ] = (
            m_ref * d_scaling ** 2.0
        )
